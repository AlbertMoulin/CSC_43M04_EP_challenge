import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer # Nécessite la bibliothèque transformers
import torch.nn.functional as F # For weighted sum normalization

# You might want to define the maximum token length here or in the configuration Hydra
MAX_TOKEN_LENGTH = 128

class ImageBranch(nn.Module):
    """
    Branche de traitement de l'image : DinoV2 + MLP.
    """
    def __init__(self, dinov2_model, mlp_layers):
        super().__init__()
        self.backbone = dinov2_model # Le modèle DinoV2 (sans la tête)
        layers = []
        # La dimension d'entrée du MLP est la dimension de sortie de DinoV2
        # Assumes the last layer of backbone is a norm layer with a normalized_shape attribute
        if hasattr(dinov2_model, 'norm') and hasattr(dinov2_model.norm, 'normalized_shape'):
             input_dim = dinov2_model.norm.normalized_shape[0]
        elif hasattr(dinov2_model, 'config') and hasattr(dinov2_model.config, 'hidden_size'):
             input_dim = dinov2_model.config.hidden_size
        else:
             # Fallback or raise error if backbone output dim is unknown
             # For dinov2_vitb14_reg, it's typically 768
             print("Warning: Could not determine DinoV2 backbone output dimension automatically. Assuming 768.")
             input_dim = 768


        # Construction des couches du MLP
        for output_dim in mlp_layers[:-1]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU()) # Utilisation de ReLU comme fonction d'activation
            input_dim = output_dim

        # Couche de sortie (prediction dimension, typically 1, but could be different for concatenation)
        layers.append(nn.Linear(input_dim, mlp_layers[-1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, image):
        # Obtenir les caractéristiques de l'image via DinoV2
        # DinoV2 forward can return different things; assume it returns the features
        # You might need to adjust this based on the specific DinoV2 implementation used by torch.hub
        # For many models, the features are the output of the last layer before the head
        x = self.backbone(image) # This might return a tuple or object depending on the model

        # Assuming the output 'x' is the tensor of features
        # If DinoV2 returns a tuple like (features, patch_features), you might need x = x[0]
        # If it returns an object, you might need x = x['last_hidden_state'] or similar.
        # For dinov2_vitb14_reg from torch.hub, the forward returns the features directly.

        # Pass the features to the MLP
        x = self.mlp(x)
        return x

class TitleBranch(nn.Module):
    """
    Branche de traitement du titre : Modèle Transformer + MLP.
    """
    def __init__(self, text_model_name, mlp_layers):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(text_model_name)

        # The input dimension of the MLP will be the hidden size of the text model's output.
        # We assume we take the output of the first token ([CLS] for BERT-like models)
        text_feature_dim = self.backbone.config.hidden_size

        layers = []
        input_dim = text_feature_dim

        # Construction des couches du MLP
        for output_dim in mlp_layers[:-1]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU()) # Utilisation de ReLU comme fonction d'activation
            input_dim = output_dim

        # Couche de sortie (prediction dimension)
        layers.append(nn.Linear(input_dim, mlp_layers[-1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):
        # Get the hidden states from the text model
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Take the hidden state of the first token ([CLS]) for sequence representation
        text_features = outputs.last_hidden_state[:, 0, :]
        # Pass the features through the MLP
        x = self.mlp(text_features)
        return x

class ChannelBranch(nn.Module):
    """
    Branche de traitement du canal YouTube : Embedding + MLP.
    """
    def __init__(self, num_channels, embedding_dim, mlp_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_channels, embedding_dim)

        layers = []
        input_dim = embedding_dim

        # Construction des couches du MLP
        for output_dim in mlp_layers[:-1]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU()) # Using ReLU as activation function
            input_dim = output_dim

        # Output layer (prediction dimension)
        layers.append(nn.Linear(input_dim, mlp_layers[-1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, channel_ids):
        embedded = self.embedding(channel_ids)
        x = self.mlp(embedded)
        return x

class DateBranch(nn.Module):
    """
    Branche de traitement de la date : MLP.
    Assumes input is a numerical representation like timestamp.
    """
    def __init__(self, input_dim, mlp_layers):
        super().__init__()
        layers = []
        current_dim = input_dim

        # Construction des couches du MLP
        for output_dim in mlp_layers[:-1]:
            layers.append(nn.Linear(current_dim, output_dim))
            layers.append(nn.ReLU()) # Using ReLU as activation function
            current_dim = output_dim

        # Output layer (prediction dimension)
        layers.append(nn.Linear(current_dim, mlp_layers[-1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, date_features):
        # Ensure date_features has the correct shape if it's a single timestamp
        # It should be [batch_size, input_dim]
        if date_features.ndim == 1:
            date_features = date_features.unsqueeze(1) # Make it [batch_size, 1] if input_dim is 1

        x = self.mlp(date_features)
        return x

class CombinedModel(nn.Module):
    """
    Modèle combiné pour la prédiction de vues à partir de l'image, du titre,
    du canal YouTube et de la date.
    Permet la concaténation+MLP ou la somme pondérée des sorties des branches.
    """
    def __init__(self,
                 image_mlp_layers,
                 text_model_name,
                 title_mlp_layers,
                 num_channels, # Number of unique channels
                 channel_embedding_dim, # Embedding dimension for channels
                 channel_mlp_layers,
                 date_mlp_layers,
                 combination_mode='concatenate_mlp', # 'concatenate_mlp' or 'weighted_sum'
                 final_mlp_layers=None, # Layers for the final MLP in concatenation mode
                 freeze_dinov2=True,
                 freeze_text_model=True,
                 max_token_length=MAX_TOKEN_LENGTH):

        super().__init__()

        self.combination_mode = combination_mode
        self.max_token_length = max_token_length

        # --- Image Branch ---
        dinov2_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        dinov2_backbone.head = nn.Identity() # Remove default head
        if freeze_dinov2:
            for param in dinov2_backbone.parameters():
                param.requires_grad = False
        self.image_branch = ImageBranch(dinov2_backbone, image_mlp_layers)

        # --- Title Branch ---
        self.title_branch = TitleBranch(text_model_name, title_mlp_layers)
        if freeze_text_model:
             # Freeze the backbone of the text model
             for param in self.title_branch.backbone.parameters():
                 param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # --- Channel Branch ---
        self.channel_branch = ChannelBranch(num_channels, channel_embedding_dim, channel_mlp_layers)

        # --- Date Branch ---
        # Assumes date input is a single timestamp, so input_dim is 1
        date_input_dim = 1 # Change if using more date features
        self.date_branch = DateBranch(date_input_dim, date_mlp_layers)


        # --- Combination Layer(s) ---
        if self.combination_mode == 'concatenate_mlp':
            if final_mlp_layers is None:
                raise ValueError("final_mlp_layers must be provided for 'concatenate_mlp' mode")

            # Calculate input dimension for the final MLP
            # This requires knowing the output dimension of each branch's final layer
            # Assuming the last layer of each branch's MLP determines its output dim
            image_output_dim = image_mlp_layers[-1]
            title_output_dim = title_mlp_layers[-1]
            channel_output_dim = channel_mlp_layers[-1]
            date_output_dim = date_mlp_layers[-1]

            concat_input_dim = image_output_dim + title_output_dim + channel_output_dim + date_output_dim

            final_layers = []
            input_dim = concat_input_dim
            for output_dim in final_mlp_layers[:-1]:
                final_layers.append(nn.Linear(input_dim, output_dim))
                final_layers.append(nn.ReLU()) # Using ReLU
                input_dim = output_dim
            # Final output layer (predicting views, so dimension 1)
            final_layers.append(nn.Linear(input_dim, final_mlp_layers[-1])) # final_mlp_layers[-1] should be 1

            self.final_combiner = nn.Sequential(*final_layers)

        elif self.combination_mode == 'weighted_sum':
            # For weighted sum, all branches should ideally output a scalar (dim 1)
            # Check if the last layer of each branch outputs dimension 1
            if not (image_mlp_layers[-1] == 1 and title_mlp_layers[-1] == 1 and
                    channel_mlp_layers[-1] == 1 and date_mlp_layers[-1] == 1):
                 print("Warning: Weighted sum mode expects branch MLPs to output dimension 1.")
                 # You might add projection layers here if branches output different dimensions
                 # For simplicity in this example, we assume the last layer is designed to output 1

            # Learnable weights for each branch
            self.weight_image = nn.Parameter(torch.tensor(0.25))
            self.weight_title = nn.Parameter(torch.tensor(0.25))
            self.weight_channel = nn.Parameter(torch.tensor(0.25))
            self.weight_date = nn.Parameter(torch.tensor(0.25))

             # Optional: ensure weights sum to 1 (can be done with softmax in forward)
        else:
            raise ValueError(f"Unknown combination_mode: {combination_mode}")


    def forward(self, batch):
        # Assume batch contains 'image', 'title', 'channel_id', and 'date_timestamp'
        image = batch["image"]
        titles = batch["title"] # List/tuple of raw title strings
        channel_ids = batch["channel_id"] # Tensor of channel integer IDs
        date_timestamps = batch["date_timestamp"] # Tensor of date timestamps

        # --- Process Image ---
        image_output = self.image_branch(image)

        # --- Process Title : Tokenization ---
        encoded_titles = self.tokenizer(
            list(titles), # Convert the list/tuple of titles to a list
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors='pt' # Return PyTorch tensors
        )

        # Send text tensors to the same device as the model
        input_ids = encoded_titles['input_ids'].to(image.device)
        attention_mask = encoded_titles['attention_mask'].to(image.device)

        # Process Title
        title_output = self.title_branch(input_ids, attention_mask)

        # --- Process Channel ---
        channel_output = self.channel_branch(channel_ids)

        # --- Process Date ---
        date_output = self.date_branch(date_timestamps)


        # --- Combine Predictions ---
        if self.combination_mode == 'concatenate_mlp':
            # Concatenate outputs from all branches
            combined_features = torch.cat(
                (image_output, title_output, channel_output, date_output),
                dim=1
            )
            # Pass through the final MLP
            final_prediction = self.final_combiner(combined_features)

        elif self.combination_mode == 'weighted_sum':
            # Ensure outputs are scalar for weighted sum (assuming last layer of branches output dim 1)
            # If not, you would need to project them to dim 1 here before summing
            image_pred = image_output.squeeze(-1) if image_output.ndim > 1 else image_output
            title_pred = title_output.squeeze(-1) if title_output.ndim > 1 else title_output
            channel_pred = channel_output.squeeze(-1) if channel_output.ndim > 1 else channel_output
            date_pred = date_output.squeeze(-1) if date_output.ndim > 1 else date_output


            # Apply softmax to weights to ensure they sum to 1 (optional but can be helpful)
            # weights = F.softmax(torch.stack([self.weight_image, self.weight_title,
            #                                  self.weight_channel, self.weight_date]), dim=0)
            # w_image, w_title, w_channel, w_date = weights

            # Simple weighted sum
            final_prediction = (self.weight_image * image_pred +
                                self.weight_title * title_pred +
                                self.weight_channel * channel_pred +
                                self.weight_date * date_pred)

            final_prediction = final_prediction.unsqueeze(1) # Add the output dimension back


        return final_prediction