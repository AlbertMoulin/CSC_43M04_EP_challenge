import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import datetime
import re

# Maximum token length for text processing
MAX_TOKEN_LENGTH = 128

class ImageBranch(nn.Module):
    """
    Image processing branch: DinoV2 + MLP.
    """
    def __init__(self, dinov2_model, mlp_layers):
        super().__init__()
        self.backbone = dinov2_model # DinoV2 model (without the head)
        layers = []
        # Input dimension for MLP is DinoV2's output dimension
        input_dim = dinov2_model.norm.normalized_shape[0] # Should be 768 for dinov2_vitb14_reg

        # Building MLP layers
        for output_dim in mlp_layers[:-1]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = output_dim

        # Output layer (single prediction)
        layers.append(nn.Linear(input_dim, mlp_layers[-1])) # mlp_layers[-1] should be 1

        self.mlp = nn.Sequential(*layers)

    def forward(self, image):
        # Get image features via DinoV2
        x = self.backbone(image)
        # Pass features through MLP
        x = self.mlp(x)
        return x

class TextBranch(nn.Module):
    """
    Text processing branch: Transformer model + MLP.
    Note: Tokenization is handled in the CombinedModel forward method.
    """
    def __init__(self, text_model_name, mlp_layers):
        super().__init__()
        # Load pre-trained text model
        self.backbone = AutoModel.from_pretrained(text_model_name, trust_remote_code=True)

        # Input dimension for MLP is the hidden size of text model output
        text_feature_dim = self.backbone.config.hidden_size # Ex: 768 for bert-base-uncased

        layers = []
        input_dim = text_feature_dim

        # Building MLP layers
        for output_dim in mlp_layers[:-1]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = output_dim

        # Output layer (single prediction)
        layers.append(nn.Linear(input_dim, mlp_layers[-1])) # mlp_layers[-1] should be 1

        self.mlp = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):
        # Get hidden states from text model
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Take hidden state of first token ([CLS]) for sequence representation
        text_features = outputs.last_hidden_state[:, 0, :]
        # Pass features through MLP
        x = self.mlp(text_features)
        return x

class MetadataBranch(nn.Module):
    """
    Metadata processing branch: Handles date and channel information.
    """
    def __init__(self, mlp_layers, num_channels=1000):
        super().__init__()
        
        # Embedding for channel ID
        self.channel_embedding = nn.Embedding(num_channels, 64)
        
        # Features: 
        # - 5 date features (year, month, day, day_of_week, hour)
        # - 64-dim channel embedding
        metadata_dim = 5 + 64
        
        layers = []
        input_dim = metadata_dim

        # Building MLP layers
        for output_dim in mlp_layers[:-1]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = output_dim

        # Output layer (single prediction)
        layers.append(nn.Linear(input_dim, mlp_layers[-1])) # mlp_layers[-1] should be 1

        self.mlp = nn.Sequential(*layers)

    def forward(self, date_features, channel_ids):
        # Get channel embeddings
        channel_emb = self.channel_embedding(channel_ids)
        
        # Concatenate date features and channel embeddings
        metadata_features = torch.cat([date_features, channel_emb], dim=1)
        
        # Pass features through MLP
        x = self.mlp(metadata_features)
        return x

class CombinedModel(nn.Module):
    """
    Combined model for predicting views from image, text, date and channel.
    """
    def __init__(
        self, 
        image_mlp_layers, 
        text_model_name, 
        text_mlp_layers, 
        metadata_mlp_layers,
        num_channels=1000,
        freeze_dinov2=True, 
        freeze_text_model=True, 
        max_token_length=MAX_TOKEN_LENGTH
    ):
        super().__init__()
        
        # Image branch setup: Load DinoV2 and create ImageBranch
        dinov2_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        dinov2_backbone.head = nn.Identity() # Remove default head
        if freeze_dinov2:
            for param in dinov2_backbone.parameters():
                param.requires_grad = False
        self.image_branch = ImageBranch(dinov2_backbone, image_mlp_layers)

        # Text branch setup: Create TextBranch
        self.text_branch = TextBranch(text_model_name, text_mlp_layers)
        if freeze_text_model:
            # Freeze text model backbone
            for param in self.text_branch.backbone.parameters():
                param.requires_grad = False

        # Metadata branch setup: Create MetadataBranch
        self.metadata_branch = MetadataBranch(metadata_mlp_layers, num_channels)

        # Tokenizer for processing raw text
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.max_token_length = max_token_length

        # Learnable weights for combination
        self.weight_image = nn.Parameter(torch.tensor(0.33))
        self.weight_text = nn.Parameter(torch.tensor(0.33))
        self.weight_metadata = nn.Parameter(torch.tensor(0.34))

    def _parse_date(self, date_str):
        """
        Parse date string into numerical features.
        Handles multiple formats including ISO format with timezone.
        Returns tensor with [year, month, day, day_of_week, hour]
        """
        try:
            # Try to handle full ISO format with time and timezone
            # First remove the timezone part if it exists
            if '+' in date_str:
                date_str = date_str.split('+')[0]
            
            # Try different date formats
            try:
                # Try format with time: "YYYY-MM-DD HH:MM:SS"
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    # Try ISO format: "YYYY-MM-DDTHH:MM:SS"
                    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    # Try just date: "YYYY-MM-DD"
                    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        
            # Extract features (normalized)
            year = (date_obj.year - 2005) / 20  # Normalize years since YouTube's founding
            month = (date_obj.month - 1) / 11   # 0-11
            day = (date_obj.day - 1) / 30       # 0-30
            day_of_week = date_obj.weekday() / 6  # 0-6
            hour = date_obj.hour / 23           # 0-23
            
            return torch.tensor([year, month, day, day_of_week, hour], dtype=torch.float32)
        
        except Exception as e:
            # If all parsing fails, return default values
            print(f"Error parsing date '{date_str}': {e}")
            return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    def forward(self, batch):
        # Extract components from batch
        image = batch["image"]
        raw_text = batch["text"]
        
        # Get date and channel features
        date_features = []
        for date_str in batch["date"]:
            date_features.append(self._parse_date(date_str))
        date_features = torch.stack(date_features).to(image.device)
        
        # Channel IDs - we expect these to be already converted to indices in dataset
        channel_ids = batch["channel_id"].to(image.device)

        # --- Text processing: Tokenization ---
        encoded_text = self.tokenizer(
            list(raw_text),
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors='pt'
        )

        # Send text tensors to same device as model
        input_ids = encoded_text['input_ids'].to(image.device)
        attention_mask = encoded_text['attention_mask'].to(image.device)

        # Get predictions from each branch
        image_prediction = self.image_branch(image)
        text_prediction = self.text_branch(input_ids, attention_mask)
        metadata_prediction = self.metadata_branch(date_features, channel_ids)

        # Normalize weights to sum to 1 using softmax
        weights = torch.softmax(
            torch.stack([self.weight_image, self.weight_text, self.weight_metadata]), 
            dim=0
        )
        
        # Combine predictions using normalized weights
        combined_prediction = (
            weights[0] * image_prediction + 
            weights[1] * text_prediction + 
            weights[2] * metadata_prediction
        )

        return combined_prediction