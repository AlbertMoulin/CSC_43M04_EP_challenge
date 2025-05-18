import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, CLIPModel
import datetime
import re

# Maximum token length for text processing
MAX_TOKEN_LENGTH = 128

class CLIPImageBranch(nn.Module):
    """
    Image processing branch using CLIP for better thumbnail attractiveness assessment.
    """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", mlp_layers=[1024, 512, 256, 1], freeze_clip=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name).vision_model
        
        # Freeze CLIP parameters if specified
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
        
        # Get CLIP image embedding dimension
        clip_dim = self.clip.config.hidden_size  # 768 for ViT-B/32
        
        # Building MLP layers
        layers = []
        input_dim = clip_dim
        
        for output_dim in mlp_layers[:-1]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.GELU())  # Using GELU activation as in CLIP/BART
            layers.append(nn.Dropout(0.1))
            input_dim = output_dim
            
        # Output layer
        layers.append(nn.Linear(input_dim, mlp_layers[-1]))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, image):
        # Extract image features from CLIP
        outputs = self.clip(image)
        image_embeds = outputs.pooler_output  # Get the [CLS] token representation
        
        # Pass through MLP
        x = self.mlp(image_embeds)
        return x

class BARTTextBranch(nn.Module):
    """
    Text processing branch using BART for better title engagement assessment.
    """
    def __init__(self, bart_model_name="facebook/bart-base", mlp_layers=[1024, 512, 256, 1], freeze_bart=True):
        super().__init__()
        self.bart = AutoModel.from_pretrained(bart_model_name)
        
        # Freeze BART parameters if specified
        if freeze_bart:
            for param in self.bart.parameters():
                param.requires_grad = False
        
        # Get BART hidden dimension
        bart_dim = self.bart.config.d_model  # 768 for bart-base
        
        # Building MLP layers
        layers = []
        input_dim = bart_dim
        
        for output_dim in mlp_layers[:-1]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            input_dim = output_dim
            
        # Output layer
        layers.append(nn.Linear(input_dim, mlp_layers[-1]))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, input_ids, attention_mask):
        # Extract text features from BART
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        # Use the encoder's pooled output for classification
        text_embeds = outputs.encoder_last_hidden_state[:, 0, :]
        
        # Pass through MLP
        x = self.mlp(text_embeds)
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





class EnhancedCombinedModel(nn.Module):
    """
    Enhanced combined model using CLIP for images and BART for text to better predict YouTube thumbnail views.
    """
    def __init__(
            self,
            image_mlp_layers=[1024, 512, 256, 1], 
            text_model_name='facebook/bart-base',
            text_mlp_layers=[1024, 512, 256, 1], 
            metadata_mlp_layers=[1024, 512, 256, 1],
            num_channels=1000,
            freeze_clip=True,
            freeze_text_model=True,
            max_token_length=128
        ):
        super().__init__()
        
        # Image branch (CLIP)
        self.image_branch = CLIPImageBranch(
            clip_model_name="openai/clip-vit-base-patch32",
            mlp_layers=image_mlp_layers,
            freeze_clip=freeze_clip
        )
        
        # Text branch (BART)
        self.text_branch = BARTTextBranch(
            bart_model_name=text_model_name,
            mlp_layers=text_mlp_layers,
            freeze_bart=freeze_text_model
        )
        
        # Metadata branch (unchanged)
        self.metadata_branch = MetadataBranch(metadata_mlp_layers, num_channels)
        
        # Tokenizer for BART
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.max_token_length = max_token_length
        
        # Learnable weights for combining the branches
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
        
        channel_ids = batch["channel_id"].to(image.device)

        # Process text with BART tokenizer
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