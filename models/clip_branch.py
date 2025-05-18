import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import datetime
import re

class CLIPBranchModel(nn.Module):
    """
    Model with three separate branches:
    1. CLIP branch for image and text
    2. Date branch
    3. Channel branch
    
    Each branch makes its own prediction, then all predictions are combined with
    learnable weights.
    """
    def __init__(
        self, 
        clip_mlp_layers=[1024, 512, 256, 1], 
        date_mlp_layers=[64, 32, 16, 1],
        channel_mlp_layers=[128, 64, 32, 1],
        clip_model_name="openai/clip-vit-base-patch32",
        num_channels=1000,
        freeze_clip=True,
        dropout_rate=0.2
    ):
        super().__init__()
        
        # Load the CLIP model
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # Freeze CLIP parameters if specified
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
        
        # Get CLIP dimension
        clip_dim = self.clip.config.projection_dim  # Typically 512 for base model
        
        # 1. CLIP branch MLP
        clip_layers = []
        input_dim = clip_dim * 2  # Combined image and text features
        
        for output_dim in clip_mlp_layers[:-1]:
            clip_layers.append(nn.Linear(input_dim, output_dim))
            clip_layers.append(nn.LayerNorm(output_dim))
            clip_layers.append(nn.GELU())
            clip_layers.append(nn.Dropout(dropout_rate))
            input_dim = output_dim
        
        clip_layers.append(nn.Linear(input_dim, clip_mlp_layers[-1]))
        self.clip_mlp = nn.Sequential(*clip_layers)
        
        # 2. Date branch MLP
        date_layers = []
        input_dim = 5  # 5 date features
        
        for output_dim in date_mlp_layers[:-1]:
            date_layers.append(nn.Linear(input_dim, output_dim))
            date_layers.append(nn.LayerNorm(output_dim))
            date_layers.append(nn.GELU())
            date_layers.append(nn.Dropout(dropout_rate))
            input_dim = output_dim
        
        date_layers.append(nn.Linear(input_dim, date_mlp_layers[-1]))
        self.date_mlp = nn.Sequential(*date_layers)
        
        # 3. Channel branch
        self.channel_embedding = nn.Embedding(num_channels, 64)
        
        channel_layers = []
        input_dim = 64  # Channel embedding dimension
        
        for output_dim in channel_mlp_layers[:-1]:
            channel_layers.append(nn.Linear(input_dim, output_dim))
            channel_layers.append(nn.LayerNorm(output_dim))
            channel_layers.append(nn.GELU())
            channel_layers.append(nn.Dropout(dropout_rate))
            input_dim = output_dim
        
        channel_layers.append(nn.Linear(input_dim, channel_mlp_layers[-1]))
        self.channel_mlp = nn.Sequential(*channel_layers)
        
        # Learnable weights for combining predictions
        self.weight_clip = nn.Parameter(torch.tensor(0.33))
        self.weight_date = nn.Parameter(torch.tensor(0.33))
        self.weight_channel = nn.Parameter(torch.tensor(0.34))
    
    def _parse_date(self, date_str):
        """
        Parse date string into numerical features.
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
        images = batch["image"]
        raw_text = batch["text"]
        batch_size = images.shape[0]
        device = images.device
        
        # 1. CLIP branch - Process images and text with CLIP
        with torch.no_grad() if not self.clip.training else torch.enable_grad():
            # Get image embeddings
            image_features = self.clip.get_image_features(images)
            
            # Process text with CLIP tokenizer and get embeddings
            clip_text_inputs = self.processor(
                text=list(raw_text),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77  # CLIP's default max length
            ).to(device)
            
            text_features = self.clip.get_text_features(**clip_text_inputs)
        
        # Normalize features as in CLIP
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # Concatenate CLIP features and get prediction
        clip_combined = torch.cat([image_features, text_features], dim=1)
        clip_prediction = self.clip_mlp(clip_combined)
        
        # 2. Date branch
        date_features = []
        for date_str in batch["date"]:
            date_features.append(self._parse_date(date_str))
        date_features = torch.stack(date_features).to(device)
        date_prediction = self.date_mlp(date_features)
        
        # 3. Channel branch
        channel_ids = batch["channel_id"].to(device)
        channel_emb = self.channel_embedding(channel_ids)
        channel_prediction = self.channel_mlp(channel_emb)
        
        # Normalize weights to sum to 1 using softmax
        weights = torch.softmax(
            torch.stack([self.weight_clip, self.weight_date, self.weight_channel]), 
            dim=0
        )
        
        # Combine predictions using normalized weights
        combined_prediction = (
            weights[0] * clip_prediction + 
            weights[1] * date_prediction + 
            weights[2] * channel_prediction
        )
        
        return combined_prediction