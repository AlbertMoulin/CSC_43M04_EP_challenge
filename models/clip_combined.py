import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import datetime
import re

class CLIPCombinedModel(nn.Module):
    """
    Model that uses CLIP for both image and text encoding, then combines with metadata.
    """
    def __init__(
        self, 
        mlp_layers=[1024, 512, 256, 1], 
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
        
        # Embedding for channel ID
        self.channel_embedding = nn.Embedding(num_channels, 64)
        
        # Determine combined feature dimension:
        # - CLIP image features (e.g., 512)
        # - CLIP text features (e.g., 512)
        # - Date features (5 features: year, month, day, day_of_week, hour)
        # - Channel embedding (64)
        clip_dim = self.clip.config.projection_dim  # Typically 512 for base model
        combined_dim = clip_dim * 2 + 5 + 64
        
        # Build MLP layers
        layers = []
        input_dim = combined_dim
        
        for output_dim in mlp_layers[:-1]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim))  # Better normalization
            layers.append(nn.GELU())  # Advanced activation
            layers.append(nn.Dropout(dropout_rate))
            input_dim = output_dim
        
        # Final output layer
        layers.append(nn.Linear(input_dim, mlp_layers[-1]))
        
        self.mlp = nn.Sequential(*layers)
    
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
        
        # Process images and text with CLIP
        with torch.no_grad() if self.clip.training is False else torch.enable_grad():
            # Get image embeddings
            image_features = self.clip.get_image_features(images)
            
            # Process text with CLIP tokenizer and get embeddings
            # Note: this assumes raw_text is a list of strings
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
        
        # Get date features
        date_features = []
        for date_str in batch["date"]:
            date_features.append(self._parse_date(date_str))
        date_features = torch.stack(date_features).to(device)
        
        # Get channel embeddings
        channel_ids = batch["channel_id"].to(device)
        channel_emb = self.channel_embedding(channel_ids)
        
        # Concatenate all features
        combined_features = torch.cat([
            image_features,
            text_features,
            date_features,
            channel_emb
        ], dim=1)
        
        # Pass through MLP
        output = self.mlp(combined_features)
        
        return output