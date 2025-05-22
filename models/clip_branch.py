import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import datetime
import re

class CLIPBranchModel(nn.Module):
    def __init__(
        self, 
        # These lists should define HIDDEN layers, and the last layer (outputting 1)
        # will be added separately.
        clip_mlp_layers=[1024, 512, 256], # Define hidden layer dimensions
        date_mlp_layers=[64, 32, 16],     # Define hidden layer dimensions
        channel_mlp_layers=[128, 64, 32], # Define hidden layer dimensions
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
        input_dim_clip = clip_dim * 2  # Combined image and text features
        
        # Iterate through specified hidden layer dimensions
        for output_dim in clip_mlp_layers:
            clip_layers.append(nn.Linear(input_dim_clip, output_dim))
            clip_layers.append(nn.LayerNorm(output_dim))
            clip_layers.append(nn.GELU())
            clip_layers.append(nn.Dropout(dropout_rate))
            input_dim_clip = output_dim
        
        # Add the final output layer (to produce a single scalar prediction per branch)
        clip_layers.append(nn.Linear(input_dim_clip, 1)) 
        self.clip_mlp = nn.Sequential(*clip_layers)
        
        # 2. Date branch MLP
        date_layers = []
        input_dim_date = 8 # year, month, day, day_of_week, hour, day_of_year, is_weekend, quarter
        
        for output_dim in date_mlp_layers:
            date_layers.append(nn.Linear(input_dim_date, output_dim))
            date_layers.append(nn.LayerNorm(output_dim))
            date_layers.append(nn.GELU())
            date_layers.append(nn.Dropout(dropout_rate))
            input_dim_date = output_dim
        
        date_layers.append(nn.Linear(input_dim_date, 1)) # Final output layer
        self.date_mlp = nn.Sequential(*date_layers)
        
        # 3. Channel branch
        self.channel_embedding = nn.Embedding(num_channels, 64) 
        
        channel_layers = []
        input_dim_channel = 64  # Channel embedding dimension
        
        for output_dim in channel_mlp_layers:
            channel_layers.append(nn.Linear(input_dim_channel, output_dim))
            channel_layers.append(nn.LayerNorm(output_dim))
            channel_layers.append(nn.GELU())
            channel_layers.append(nn.Dropout(dropout_rate))
            input_dim_channel = output_dim
        
        channel_layers.append(nn.Linear(input_dim_channel, 1)) # Final output layer
        self.channel_mlp = nn.Sequential(*channel_layers)
        
        # Learnable weights for combining predictions
        self.weight_clip = nn.Parameter(torch.tensor(0.33))
        self.weight_date = nn.Parameter(torch.tensor(0.33))
        self.weight_channel = nn.Parameter(torch.tensor(0.34))
    
    def _parse_date(self, date_str):
        # ... (as previously defined, assuming this part is correct) ...
        """
        Parse date string into numerical features.
        Returns tensor with [year, month, day, day_of_week, hour, day_of_year, is_weekend, quarter]
        """
        try:
            # First, clean the date string: remove timezone info like '+00:00' or 'Z'
            date_str = re.sub(r'[Zz\+\-][0-9\:]+$', '', date_str).strip()
            
            # Try different date formats
            date_obj = None
            formats_to_try = [
                "%Y-%m-%d %H:%M:%S",  # "YYYY-MM-DD HH:MM:SS"
                "%Y-%m-%dT%H:%M:%S", # "YYYY-MM-DDTHH:MM:SS" (ISO format)
                "%Y-%m-%d %H:%M",    # "YYYY-MM-DD HH:MM"
                "%Y-%m-%d",         # "YYYY-MM-DD"
                "%Y/%m/%d %H:%M:%S", # "YYYY/MM/DD HH:MM:SS"
                "%Y/%m/%d",         # "YYYY/MM/DD"
            ]
            
            for fmt in formats_to_try:
                try:
                    date_obj = datetime.datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            if date_obj is None:
                # Fallback for very minimal date: "YYYY-MM-DD" (from your Dataset's default)
                try:
                    date_obj = datetime.datetime.strptime(date_str.split(' ')[0], "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"No matching format found for date string: '{date_str}'")

            # Extract features (normalized for neural network inputs)
            year = (date_obj.year - 2005) / 20 # Assuming years between ~2005-2025
            month = (date_obj.month - 1) / 11 
            day = (date_obj.day - 1) / 30     
            day_of_week = date_obj.weekday() / 6 
            hour = date_obj.hour / 23          
            day_of_year = (date_obj.timetuple().tm_yday - 1) / 365 
            is_weekend = float(date_obj.weekday() >= 5) 
            quarter = (date_obj.month - 1) // 3 / 3 

            return torch.tensor([year, month, day, day_of_week, hour, day_of_year, is_weekend, quarter], dtype=torch.float32)
        
        except Exception as e:
            # If all parsing fails, return default (zero) values.
            print(f"Warning: Error parsing date '{date_str}': {e}. Returning zeros for date features.")
            return torch.tensor([0.0] * 8, dtype=torch.float32)
    
    def forward(self, batch):
        images = batch["image"]
        raw_text = batch["text"]
        device = images.device
        
        # 1. CLIP branch
        with torch.no_grad() if not self.clip.training and self.training else torch.enable_grad():
            image_features = self.clip.get_image_features(images)
            
            clip_text_inputs = self.processor(
                text=list(raw_text),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)
            
            text_features = self.clip.get_text_features(**clip_text_inputs)
        
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        clip_combined = torch.cat([image_features, text_features], dim=1)
        clip_prediction = self.clip_mlp(clip_combined) # This should now output (batch_size, 1)
        
        # 2. Date branch
        date_features = []
        for date_str in batch["date"]:
            date_features.append(self._parse_date(date_str))
        date_features = torch.stack(date_features).to(device)
        date_prediction = self.date_mlp(date_features) # This should now output (batch_size, 1)
        
        # 3. Channel branch
        channel_ids = batch["channel_id"].to(device)
        channel_emb = self.channel_embedding(channel_ids)
        channel_prediction = self.channel_mlp(channel_emb) # This should now output (batch_size, 1)
        
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