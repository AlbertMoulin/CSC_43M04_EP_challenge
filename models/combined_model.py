import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, CLIPModel
import datetime
import math
import re

# Maximum token length for text processing
MAX_TOKEN_LENGTH = 128

class ResidualBlock(nn.Module):
    """
    Residual block for stable training of deeper networks.
    """
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        return x + self.layers(x)

class EnhancedCLIPImageBranch(nn.Module):
    """
    Enhanced image processing branch using CLIP and attention pooling.
    """
    def __init__(self, clip_model_name="openai/clip-vit-base-patch16", mlp_layers=[1024, 512, 256, 1], freeze_clip=True, dropout_rate=0.2):
        super().__init__()
        # Use ViT-B/16 for higher resolution features rather than ViT-B/32
        self.clip = CLIPModel.from_pretrained(clip_model_name).vision_model
        
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
        
        clip_dim = self.clip.config.hidden_size
        
        # Add attention pooling for better feature aggregation
        self.attention_pool = nn.Sequential(
            nn.Linear(clip_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        
        # Build MLP with residual connections
        layers = []
        input_dim = clip_dim
        
        for output_dim in mlp_layers[:-1]:
            if input_dim == output_dim:  # Add residual connection if dims match
                layers.append(ResidualBlock(input_dim, dropout_rate))
            else:
                layers.append(nn.Linear(input_dim, output_dim))
                layers.append(nn.LayerNorm(output_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout_rate))
            input_dim = output_dim
            
        layers.append(nn.Linear(input_dim, mlp_layers[-1]))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, image):
        # Get CLIP features with hidden states
        outputs = self.clip(image, output_hidden_states=True)
        
        # Get the last hidden state
        last_hidden = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        
        # Apply attention pooling over the sequence dimension
        attn_weights = F.softmax(self.attention_pool(last_hidden).squeeze(-1), dim=1)
        weighted_features = torch.bmm(
            attn_weights.unsqueeze(1), 
            last_hidden
        ).squeeze(1)
        
        # Pass through MLP
        x = self.mlp(weighted_features)
        return x, weighted_features  # Return features for cross-modal integration

class EnhancedBARTTextBranch(nn.Module):
    """
    Enhanced text processing branch using BART with cross-attention.
    """
    def __init__(self, bart_model_name="facebook/bart-base", mlp_layers=[1024, 512, 256, 1], freeze_bart=True, dropout_rate=0.2):
        super().__init__()
        self.bart = AutoModel.from_pretrained(bart_model_name)
        
        if freeze_bart:
            for param in self.bart.parameters():
                param.requires_grad = False
        
        # Get BART hidden dimension
        bart_dim = self.bart.config.d_model
        
        # Add cross-attention between encoder and decoder outputs
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=bart_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # MLP with residual connections
        layers = []
        input_dim = bart_dim
        
        for output_dim in mlp_layers[:-1]:
            if input_dim == output_dim:
                layers.append(ResidualBlock(input_dim, dropout_rate))
            else:
                layers.append(nn.Linear(input_dim, output_dim))
                layers.append(nn.LayerNorm(output_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout_rate))
            input_dim = output_dim
            
        layers.append(nn.Linear(input_dim, mlp_layers[-1]))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, input_ids, attention_mask):
        # Get BART encoder outputs
        outputs = self.bart(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get encoder representation
        encoder_output = outputs.encoder_last_hidden_state
        
        # Use cross-attention between encoder output and CLS token
        cls_token = encoder_output[:, 0, :].unsqueeze(1)
        attended_output, _ = self.cross_attention(
            query=cls_token,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=(1 - attention_mask).bool() if attention_mask is not None else None
        )
        
        # Squeeze and pass through MLP
        attended_features = attended_output.squeeze(1)
        x = self.mlp(attended_features)
        return x, attended_features  # Return features for cross-modal integration

class EnhancedMetadataBranch(nn.Module):
    """
    Enhanced metadata processing branch with advanced date processing and deeper embeddings.
    """
    def __init__(self, mlp_layers=[1024, 512, 256, 1], num_channels=1000, dropout_rate=0.2):
        super().__init__()
        
        # Enhanced channel embedding with more dimensions
        self.channel_embedding = nn.Embedding(num_channels, 128)  # Increase from 64 to 128
        
        # Add temporal features encoding
        self.temporal_encoder = nn.Sequential(
            nn.Linear(11, 64),  # Process the 11 date features
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Features: 
        # - 64-dim processed date features
        # - 128-dim channel embedding
        metadata_dim = 64 + 128
        
        # Build deeper MLP layers with residual connections
        layers = []
        input_dim = metadata_dim
        
        for output_dim in mlp_layers[:-1]:
            if input_dim == output_dim and input_dim > 32:  # Add residual connection if dims match
                layers.append(ResidualBlock(input_dim, dropout_rate))
            else:
                layers.append(nn.Linear(input_dim, output_dim))
                layers.append(nn.LayerNorm(output_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout_rate))
            input_dim = output_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, mlp_layers[-1]))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, date_features, channel_ids):
        # Process date features through temporal encoder
        date_encoding = self.temporal_encoder(date_features)
        
        # Get channel embeddings
        channel_emb = self.channel_embedding(channel_ids)
        
        # Concatenate date features and channel embeddings
        metadata_features = torch.cat([date_encoding, channel_emb], dim=1)
        
        # Pass through MLP
        x = self.mlp(metadata_features)
        return x, metadata_features  # Return features for cross-modal integration

class CrossModalIntegration(nn.Module):
    """
    Cross-modal integration module to capture interactions between different modalities.
    """
    def __init__(self, image_dim, text_dim, metadata_dim, hidden_dim=512, output_dim=1, dropout_rate=0.2):
        super().__init__()
        
        # Project all features to a common dimension
        self.image_projection = nn.Linear(image_dim, hidden_dim) if image_dim != hidden_dim else nn.Identity()
        self.text_projection = nn.Linear(text_dim, hidden_dim) if text_dim != hidden_dim else nn.Identity()
        self.metadata_projection = nn.Linear(metadata_dim, hidden_dim)
        
        # Cross-attention between image and text
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Final integration MLP
        self.integration_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, image_features, text_features, metadata_features):
        # Project all features to common dimension
        image_proj = self.image_projection(image_features)
        text_proj = self.text_projection(text_features)
        metadata_proj = self.metadata_projection(metadata_features)
        
        # Reshape for attention if needed
        if len(image_proj.shape) == 2:
            image_proj = image_proj.unsqueeze(1)
        if len(text_proj.shape) == 2:
            text_proj = text_proj.unsqueeze(1)
        
        # Cross-attention: image attends to text
        enhanced_image, _ = self.cross_attention(
            query=image_proj,
            key=text_proj,
            value=text_proj
        )
        
        # Concatenate all features
        enhanced_image = enhanced_image.squeeze(1)
        metadata_proj = metadata_proj.squeeze(1) if len(metadata_proj.shape) > 2 else metadata_proj
        
        integrated = torch.cat([enhanced_image, text_proj.squeeze(1), metadata_proj], dim=1)
        
        # Final integration
        output = self.integration_mlp(integrated)
        
        return output

class EnhancedCombinedModel(nn.Module):
    """
    Enhanced combined model using CLIP for images, BART for text, and advanced metadata processing.
    Includes cross-modal integration and adaptive weighting.
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
            max_token_length=MAX_TOKEN_LENGTH,
            dropout_rate=0.2
        ):
        super().__init__()
        
        # Image branch (Enhanced CLIP)
        self.image_branch = EnhancedCLIPImageBranch(
            clip_model_name="openai/clip-vit-base-patch16",
            mlp_layers=image_mlp_layers,
            freeze_clip=freeze_clip,
            dropout_rate=dropout_rate
        )
        
        # Text branch (Enhanced BART)
        self.text_branch = EnhancedBARTTextBranch(
            bart_model_name=text_model_name,
            mlp_layers=text_mlp_layers,
            freeze_bart=freeze_text_model,
            dropout_rate=dropout_rate
        )
        
        # Metadata branch (Enhanced)
        self.metadata_branch = EnhancedMetadataBranch(
            mlp_layers=metadata_mlp_layers,
            num_channels=num_channels,
            dropout_rate=dropout_rate
        )
        
        # Get the dimensions for cross-modal integration
        # These would normally be determined by the models, but for simplicity we'll use constants
        clip_dim = 768  # for ViT-B/16
        bart_dim = 768  # for BART base
        metadata_dim = 64 + 128  # From our enhanced metadata branch
        
        # Cross-modal integration
        self.cross_modal = CrossModalIntegration(
            image_dim=clip_dim,
            text_dim=bart_dim,
            metadata_dim=metadata_dim,
            hidden_dim=512,
            output_dim=1,
            dropout_rate=dropout_rate
        )
        
        # Tokenizer for BART
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.max_token_length = max_token_length
        
        # Learnable weights for combining branch predictions with cross-modal output
        self.weight_image = nn.Parameter(torch.tensor(0.2))
        self.weight_text = nn.Parameter(torch.tensor(0.2))
        self.weight_metadata = nn.Parameter(torch.tensor(0.3))
        self.weight_cross_modal = nn.Parameter(torch.tensor(0.3))
        
    def _parse_date_advanced(self, date_str):
        """
        Enhanced date parsing with more sophisticated temporal features.
        """
        try:
            # Parse date object as before
            if '+' in date_str:
                date_str = date_str.split('+')[0]
            
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            
            # Extract basic features
            year = (date_obj.year - 2005) / 20  # Normalize years since YouTube's founding
            month = (date_obj.month - 1) / 11   # 0-11
            day = (date_obj.day - 1) / 30       # 0-30
            day_of_week = date_obj.weekday() / 6  # 0-6
            hour = date_obj.hour / 23           # 0-23
            
            # New temporal features
            is_weekend = 1.0 if date_obj.weekday() >= 5 else 0.0  # Weekend flag
            
            # Season encoding (Northern Hemisphere)
            month_rad = 2 * math.pi * date_obj.month / 12
            sin_season = math.sin(month_rad)  # Peaks in summer
            cos_season = math.cos(month_rad)  # Peaks in winter/spring
            
            # Time of day encoding (peaks at different parts of day)
            hour_rad = 2 * math.pi * date_obj.hour / 24
            sin_time = math.sin(hour_rad)  # Peaks at 6am/6pm
            cos_time = math.cos(hour_rad)  # Peaks at midnight/noon
            
            # YouTube trend periods (rough approximations)
            is_holiday_season = 1.0 if (date_obj.month == 12 or date_obj.month == 11) else 0.0
            
            return torch.tensor([
                year, month, day, day_of_week, hour,
                is_weekend, sin_season, cos_season, sin_time, cos_time, is_holiday_season
            ], dtype=torch.float32)
            
        except Exception as e:
            # If all parsing fails, return default values
            print(f"Error parsing date '{date_str}': {e}")
            return torch.zeros(11, dtype=torch.float32)
        
    def forward(self, batch):
        # Extract components from batch
        image = batch["image"]
        raw_text = batch["text"]
        
        # Get enhanced date features
        date_features = []
        for date_str in batch["date"]:
            date_features.append(self._parse_date_advanced(date_str))
        date_features = torch.stack(date_features).to(image.device)
        
        # Channel IDs
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

        # Get predictions and features from each branch
        image_prediction, image_features = self.image_branch(image)
        text_prediction, text_features = self.text_branch(input_ids, attention_mask)
        metadata_prediction, metadata_features = self.metadata_branch(date_features, channel_ids)
        
        # Get cross-modal prediction
        cross_modal_prediction = self.cross_modal(
            image_features, 
            text_features, 
            metadata_features
        )

        # Normalize weights to sum to 1 using softmax
        weights = torch.softmax(
            torch.stack([
                self.weight_image, 
                self.weight_text, 
                self.weight_metadata, 
                self.weight_cross_modal
            ]), 
            dim=0
        )
        
        # Combine predictions using normalized weights
        combined_prediction = (
            weights[0] * image_prediction + 
            weights[1] * text_prediction + 
            weights[2] * metadata_prediction +
            weights[3] * cross_modal_prediction
        )

        return combined_prediction