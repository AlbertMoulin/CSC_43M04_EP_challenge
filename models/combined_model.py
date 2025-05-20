import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import datetime
import re

# Maximum token length for text processing
MAX_TOKEN_LENGTH = 128

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention layer.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Apply layer normalization first
        residual = x
        x = self.layer_norm(x)
        
        # Linear projections
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final projection and dropout
        output = self.proj(attn_output)
        output = self.dropout(output)
        
        # Add residual connection
        output = output + residual
        
        return output

class ImageBranch(nn.Module):
    """
    Image processing branch: DinoV2 without head
    """
    def __init__(self, dinov2_model):
        super().__init__()
        self.backbone = dinov2_model # DinoV2 model (without the head)
        
    def forward(self, image):
        return self.backbone(image)

class TextBranch(nn.Module):
    """
    Text processing branch: Transformer model
    """
    def __init__(self, text_model_name):
        super().__init__()
        # Load pre-trained text model
        self.backbone = AutoModel.from_pretrained(text_model_name, trust_remote_code=True)
        
    def forward(self, input_ids, attention_mask):
        # Get hidden states from text model
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Take hidden state of first token ([CLS]) for sequence representation
        text_features = outputs.last_hidden_state[:, 0, :]
        return text_features

class MetadataBranch(nn.Module):
    """
    Metadata processing branch: Handles date and channel information.
    """
    def __init__(self, num_channels=1000, embedding_dim=64):
        super().__init__()
        
        # Embedding for channel ID
        self.channel_embedding = nn.Embedding(num_channels, embedding_dim)
        
        # Linear projection for date features
        self.date_projection = nn.Linear(5, embedding_dim)
        
    def forward(self, date_features, channel_ids):
        # Get channel embeddings
        channel_emb = self.channel_embedding(channel_ids)
        
        # Project date features to the same dimension as channel embeddings
        date_emb = self.date_projection(date_features)
        
        # Concatenate date and channel features
        metadata_features = torch.cat([date_emb, channel_emb], dim=1)
        
        return metadata_features

class EnhancedCombinedModel(nn.Module):
    """
    Enhanced combined model for predicting views from image, text, date and channel.
    Features:
    - Combines image, text, and a configurable percentage of metadata features
    - Applies self-attention to the combined representation
    - Processes through final MLP
    """
    def __init__(
        self, 
        image_mlp_layers, 
        text_model_name, 
        text_mlp_layers, 
        metadata_mlp_layers,
        final_mlp_layers,
        num_channels=1000,
        freeze_dinov2=True, 
        freeze_text_model=True,
        metadata_percentage=0.2,
        attention_heads=8,
        max_token_length=MAX_TOKEN_LENGTH
    ):
        super().__init__()
        
        # Image branch setup: Load DinoV2 and create ImageBranch
        dinov2_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        dinov2_backbone.head = nn.Identity() # Remove default head
        if freeze_dinov2:
            for param in dinov2_backbone.parameters():
                param.requires_grad = False
        self.image_branch = ImageBranch(dinov2_backbone)
        self.image_dim = dinov2_backbone.norm.normalized_shape[0]  # Should be 768 for dinov2_vitb14_reg

        # Text branch setup
        self.text_branch = TextBranch(text_model_name)
        if freeze_text_model:
            # Freeze text model backbone
            for param in self.text_branch.backbone.parameters():
                param.requires_grad = False
        self.text_dim = self.text_branch.backbone.config.hidden_size  # Ex: 768 for bert-base-uncased

        # Metadata branch setup
        self.metadata_branch = MetadataBranch(num_channels)
        self.metadata_dim = 64 * 2  # 64-dim for date, 64-dim for channel
        
        # Store the metadata percentage 
        self.metadata_percentage = metadata_percentage

        # Feature dimension after combination
        self.combined_dim = self.image_dim + self.text_dim + int(self.metadata_dim * metadata_percentage)
        
        # Linear projections to adjust feature dimensions before combining
        self.image_projection = nn.Linear(self.image_dim, self.image_dim)
        self.text_projection = nn.Linear(self.text_dim, self.text_dim)
        self.metadata_projection = nn.Linear(self.metadata_dim, int(self.metadata_dim * metadata_percentage))
        
        # Self-attention layers
        self.self_attention1 = MultiHeadSelfAttention(self.combined_dim, attention_heads)
        self.self_attention2 = MultiHeadSelfAttention(self.combined_dim, attention_heads)
        
        # Final MLP
        layers = []
        input_dim = self.combined_dim
        
        for output_dim in final_mlp_layers[:-1]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LayerNorm(output_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            input_dim = output_dim
            
        # Output layer
        layers.append(nn.Linear(input_dim, final_mlp_layers[-1]))
        
        self.final_mlp = nn.Sequential(*layers)

        # Tokenizer for processing raw text
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.max_token_length = max_token_length

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
        image = batch["image"]
        raw_text = batch["text"]
        batch_size = image.size(0)
        device = image.device
        
        # Get date and channel features
        date_features = []
        for date_str in batch["date"]:
            date_features.append(self._parse_date(date_str))
        date_features = torch.stack(date_features).to(device)
        
        # Channel IDs - we expect these to be already converted to indices in dataset
        channel_ids = batch["channel_id"].to(device)

        # --- Text processing: Tokenization ---
        encoded_text = self.tokenizer(
            list(raw_text),
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors='pt'
        )

        # Send text tensors to the same device as model
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)

        # Get features from each branch
        image_features = self.image_branch(image)
        text_features = self.text_branch(input_ids, attention_mask)
        metadata_features = self.metadata_branch(date_features, channel_ids)
        
        # Apply projections
        image_features = self.image_projection(image_features)
        text_features = self.text_projection(text_features)
        metadata_features = self.metadata_projection(metadata_features)
        
        # Combine features
        combined_features = torch.cat([image_features, text_features, metadata_features], dim=1)
        
        # Reshape for self-attention (batch_size, seq_length=1, feature_dim)
        combined_features = combined_features.unsqueeze(1)
        
        # Apply self-attention twice
        attended_features = self.self_attention1(combined_features)
        attended_features = self.self_attention2(attended_features)
        
        # Squeeze back to (batch_size, feature_dim)
        attended_features = attended_features.squeeze(1)
        
        # Apply final MLP
        output = self.final_mlp(attended_features)
        
        return output