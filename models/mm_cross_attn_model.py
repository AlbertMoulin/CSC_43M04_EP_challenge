import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, ViTModel, ViTImageProcessor
import datetime
import re

class SelfAttentionLayer(nn.Module):
    """
    Self-attention layer as described in the paper.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(x, x, x)
        output = self.dropout(attn_output)
        return output + residual

class CrossAttentionLayer(nn.Module):
    """
    Cross-attention layer for aligning visual and textual features.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm_query = nn.LayerNorm(embed_dim)
        self.layer_norm_key = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        residual = query
        query = self.layer_norm_query(query)
        key = self.layer_norm_key(key)
        
        attn_output, _ = self.multihead_attn(query, key, value)
        output = self.dropout(attn_output)
        return output + residual

class ImprovedMLPHead(nn.Module):
    """
    MLP head with layer normalization and GELU activations for better performance.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

class AdaptiveGatingMechanism(nn.Module):
    """
    Adaptive gating mechanism for dynamically weighting modalities.
    """
    def __init__(self, input_dim, context_dim=None):
        super().__init__()
        self.with_context = context_dim is not None
        
        if self.with_context:
            # If using context, create a gate for combined input+context
            combined_dim = input_dim + context_dim
            self.gate = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.GELU(),
                nn.Linear(combined_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            # If no context, just use the input directly
            self.gate = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.GELU(),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid()
            )
        
    def forward(self, x, context=None):
        if self.with_context and context is not None:
            # If context is provided, concatenate with input
            x_combined = torch.cat([x, context], dim=-1)
            gate_value = self.gate(x_combined)
        else:
            gate_value = self.gate(x)
        return x * gate_value

class MultiModalCrossAttentionNetwork(nn.Module):
    """
    Multi-Modality Cross Attention Network for YouTube view prediction.
    
    Architecture:
    1. Vision Transformer for image processing
    2. BERT-based model for text processing
    3. Self-attention for intra-modal feature enhancement
    4. Cross-attention for inter-modal feature alignment
    5. Adaptive gating mechanism for modality weighting
    6. Metadata branch for channel and date information
    """
    def __init__(
        self,
        vit_model_name="google/vit-base-patch16-224",
        text_model_name="bert-base-uncased",
        num_attention_heads=8,
        img_hidden_dim=768,
        text_hidden_dim=768,
        fusion_hidden_dim=1024,
        final_mlp_layers=[1024, 512, 256, 1],
        num_channels=1000,
        dropout_rate=0.2,
        freeze_vit=True,
        freeze_text_model=True,
        max_token_length=128
    ):
        super().__init__()
        
        # Image branch: Vision Transformer
        self.vit_model = ViTModel.from_pretrained(vit_model_name)
        self.image_processor = ViTImageProcessor.from_pretrained(vit_model_name)
        
        if freeze_vit:
            for param in self.vit_model.parameters():
                param.requires_grad = False
                
        # Text branch: BERT-based model
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        if freeze_text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False
                
        # Dimensions
        self.img_dim = img_hidden_dim
        self.text_dim = text_hidden_dim
        self.fusion_dim = fusion_hidden_dim
        self.max_token_length = max_token_length
        
        # Metadata branch
        self.channel_embedding = nn.Embedding(num_channels, 64)
        self.date_projection = nn.Linear(5, 64)  # 5 date features -> 64D
        self.metadata_fusion = nn.Linear(128, 256)  # 64 (channel) + 64 (date) -> 256
        
        # Feature projections for dimensionality alignment
        self.img_projection = nn.Linear(self.img_dim, self.fusion_dim)
        self.text_projection = nn.Linear(self.text_dim, self.fusion_dim)
        self.metadata_projection = nn.Linear(256, self.fusion_dim)
        
        # Self-attention layers for intra-modal processing
        self.img_self_attn = SelfAttentionLayer(self.fusion_dim, num_attention_heads, dropout_rate)
        self.text_self_attn = SelfAttentionLayer(self.fusion_dim, num_attention_heads, dropout_rate)
        
        # Cross-attention layers for inter-modal alignment
        self.img_text_cross_attn = CrossAttentionLayer(self.fusion_dim, num_attention_heads, dropout_rate)
        self.text_img_cross_attn = CrossAttentionLayer(self.fusion_dim, num_attention_heads, dropout_rate)
        
        # Adaptive gating mechanisms
        self.img_gate = AdaptiveGatingMechanism(self.fusion_dim)
        self.text_gate = AdaptiveGatingMechanism(self.fusion_dim)
        self.metadata_gate = AdaptiveGatingMechanism(self.fusion_dim)
        
        # Fusion MLP
        self.fusion_mlp = ImprovedMLPHead(
            self.fusion_dim * 3,  # Image + Text + Metadata
            final_mlp_layers[:-1],
            final_mlp_layers[-1],
            dropout_rate
        )
    
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
        
            # Extract features (normalized as suggested in the document)
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
        batch_size = images.size(0)
        device = images.device
        
        # --- Process date and channel information ---
        date_features = []
        for date_str in batch["date"]:
            date_features.append(self._parse_date(date_str))
        date_features = torch.stack(date_features).to(device)
        
        channel_ids = batch["channel_id"].to(device)
        
        # --- Image processing using ViT ---
        with torch.no_grad() if not self.vit_model.training else torch.enable_grad():
            # No need to preprocess as the DataLoader already does this
            vit_output = self.vit_model(images)
            img_features = vit_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        # --- Text processing using BERT ---
        encoded_text = self.tokenizer(
            list(raw_text),
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad() if not self.text_model.training else torch.enable_grad():
            text_output = self.text_model(**encoded_text)
            text_features = text_output.last_hidden_state[:, 0, :]  # [CLS] token
        
        # --- Metadata processing ---
        channel_emb = self.channel_embedding(channel_ids)
        date_emb = self.date_projection(date_features)
        metadata_features = torch.cat([channel_emb, date_emb], dim=1)
        metadata_features = self.metadata_fusion(metadata_features)
        
        # --- Project features to common dimension ---
        img_proj = self.img_projection(img_features).unsqueeze(1)  # Add sequence dimension
        text_proj = self.text_projection(text_features).unsqueeze(1)
        metadata_proj = self.metadata_projection(metadata_features).unsqueeze(1)
        
        # --- Self-attention for intra-modal processing ---
        img_self = self.img_self_attn(img_proj)
        text_self = self.text_self_attn(text_proj)
        
        # --- Cross-attention for inter-modal alignment ---
        img_cross = self.img_text_cross_attn(img_self, text_self, text_self)
        text_cross = self.text_img_cross_attn(text_self, img_self, img_self)
        
        # --- Apply adaptive gating mechanisms ---
        # Get combined cross-attention features for gating
        img_cross_feat = img_cross.squeeze(1)
        text_cross_feat = text_cross.squeeze(1)
        img_self_feat = img_self.squeeze(1)
        text_self_feat = text_self.squeeze(1)
        
        # Apply adaptive gating
        img_gated = self.img_gate(img_cross_feat)
        text_gated = self.text_gate(text_cross_feat)
        metadata_gated = self.metadata_gate(metadata_proj.squeeze(1))
        
        # --- Concatenate all modalities for final prediction ---
        combined_features = torch.cat([img_gated, text_gated, metadata_gated], dim=1)
        
        # --- Final prediction ---
        output = self.fusion_mlp(combined_features)
        
        return output