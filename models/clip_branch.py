import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import datetime
import math

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention mechanism for inter-modal alignment
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
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Apply layer normalization
        residual = query
        query = self.layer_norm(query)
        
        # Linear projections and reshape for multi-head attention
        q = self.query(query).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(key).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(value).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.embed_dim)
        
        # Final projection and residual connection
        output = self.proj(attn_output)
        output = self.dropout(output) + residual
        
        return output

class SelfAttentionBlock(nn.Module):
    """
    Self-attention block for intra-modal processing
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x

class SqueezeExcitationNetwork(nn.Module):
    """
    Squeeze-and-Excitation Network for channel-wise feature recalibration
    """
    def __init__(self, embed_dim, reduction_ratio=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // reduction_ratio, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, embed_dim)
        # Add sequence dimension for pooling
        x_expanded = x.unsqueeze(-1)  # (batch_size, embed_dim, 1)
        
        # Global average pooling
        pooled = self.global_pool(x_expanded).squeeze(-1)  # (batch_size, embed_dim)
        
        # Channel attention weights
        weights = self.fc(pooled)  # (batch_size, embed_dim)
        
        # Apply attention weights
        return x * weights

class AdaptiveGatingMechanism(nn.Module):
    """
    Adaptive gating mechanism for dynamic modality weighting
    """
    def __init__(self, embed_dim, num_modalities):
        super().__init__()
        self.num_modalities = num_modalities
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, *modality_features):
        # Concatenate all modality features
        combined = torch.cat(modality_features, dim=-1)
        
        # Compute gating weights
        gates = self.gate_network(combined)
        
        # Apply gates to each modality
        gated_features = []
        for i, features in enumerate(modality_features):
            gated = features * gates[:, i:i+1]
            gated_features.append(gated)
            
        return torch.stack(gated_features, dim=1).sum(dim=1)

class EnhancedCLIPBranchModel(nn.Module):
    """
    Enhanced CLIP Branch Model with hierarchical multimodal fusion:
    1. Self-attention intra-modal processing
    2. Cross-attention inter-modal alignment  
    3. Squeeze-and-Excitation feature enhancement
    4. Adaptive gating for dynamic fusion
    5. Separate PCA-like dimensionality reduction for each modality
    """
    def __init__(
        self, 
        clip_model_name="openai/clip-vit-base-patch32",
        embed_dim=512,
        num_heads=8,
        num_channels=1000,
        freeze_clip=True,
        dropout_rate=0.1,
        metadata_embed_dim=128,
        final_mlp_layers=[1024, 512, 256, 1]
    ):
        super().__init__()
        
        # Load CLIP model
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
        
        # Get original CLIP dimensions
        clip_dim = self.clip.config.projection_dim  # Usually 512
        
        # === Modality-specific dimensionality reduction (inspired by V4 strategy) ===
        self.image_reducer = nn.Sequential(
            nn.Linear(clip_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.text_reducer = nn.Sequential(
            nn.Linear(clip_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # === Self-attention for intra-modal processing ===
        self.image_self_attn = SelfAttentionBlock(embed_dim, num_heads, dropout_rate)
        self.text_self_attn = SelfAttentionBlock(embed_dim, num_heads, dropout_rate)
        
        # === Cross-attention for inter-modal alignment ===
        self.image_to_text_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout_rate)
        self.text_to_image_attn = MultiHeadCrossAttention(embed_dim, num_heads, dropout_rate)
        
        # === Squeeze-and-Excitation for feature enhancement ===
        self.image_senet = SqueezeExcitationNetwork(embed_dim)
        self.text_senet = SqueezeExcitationNetwork(embed_dim)
        self.metadata_senet = SqueezeExcitationNetwork(metadata_embed_dim)
        
        # === Metadata processing ===
        # Date features
        self.date_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 64)
        )
        
        # Channel embedding
        self.channel_embedding = nn.Embedding(num_channels, 64)
        
        # Metadata fusion
        self.metadata_fusion = nn.Sequential(
            nn.Linear(128, metadata_embed_dim),  # 64 + 64
            nn.LayerNorm(metadata_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # === Adaptive gating mechanism ===
        self.adaptive_gating = AdaptiveGatingMechanism(
            embed_dim=embed_dim, 
            num_modalities=2  # image and text
        )
        
        # === Final fusion and prediction ===
        # Cross Network for explicit feature interactions
        self.cross_network = CrossNetwork(
            input_dim=embed_dim * 2 + metadata_embed_dim,  # fused image-text + metadata
            num_layers=3
        )
        
        # Final MLP
        layers = []
        input_dim = embed_dim * 2 + metadata_embed_dim
        
        for output_dim in final_mlp_layers[:-1]:
            layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = output_dim
            
        layers.append(nn.Linear(input_dim, final_mlp_layers[-1]))
        self.final_mlp = nn.Sequential(*layers)
    
    def _parse_date(self, date_str):
        """Parse date string into numerical features"""
        try:
            if '+' in date_str:
                date_str = date_str.split('+')[0]
            
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        
            # Normalized features
            year = (date_obj.year - 2005) / 20
            month = (date_obj.month - 1) / 11
            day = (date_obj.day - 1) / 30
            day_of_week = date_obj.weekday() / 6
            hour = date_obj.hour / 23
            
            return torch.tensor([year, month, day, day_of_week, hour], dtype=torch.float32)
        
        except Exception as e:
            print(f"Error parsing date '{date_str}': {e}")
            return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    
    def forward(self, batch):
        images = batch["image"]
        raw_text = batch["text"]
        device = images.device
        
        # === CLIP feature extraction ===
        with torch.no_grad() if not self.clip.training else torch.enable_grad():
            # Image features
            image_features = self.clip.get_image_features(images)
            
            # Text features
            clip_text_inputs = self.processor(
                text=list(raw_text),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)
            text_features = self.clip.get_text_features(**clip_text_inputs)
        
        # Normalize CLIP features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        
        # === Modality-specific dimensionality reduction (V4 strategy) ===
        image_reduced = self.image_reducer(image_features)
        text_reduced = self.text_reducer(text_features)
        
        # === Self-attention for intra-modal processing ===
        # Add sequence dimension for self-attention
        image_attended = self.image_self_attn(image_reduced.unsqueeze(1)).squeeze(1)
        text_attended = self.text_self_attn(text_reduced.unsqueeze(1)).squeeze(1)
        
        # === Cross-attention for inter-modal alignment ===
        image_cross = self.image_to_text_attn(image_attended, text_attended, text_attended)
        text_cross = self.text_to_image_attn(text_attended, image_attended, image_attended)
        
        # === Squeeze-and-Excitation enhancement ===
        image_enhanced = self.image_senet(image_cross)
        text_enhanced = self.text_senet(text_cross)
        
        # === Adaptive gating for dynamic fusion ===
        fused_visual_text = self.adaptive_gating(image_enhanced, text_enhanced)
        
        # === Metadata processing ===
        # Date features
        date_features = []
        for date_str in batch["date"]:
            date_features.append(self._parse_date(date_str))
        date_features = torch.stack(date_features).to(device)
        date_emb = self.date_mlp(date_features)
        
        # Channel features
        channel_ids = batch["channel_id"].to(device)
        channel_emb = self.channel_embedding(channel_ids)
        
        # Fuse metadata
        metadata_combined = torch.cat([date_emb, channel_emb], dim=1)
        metadata_features = self.metadata_fusion(metadata_combined)
        metadata_enhanced = self.metadata_senet(metadata_features)
        
        # === Final feature combination ===
        # Concatenate enhanced image, text, and metadata features
        combined_features = torch.cat([
            image_enhanced, 
            text_enhanced, 
            metadata_enhanced
        ], dim=1)
        
        # === Cross Network for explicit interactions ===
        cross_output = self.cross_network(combined_features)
        
        # === Final prediction ===
        # Combine original features with cross network output
        final_input = combined_features + cross_output
        output = self.final_mlp(final_input)
        
        return output

class CrossNetwork(nn.Module):
    """
    Cross Network for modeling explicit feature interactions
    """
    def __init__(self, input_dim, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.cross_layers = nn.ModuleList([
            nn.Linear(input_dim, input_dim, bias=True) for _ in range(num_layers)
        ])
        
    def forward(self, x0):
        x = x0
        for layer in self.cross_layers:
            # Cross interaction: x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
            cross_term = layer(x)  # W_l * x_l + b_l
            x = x0 * cross_term + x  # x_0 * cross_term + x_l
        return x