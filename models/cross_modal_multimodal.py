import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between image and text features
    """
    def __init__(self, image_dim=768, text_dim=1152, hidden_dim=512, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Project both modalities to same dimension
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=0.1
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, image_features, text_features):
        # Keep everything in float32
        # Project to same dimension
        img_proj = self.image_proj(image_features)  # [B, hidden_dim]
        txt_proj = self.text_proj(text_features)    # [B, hidden_dim]
        
        # Add sequence dimension for attention
        img_seq = img_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        txt_seq = txt_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # Cross attention: text queries attend to image keys/values
        attended_text, attention_weights = self.cross_attention(
            query=txt_seq,
            key=img_seq, 
            value=img_seq
        )
        
        # Cross attention: image queries attend to text keys/values  
        attended_image, _ = self.cross_attention(
            query=img_seq,
            key=txt_seq,
            value=txt_seq
        )
        
        # Remove sequence dimension and apply layer norm
        attended_text = self.layer_norm(attended_text.squeeze(1))    # [B, hidden_dim]
        attended_image = self.layer_norm(attended_image.squeeze(1))  # [B, hidden_dim]
        
        # Concatenate attended features
        cross_modal_features = torch.cat([attended_image, attended_text], dim=1)
        
        return cross_modal_features  # [B, hidden_dim * 2]


class TextAttentionPooling(nn.Module):
    """
    Attention-based pooling for text sequences instead of just using last token
    """
    def __init__(self, text_dim=1152):
        super().__init__()
        self.attention_weights = nn.Linear(text_dim, 1)
        
    def forward(self, hidden_states, attention_mask):
        # hidden_states: [B, seq_len, hidden_dim]
        # attention_mask: [B, seq_len]
        
        # Compute attention scores
        scores = self.attention_weights(hidden_states).squeeze(-1)  # [B, seq_len]
        
        # Mask out padding tokens
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, seq_len, 1]
        
        # Weighted sum
        pooled_text = (hidden_states * attention_weights).sum(dim=1)  # [B, hidden_dim]
        
        return pooled_text


class MLP(nn.Module):
    """
    Simple MLP with float16 support
    """
    def __init__(self, input_dim, output_dim, hidden_dim=[1024, 1024, 1024], dropout_rate=0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dim:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = h_dim
            
        # Final layer outputs float32 for loss computation stability
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class EnhancedCrossModalModel(nn.Module):
    """
    Your model with cross-modal attention integrated
    """
    def __init__(self, 
                 image_model_frozen=True,
                 text_model_frozen=True,
                 final_mlp_layers=[1024, 1024, 1024],
                 max_token_length=256,
                 dropout_rate=0.1,
                 text_model_name="google/gemma-3-1b-it",
                 proportion_date=0.1,
                 proportion_channel=0.1,
                 cross_modal_hidden_dim=512,
                 cross_modal_heads=8):
        super().__init__()
        
        # Image branch (unchanged)
        self.image_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.image_backbone.head = nn.Identity()
        image_dim = self.image_backbone.norm.normalized_shape[0]  # 768
        
        if image_model_frozen:
            for param in self.image_backbone.parameters():
                param.requires_grad = False
        
        # Text branch - use smaller model
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # Handle different model types
        if "gemma" in text_model_name.lower():
            self.text_backbone = AutoModelForCausalLM.from_pretrained(
                text_model_name, 
                torch_dtype=torch.float32
            )
            text_dim = self.text_backbone.config.hidden_size
            self.is_causal_lm = True
        else:
            # For BERT-style models
            from transformers import AutoModel
            self.text_backbone = AutoModel.from_pretrained(text_model_name)
            text_dim = self.text_backbone.config.hidden_size
            self.is_causal_lm = False
            
        self.max_token_length = max_token_length
        
        if text_model_frozen:
            for param in self.text_backbone.parameters():
                param.requires_grad = False
        
        # NEW: Text attention pooling
        self.text_attention_pooling = TextAttentionPooling(text_dim)
        
        # NEW: Cross-modal attention
        self.cross_modal_attention = CrossModalAttention(
            image_dim=image_dim,
            text_dim=text_dim, 
            hidden_dim=cross_modal_hidden_dim,
            num_heads=cross_modal_heads
        )
        
        # Date and channel embeddings (unchanged)
        self.proportion_date = proportion_date
        self.proportion_channel = proportion_channel
        cross_modal_output_dim = cross_modal_hidden_dim * 2  # Since we concatenate attended features
        
        self.ind_date_dim = int(self.proportion_date * cross_modal_output_dim)
        self.ind_channel_dim = int(self.proportion_channel * cross_modal_output_dim)
        
        # Final MLP
        final_input_dim = cross_modal_output_dim + self.ind_date_dim + self.ind_channel_dim
        self.mlp = MLP(
            input_dim=final_input_dim,
            output_dim=1, 
            hidden_dim=final_mlp_layers, 
            dropout_rate=dropout_rate
        )
        
    def forward(self, batch):
        device = batch["image"].device
        
        # Image processing (keep in float32)
        image_features = self.image_backbone(batch["image"])
        
        # Enhanced text processing
        raw_text = list(batch["text"])
        
        encoded_text = self.tokenizer(
            raw_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)
        
        # Get text features based on model type
        if self.is_causal_lm:
            # For Gemma-style models
            text_outputs = self.text_backbone(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True
            )
            all_hidden_states = text_outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]
        else:
            # For BERT-style models
            text_outputs = self.text_backbone(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            all_hidden_states = text_outputs.last_hidden_state  # [B, seq_len, hidden_dim]
        
        # NEW: Use attention pooling instead of just last token
        text_features = self.text_attention_pooling(all_hidden_states, attention_mask)
        
        # NEW: Cross-modal attention (both inputs are float32)
        cross_modal_features = self.cross_modal_attention(image_features, text_features)
        
        # Date embedding 
        if "date" in batch:
            date = batch["date"].unsqueeze(1).to(device)
            date_embedding = date.repeat(1, self.ind_date_dim).to(dtype=torch.float32)
        
        # Channel embedding
        if "channel" in batch:
            channel = batch["channel"].unsqueeze(1).to(device)
            channel_embedding = channel.repeat(1, self.ind_channel_dim).to(dtype=torch.float32)

        # Combine all features (all in float32)
        combined_features = torch.cat([cross_modal_features, date_embedding, channel_embedding], dim=1)
        
        # Final prediction
        output = self.mlp(combined_features)
        return output