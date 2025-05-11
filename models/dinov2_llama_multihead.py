import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class MultiHeadAttentionLayer(nn.Module):
    """
    Couche Multi-Head Attention personnalisée pour fusion multimodale
    """
    def __init__(self, hidden_dim, n_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections pour Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Projection de sortie
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Normalisation et dropout
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Skip connection input
        residual = x
        
        # Normalize
        x = self.norm(x)
        
        # Compute Q, K, V
        q = self.q_proj(x)  # [batch_size, seq_len, hidden_dim]
        k = self.k_proj(x)  # [batch_size, seq_len, hidden_dim]
        v = self.v_proj(x)  # [batch_size, seq_len, hidden_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch_size, n_heads, seq_len, seq_len]
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        out = torch.matmul(attn_weights, v)  # [batch_size, n_heads, seq_len, head_dim]
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)  # [batch_size, seq_len, hidden_dim]
        
        # Output projection
        out = self.out_proj(out)
        
        # Skip connection
        out = residual + self.dropout(out)
        
        return out


class DinoV2LLamaMultiHead(nn.Module):
    """
    Modèle multimodal combinant DinoV2 et LLama avec Multi-Head Attention
    """
    def __init__(
        self,
        # Paramètres DinoV2
        freeze_dinov2=True,
        dinov2_output_dim=768,  # Dimension de sortie après projection
        
        # Paramètres LLama
        llama_model_name="microsoft/DialoGPT-small",  # Alternative légère à LLama
        freeze_llama=True,
        text_max_length=128,
        
        # Paramètres Multi-Head Attention
        n_attention_layers=3,
        n_attention_heads=8,
        attention_dropout=0.1,
        
        # Paramètres MLP final
        final_hidden_dims=[512, 256, 128],
        final_dropout=0.2
    ):
        super().__init__()
        
        # --- DinoV2 ---
        self.dinov2_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.dinov2_backbone.head = nn.Identity()
        self.dinov2_dim = self.dinov2_backbone.norm.normalized_shape[0]  # 768 for vitb14
        
        if freeze_dinov2:
            for param in self.dinov2_backbone.parameters():
                param.requires_grad = False
        
        # MLP pour projeter DinoV2 vers la dimension de LLama
        self.dinov2_projection = nn.Sequential(
            nn.Linear(self.dinov2_dim, dinov2_output_dim),
            nn.LayerNorm(dinov2_output_dim),
            nn.GELU(),
            nn.Dropout(attention_dropout)
        )
        
        # --- LLama (ou alternative) ---
        # Note: Pour LLama original, utilisez "meta-llama/Llama-2-7b-hf" 
        # mais il nécessite une authentification HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
        self.llama_model = AutoModel.from_pretrained(llama_model_name)
        self.llama_dim = self.llama_model.config.hidden_size
        self.text_max_length = text_max_length
        
        # Ajouter un token de padding si nécessaire
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if freeze_llama:
            for param in self.llama_model.parameters():
                param.requires_grad = False
        
        # Projection pour aligner LLama avec la dimension commune
        self.llama_projection = nn.Sequential(
            nn.Linear(self.llama_dim, dinov2_output_dim),
            nn.LayerNorm(dinov2_output_dim),
            nn.GELU(),
            nn.Dropout(attention_dropout)
        )
        
        # --- Multi-Head Attention Layers ---
        self.attention_layers = nn.ModuleList([
            MultiHeadAttentionLayer(
                hidden_dim=dinov2_output_dim,
                n_heads=n_attention_heads,
                dropout=attention_dropout
            ) for _ in range(n_attention_layers)
        ])
        
        # --- MLP Final ---
        final_layers = []
        prev_dim = dinov2_output_dim * 2  # On concatène image et texte
        
        for hidden_dim in final_hidden_dims:
            final_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(final_dropout)
            ])
            prev_dim = hidden_dim
        
        # Couche de sortie pour la régression
        final_layers.append(nn.Linear(prev_dim, 1))
        
        self.final_mlp = nn.Sequential(*final_layers)
        
    def forward(self, x):
        """
        Forward pass du modèle
        
        Args:
            x: dictionnaire contenant:
               - "image": tensor de forme [batch_size, 3, H, W]
               - "text": liste de textes [batch_size]
        
        Returns:
            Prédictions de nombre de vues [batch_size, 1]
        """
        batch_size = x["image"].size(0)
        device = x["image"].device
        
        # --- Traitement de l'image avec DinoV2 ---
        image_features = self.dinov2_backbone(x["image"])  # [batch_size, dinov2_dim]
        image_features = self.dinov2_projection(image_features)  # [batch_size, dinov2_output_dim]
        
        # --- Traitement du texte avec LLama ---
        text_inputs = self.tokenizer(
            x["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length
        ).to(device)
        
        # Forward pass à travers LLama
        text_outputs = self.llama_model(**text_inputs)
        
        # Utiliser le pooling ou la dernière couche cachée
        if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
            text_features = text_outputs.pooler_output
        else:
            # Prendre la moyenne des embeddings de la séquence
            text_features = text_outputs.last_hidden_state.mean(dim=1)
        
        text_features = self.llama_projection(text_features)  # [batch_size, dinov2_output_dim]
        
        # --- Préparation pour Multi-Head Attention ---
        # Ajouter une dimension de séquence pour traiter comme des séquences
        image_sequence = image_features.unsqueeze(1)  # [batch_size, 1, dinov2_output_dim]
        text_sequence = text_features.unsqueeze(1)    # [batch_size, 1, dinov2_output_dim]
        
        # Concaténer les séquences
        combined_sequence = torch.cat([image_sequence, text_sequence], dim=1)  # [batch_size, 2, dinov2_output_dim]
        
        # --- Application des couches Multi-Head Attention ---
        attention_output = combined_sequence
        for attention_layer in self.attention_layers:
            attention_output = attention_layer(attention_output)
        
        # Extraire les features finales
        final_image_features = attention_output[:, 0, :]  # [batch_size, dinov2_output_dim]
        final_text_features = attention_output[:, 1, :]   # [batch_size, dinov2_output_dim]
        
        # --- Concaténation finale ---
        final_features = torch.cat([final_image_features, final_text_features], dim=1)  # [batch_size, dinov2_output_dim * 2]
        
        # --- Prédiction finale ---
        output = self.final_mlp(final_features)  # [batch_size, 1]
        
        return output