import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class SelfAttentionLayer(nn.Module):
    """
    Couche d'auto-attention avec LayerNorm et skip connection
    """
    def __init__(self, hidden_dim, n_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, hidden_dim]
        """
        # Self-attention avec skip connection
        attn_output, _ = self.attention(x, x, x)
        output = self.norm(x + self.dropout(attn_output))
        return output

class DinoV2BertSelfAttention(nn.Module):
    """
    Modèle combinant DinoV2 et BERT avec 2 couches d'auto-attention chacun,
    suivi d'une concaténation et d'un MLP final
    """
    def __init__(
        self,
        # Paramètres DinoV2
        dinov2_hidden_dims=[1024, 512],
        freeze_dinov2=True,
        
        # Paramètres BERT
        bert_model="bert-base-uncased",
        bert_hidden_dims=[768, 512],
        freeze_bert=True,
        text_max_length=128,
        
        # Paramètres auto-attention
        n_attention_heads=8,
        attention_dropout=0.1,
        
        # Paramètres MLP final
        final_hidden_dims=[512, 256, 128],
        final_dropout=0.2
    ):
        super().__init__()
        
        # --- DinoV2 ---
        self.backbone_dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone_dinov2.head = nn.Identity()
        self.dinov2_dim = self.backbone_dinov2.norm.normalized_shape[0]
        
        if freeze_dinov2:
            for param in self.backbone_dinov2.parameters():
                param.requires_grad = False
        
        # MLP pour projeter DinoV2 features
        self.dinov2_proj = self._build_projection_mlp(self.dinov2_dim, dinov2_hidden_dims, attention_dropout)
        self.dinov2_final_dim = dinov2_hidden_dims[-1] if dinov2_hidden_dims else self.dinov2_dim
        
        # 2 couches d'auto-attention pour DinoV2
        self.dinov2_self_attn1 = SelfAttentionLayer(self.dinov2_final_dim, n_attention_heads, attention_dropout)
        self.dinov2_self_attn2 = SelfAttentionLayer(self.dinov2_final_dim, n_attention_heads, attention_dropout)
        
        # --- BERT ---
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.backbone_bert = BertModel.from_pretrained(bert_model)
        self.bert_dim = self.backbone_bert.config.hidden_size
        self.text_max_length = text_max_length
        
        if freeze_bert:
            for param in self.backbone_bert.parameters():
                param.requires_grad = False
        
        # MLP pour projeter BERT features
        self.bert_proj = self._build_projection_mlp(self.bert_dim, bert_hidden_dims, attention_dropout)
        self.bert_final_dim = bert_hidden_dims[-1] if bert_hidden_dims else self.bert_dim
        
        # 2 couches d'auto-attention pour BERT
        self.bert_self_attn1 = SelfAttentionLayer(self.bert_final_dim, n_attention_heads, attention_dropout)
        self.bert_self_attn2 = SelfAttentionLayer(self.bert_final_dim, n_attention_heads, attention_dropout)
        
        # --- MLP Final ---
        combined_dim = self.dinov2_final_dim + self.bert_final_dim
        self.final_mlp = self._build_final_mlp(combined_dim, final_hidden_dims, final_dropout)
        
    def _build_projection_mlp(self, input_dim, hidden_dims, dropout_rate):
        """Construit le MLP de projection pour DinoV2 ou BERT"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        return nn.Sequential(*layers) if layers else nn.Identity()
    
    def _build_final_mlp(self, input_dim, hidden_dims, dropout_rate):
        """Construit le MLP final pour la prédiction"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Couche de sortie pour la régression
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
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
        dinov2_features = self.backbone_dinov2(x["image"])  # [batch_size, dinov2_dim]
        dinov2_features = self.dinov2_proj(dinov2_features)  # [batch_size, dinov2_final_dim]
        
        # Ajouter une dimension pour les couches d'attention (séquence de longueur 1)
        dinov2_features = dinov2_features.unsqueeze(1)  # [batch_size, 1, dinov2_final_dim]
        
        # Application des 2 couches d'auto-attention
        dinov2_features = self.dinov2_self_attn1(dinov2_features)
        dinov2_features = self.dinov2_self_attn2(dinov2_features)
        
        # Retirer la dimension de séquence
        dinov2_features = dinov2_features.squeeze(1)  # [batch_size, dinov2_final_dim]
        
        # --- Traitement du texte avec BERT ---
        text_inputs = self.tokenizer(
            x["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length
        ).to(device)
        
        bert_outputs = self.backbone_bert(**text_inputs)
        bert_features = bert_outputs.pooler_output  # [batch_size, bert_dim]
        bert_features = self.bert_proj(bert_features)  # [batch_size, bert_final_dim]
        
        # Ajouter une dimension pour les couches d'attention (séquence de longueur 1)
        bert_features = bert_features.unsqueeze(1)  # [batch_size, 1, bert_final_dim]
        
        # Application des 2 couches d'auto-attention
        bert_features = self.bert_self_attn1(bert_features)
        bert_features = self.bert_self_attn2(bert_features)
        
        # Retirer la dimension de séquence
        bert_features = bert_features.squeeze(1)  # [batch_size, bert_final_dim]
        
        # --- Concaténation des caractéristiques ---
        combined_features = torch.cat([dinov2_features, bert_features], dim=1)  # [batch_size, combined_dim]
        
        # --- Prédiction finale ---
        output = self.final_mlp(combined_features)  # [batch_size, 1]
        
        return output