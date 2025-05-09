import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class CrossAttention(nn.Module):
    """
    Module d'attention croisée entre deux modalités
    """
    def __init__(self, query_dim, key_dim, value_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections linéaires pour les requêtes, clés et valeurs
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(key_dim, query_dim)
        self.to_v = nn.Linear(value_dim, query_dim)
        
        # Projection finale
        self.to_out = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query, key, value):
        """
        query: [batch_size, query_len, query_dim]
        key: [batch_size, key_len, key_dim]
        value: [batch_size, value_len, value_dim]
        """
        batch_size = query.shape[0]
        
        # Projections
        q = self.to_q(query)  # [batch_size, query_len, query_dim]
        k = self.to_k(key)    # [batch_size, key_len, query_dim]
        v = self.to_v(value)  # [batch_size, value_len, query_dim]
        
        # Reshape pour l'attention multi-têtes
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch, heads, query_len, head_dim]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch, heads, key_len, head_dim]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [batch, heads, value_len, head_dim]
        
        # Calcul de l'attention
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [batch, heads, query_len, key_len]
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Application de l'attention
        out = torch.matmul(attn_weights, v)  # [batch, heads, query_len, head_dim]
        
        # Reshape pour la sortie
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Projection finale
        return self.to_out(out)


class CrossModalityFusion(nn.Module):
    """
    Module de fusion multimodale avec attention croisée bidirectionnelle
    """
    def __init__(self, img_dim, text_dim, fusion_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Normalisation des entrées
        self.norm_img = nn.LayerNorm(img_dim)
        self.norm_text = nn.LayerNorm(text_dim)
        
        # Attention croisée dans les deux sens
        self.img_to_text_attn = CrossAttention(
            query_dim=text_dim,
            key_dim=img_dim,
            value_dim=img_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.text_to_img_attn = CrossAttention(
            query_dim=img_dim,
            key_dim=text_dim,
            value_dim=text_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Fusion des deux représentations enrichies
        fusion_input_dim = img_dim + text_dim
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        
    def forward(self, img_features, text_features):
        """
        img_features: [batch_size, img_dim]
        text_features: [batch_size, text_dim]
        """
        batch_size = img_features.shape[0]
        
        # Ajouter une dimension de séquence
        img_features = img_features.unsqueeze(1)  # [batch_size, 1, img_dim]
        text_features = text_features.unsqueeze(1)  # [batch_size, 1, text_dim]
        
        # Normalisation des entrées
        img_norm = self.norm_img(img_features)
        text_norm = self.norm_text(text_features)
        
        # Attention croisée bidirectionnelle
        img_enhanced = img_features + self.text_to_img_attn(img_norm, text_norm, text_norm)
        text_enhanced = text_features + self.img_to_text_attn(text_norm, img_norm, img_norm)
        
        # Squeeze pour enlever la dimension de séquence
        img_enhanced = img_enhanced.squeeze(1)
        text_enhanced = text_enhanced.squeeze(1)
        
        # Concaténation des caractéristiques enrichies
        fused_features = torch.cat([img_enhanced, text_enhanced], dim=1)
        
        # MLP final
        output = self.fusion_mlp(fused_features)
        
        return output


class DinoV2BertWithCrossAttention(nn.Module):
    """
    Modèle multimodal complet combinant DinoV2 et BERT avec attention croisée
    """
    def __init__(
        self,
        image_hidden_dims=[1024, 512],
        text_hidden_dims=[768, 512],
        fusion_dim=512,
        output_hidden_dims=[256, 128],
        dropout_rate=0.2,
        freeze_bert=True,
        freeze_dinov2=True,
        num_attention_heads=8,
        bert_model="bert-base-uncased",
        text_max_length=128
    ):
        super().__init__()
        
        # Partie DinoV2 pour l'analyse d'image
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.image_dim = self.backbone.norm.normalized_shape[0]
        
        # Gel du backbone DinoV2
        if freeze_dinov2:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
        # Partie BERT pour l'analyse de texte
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.text_model = BertModel.from_pretrained(bert_model)
        self.text_dim = self.text_model.config.hidden_size  # généralement 768
        self.text_max_length = text_max_length
        
        # Gel de BERT si spécifié
        if freeze_bert:
            for param in self.text_model.parameters():
                param.requires_grad = False
                
        # MLP pour l'analyse d'image
        image_layers = []
        prev_dim = self.image_dim
        
        for hidden_dim in image_hidden_dims:
            image_layers.append(nn.Linear(prev_dim, hidden_dim))
            image_layers.append(nn.LayerNorm(hidden_dim))
            image_layers.append(nn.GELU())
            image_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        self.image_mlp = nn.Sequential(*image_layers)
        self.image_output_dim = prev_dim
        
        # MLP pour l'analyse de texte
        text_layers = []
        prev_dim = self.text_dim
        
        for hidden_dim in text_hidden_dims:
            text_layers.append(nn.Linear(prev_dim, hidden_dim))
            text_layers.append(nn.LayerNorm(hidden_dim))
            text_layers.append(nn.GELU())
            text_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        self.text_mlp = nn.Sequential(*text_layers)
        self.text_output_dim = prev_dim
        
        # Module d'attention croisée pour la fusion
        self.cross_modal_fusion = CrossModalityFusion(
            img_dim=self.image_output_dim,
            text_dim=self.text_output_dim,
            fusion_dim=fusion_dim,
            num_heads=num_attention_heads,
            dropout=dropout_rate
        )
        
        # Couches finales pour la régression
        output_layers = []
        prev_dim = fusion_dim
        
        for hidden_dim in output_hidden_dims:
            output_layers.append(nn.Linear(prev_dim, hidden_dim))
            output_layers.append(nn.LayerNorm(hidden_dim))
            output_layers.append(nn.GELU())
            output_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Couche finale pour la régression
        output_layers.append(nn.Linear(prev_dim, 1))
        
        self.output_mlp = nn.Sequential(*output_layers)
        
    def forward(self, x):
        """
        Forward pass du modèle
        
        Args:
            x: dictionnaire contenant:
               - "image": tensor de forme [batch_size, 3, H, W]
               - "text": liste de textes [batch_size]
        
        Returns:
            Prédictions de nombre de vues
        """
        batch_size = x["image"].size(0)
        device = x["image"].device
        
        # Extraction des caractéristiques de l'image
        image_features = self.backbone(x["image"])
        image_features = self.image_mlp(image_features)
        
        # Extraction des caractéristiques du texte
        text_inputs = self.tokenizer(
            x["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length
        ).to(device)
        
        text_outputs = self.text_model(**text_inputs)
        text_features = text_outputs.pooler_output
        text_features = self.text_mlp(text_features)
        
        # Fusion des caractéristiques par attention croisée
        fused_features = self.cross_modal_fusion(image_features, text_features)
        
        # Prédiction finale
        output = self.output_mlp(fused_features)
        
        return output