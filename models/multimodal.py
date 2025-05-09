import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class MultiModalModel(nn.Module):
    """
    Modèle multimodal combinant DinoV2 pour l'analyse d'images et BERT pour l'analyse de texte
    """
    def __init__(
        self,
        image_hidden_dims=[1024, 512, 256],
        text_hidden_dims=[768, 512],
        fusion_hidden_dims=[512, 256],
        dropout_rate=0.2,
        freeze_bert=True,
        bert_model="bert-base-uncased",
        text_max_length=128
    ):
        super().__init__()
        
        # Partie DinoV2 pour l'analyse d'image
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.image_dim = self.backbone.norm.normalized_shape[0]
        
        # Gel du backbone DinoV2
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
        
        # Fusion des caractéristiques vision et texte
        fusion_layers = []
        prev_dim = self.image_output_dim + self.text_output_dim
        
        for hidden_dim in fusion_hidden_dims:
            fusion_layers.append(nn.Linear(prev_dim, hidden_dim))
            fusion_layers.append(nn.LayerNorm(hidden_dim))
            fusion_layers.append(nn.GELU())
            fusion_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Couche finale pour la régression
        fusion_layers.append(nn.Linear(prev_dim, 1))
        
        self.fusion_mlp = nn.Sequential(*fusion_layers)
        
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
        
        # Fusion des caractéristiques
        combined_features = torch.cat([image_features, text_features], dim=1)
        output = self.fusion_mlp(combined_features)
        
        return output