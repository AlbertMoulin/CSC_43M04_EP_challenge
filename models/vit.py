import torch
import torch.nn as nn
# import timm
from transformers import ViTModel

class ViTForRegression(nn.Module):
    def __init__(
        self,
        pretrained=True,
        frozen_layers=8,
        hidden_dims=[1024, 512, 256],
        dropout_rate=0.2,
        use_text=False,
        vit_model="google/vit-base-patch16-224"
    ):
        super().__init__()
        
        # Chargement du modèle ViT préentraîné
        self.backbone = ViTModel.from_pretrained(vit_model)
        self.hidden_size = self.backbone.config.hidden_size  # Typiquement 768
        
        # Gel des couches si spécifié
        if frozen_layers > 0:
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False
                
            for i, layer in enumerate(self.backbone.encoder.layer):
                if i < frozen_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Configuration pour l'utilisation de texte
        self.use_text = use_text
        if use_text:
            # On utilisera un tokenizer BERT simple pour le texte
            from transformers import BertTokenizer, BertModel
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.text_model = BertModel.from_pretrained('bert-base-uncased')
            text_dim = 768  # Dimension d'embedding BERT
            
            # Tête de régression avec fusion de la vision et du texte
            head_input_dim = self.hidden_size + text_dim
        else:
            head_input_dim = self.hidden_size
        
        # Construction de la tête de régression (MLP)
        layers = []
        prev_dim = head_input_dim
        
        # Création des couches cachées
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Couche de sortie
        layers.append(nn.Linear(prev_dim, 1))
        
        self.regression_head = nn.Sequential(*layers)
    
    def forward(self, x):
        # Traitement de l'image avec ViT
        img_outputs = self.backbone(x["image"])
        img_embedding = img_outputs.pooler_output  # [batch_size, hidden_size]
        
        # Traitement du texte si activé
        if self.use_text and "text" in x:
            # Tokenization des textes d'entrée
            tokens = self.tokenizer(
                x["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=128,
                return_tensors="pt"
            ).to(x["image"].device)
            
            # Obtention des embeddings textuels
            with torch.no_grad():
                text_outputs = self.text_model(**tokens)
            text_embedding = text_outputs.pooler_output  # [batch_size, 768]
            
            # Concaténation des embeddings vision et texte
            combined_embedding = torch.cat([img_embedding, text_embedding], dim=1)
            output = self.regression_head(combined_embedding)
        else:
            # Utilisation uniquement de l'image
            output = self.regression_head(img_embedding)
            
        return output

# # Variante qui utilise timm pour charger un ViT (plus d'options et potentiellement plus rapide)
# class TimmViT(nn.Module):
#     def __init__(
#         self,
#         model_name='vit_base_patch16_224',
#         pretrained=True,
#         frozen_backbone=False,
#         hidden_dims=[1024, 512, 256],
#         dropout_rate=0.2,
#         use_text=False
#     ):
#         super().__init__()
        
#         # Chargement du modèle ViT de timm
#         self.backbone = timm.create_model(
#             model_name,
#             pretrained=pretrained,
#             num_classes=0  # Supprime la tête de classification
#         )
#         self.hidden_size = self.backbone.num_features
        
#         # Gel du backbone si demandé
#         if frozen_backbone:
#             for param in self.backbone.parameters():
#                 param.requires_grad = False
        
#         # Configuration pour l'utilisation de texte
#         self.use_text = use_text
#         if use_text:
#             from transformers import BertTokenizer, BertModel
#             self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#             self.text_model = BertModel.from_pretrained('bert-base-uncased')
#             text_dim = 768
#             head_input_dim = self.hidden_size + text_dim
#         else:
#             head_input_dim = self.hidden_size
        
#         # Construction de la tête de régression
#         layers = []
#         prev_dim = head_input_dim
        
#         for hidden_dim in hidden_dims:
#             layers.append(nn.Linear(prev_dim, hidden_dim))
#             layers.append(nn.LayerNorm(hidden_dim))
#             layers.append(nn.GELU())
#             layers.append(nn.Dropout(dropout_rate))
#             prev_dim = hidden_dim
        
#         layers.append(nn.Linear(prev_dim, 1))
        
#         self.regression_head = nn.Sequential(*layers)
    
#     def forward(self, x):
#         # Traitement de l'image
#         img_embedding = self.backbone(x["image"])
        
#         # Traitement du texte si activé
#         if self.use_text and "text" in x:
#             tokens = self.tokenizer(
#                 x["text"], 
#                 padding="max_length", 
#                 truncation=True, 
#                 max_length=128,
#                 return_tensors="pt"
#             ).to(x["image"].device)
            
#             with torch.no_grad():
#                 text_outputs = self.text_model(**tokens)
#             text_embedding = text_outputs.pooler_output
            
#             combined_embedding = torch.cat([img_embedding, text_embedding], dim=1)
#             output = self.regression_head(combined_embedding)
#         else:
#             output = self.regression_head(img_embedding)
            
#         return output