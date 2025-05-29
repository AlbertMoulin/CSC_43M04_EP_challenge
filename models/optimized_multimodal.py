import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class MultiModalAttentionFusion(nn.Module):
    """
    Module de fusion avec attention cross-modale
    """
    def __init__(self, vision_dim, text_dim, numeric_dim, hidden_dim=256):
        super().__init__()
        
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.numeric_proj = nn.Linear(numeric_dim, hidden_dim)
        
        # Cross-attention entre vision et texte
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Self-attention pour la fusion finale
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # MLP final
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, vision_emb, text_emb, numeric_emb):
        batch_size = vision_emb.size(0)
        
        # Projeter toutes les modalités dans le même espace
        vision_proj = self.vision_proj(vision_emb).unsqueeze(1)  # [B, 1, hidden_dim]
        text_proj = self.text_proj(text_emb).unsqueeze(1)        # [B, 1, hidden_dim]
        numeric_proj = self.numeric_proj(numeric_emb).unsqueeze(1) # [B, 1, hidden_dim]
        
        # Cross-attention vision-texte
        vision_attended, _ = self.cross_attention(
            vision_proj, text_proj, text_proj
        )
        vision_attended = self.layer_norm1(vision_attended + vision_proj)
        
        # Combiner toutes les modalités
        all_modalities = torch.cat([vision_attended, text_proj, numeric_proj], dim=1)  # [B, 3, hidden_dim]
        
        # Self-attention pour fusion finale
        fused, _ = self.self_attention(all_modalities, all_modalities, all_modalities)
        fused = self.layer_norm2(fused + all_modalities)
        
        # Flatten et prédiction finale
        fused_flat = fused.view(batch_size, -1)  # [B, 3 * hidden_dim]
        prediction = self.fusion_mlp(fused_flat)
        
        return prediction.squeeze()


class OptimizedMultiModalModel(nn.Module):
    """
    Modèle multimodal optimisé basé sur l'analyse des features critiques
    """
    def __init__(self, 
                 text_model_name="bert-base-uncased",  # Plus léger que Gemma
                 max_token_length=512,  # Plus long pour title + description
                 freeze_vision=True,
                 freeze_text=True,
                 dropout_rate=0.3):
        super().__init__()
        
        # === VISION ENCODER ===
        print("Loading DINOv2 vision model...")
        
        # Désactiver xFormers warnings temporairement
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.vision_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg", force_reload=False)
        
        self.vision_backbone.head = nn.Identity()
        vision_dim = 768  # DINOv2 ViT-B/14
        
        if freeze_vision:
            for param in self.vision_backbone.parameters():
                param.requires_grad = False
            print("Vision backbone frozen")
            
        # Vision head
        self.vision_head = nn.Sequential(
            nn.Linear(vision_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256)
        )
        
        # === TEXT ENCODER ===
        print(f"Loading {text_model_name} text model...")
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Chargement GPU-safe du modèle BERT
        try:
            # Essayer de charger avec configuration optimisée si GPU disponible
            if torch.cuda.is_available():
                self.text_backbone = AutoModel.from_pretrained(text_model_name)
            else:
                # Configuration CPU-friendly
                from transformers import BertConfig, BertModel
                config = BertConfig.from_pretrained(text_model_name)
                # Désactiver certaines optimisations problématiques sur CPU
                config.attention_type = "original_full"
                self.text_backbone = BertModel.from_pretrained(text_model_name, config=config)
        except Exception as e:
            print(f"Warning: Issue loading {text_model_name}, falling back to basic config: {e}")
            # Fallback simple
            from transformers import BertModel
            self.text_backbone = BertModel.from_pretrained("bert-base-uncased")
            
        text_dim = self.text_backbone.config.hidden_size  # 768 pour BERT
        self.max_token_length = max_token_length
        
        if freeze_text:
            for param in self.text_backbone.parameters():
                param.requires_grad = False
            print("Text backbone frozen")
            
        # Text head
        self.text_head = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256)
        )
        
        # === NUMERIC FEATURES ENCODER ===
        # 17 features critiques identifiées (pas 16)
        numeric_dim = 17  # CORRECTION: 17 features dans le dataset
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # === FUSION MODULE ===
        self.fusion = MultiModalAttentionFusion(
            vision_dim=256,  # Output du vision head
            text_dim=256,    # Output du text head  
            numeric_dim=32,  # Output du numeric encoder
            hidden_dim=256
        )
        
        print("Model initialized successfully!")
        
    def forward(self, batch):
        device = batch["image"].device
        batch_size = batch["image"].size(0)
        
        # === VISION PATHWAY ===
        with torch.set_grad_enabled(not hasattr(self, '_vision_frozen') or not self._vision_frozen):
            vision_features = self.vision_backbone(batch["image"])
        vision_emb = self.vision_head(vision_features)
        
        # === TEXT PATHWAY ===
        # Tokenizer le texte (title + description)
        texts = batch["text"] if isinstance(batch["text"], list) else batch["text"].tolist()
        
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_token_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Passer par le modèle texte
        with torch.set_grad_enabled(not hasattr(self, '_text_frozen') or not self._text_frozen):
            text_outputs = self.text_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Utiliser [CLS] token (premier token) pour BERT
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_emb = self.text_head(text_features)
        
        # === NUMERIC FEATURES PATHWAY ===
        numeric_features = batch["numeric_features"].to(device)
        numeric_emb = self.numeric_encoder(numeric_features)
        
        # === FUSION ===
        prediction = self.fusion(vision_emb, text_emb, numeric_emb)
        
        return prediction
        
    def get_feature_importance(self, batch):
        """
        Analyse l'importance des différentes modalités
        """
        self.eval()
        with torch.no_grad():
            device = batch["image"].device
            
            # Prédiction complète
            full_pred = self.forward(batch)
            
            # Prédiction vision seule (zéroiser les autres)
            batch_vision = batch.copy()
            batch_vision["text"] = [""] * len(batch["text"])
            batch_vision["numeric_features"] = torch.zeros_like(batch["numeric_features"])
            vision_pred = self.forward(batch_vision)
            
            # Importance relative
            vision_importance = torch.abs(vision_pred - full_pred.mean()).mean()
            
            return {
                'full_prediction': full_pred.mean().item(),
                'vision_importance': vision_importance.item()
            }


# === CLASSE DE COMPATIBILITÉ ===
class EnhancedPhase1LargeMLP(OptimizedMultiModalModel):
    """
    Classe de compatibilité pour remplacer l'ancien modèle
    """
    def __init__(self, 
                 image_model_frozen=True,
                 text_model_frozen=True,
                 final_mlp_layers=None,  # Ignoré
                 image_head=None,       # Ignoré
                 text_head=None,        # Ignoré
                 max_token_length=512,
                 dropout_rate=0.3,
                 text_model_name="bert-base-uncased",  # Changé de Gemma à BERT
                 proportion_date=None,  # Ignoré
                 proportion_channel=None):  # Ignoré
        
        super().__init__(
            text_model_name=text_model_name,
            max_token_length=max_token_length,
            freeze_vision=image_model_frozen,
            freeze_text=text_model_frozen,
            dropout_rate=dropout_rate
        )
        print("Compatibility wrapper initialized - using optimized architecture")


# === FONCTION DE TEST ===
def test_model_architecture():
    """Test basic du modèle"""
    
    # Créer un batch de test
    batch_size = 4
    batch = {
        "image": torch.randn(batch_size, 3, 224, 224),
        "text": [
            "CGI Animated Short Film: Test Video [SEP] This is a test description with more details about the content.",
            "Tutorial: How to use AI [SEP] Learn artificial intelligence in this comprehensive guide.",
            "Gaming highlights compilation [SEP] Best moments from recent gaming sessions.",
            "Short film about nature [SEP] Beautiful documentary about wildlife and landscapes."
        ],
        "numeric_features": torch.randn(batch_size, 16),  # 16 features critiques
        "target": torch.tensor([100000., 50000., 200000., 80000.])
    }
    
    # Créer le modèle
    model = OptimizedMultiModalModel(
        text_model_name="bert-base-uncased",
        max_token_length=256,
        freeze_vision=True,
        freeze_text=True
    )
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(batch)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions: {predictions}")
        
    print("Model test successful!")
    return model


if __name__ == "__main__":
    test_model_architecture()