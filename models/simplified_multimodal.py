import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class MLP(nn.Module):
    """
    MLP amélioré avec LayerNorm bien placé
    """
    def __init__(self, input_dim, output_dim, hidden_dim=[1024, 1024, 1024], 
                 dropout_rate=0.1, use_layer_norm=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm
        
        layers = []
        
        # Première couche
        layers.append(nn.Linear(input_dim, hidden_dim[0]))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim[0]))  # LayerNorm APRÈS Linear
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Couches cachées
        for i in range(1, len(hidden_dim)):
            layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim[i]))  # LayerNorm sur chaque couche cachée
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Couche finale (SANS LayerNorm pour la sortie)
        layers.append(nn.Linear(hidden_dim[-1], output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class ViralAwareMultimodal(nn.Module):
    """
    Modèle multimodal avec classification virale/normale
    """
    def __init__(self, 
                 image_model_frozen=True,
                 text_model_frozen=True,
                 final_mlp_layers=[1024, 1024, 1024],
                 image_head=[1024, 1024, 1024],
                 text_head=[1024, 1024, 1024],
                 max_token_length=256,
                 dropout_rate=0.1,
                 text_model_name="google/gemma-3-1b-it",
                 proportion_date=0.1,
                 proportion_channel=0.1,
                 viral_threshold_percentile=90,
                 viral_weight=0.3):
        super().__init__()
        
        # Image branch
        self.image_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        image_dim = self.image_backbone.norm.normalized_shape[0]  # 768
        self.image_backbone.head = MLP(input_dim=image_dim, output_dim=image_dim, hidden_dim=image_head, dropout_rate=dropout_rate)
        
        if image_model_frozen:
            for param in self.image_backbone.parameters():
                param.requires_grad = False
        
        # Text branch
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_backbone = AutoModelForCausalLM.from_pretrained(text_model_name,torch_dtype=torch.float16)
        text_dim = self.text_backbone.config.hidden_size # 1152
        self.max_token_length = max_token_length
        self.text_head_mlp = MLP(input_dim=text_dim, output_dim=text_dim, hidden_dim=text_head, dropout_rate=dropout_rate)

        if text_model_frozen:
            for param in self.text_backbone.parameters():
                param.requires_grad = False
        
        # Metadata dimensions
        self.proportion_date = proportion_date
        self.ind_date_dim = max(32, int(self.proportion_date * (image_dim + text_dim)))  # Minimum de 32 dimensions
        self.proportion_channel = proportion_channel
        self.ind_channel_dim = int(self.proportion_channel * (image_dim + text_dim))

        # Classification virale
        self.viral_threshold_percentile = viral_threshold_percentile
        self.viral_weight = viral_weight
        
        # Classificateur viral (détermine si la vidéo sera virale ou non)
        combined_dim = image_dim + text_dim + self.ind_date_dim + self.ind_channel_dim
        self.viral_classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probabilité d'être viral
        )
        
        # Deux régresseurs séparés
        self.viral_regressor = MLP(
            input_dim=combined_dim, 
            output_dim=1, 
            hidden_dim=final_mlp_layers, 
            dropout_rate=dropout_rate
        )
        
        self.normal_regressor = MLP(
            input_dim=combined_dim, 
            output_dim=1, 
            hidden_dim=final_mlp_layers, 
            dropout_rate=dropout_rate
        )
        
        # Buffer pour le seuil viral (mis à jour pendant l'entraînement)
        self.register_buffer('viral_threshold', torch.tensor(1000000.0))  # Valeur initiale

        # Enhanced date encoding layers
        self._setup_date_embedding()
        
    def _setup_date_embedding(self):
        """Configure l'embedding avancé pour les dates"""
        # Embedding sinusoidal pour capturer les patterns cycliques
        self.date_embedding_dim = 16  # Dimensions pour l'embedding sinusoidal
        
        # MLP pour transformer l'embedding de date
        self.date_mlp = nn.Sequential(
            nn.Linear(self.date_embedding_dim + 1, 64),  # +1 pour la date normalisée
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, self.ind_date_dim),
            nn.Tanh()  # Normalise la sortie
        )
        
    def _create_date_embedding(self, dates, device, dtype):
        """
        Crée un embedding riche pour les dates qui capture :
        - Patterns saisonniers (jour de l'année)
        - Patterns hebdomadaires (jour de la semaine)
        - Tendances temporelles
        """
        batch_size = len(dates)
        # S'assurer que tout est en float32 dès le début
        dates = dates.to(device=device, dtype=torch.float32)
        
        # 1. Date normalisée (déjà fournie, représente le jour dans l'année [0,1])
        normalized_date = dates.unsqueeze(1)  # [batch_size, 1]
        
        # 2. Embeddings sinusoidaux pour capturer les cycles
        # Créer différentes fréquences pour capturer différents patterns temporels
        freqs = torch.linspace(1, self.date_embedding_dim // 2, self.date_embedding_dim // 2, 
                              device=device, dtype=torch.float32)
        
        # Patterns annuels (basés sur la date normalisée [0,1])
        angles = 2 * torch.pi * dates.unsqueeze(1) * freqs.unsqueeze(0)  # [batch_size, date_embedding_dim//2]
        
        # Fonctions sin et cos pour capturer les cycles
        sin_embedding = torch.sin(angles)  # [batch_size, date_embedding_dim//2]
        cos_embedding = torch.cos(angles)  # [batch_size, date_embedding_dim//2]
        
        # Combiner sin et cos
        sinusoidal_embedding = torch.cat([sin_embedding, cos_embedding], dim=1)  # [batch_size, date_embedding_dim]
        
        # 3. Combiner date normalisée et embedding sinusoidal
        full_date_features = torch.cat([normalized_date, sinusoidal_embedding], dim=1)  # [batch_size, date_embedding_dim + 1]
        
        # 4. Passer par le MLP pour obtenir la dimension finale
        date_embedding = self.date_mlp(full_date_features)  # [batch_size, ind_date_dim]
        
        # Convertir au bon dtype final
        return date_embedding.to(dtype=dtype)

        
        
    def _update_viral_threshold(self, targets):
        """Met à jour le seuil viral basé sur les vraies valeurs"""
        if self.training and targets is not None:
            with torch.no_grad():
                threshold = torch.quantile(targets, self.viral_threshold_percentile / 100.0)
                # Mise à jour avec momentum pour la stabilité
                self.viral_threshold = 0.9 * self.viral_threshold + 0.1 * threshold
    
    def forward(self, batch, targets=None):
        # Mise à jour du seuil viral si en mode entraînement
        if targets is not None:
            self._update_viral_threshold(targets)
        
        # Image processing
        image_features = self.image_backbone(batch["image"])
        
        # Text processing
        raw_text = list(batch["text"])
        
        encoded_text = self.tokenizer(
            raw_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        device = batch["image"].device
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)
        
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = text_outputs.hidden_states[-1]
        last_token_hidden = last_hidden_state[:, -1, :]
        last_token_hidden = last_token_hidden.to(dtype=torch.float32)

        # Enhanced date embedding
        if "date" in batch:
            date_embedding = self._create_date_embedding(batch["date"], device, image_features.dtype)
        else:
            date_embedding = torch.zeros(batch["image"].size(0), self.ind_date_dim, 
                                       device=device, dtype=image_features.dtype)

        if "channel" in batch:
            channel = batch["channel"].unsqueeze(1).to(device)
            channel_embedding = channel.repeat(1, self.ind_channel_dim).to(dtype=image_features.dtype)

        # Combine features
        combined_features = torch.cat((image_features, last_token_hidden, date_embedding, channel_embedding), dim=1)
        
        # Classification virale (probabilité d'être viral)
        viral_prob = self.viral_classifier(combined_features)
        
        # Prédictions des deux régresseurs
        viral_pred = self.viral_regressor(combined_features)
        normal_pred = self.normal_regressor(combined_features)
        
        # Combinaison pondérée des prédictions
        # viral_prob proche de 1 = vidéo virale, proche de 0 = vidéo normale
        final_pred = viral_prob * viral_pred + (1 - viral_prob) * normal_pred
        
        # Retourner les prédictions et informations supplémentaires
        result = {
            'prediction': final_pred.squeeze(),
            'viral_probability': viral_prob.squeeze(),
            'viral_prediction': viral_pred.squeeze(),
            'normal_prediction': normal_pred.squeeze(),
            'viral_threshold': self.viral_threshold.item()
        }
        
        return result
    
    def get_viral_stats(self, batch, targets=None):
        """Retourne des statistiques sur la classification viral/normal"""
        with torch.no_grad():
            if targets is not None:
                viral_mask = targets > self.viral_threshold
                stats = {
                    'viral_threshold': self.viral_threshold.item(),
                    'viral_count': viral_mask.sum().item(),
                    'normal_count': (~viral_mask).sum().item(),
                    'viral_percentage': (viral_mask.sum().float() / len(targets) * 100).item()
                }
            else:
                # Utiliser les prédictions de probabilité virale
                result = self.forward(batch)
                viral_probs = result['viral_probability']
                predicted_viral = viral_probs > 0.5
                stats = {
                    'viral_threshold': self.viral_threshold.item(),
                    'predicted_viral_count': predicted_viral.sum().item(),
                    'predicted_normal_count': (~predicted_viral).sum().item(),
                    'avg_viral_probability': viral_probs.mean().item()
                }
        return stats


# Classe de compatibilité pour remplacer facilement l'ancien modèle
class EnhancedPhase1LargeMLP(ViralAwareMultimodal):
    """
    Wrapper de compatibilité - utilise le nouveau modèle viral-aware
    mais retourne seulement la prédiction pour compatibilité
    """
    def forward(self, batch, targets=None):
        # Si appelé avec targets (mode entraînement), passer les targets
        if "target" in batch:
            result = super().forward(batch, batch["target"])
        else:
            result = super().forward(batch, targets)
        
        # Retourner seulement la prédiction pour compatibilité avec l'ancien code
        return result['prediction']