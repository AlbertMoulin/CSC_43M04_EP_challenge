import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer # Nécessite la bibliothèque transformers

# Vous pourriez vouloir définir la longueur maximale des tokens ici ou dans la configuration Hydra
MAX_TOKEN_LENGTH = 128

class ImageBranch(nn.Module):
    """
    Branche de traitement de l'image : DinoV2 + MLP.
    """
    def __init__(self, dinov2_model, mlp_layers):
        super().__init__()
        self.backbone = dinov2_model # Le modèle DinoV2 (sans la tête)
        layers = []
        # La dimension d'entrée du MLP est la dimension de sortie de DinoV2
        input_dim = dinov2_model.norm.normalized_shape[0] # Devrait être 768 pour dinov2_vitb14_reg

        # Construction des couches du MLP
        for output_dim in mlp_layers[:-1]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU()) # Utilisation de ReLU comme fonction d'activation
            input_dim = output_dim

        # Couche de sortie (prédiction unique)
        layers.append(nn.Linear(input_dim, mlp_layers[-1])) # mlp_layers[-1] doit être 1

        self.mlp = nn.Sequential(*layers)

    def forward(self, image):
        # Obtenir les caractéristiques de l'image via DinoV2
        x = self.backbone(image)
        # Passer les caractéristiques à travers le MLP
        x = self.mlp(x)
        return x

class TextBranch(nn.Module):
    """
    Branche de traitement du texte : Modèle Transformer + MLP.
    Note : La tokenisation est gérée dans la méthode forward du CombinedModel.
    Cette branche prend directement les input_ids et attention_mask.
    """
    def __init__(self, text_model_name, mlp_layers):
        super().__init__()
        # Charger un modèle de texte pré-entraîné de la bibliothèque transformers
        self.backbone = AutoModel.from_pretrained(text_model_name)

        # La dimension d'entrée du MLP sera la taille cachée de la sortie du modèle texte.
        # Nous supposons que nous prenons la sortie du premier token ([CLS] pour les modèles de type BERT)
        text_feature_dim = self.backbone.config.hidden_size # Ex: 768 pour bert-base-uncased

        layers = []
        input_dim = text_feature_dim

        # Construction des couches du MLP
        for output_dim in mlp_layers[:-1]:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.ReLU()) # Utilisation de ReLU comme fonction d'activation
            input_dim = output_dim

        # Couche de sortie (prédiction unique)
        layers.append(nn.Linear(input_dim, mlp_layers[-1])) # mlp_layers[-1] doit être 1

        self.mlp = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):
        # Obtenir les états cachés du modèle texte
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Prendre l'état caché du premier token ([CLS]) pour la représentation de la séquence
        text_features = outputs.last_hidden_state[:, 0, :]
        # Passer les caractéristiques à travers le MLP
        x = self.mlp(text_features)
        return x

class CombinedModel(nn.Module):
    """
    Modèle combiné pour la prédiction de vues à partir de l'image et du texte.
    Prend les données brutes de texte du Dataset et effectue la tokenisation.
    """
    def __init__(self, image_mlp_layers, text_model_name, text_mlp_layers, freeze_dinov2=True, freeze_text_model=True, max_token_length=MAX_TOKEN_LENGTH):
        super().__init__()
        # Configuration de la branche image : Charger DinoV2 et créer ImageBranch
        dinov2_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        dinov2_backbone.head = nn.Identity() # Supprimer la tête par défaut
        if freeze_dinov2:
            for param in dinov2_backbone.parameters():
                param.requires_grad = False
        self.image_branch = ImageBranch(dinov2_backbone, image_mlp_layers)

        # Configuration de la branche texte : Créer TextBranch
        self.text_branch = TextBranch(text_model_name, text_mlp_layers)
        if freeze_text_model:
             # Geler le backbone du modèle texte
             for param in self.text_branch.backbone.parameters():
                 param.requires_grad = False

        # Tokenizer pour traiter le texte brut
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.max_token_length = max_token_length

        # Poids apprenables pour la combinaison
        self.weight_image = nn.Parameter(torch.tensor(0.5))
        self.weight_text = nn.Parameter(torch.tensor(0.5))

    def forward(self, batch):
        # Supposer que le batch contient 'image' et 'text' (le texte brut du titre/metadata)
        image = batch["image"]
        raw_text = batch["text"] # Texte brut chargé par le Dataset

        # --- Traitement du texte : Tokenisation ---
        # Le tokenizer s'attend généralement à une liste de chaînes
        # et retourne un dictionnaire de tensors.
        encoded_text = self.tokenizer(
            list(raw_text), # Convertir le tuple/liste de textes en liste
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors='pt' # Retourne des tensors PyTorch
        )

        # Envoyer les tensors de texte sur le même appareil que le modèle
        input_ids = encoded_text['input_ids'].to(image.device) # Utiliser l'appareil de l'image
        attention_mask = encoded_text['attention_mask'].to(image.device)

        # Obtenir les prédictions de chaque branche
        image_prediction = self.image_branch(image)
        text_prediction = self.text_branch(input_ids, attention_mask)

        # Combiner les prédictions en utilisant les poids apprenables
        combined_prediction = self.weight_image * image_prediction + self.weight_text * text_prediction

        return combined_prediction