import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import pandas as pd
from datetime import datetime

class FeatureGate(nn.Module):
    #pour adapter dynamiquement l’importance de chaque feature
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.gate(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        out = self.linear1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return self.act(out + x)

class ImprovedMLPRegressor(nn.Module):
    def __init__(self, input_dim=1024, output_dim = 1, hidden_dim=[1024, 1024, 1024], dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim[0])
        self.blocks = nn.ModuleList()
        current_dim = hidden_dim[0]
        self.feature_gate = FeatureGate(current_dim)  # Gate pour les features

        for dim in hidden_dim[1:len(hidden_dim)]:
            self.blocks.append(nn.Linear(current_dim, dim))  # Projection linéaire
            self.blocks.append(nn.LayerNorm(dim))               # Normalisation
            self.blocks.append(nn.BatchNorm1d(dim))            # Normalisation par lot
            self.blocks.append(nn.LeakyReLU())                     # Activation
            self.blocks.append(ResidualBlock(dim, dropout=dropout))  # Bloc résiduel avec dropout
            self.blocks.append(nn.Dropout(dropout))               # Dropout
            current_dim = dim

        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.feature_gate(x)  # Appliquer le gate sur les features
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


class MultimodalCombined(nn.Module):
    """
    Enhanced multimodal model with channel and date features
    """
    def __init__(self, 
                text_model_name: "distilbert-base-uncased",
                image_model_frozen=True,
                text_model_frozen=True,
                mlp_indim = 1024,
                mlp_hidden_dim = [512, 256, 128],
                text_proportion=20,
                channel_proportion=10,
                date_proportion=10,
                img_proportion=50,
                year_proportion=10,
                max_token_length=256,
                dropout_rate=0.1,
                mode="regression",  # "regression" ou "classification"
                num_classes=20):   
        super().__init__()
        
        assert int(text_proportion + channel_proportion + date_proportion + img_proportion + year_proportion) == 100, f"Proportions must sum to 100. Sum = {text_proportion + channel_proportion + date_proportion + img_proportion + year_proportion}"
       
        text_proportion = text_proportion / 100
        channel_proportion = channel_proportion / 100
        date_proportion = date_proportion / 100
        img_proportion = img_proportion / 100
        year_proportion = year_proportion / 100
        # Image branch
        self.outimg_dim = int(mlp_indim*img_proportion)
        self.image_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.image_backbone.head = nn.Identity()
        image_dim = self.image_backbone.norm.normalized_shape[0]  # 768

        if mlp_indim is not None:
            self.img_proj = nn.Linear(image_dim, self.outimg_dim)
        else:
            self.img_proj = nn.Identity()
        
        
        for param in self.image_backbone.parameters():
            param.requires_grad = False

        if not image_model_frozen:
            for param in self.image_backbone.blocks[-2:].parameters():
                param.requires_grad = True
        
        # Text branch
        self.outtext_dim = int(mlp_indim*text_proportion)
        self.text_backbone = None
        if "bert" in text_model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            self.text_backbone = AutoModel.from_pretrained(text_model_name)
            text_dim = self.text_backbone.config.hidden_size
            self.max_token_length = max_token_length
        elif "gemma" in text_model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            self.text_backbone = AutoModelForCausalLM.from_pretrained(text_model_name)
            text_dim = self.text_backbone.config.hidden_size # 1152
            self.max_token_length = max_token_length
        else:
            raise ValueError("Unsupported text model. Please use a supported model name.")
        
        if mlp_indim is not None:
            self.text_proj = nn.Linear(text_dim, self.outtext_dim)
        else:
            self.text_proj = nn.Identity()

        for param in self.text_backbone.parameters():
            param.requires_grad = False

        if not text_model_frozen:
            if hasattr(self.text_backbone, "encoder"):
                for param in self.text_backbone.encoder.layer[-2].parameters():
                    param.requires_grad = True

        # Date (already preprocessed)
        if mlp_indim is not None:
            self.outdate_dim = int(mlp_indim*date_proportion)
            self.date_proj = nn.Linear(5, int(mlp_indim*date_proportion))
        else:
            self.date_proj = nn.Identity()
        
        # Year (already preprocessed)
        if mlp_indim is not None:
            self.outyear_dim = int(mlp_indim*year_proportion)
            self.year_proj = nn.Linear(1, self.outyear_dim)
        else:
            self.year_proj = nn.Identity()
        
        # Channel embeddings
        self.channel_embedding_dim = (mlp_indim - self.outimg_dim - self.outtext_dim - self.outdate_dim - self.outyear_dim)
        # We'll initialize this after seeing the data, but reserve space
        self.channel_embedding = None
        self.channel_to_idx = {}
        self._channel_initialized = False

        # Growth of the channel
        # self.growth_head = nn.Sequential(
        #     nn.Linear(self.channel_embedding_dim + self.outyear_dim, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 1)
        # )

        # major channel in training set
        self.major_channel_lin = nn.Linear(1, 16)  # 1 feature for major channel

        # MLP
        self.shared_mlp = ImprovedMLPRegressor(
            input_dim=mlp_indim+16,  # +16 pour major channel
            hidden_dim=mlp_hidden_dim[:-1],  # On garde toutes les couches sauf la dernière
            output_dim=mlp_hidden_dim[-1],   # La sortie est la dernière dimension cachée
            dropout=dropout_rate
        )

        # Tête de classification (pour mode="classification")
        self.classification_head = nn.Linear(mlp_hidden_dim[-1], num_classes)

        # Tête de régression (pour mode="regression")
        self.regression_head = nn.Linear(mlp_hidden_dim[-1], 1)

        # Mode actuel
        self.mode = mode


    def initialize_channel_embedding(self, unique_channels):
        """Initialize channel embedding after seeing all unique channels"""
        if self._channel_initialized:
            return
            
        self.channel_to_idx = {channel: idx for idx, channel in enumerate(unique_channels)}
        num_channels = len(unique_channels)
        
        # Create the embedding and register it as a proper module
        self.channel_embedding = nn.Embedding(num_channels, self.channel_embedding_dim)
        
        # If model is already on a device, move the embedding there too
        device = next(self.parameters()).device
        self.channel_embedding = self.channel_embedding.to(device)
        
        # Register as a submodule so it gets moved with the model
        self.add_module('channel_embedding', self.channel_embedding)
        
        self._channel_initialized = True
    
    def forward(self, batch):
        # Image processing
        image_features = self.image_backbone(batch["image"])
        image_features = self.img_proj(image_features)

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

        if isinstance(self.text_backbone, (AutoModel, )):
            # BERT, DistilBERT, etc. → on prend le [CLS] token
            text_embedding = last_hidden_state[:, 0, :]
        else:
            # GPT, Gemma, etc. → on prend le dernier token
            text_embedding = last_hidden_state[:, -1, :]

        text_embedding = self.text_proj(text_embedding)

        # Channel processing
        if not self._channel_initialized or self.channel_embedding is None:
            raise ValueError("Channel embedding not initialized. Call initialize_channel_embedding() first.")
        
        channels = batch["channel"]
        channel_indices = torch.tensor([self.channel_to_idx[ch] for ch in channels], device=device)
        channel_features = self.channel_embedding(channel_indices)
        
        # Date already preprocessed
        date_features = batch["date"]
        date_features = self.date_proj(date_features)

        # Year already preprocessed
        year_features = batch["year_norm"]
        year_features = self.year_proj(year_features.unsqueeze(1))

        # is the channel major in the training set?
        is_train_major_channel = batch["is_train_major_channel"].unsqueeze(1).float()
        is_train_major_channel = self.major_channel_lin(is_train_major_channel)
        

        # Combine all features 
        #growth_input = torch.cat([channel_features, year_features], dim=1)
        #growth_effect = self.growth_head(growth_input)

        combined_features = torch.cat([
            image_features,
            text_embedding,
            channel_features,
            date_features,
            year_features,
            is_train_major_channel], dim=1)

        # output = self.final_mlp(combined_features)
        # output = output + growth_effect  # le modèle apprend la croissance spécifique à la chaîne et à l'année
        # return output.squeeze(1)
    
        shared_features = self.shared_mlp(combined_features)

        # Utilise la tête appropriée selon le mode
        if self.mode == "classification":
            return self.classification_head(shared_features)
        else:  # mode="regression"
            return self.regression_head(shared_features)
        

    def set_mode(self, mode, freeze_backbone=True):
        """Change le mode du modèle entre 'classification' et 'regression'"""
        assert mode in ["classification", "regression"], "Mode must be 'classification' or 'regression'"
        self.mode = mode
        
        # Optionnel: gèle le backbone si demandé
        if freeze_backbone:
            # Gèle tout sauf la tête appropriée
            for param in self.parameters():
                param.requires_grad = False
                
            if mode == "classification":
                for param in self.classification_head.parameters():
                    param.requires_grad = True
            else:
                for param in self.regression_head.parameters():
                    param.requires_grad = True

    def load_backbone(self, state_dict):
        """Charge uniquement les poids du backbone (tout sauf les têtes)"""
        # Filtre les clés pour exclure les têtes
        backbone_dict = {k: v for k, v in state_dict.items() 
                        if not k.startswith('classification_head') 
                        and not k.startswith('regression_head')}
        
        # Charge les poids filtrés
        missing_keys, unexpected_keys = self.load_state_dict(backbone_dict, strict=False)
        print(f"Loaded backbone. Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")
