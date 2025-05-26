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
        self.act = nn.ReLU()
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
            self.blocks.append(nn.ReLU())                     # Activation
            #self.blocks.append(ResidualBlock(dim, dropout=dropout))  # Bloc résiduel avec dropout
            current_dim = dim

        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.feature_gate(x)  # Appliquer le gate sur les features
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x).squeeze(1)


class MultimodalCombined(nn.Module):
    """
    Enhanced multimodal model with channel and date features
    """
    def __init__(self, 
                text_model_name: "distilbert-base-uncased",
                image_model_frozen=True,
                text_model_frozen=True,
                small_mlp_indim = 768,
                small_mlp_num_layers = 2,
                final_mlp_indim = 64,
                final_mlp_hidden_dim = [32, 16],
                text_proportion=0.2,
                channel_proportion=0.1,
                date_proportion=0.1,
                img_proportion=0.5,
                year_proportion=0.1,
                max_token_length=256,
                dropout_rate=0.1):
        super().__init__()
        
        assert int(text_proportion + channel_proportion + date_proportion + img_proportion + year_proportion) == 1, f"Proportions must sum to 1. Sum = {text_proportion + channel_proportion + date_proportion + img_proportion + year_proportion}"
       
        # Image branch
        self.outimg_dim = int(small_mlp_indim*img_proportion)
        self.image_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.image_backbone.head = nn.Identity()
        image_dim = self.image_backbone.norm.normalized_shape[0]  # 768

        if small_mlp_indim is not None:
            self.img_proj = nn.Linear(image_dim, self.outimg_dim)
        else:
            self.img_proj = nn.Identity()
        
        if image_model_frozen:
            for param in self.image_backbone.parameters():
                param.requires_grad = False
        
        # Text branch
        self.outtext_dim = int(small_mlp_indim*text_proportion)
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
        
        if small_mlp_indim is not None:
            self.text_proj = nn.Linear(text_dim, self.outtext_dim)
        else:
            self.text_proj = nn.Identity()

        if text_model_frozen:
            for param in self.text_backbone.parameters():
                param.requires_grad = False

        # Date (already preprocessed)
        if small_mlp_indim is not None:
            self.outdate_dim = int(small_mlp_indim*date_proportion)
            self.date_proj = nn.Linear(5, int(small_mlp_indim*date_proportion))
        else:
            self.date_proj = nn.Identity()
        
        # Year (already preprocessed)
        if small_mlp_indim is not None:
            self.outyear_dim = int(small_mlp_indim*year_proportion)
            self.year_proj = nn.Linear(1, self.outyear_dim)
        else:
            self.year_proj = nn.Identity()
        
        # Channel embeddings
        self.channel_embedding_dim = small_mlp_indim - self.outimg_dim - self.outtext_dim - self.outdate_dim - self.outyear_dim
        # We'll initialize this after seeing the data, but reserve space
        self.channel_embedding = None
        self.channel_to_idx = {}
        self._channel_initialized = False

        # Dynamic weights for concatenation
        self.branch_weights = nn.Parameter(torch.ones(2))

        # Final MLP - input size will be image + text + channel + temporal
        self.mlp_channel_text_img = ImprovedMLPRegressor(input_dim=
            self.outimg_dim + self.outtext_dim + self.channel_embedding_dim,
            hidden_dim=[small_mlp_indim//(2**i) for i in range(1,small_mlp_num_layers+1)],
            output_dim=final_mlp_indim//2,
            dropout=dropout_rate
        )
        self.mlp_channel_year_date = ImprovedMLPRegressor(input_dim=
            self.outyear_dim + self.outdate_dim + self.channel_embedding_dim,
            hidden_dim=[small_mlp_indim//(2**i) for i in range(1,small_mlp_num_layers+1)],
            output_dim=final_mlp_indim//2,
            dropout=dropout_rate
        )
        self.final_mlp = ImprovedMLPRegressor(input_dim=final_mlp_indim,
            hidden_dim=final_mlp_hidden_dim,
            output_dim=1,  # Final output is a single value
            dropout=dropout_rate
        )

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
        
        
        # Concatenate features

        channel_text_img_features = torch.cat([
            image_features, 
            text_embedding, 
            channel_features, 
        ], dim=1)
        channel_year_date_features = torch.cat([
            year_features, 
            date_features, 
            channel_features, 
        ], dim=1)

        # Apply MLPs with dynamic weights

        mlp1_out = self.mlp_channel_text_img(channel_text_img_features)
        mlp2_out = self.mlp_channel_year_date(channel_year_date_features)
        w = torch.softmax(self.branch_weights, dim=0) #to sum to 1
        mlp1_out = w[0] * mlp1_out
        mlp2_out = w[1] * mlp2_out
        combined_features = torch.cat([mlp1_out, mlp2_out], dim=1)
    
        # Apply final MLP
        output = self.final_mlp(combined_features)
        return output