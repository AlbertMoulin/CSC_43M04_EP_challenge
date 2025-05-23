import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from datetime import datetime

class MLP(nn.Module):
    """
    Simple MLP
    """
    def __init__(self, input_dim, output_dim, hidden_dim : list = [1024, 1024,1024],dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = []
        for i in range(len(hidden_dim)):
            if i == 0:
                self.hidden_layers.append(nn.Linear(input_dim, hidden_dim[i]))
            else:
                self.hidden_layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            self.hidden_layers.append(nn.ReLU())
            self.hidden_layers.append(nn.Dropout(dropout_rate))
        self.hidden_layers.append(nn.Linear(hidden_dim[-1], output_dim))
        self.hidden_layers = nn.ModuleList(self.hidden_layers)
        # Initialize the MLP with the specified hidden layers
        self.mlp = nn.Sequential(*self.hidden_layers)
    
    def forward(self, x):
        # Forward pass through the MLP
        x = self.mlp(x)
        return x

class DateEncoder(nn.Module):
    """
    Encodes date information focusing on time of year patterns
    """
    def __init__(self, date_embedding_dim=32):
        super().__init__()
        self.date_embedding_dim = date_embedding_dim
        
        # Learnable embeddings for different time components
        self.month_embedding = nn.Embedding(12, date_embedding_dim // 4)  # 0-11 for months
        self.day_of_week_embedding = nn.Embedding(7, date_embedding_dim // 4)  # 0-6 for days
        self.day_of_month_embedding = nn.Embedding(31, date_embedding_dim // 4)  # 0-30 for days of month
        
        # MLP to combine all date features
        self.date_mlp = nn.Sequential(
            nn.Linear(date_embedding_dim // 4 * 3 + 4, date_embedding_dim),  # +4 for cyclical features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(date_embedding_dim, date_embedding_dim)
        )
    
    def forward(self, dates):
        """
        Args:
            dates: tensor of shape (batch_size,) containing datetime objects or timestamps
        """
        batch_size = dates.shape[0]
        device = dates.device
        
        # Extract date components
        months = torch.zeros(batch_size, dtype=torch.long, device=device)
        days_of_week = torch.zeros(batch_size, dtype=torch.long, device=device)
        days_of_month = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Cyclical features for better representation of time periodicity
        month_sin = torch.zeros(batch_size, device=device)
        month_cos = torch.zeros(batch_size, device=device)
        day_sin = torch.zeros(batch_size, device=device)
        day_cos = torch.zeros(batch_size, device=device)
        
        for i in range(batch_size):
            # Assuming dates[i] is a timestamp or can be converted to datetime
            if isinstance(dates[i].item(), (int, float)):
                dt = datetime.fromtimestamp(dates[i].item())
            else:
                dt = dates[i]
            
            months[i] = dt.month - 1  # 0-11
            days_of_week[i] = dt.weekday()  # 0-6
            days_of_month[i] = dt.day - 1  # 0-30
            
            # Cyclical encoding
            month_sin[i] = np.sin(2 * np.pi * dt.month / 12)
            month_cos[i] = np.cos(2 * np.pi * dt.month / 12)
            day_sin[i] = np.sin(2 * np.pi * dt.day / 31)
            day_cos[i] = np.cos(2 * np.pi * dt.day / 31)
        
        # Get embeddings
        month_emb = self.month_embedding(months)
        dow_emb = self.day_of_week_embedding(days_of_week)
        dom_emb = self.day_of_month_embedding(days_of_month)
        
        # Combine all features
        cyclical_features = torch.stack([month_sin, month_cos, day_sin, day_cos], dim=1)
        all_features = torch.cat([month_emb, dow_emb, dom_emb, cyclical_features], dim=1)
        
        return self.date_mlp(all_features)

class EnhancedMultimodalWithDate(nn.Module):
    """
    Enhanced multimodal model with date features
    """
    def __init__(self, 
                 image_model_frozen=True,
                 text_model_frozen=True,
                 final_mlp_layers=[1024, 1024, 1024],
                 max_token_length=256,
                 dropout_rate=0.1,
                 text_model_name="google/gemma-3-1b-it",
                 date_embedding_dim=32,
                 use_date_features=True):
        super().__init__()
        
        self.use_date_features = use_date_features
        
        # Image branch
        self.image_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.image_backbone.head = nn.Identity()
        image_dim = self.image_backbone.norm.normalized_shape[0]  # 768
        
        if image_model_frozen:
            for param in self.image_backbone.parameters():
                param.requires_grad = False
        
        # Text branch
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_backbone = AutoModelForCausalLM.from_pretrained(text_model_name, torch_dtype=torch.float16)
        text_dim = self.text_backbone.config.hidden_size  # 1152
        self.max_token_length = max_token_length
        
        if text_model_frozen:
            for param in self.text_backbone.parameters():
                param.requires_grad = False
        
        # Date branch
        if self.use_date_features:
            self.date_encoder = DateEncoder(date_embedding_dim)
            combined_dim = image_dim + text_dim + date_embedding_dim
        else:
            combined_dim = image_dim + text_dim
        
        # Final MLP
        self.mlp = MLP(
            input_dim=combined_dim, 
            output_dim=1, 
            hidden_dim=final_mlp_layers, 
            dropout_rate=dropout_rate
        )
    
    def forward(self, batch):
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
        
        # Combine features
        if self.use_date_features and "date" in batch:
            # Date processing
            date_features = self.date_encoder(batch["date"])
            combined_features = torch.cat((image_features, last_token_hidden, date_features), dim=1)
        else:
            combined_features = torch.cat((image_features, last_token_hidden), dim=1)
        
        # MLP processing
        mlp_output = self.mlp(combined_features)
        return mlp_output