import torch
import torch.nn as nn
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


class EnhancedMultimodalWithMetadata(nn.Module):
    """
    Enhanced multimodal model with channel and date features
    """
    def __init__(self, 
                 image_model_frozen=True,
                 text_model_frozen=True,
                 final_mlp_layers=[1024, 1024, 1024],
                 max_token_length=256,
                 dropout_rate=0.1,
                 text_model_name="google/gemma-3-1b-it",
                 channel_embedding_dim=64,
                 temporal_embedding_dim=32):
        super().__init__()
        
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
        text_dim = self.text_backbone.config.hidden_size # 1152
        self.max_token_length = max_token_length
        
        if text_model_frozen:
            for param in self.text_backbone.parameters():
                param.requires_grad = False
        
        # Channel embeddings
        self.channel_embedding_dim = channel_embedding_dim
        # We'll initialize this after seeing the data, but reserve space
        self.channel_embedding = None
        self.channel_to_idx = {}
        self._channel_initialized = False
        
        # Temporal embeddings
        self.temporal_embedding_dim = temporal_embedding_dim
        # Day of week (0-6), Month (1-12), Day of month (1-31)
        self.day_of_week_embedding = nn.Embedding(7, temporal_embedding_dim // 3)
        self.month_embedding = nn.Embedding(12, temporal_embedding_dim // 3)
        self.day_of_month_embedding = nn.Embedding(31, temporal_embedding_dim - 2 * (temporal_embedding_dim // 3))
        
        # Final MLP - input size will be image + text + channel + temporal
        total_feature_dim = image_dim + text_dim + channel_embedding_dim + temporal_embedding_dim
        self.mlp = MLP(input_dim=total_feature_dim, output_dim=1, hidden_dim=final_mlp_layers, dropout_rate=dropout_rate)
        
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
        
    def extract_temporal_features(self, date_strings):
        """Extract temporal features from date strings"""
        device = next(self.parameters()).device
        
        day_of_week_list = []
        month_list = []
        day_of_month_list = []
        
        for date_str in date_strings:
            # Parse the date string (format: 2024-12-19 03:30:00+00:00)
            date_obj = pd.to_datetime(date_str)
            
            day_of_week_list.append(date_obj.weekday())  # 0=Monday, 6=Sunday
            month_list.append(date_obj.month - 1)  # Convert to 0-indexed
            day_of_month_list.append(date_obj.day - 1)  # Convert to 0-indexed
        
        day_of_week_tensor = torch.tensor(day_of_week_list, device=device)
        month_tensor = torch.tensor(month_list, device=device)
        day_of_month_tensor = torch.tensor(day_of_month_list, device=device)
        
        return day_of_week_tensor, month_tensor, day_of_month_tensor
    
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
        
        # Channel processing
        if not self._channel_initialized or self.channel_embedding is None:
            raise ValueError("Channel embedding not initialized. Call initialize_channel_embedding() first.")
        
        channels = batch["channel"]
        channel_indices = torch.tensor([self.channel_to_idx[ch] for ch in channels], device=device)
        channel_features = self.channel_embedding(channel_indices)
        
        # Temporal processing
        dates = batch["date"]
        day_of_week, month, day_of_month = self.extract_temporal_features(dates)
        
        day_of_week_emb = self.day_of_week_embedding(day_of_week)
        month_emb = self.month_embedding(month)
        day_of_month_emb = self.day_of_month_embedding(day_of_month)
        
        temporal_features = torch.cat([day_of_week_emb, month_emb, day_of_month_emb], dim=1)
        
        # Concatenate all features
        combined_features = torch.cat([
            image_features, 
            last_token_hidden, 
            channel_features, 
            temporal_features
        ], dim=1)
        
        # Final MLP
        output = self.mlp(combined_features)
        return output