import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from datetime import datetime


class ImprovedHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Add multiple hidden layers with non-linearities
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class MultimodalViewModel(nn.Module):
    def __init__(self, 
                 dinov2_frozen=True,
                 bert_frozen=True,
                 dropout_rate=0.2,
                 dataset_path="dataset"):
        super().__init__()
        
        # Image encoder - DinoV2
        self.image_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.image_backbone.head = nn.Identity()
        self.image_dim = self.image_backbone.norm.normalized_shape[0]  # 768
        
        if dinov2_frozen:
            for param in self.image_backbone.parameters():
                param.requires_grad = False
        
        # Text encoder - BERT
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.text_backbone = AutoModel.from_pretrained('bert-base-uncased')
        self.text_dim = self.text_backbone.config.hidden_size  # 768
        
        if bert_frozen:
            for param in self.text_backbone.parameters():
                param.requires_grad = False
        
        # Metadata encoder
        self.metadata_dim = 64  # Embedding dimension for metadata
        
        # Load and prepare channel/date mappings
        self._prepare_metadata_mappings(dataset_path)
        
        # Channel embeddings
        self.channel_embedding = nn.Embedding(
            num_embeddings=len(self.channel_to_idx), 
            embedding_dim=32
        )
        
        # Date features (we'll use engineered features)
        self.date_encoder = nn.Linear(6, 32)  # year, month, day, day_of_week, day_of_year, is_weekend
        
        # Individual prediction heads
        self.image_head = ImprovedHead(
            input_dim=self.image_dim,
            hidden_dims=[512]*6,
            output_dim=1,
            dropout_rate=dropout_rate
        )
        
        self.text_head = ImprovedHead(
            input_dim=self.text_dim,
            hidden_dims=[512]*6,
            output_dim=1,
            dropout_rate=dropout_rate
        )
        
        self.metadata_head = ImprovedHead(
            input_dim=self.metadata_dim,
            hidden_dims=[512]*6,
            output_dim=1,
            dropout_rate=dropout_rate
        )
        
        # Learnable weights for weighted sum (will be normalized with softmax)
        self.modality_weights = nn.Parameter(torch.ones(3))  # image, text, metadata
        
        # Optional: Add a fusion layer that can learn cross-modal interactions
        self.fusion_layer = nn.Sequential(
            nn.Linear(3, 8),  # 3 predictions -> 8 hidden
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(8, 3),  # 8 hidden -> 3 weights
            nn.Softmax(dim=1)
        )
        
    def _prepare_metadata_mappings(self, dataset_path):
        """Prepare mappings for channels from the training data."""
        try:
            # Read training data to get unique channels
            train_df = pd.read_csv(f"{dataset_path}/train_val.csv")
            unique_channels = train_df['channel'].unique()
            
            # Create channel to index mapping
            self.channel_to_idx = {channel: idx for idx, channel in enumerate(unique_channels)}
            self.channel_to_idx['<UNK>'] = len(unique_channels)  # For unknown channels
            
            print(f"Found {len(unique_channels)} unique channels")
            
        except Exception as e:
            print(f"Warning: Could not load channel data: {e}")
            # Fallback - will need to be updated with actual data
            self.channel_to_idx = {'<UNK>': 0}
    
    def _encode_metadata(self, batch):
        """Encode channel and date information."""
        device = next(self.parameters()).device
        
        # Channel encoding
        channel_ids = batch['channel_idx'].to(device)
        channel_emb = self.channel_embedding(channel_ids)
        
        # Date encoding
        date_features = batch['date_features'].to(device)
        date_emb = self.date_encoder(date_features)
        
        # Concatenate channel and date embeddings
        metadata_features = torch.cat([channel_emb, date_emb], dim=1)
        
        return metadata_features
    
    def _encode_text(self, text_list):
        """Encode text using BERT."""
        # Tokenize the text
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Get BERT features
        with torch.no_grad() if hasattr(self.text_backbone, 'training') and not self.text_backbone.training else torch.enable_grad():
            outputs = self.text_backbone(**encoded)
            # Use [CLS] token representation
            text_features = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        return text_features
    
    def forward(self, batch):
        # Extract image features
        image_features = self.image_backbone(batch["image"])  # [batch_size, 768]
        
        # Extract text features
        text_features = self._encode_text(batch["text"])  # [batch_size, 768]
        
        # Extract metadata features
        metadata_features = self._encode_metadata(batch)  # [batch_size, 64]
        
        # Get individual predictions
        image_pred = self.image_head(image_features)  # [batch_size, 1]
        text_pred = self.text_head(text_features)  # [batch_size, 1]
        metadata_pred = self.metadata_head(metadata_features)  # [batch_size, 1]
        
        # Stack predictions for potential fusion learning
        all_predictions = torch.cat([image_pred, text_pred, metadata_pred], dim=1)  # [batch_size, 3]
        
        # Option 1: Simple learnable weighted sum
        normalized_weights = torch.softmax(self.modality_weights, dim=0)
        simple_weighted = (
            normalized_weights[0] * image_pred +
            normalized_weights[1] * text_pred +
            normalized_weights[2] * metadata_pred
        )
        
        # Option 2: Learned fusion (more sophisticated)
        # fusion_weights = self.fusion_layer(all_predictions)  # [batch_size, 3]
        # learned_weighted = torch.sum(all_predictions * fusion_weights, dim=1, keepdim=True)
        
        # For now, use simple weighted sum (you can experiment with fusion_layer)
        final_prediction = simple_weighted
        
        return final_prediction