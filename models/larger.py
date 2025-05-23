import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np


class LightweightChannelModel(nn.Module):
    """
    Memory-efficient model that leverages channel insights without complex architectures.
    
    Focus: Use the key insights from analysis without heavy models.
    """
    def __init__(self, 
                 dinov2_frozen=True,
                 bert_frozen=True,
                 max_token_length=256,  # Reduced
                 text_model_name='bert-base-uncased',
                 dataset_path="dataset"):
        super().__init__()
        
        # Load channel insights
        self._load_channel_insights(dataset_path)
        
        # Image branch - SIMPLIFIED
        self.image_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.image_backbone.head = nn.Identity()
        image_dim = self.image_backbone.norm.normalized_shape[0]
        
        if dinov2_frozen:
            for param in self.image_backbone.parameters():
                param.requires_grad = False
        
        # Text branch - SIMPLIFIED  
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_backbone = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_backbone.config.hidden_size
        self.max_token_length = max_token_length
        
        if bert_frozen:
            for param in self.text_backbone.parameters():
                param.requires_grad = False
        
        # LIGHTWEIGHT MLPs
        self.image_mlp = nn.Sequential(
            nn.Linear(image_dim, 256),  # Much smaller
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Text + channel features
        channel_features_dim = 8  # Key channel features only
        combined_text_dim = text_dim + channel_features_dim
        
        self.text_mlp = nn.Sequential(
            nn.Linear(combined_text_dim, 512),  # Smaller
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Weights based on analysis insights
        self.weight_image = nn.Parameter(torch.tensor(0.3))
        self.weight_text = nn.Parameter(torch.tensor(0.7))  # Text+channel gets more weight
    
    def _load_channel_insights(self, dataset_path):
        """Load key channel insights from analysis."""
        try:
            train_df = pd.read_csv(f"{dataset_path}/train_val.csv")
            
            # Key insights from your analysis:
            # 1. Elite channels: 63% short films, 82% social presence, 2.5M avg
            # 2. High channels: 22% short films, 54% social, 314K avg  
            # 3. Channel reputation correlation: 0.166 (strongest!)
            
            channel_stats = train_df.groupby('channel').agg({
                'views': ['mean', 'median', 'std', 'count'],
                'title': [
                    lambda x: x.str.lower().str.contains('short film|animated short').mean(),
                    lambda x: x.str.lower().str.contains('vfx|cgi|breakdown').mean(),
                ],
                'description': [
                    lambda x: x.fillna('').str.lower().str.contains('subscribe|like|share').mean(),
                    lambda x: x.fillna('').str.lower().str.contains('facebook|twitter|instagram').mean(),
                ]
            }).round(4)
            
            channel_stats.columns = ['avg_views', 'median_views', 'std_views', 'video_count',
                                   'short_film_ratio', 'vfx_ratio', 'cta_ratio', 'social_ratio']
            
            # Create simplified channel features
            self.channel_features = {}
            
            max_avg_views = channel_stats['avg_views'].max()
            
            for channel in channel_stats.index:
                stats = channel_stats.loc[channel]
                
                # Simplified feature vector (8 features)
                features = [
                    # Core performance (normalized)
                    stats['avg_views'] / max_avg_views,  # 0-1 range
                    min(stats['video_count'] / 100.0, 1.0),  # Channel maturity
                    
                    # Content type patterns (from analysis)
                    stats['short_film_ratio'],
                    stats['vfx_ratio'],
                    
                    # Professional indicators (strong predictors from analysis)
                    stats['cta_ratio'],
                    stats['social_ratio'],
                    
                    # Performance tier (from analysis)
                    1.0 if stats['avg_views'] > 1000000 else 0.0,  # Elite
                    1.0 if stats['avg_views'] > 100000 else 0.0,   # High+
                ]
                
                self.channel_features[channel] = features
            
            # Default for unknown channels (shouldn't happen but safety)
            self.channel_features['<UNK>'] = [0.1, 0.1, 0.2, 0.1, 0.5, 0.4, 0.0, 0.0]
            
            print(f"✅ Loaded lightweight channel features for {len(channel_stats)} channels")
            
        except Exception as e:
            print(f"❌ Error loading channel insights: {e}")
            self.channel_features = {'<UNK>': [0.5] * 8}
    
    def _get_channel_features(self, channels):
        """Get channel features for batch."""
        features = []
        for channel in channels:
            features.append(self.channel_features.get(channel, self.channel_features['<UNK>']))
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_content_features(self, batch):
        """Extract basic content features that matter."""
        features = []
        channels = batch.get("channel", ["<UNK>"] * len(batch["id"]))
        
        for i, text in enumerate(batch["text"]):
            channel = channels[i]
            channel_feats = self.channel_features.get(channel, self.channel_features['<UNK>'])
            
            if not isinstance(text, str):
                text = ""
            
            # Parse title and description
            if " + " in text:
                parts = text.split(" + ", 1)
                title = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ""
            else:
                title = text.strip()
                description = ""
            
            # Content analysis relative to channel norms
            is_short_film = int(any(word in title.lower() for word in ['short film', 'animated short']))
            is_vfx = int(any(word in title.lower() for word in ['vfx', 'cgi', 'breakdown']))
            has_cta = int(any(word in description.lower() for word in ['subscribe', 'like', 'share']))
            has_social = int(any(word in description.lower() for word in ['facebook', 'twitter', 'instagram']))
            
            # Key insight: Does this content match the channel's successful pattern?
            channel_short_film_ratio = channel_feats[2]  # Expected short film ratio
            channel_vfx_ratio = channel_feats[3]  # Expected VFX ratio
            channel_cta_ratio = channel_feats[4]  # Expected CTA ratio
            channel_social_ratio = channel_feats[5]  # Expected social ratio
            
            # Pattern matching scores (positive = matches successful pattern)
            short_film_match = is_short_film if channel_short_film_ratio > 0.3 else (1 - is_short_film)
            vfx_match = is_vfx if channel_vfx_ratio > 0.3 else (1 - is_vfx)
            cta_match = has_cta if channel_cta_ratio > 0.5 else (1 - has_cta)
            social_match = has_social if channel_social_ratio > 0.5 else (1 - has_social)
            
            # Simple quality indicators
            title_quality = min(len(title) / 50.0, 1.0)
            desc_quality = min(len(description) / 1000.0, 1.0)
            
            content_features = [
                short_film_match,
                vfx_match, 
                cta_match,
                social_match,
                title_quality,
                desc_quality,
                # Channel context
                channel_feats[0],  # Channel reputation
                channel_feats[6],  # Is elite channel
            ]
            
            features.append(content_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self, batch):
        device = next(self.parameters()).device
        
        # Image processing (simplified)
        image_features = self.image_backbone(batch["image"])
        image_pred = self.image_mlp(image_features)
        
        # Text processing with channel context
        raw_text = list(batch["text"])
        processed_text = [text.strip() if isinstance(text, str) else "[EMPTY]" for text in raw_text]
        
        # BERT encoding (shorter sequences to save memory)
        encoded_text = self.tokenizer(
            processed_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_token_length,
            return_tensors='pt',
            add_special_tokens=True
        )
        
        input_ids = encoded_text['input_ids'].to(device)
        attention_mask = encoded_text['attention_mask'].to(device)
        
        text_outputs = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
        bert_features = text_outputs.last_hidden_state[:, 0, :]
        
        # Content analysis features
        content_features = self._extract_content_features(batch).to(device)
        
        # Combine text + content features
        combined_features = torch.cat([bert_features, content_features], dim=1)
        text_pred = self.text_mlp(combined_features)
        
        # Final prediction
        final_pred = self.weight_image * image_pred + self.weight_text * text_pred
        
        return final_pred


class UltraLightModel(nn.Module):
    """
    Ultra-lightweight model for memory-constrained environments.
    Uses only the most essential features from your analysis.
    """
    def __init__(self, dataset_path="dataset"):
        super().__init__()
        
        # Load only essential channel data
        self._load_essential_data(dataset_path)
        
        # Tiny networks
        self.channel_net = nn.Sequential(
            nn.Linear(4, 32),  # Only 4 key channel features
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.content_net = nn.Sequential(
            nn.Linear(6, 24),  # Only 6 key content features  
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 1)
        )
        
        # Simple combination
        self.combiner = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def _load_essential_data(self, dataset_path):
        """Load only the most essential channel data."""
        try:
            train_df = pd.read_csv(f"{dataset_path}/train_val.csv")
            
            # Ultra-simple channel features (only the most predictive)
            channel_stats = train_df.groupby('channel').agg({
                'views': 'mean',
                'title': lambda x: x.str.lower().str.contains('short film|animated short').mean(),
                'description': [
                    lambda x: x.fillna('').str.lower().str.contains('subscribe|like|share').mean(),
                    lambda x: x.fillna('').str.lower().str.contains('facebook|twitter|instagram').mean(),
                ]
            })
            
            channel_stats.columns = ['avg_views', 'short_film_ratio', 'cta_ratio', 'social_ratio']
            
            # Normalize
            max_views = channel_stats['avg_views'].max()
            
            self.channel_data = {}
            for channel in channel_stats.index:
                stats = channel_stats.loc[channel]
                self.channel_data[channel] = [
                    stats['avg_views'] / max_views,  # Normalized reputation
                    stats['short_film_ratio'],       # Content type
                    stats['cta_ratio'],              # Professional marker
                    stats['social_ratio'],           # Professional marker
                ]
            
            self.channel_data['<UNK>'] = [0.1, 0.2, 0.5, 0.4]
            
            print(f"✅ Ultra-light model with {len(channel_stats)} channels")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self.channel_data = {'<UNK>': [0.5, 0.2, 0.5, 0.4]}
    
    def forward(self, batch):
        device = next(self.parameters()).device
        
        # Channel features
        channels = batch.get("channel", ["<UNK>"] * len(batch["id"]))
        channel_features = []
        
        for channel in channels:
            channel_features.append(self.channel_data.get(channel, self.channel_data['<UNK>']))
        
        channel_features = torch.tensor(channel_features, device=device, dtype=torch.float32)
        channel_pred = self.channel_net(channel_features)
        
        # Ultra-simple content features
        content_features = []
        for text in batch["text"]:
            if not isinstance(text, str):
                text = ""
            
            title = text.split(" + ")[0] if " + " in text else text
            description = text.split(" + ", 1)[1] if " + " in text and len(text.split(" + ")) > 1 else ""
            
            features = [
                min(len(title) / 50.0, 1.0),  # Title length
                int(any(word in title.lower() for word in ['short film', 'animated short'])),
                int(any(word in title.lower() for word in ['vfx', 'cgi', 'breakdown'])),
                int(any(word in description.lower() for word in ['subscribe', 'like', 'share'])),
                int(any(word in description.lower() for word in ['facebook', 'twitter', 'instagram'])),
                min(len(description) / 1000.0, 1.0),  # Description length
            ]
            content_features.append(features)
        
        content_features = torch.tensor(content_features, device=device, dtype=torch.float32)
        content_pred = self.content_net(content_features)
        
        # Combine
        combined = torch.cat([channel_pred, content_pred], dim=1)
        final_pred = self.combiner(combined)
        
        return final_pred