import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np


class PerfectChannelModel(nn.Module):
    """
    Model leveraging 100% channel coverage for maximum performance.
    
    With perfect channel knowledge, this should achieve sub-2.0 MSLE!
    """
    def __init__(self, 
                 dinov2_frozen=True,
                 bert_frozen=True,
                 max_token_length=512,
                 text_model_name='bert-base-uncased',
                 dataset_path="dataset"):
        super().__init__()
        
        # Load perfect channel knowledge
        self._load_perfect_channel_data(dataset_path)
        
        # Image branch (smaller role now)
        self.image_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.image_backbone.head = nn.Identity()
        image_dim = self.image_backbone.norm.normalized_shape[0]
        
        if dinov2_frozen:
            for param in self.image_backbone.parameters():
                param.requires_grad = False
        
        # Text branch
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_backbone = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_backbone.config.hidden_size
        self.max_token_length = max_token_length
        
        if bert_frozen:
            for param in self.text_backbone.parameters():
                param.requires_grad = False
        
        # Tier-specific models (adaptive to channel performance level)
        self.elite_model = TierSpecificModel(image_dim, text_dim, "elite")
        self.high_model = TierSpecificModel(image_dim, text_dim, "high") 
        self.medium_model = TierSpecificModel(image_dim, text_dim, "medium")
        self.low_model = TierSpecificModel(image_dim, text_dim, "low")
        
        # Channel baseline predictor (VERY STRONG!)
        self.channel_baseline_weight = nn.Parameter(torch.tensor(0.6))  # 60% from channel history!
        self.model_adjustment_weight = nn.Parameter(torch.tensor(0.4))  # 40% from content
        
        # Tier routing
        self.tier_models = {
            'Elite': self.elite_model,
            'High': self.high_model, 
            'Medium': self.medium_model,
            'Low': self.low_model
        }
    
    def _load_perfect_channel_data(self, dataset_path):
        """Load comprehensive channel data with perfect coverage."""
        try:
            # Load the saved channel lookup table
            self.channel_lookup = pd.read_csv('channel_lookup_table.csv', index_col=0)
            
            # Also load raw training data for baseline predictions
            train_df = pd.read_csv(f"{dataset_path}/train_val.csv")
            
            # Channel baseline predictions (median for robustness)
            self.channel_baselines = train_df.groupby('channel')['views'].median().to_dict()
            
            # Channel tier mapping
            self.channel_tiers = self.channel_lookup['performance_tier'].to_dict()
            
            # Channel statistics for advanced features
            self.channel_stats = {}
            for channel in self.channel_lookup.index:
                stats = self.channel_lookup.loc[channel]
                self.channel_stats[channel] = {
                    'baseline_views': self.channel_baselines.get(channel, 10000),
                    'tier': stats['performance_tier'],
                    'vfx_ratio': stats['vfx_ratio'],
                    'short_film_ratio': stats['short_film_ratio'],
                    'cta_ratio': stats['cta_ratio'],
                    'social_ratio': stats['social_ratio'],
                    'consistency': stats['consistency'],
                    'video_count': stats['video_count'],
                    'avg_views': stats['avg_views']
                }
            
            print(f"✅ Loaded perfect channel data for {len(self.channel_stats)} channels")
            print(f"Baseline range: {min(self.channel_baselines.values()):,.0f} - {max(self.channel_baselines.values()):,.0f}")
            
        except Exception as e:
            print(f"❌ Error loading channel data: {e}")
            self.channel_stats = {}
            self.channel_baselines = {}
            self.channel_tiers = {}
    
    def _get_channel_baseline(self, channel):
        """Get robust channel baseline prediction."""
        return self.channel_baselines.get(channel, 10000)  # Default for unknown (shouldn't happen!)
    
    def _get_channel_tier(self, channel):
        """Get channel performance tier."""
        return self.channel_tiers.get(channel, 'Medium')
    
    def _compute_content_deviation_features(self, batch):
        """Compute how this content deviates from channel norms."""
        features = []
        
        channels = batch.get("channel", ["<UNK>"] * len(batch["id"]))
        
        for i, text in enumerate(batch["text"]):
            channel = channels[i]
            channel_stats = self.channel_stats.get(channel, {})
            
            if not isinstance(text, str):
                text = ""
            
            # Parse content
            if " + " in text:
                parts = text.split(" + ", 1)
                title = parts[0].strip()
                description = parts[1].strip() if len(parts) > 1 else ""
            else:
                title = text.strip()
                description = ""
            
            # Content analysis
            is_vfx_content = int(any(word in title.lower() for word in ['vfx', 'cgi', 'breakdown']))
            is_short_film = int(any(word in title.lower() for word in ['short film', 'animated short']))
            has_cta = int(any(word in description.lower() for word in ['subscribe', 'like', 'share']))
            has_social = int(any(word in description.lower() for word in ['facebook', 'twitter', 'instagram']))
            
            # Deviation from channel norms
            expected_vfx = channel_stats.get('vfx_ratio', 0.1)
            expected_short_film = channel_stats.get('short_film_ratio', 0.2)
            expected_cta = channel_stats.get('cta_ratio', 0.5)
            expected_social = channel_stats.get('social_ratio', 0.4)
            
            # Deviation scores (positive = above normal for this channel)
            vfx_deviation = is_vfx_content - expected_vfx
            short_film_deviation = is_short_film - expected_short_film
            cta_deviation = has_cta - expected_cta
            social_deviation = has_social - expected_social
            
            # Content quality indicators
            title_quality_score = (
                min(len(title) / 50.0, 1.0) * 0.3 +  # Good length
                int('"' in title) * 0.2 +  # Has quotes
                int(':' in title) * 0.2 +  # Has subtitle
                int(any(c.isupper() for c in title)) * 0.3  # Has caps
            )
            
            desc_quality_score = min(len(description) / 1000.0, 1.0)
            
            feature_vector = [
                vfx_deviation,
                short_film_deviation,
                cta_deviation,
                social_deviation,
                title_quality_score,
                desc_quality_score,
                # Channel context features
                channel_stats.get('video_count', 50) / 100.0,  # Channel maturity
                max(-1, min(1, channel_stats.get('consistency', 0))),  # Channel consistency
            ]
            
            features.append(feature_vector)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self, batch):
        device = next(self.parameters()).device
        channels = batch.get("channel", ["<UNK>"] * len(batch["id"]))
        
        # Get channel baseline predictions (VERY STRONG!)
        baseline_predictions = []
        channel_tiers = []
        
        for channel in channels:
            baseline = self._get_channel_baseline(channel)
            tier = self._get_channel_tier(channel)
            baseline_predictions.append(baseline)
            channel_tiers.append(tier)
        
        baseline_preds = torch.tensor(baseline_predictions, device=device, dtype=torch.float32).unsqueeze(1)
        
        # Content analysis
        image_features = self.image_backbone(batch["image"])
        
        # Text processing
        raw_text = list(batch["text"])
        processed_text = [text.strip() if isinstance(text, str) else "[EMPTY]" for text in raw_text]
        
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
        text_features = text_outputs.last_hidden_state[:, 0, :]
        
        # Content deviation features
        deviation_features = self._compute_content_deviation_features(batch).to(device)
        
        # Tier-specific adjustments
        tier_adjustments = []
        
        for i, tier in enumerate(channel_tiers):
            # Get features for this sample
            img_feat = image_features[i:i+1]
            txt_feat = text_features[i:i+1]
            dev_feat = deviation_features[i:i+1]
            
            # Get tier-specific model
            tier_model = self.tier_models[tier]
            adjustment = tier_model(img_feat, txt_feat, dev_feat)
            tier_adjustments.append(adjustment)
        
        tier_adjustments = torch.cat(tier_adjustments, dim=0)
        
        # Final prediction: Strong channel baseline + learned adjustments
        final_pred = (self.channel_baseline_weight * baseline_preds + 
                     self.model_adjustment_weight * tier_adjustments)
        
        return final_pred


class TierSpecificModel(nn.Module):
    """Tier-specific adjustment model."""
    def __init__(self, image_dim, text_dim, tier_type):
        super().__init__()
        
        self.tier_type = tier_type
        
        if tier_type == "elite":
            # Elite channels: content quality matters most
            self.image_head = nn.Sequential(
                nn.Linear(image_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64)
            )
            
            self.text_head = nn.Sequential(
                nn.Linear(text_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 128)
            )
            
            self.fusion = nn.Sequential(
                nn.Linear(64 + 128 + 8, 128),  # image + text + deviation features
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            
        elif tier_type == "high":
            # High tier: balanced approach
            self.image_head = nn.Sequential(
                nn.Linear(image_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64)
            )
            
            self.text_head = nn.Sequential(
                nn.Linear(text_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 64)
            )
            
            self.fusion = nn.Sequential(
                nn.Linear(64 + 64 + 8, 96),
                nn.ReLU(),
                nn.Linear(96, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
        else:  # medium and low
            # Medium/Low tier: simpler models
            self.image_head = nn.Sequential(
                nn.Linear(image_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 32)
            )
            
            self.text_head = nn.Sequential(
                nn.Linear(text_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 32)
            )
            
            self.fusion = nn.Sequential(
                nn.Linear(32 + 32 + 8, 48),
                nn.ReLU(),
                nn.Linear(48, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
    
    def forward(self, image_features, text_features, deviation_features):
        img_processed = self.image_head(image_features)
        txt_processed = self.text_head(text_features)
        
        combined = torch.cat([img_processed, txt_processed, deviation_features], dim=1)
        adjustment = self.fusion(combined)
        
        return adjustment


class SimpleChannelBaseline(nn.Module):
    """
    Ultra-simple baseline using just channel medians + tiny adjustments.
    This should be VERY strong given perfect channel coverage!
    """
    def __init__(self, dataset_path="dataset"):
        super().__init__()
        
        # Load channel baselines
        train_df = pd.read_csv(f"{dataset_path}/train_val.csv")
        self.channel_medians = train_df.groupby('channel')['views'].median().to_dict()
        
        # Tiny adjustment network
        self.adjustment = nn.Sequential(
            nn.Linear(5, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # Bounded adjustments
        )
        
        print(f"✅ Simple baseline for {len(self.channel_medians)} channels")
    
    def forward(self, batch):
        device = next(self.parameters()).device
        channels = batch.get("channel", ["<UNK>"] * len(batch["id"]))
        
        # Channel baselines
        baselines = [self.channel_medians.get(ch, 10000) for ch in channels]
        baselines = torch.tensor(baselines, device=device, dtype=torch.float32).unsqueeze(1)
        
        # Tiny content adjustments
        adjustments = []
        for text in batch["text"]:
            if not isinstance(text, str):
                text = ""
            
            # Very basic features
            title = text.split(" + ")[0] if " + " in text else text
            features = [
                min(len(title) / 50.0, 1.0),
                int('"' in title),
                int(':' in title),
                int('!' in title),
                int(any(c.isupper() for c in title))
            ]
            adjustments.append(features)
        
        adjustments = torch.tensor(adjustments, device=device, dtype=torch.float32)
        adjustment_factors = self.adjustment(adjustments)
        
        # Final: baseline * (1 + small_adjustment)
        final_pred = baselines * (1 + 0.1 * adjustment_factors)
        
        return final_pred