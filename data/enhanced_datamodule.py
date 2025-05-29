# data/enhanced_datamodule.py
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np

# Importer le nouveau dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Créer le dataset amélioré ici pour éviter les imports
import torch
from PIL import Image
import re
from datetime import datetime


class EnhancedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata):
        self.dataset_path = dataset_path
        self.split = split
        
        # Lire les infos CSV
        print(f"Loading {dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        info["description"] = info["description"].fillna("")
        info["title"] = info["title"].fillna("")
        
        # === FEATURE ENGINEERING CRITIQUE ===
        self._engineer_critical_features(info)
        
        # === CHANNEL PERFORMANCE STATS ===
        if split != "test":  # Seulement pour train/val
            info = self._add_channel_stats(info, dataset_path)
        else:
            # Pour le test, charger les stats depuis le train
            info = self._load_channel_stats_for_test(info, dataset_path)
            
        # === FEATURES TEMPORELLES AMÉLIORÉES ===
        self._engineer_temporal_features(info)
        
        # Concaténer title + description (CRITIQUE)
        info["full_text"] = info["title"] + " [SEP] " + info["description"]
        
        # Targets
        if "views" in info.columns:
            self.targets = info["views"].values
            
        # Stocker toutes les features
        self.ids = info["id"].values
        self.full_text = info["full_text"].values
        self.channels = info["channel"].values
        
        # Features numériques critiques
        feature_columns = [
            'title_length', 'desc_length', 'title_word_count', 'desc_word_count',
            'video_age_days', 'is_short', 'is_cgi_animation', 'is_film',
            'has_caps_title', 'has_numbers_in_title', 'year_normalized',
            'month', 'day_of_week', 'is_weekend',
            'channel_avg_views', 'channel_consistency', 'channel_total_videos'
        ]
        
        # Vérifier que toutes les colonnes existent
        missing_cols = [col for col in feature_columns if col not in info.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}, filling with zeros")
            for col in missing_cols:
                info[col] = 0
                
        self.numeric_features = info[feature_columns].values.astype(np.float32)
        
        # Channel encoding
        self.channel_codes = pd.Categorical(self.channels).codes
        
        self.transforms = transforms
        
        print(f"Dataset {split} loaded: {len(self)} samples")
        print(f"Features shape: {self.numeric_features.shape}")
        
    def _engineer_critical_features(self, info):
        """Ajoute toutes les features critiques identifiées dans l'analyse"""
        
        # === FEATURES TEXTUELLES ===
        info['title_length'] = info['title'].str.len()
        info['desc_length'] = info['description'].str.len()
        info['title_word_count'] = info['title'].str.split().str.len()
        info['desc_word_count'] = info['description'].str.split().str.len()
        
        # === FEATURES CATÉGORIELLES CRITIQUES ===
        
        # YouTube Shorts detection (+118.6% lift)
        shorts_pattern = r'\b(shorts?|short)\b'
        info['is_short'] = (
            info['title'].str.contains(shorts_pattern, case=False, na=False) |
            info['description'].str.contains(shorts_pattern, case=False, na=False)
        ).astype(int)
        
        # CGI/Animation detection (+36.2% lift)
        cgi_pattern = r'\b(cgi|animated?|animation|3d|vfx)\b'
        info['is_cgi_animation'] = (
            info['title'].str.contains(cgi_pattern, case=False, na=False) |
            info['description'].str.contains(cgi_pattern, case=False, na=False)
        ).astype(int)
        
        # Film detection (+235.5% lift - MASSIF!)
        film_pattern = r'\b(film|movie|cinema|short film)\b'
        info['is_film'] = info['title'].str.contains(film_pattern, case=False, na=False).astype(int)
        
        # === FEATURES TITRE ===
        info['has_caps_title'] = info['title'].str.contains(r'[A-Z]{3,}', na=False).astype(int)
        info['has_numbers_in_title'] = info['title'].str.contains(r'\d+', na=False).astype(int)
        info['has_question_mark'] = info['title'].str.contains(r'\?', na=False).astype(int)
        info['has_exclamation'] = info['title'].str.contains(r'!', na=False).astype(int)
        
        print(f"Feature engineering completed:")
        print(f"  - Shorts: {info['is_short'].sum()} ({info['is_short'].mean()*100:.1f}%)")
        print(f"  - CGI: {info['is_cgi_animation'].sum()} ({info['is_cgi_animation'].mean()*100:.1f}%)")
        print(f"  - Films: {info['is_film'].sum()} ({info['is_film'].mean()*100:.1f}%)")
        
    def _add_channel_stats(self, info, dataset_path):
        """Ajoute les statistiques de performance par chaîne (CRITIQUE)"""
        
        # Calculer les stats par chaîne depuis le dataset complet
        if self.split == "train_val":
            full_info = info.copy()
        else:
            # Charger train_val pour avoir toutes les stats
            full_info = pd.read_csv(f"{dataset_path}/train_val.csv")
            full_info["description"] = full_info["description"].fillna("")
            
        # Stats par chaîne
        channel_stats = full_info.groupby('channel')['views'].agg([
            'mean', 'median', 'std', 'count'
        ]).reset_index()
        
        channel_stats.columns = ['channel', 'channel_avg_views', 'channel_median_views', 
                                'channel_std_views', 'channel_total_videos']
        
        # Score de consistance (inverse du coefficient de variation)
        channel_stats['channel_consistency'] = 1 / (1 + channel_stats['channel_std_views'] / channel_stats['channel_avg_views'])
        channel_stats['channel_consistency'] = channel_stats['channel_consistency'].fillna(0)
        
        # Sauvegarder les stats pour le test
        channel_stats.to_csv(f"{dataset_path}/channel_stats.csv", index=False)
        
        # Merger avec les données
        info = info.merge(channel_stats, on='channel', how='left')
        
        print(f"Channel stats added: {len(channel_stats)} unique channels")
        
        # Remplir les valeurs manquantes
        stats_cols = ['channel_avg_views', 'channel_median_views', 'channel_std_views', 
                      'channel_total_videos', 'channel_consistency']
        info[stats_cols] = info[stats_cols].fillna(0)
            
        return info
        
    def _load_channel_stats_for_test(self, info, dataset_path):
        """Charge les stats de chaîne pour le dataset test"""
        try:
            channel_stats = pd.read_csv(f"{dataset_path}/channel_stats.csv")
            info = info.merge(channel_stats, on='channel', how='left')
            
            # Remplir les valeurs manquantes pour les nouvelles chaînes
            stats_cols = ['channel_avg_views', 'channel_median_views', 'channel_std_views', 
                          'channel_total_videos', 'channel_consistency']
            info[stats_cols] = info[stats_cols].fillna(0)
                
            print(f"Channel stats loaded for test set")
        except FileNotFoundError:
            print("Warning: channel_stats.csv not found, using zeros")
            for col in ['channel_avg_views', 'channel_median_views', 'channel_std_views', 
                       'channel_total_videos', 'channel_consistency']:
                info[col] = 0
                
        return info
        
    def _engineer_temporal_features(self, info):
        """Features temporelles améliorées"""
        
        # Convertir les dates
        info['datetime'] = pd.to_datetime(info['date'])
        
        # Âge des vidéos (CRITIQUE - feature #3)
        # Gestion timezone-safe
        if info['datetime'].dt.tz is not None:
            # Si les données ont une timezone, l'utiliser
            reference_date = pd.to_datetime('2024-01-01', utc=True)
            if info['datetime'].dt.tz != reference_date.tz:
                reference_date = reference_date.tz_convert(info['datetime'].dt.tz)
        else:
            # Si pas de timezone, utiliser date naive
            reference_date = pd.to_datetime('2024-01-01')
            
        info['video_age_days'] = (reference_date - info['datetime']).dt.days
        
        # Features temporelles
        info['year'] = info['datetime'].dt.year
        info['month'] = info['datetime'].dt.month
        info['day_of_week'] = info['datetime'].dt.dayofweek
        info['is_weekend'] = (info['day_of_week'] >= 5).astype(int)
        
        # Normalisation de l'année
        year_min, year_max = info['year'].min(), info['year'].max()
        if year_max > year_min:
            info['year_normalized'] = (info['year'] - year_min) / (year_max - year_min)
        else:
            info['year_normalized'] = 0.5
        
    def __len__(self):
        return len(self.ids)
        
    def __getitem__(self, idx):
        # Charger l'image
        image = Image.open(
            f"{self.dataset_path}/{self.split}/{self.ids[idx]}.jpg"
        ).convert("RGB")
        image = self.transforms(image)
        
        # Préparer le batch
        batch = {
            "id": self.ids[idx],
            "image": image,
            "text": self.full_text[idx],  # Title + description
            "numeric_features": torch.tensor(self.numeric_features[idx], dtype=torch.float32),
            "channel": self.channel_codes[idx]
        }
        
        # Ajouter target si disponible
        if hasattr(self, "targets"):
            batch["target"] = torch.tensor(np.log1p(self.targets[idx]), dtype=torch.float32)
            
        return batch


class EnhancedDataModule:
    def __init__(
        self,
        dataset_path,
        train_transform,
        test_transform,
        batch_size,
        num_workers,
        metadata=["title", "description"],  # Par défaut title + description
        val_split=0.25,
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        self.val_split = val_split
        
        print(f"Enhanced DataModule initialized:")
        print(f"  - Metadata: {metadata}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Val split: {val_split}")

    def _get_temporal_split_indices(self):
        """Get train/val indices based on temporal split"""
        train_val_info = pd.read_csv(f"{self.dataset_path}/train_val.csv")
        
        # Convert date strings to datetime objects for sorting
        train_val_info['datetime'] = pd.to_datetime(train_val_info['date'])
        
        # Sort by date but keep original indices
        train_val_info_sorted = train_val_info.sort_values('datetime')
        
        # Calculate split point (most recent for validation)
        total_size = len(train_val_info_sorted)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size
        
        # Get the ORIGINAL indices (before sorting)
        train_indices = train_val_info_sorted.iloc[:train_size].index.tolist()
        val_indices = train_val_info_sorted.iloc[train_size:].index.tolist()
        
        # Print info
        train_date_range = (
            train_val_info_sorted.iloc[0]['datetime'].strftime('%Y-%m-%d'),
            train_val_info_sorted.iloc[train_size-1]['datetime'].strftime('%Y-%m-%d')
        )
        val_date_range = (
            train_val_info_sorted.iloc[train_size]['datetime'].strftime('%Y-%m-%d'),
            train_val_info_sorted.iloc[-1]['datetime'].strftime('%Y-%m-%d')
        )
        
        print(f"Temporal split:")
        print(f"  Train: {len(train_indices)} samples from {train_date_range[0]} to {train_date_range[1]}")
        print(f"  Val: {len(val_indices)} samples from {val_date_range[0]} to {val_date_range[1]}")
        
        return train_indices, val_indices

    def train_dataloader(self):
        """Train dataloader with enhanced features"""
        full_dataset = EnhancedDataset(
            self.dataset_path,
            "train_val",
            transforms=self.train_transform,
            metadata=self.metadata,
        )
        
        # Get temporal split indices
        train_indices, _ = self._get_temporal_split_indices()
        train_subset = Subset(full_dataset, train_indices)
        
        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # Optimisation GPU
        )

    def val_dataloader(self):
        """Validation dataloader with enhanced features"""
        full_dataset = EnhancedDataset(
            self.dataset_path,
            "train_val",
            transforms=self.test_transform,
            metadata=self.metadata,
        )
        
        # Get temporal split indices
        _, val_indices = self._get_temporal_split_indices()
        val_subset = Subset(full_dataset, val_indices)
        
        return DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        """Test dataloader with enhanced features"""
        dataset = EnhancedDataset(
            self.dataset_path,
            "test",
            transforms=self.test_transform,
            metadata=self.metadata,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )