import torch
import pandas as pd
from PIL import Image
from datetime import datetime
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata, indices=None):
        self.dataset_path = dataset_path
        self.split = split
        # - read the info csvs
        print(f"{dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        info["description"] = info["description"].fillna("")
        info["meta"] = info[metadata].agg(" + ".join, axis=1)
        
        # Apply indices subset if provided (for train/val split)
        if indices is not None:
            info = info.iloc[indices].reset_index(drop=True)
            print(f"Using subset of {len(indices)} samples from {split}")
        
        if "views" in info.columns:
            self.targets = info["views"].values

        # - ids
        self.ids = info["id"].values
        # - text
        self.text = info["meta"].values
        
        # - channel and date information (if available)
        self.channels = info["channel"].values if "channel" in info.columns else None
        self.dates = info["date"].values if "date" in info.columns else None
        
        # Process dates if available
        if self.dates is not None:
            self.processed_dates = self._process_dates(self.dates)
        else:
            # Create dummy date features for test set
            self.processed_dates = np.random.randn(len(self.ids), 6)

        # - transforms
        self.transforms = transforms
        
        # Create channel to index mapping
        if self.channels is not None:
            unique_channels = pd.unique(info["channel"])
            self.channel_to_idx = {channel: idx for idx, channel in enumerate(unique_channels)}
            self.channel_to_idx['<UNK>'] = len(unique_channels)
        else:
            self.channel_to_idx = {'<UNK>': 0}
    
    def _process_dates(self, dates):
        """Convert date strings to engineered features."""
        processed = []
        for date_str in dates:
            try:
                # Parse the date string (adjust format as needed)
                if pd.isna(date_str):
                    # Handle missing dates with default values
                    date_features = [2020, 1, 1, 0, 1, 0]  # Default date
                else:
                    # Assuming date format is YYYY-MM-DD, adjust as needed
                    date_obj = pd.to_datetime(date_str)
                    
                    # Extract features
                    year = date_obj.year - 2010  # Normalize year
                    month = date_obj.month
                    day = date_obj.day
                    day_of_week = date_obj.weekday()
                    day_of_year = date_obj.timetuple().tm_yday
                    is_weekend = 1 if day_of_week >= 5 else 0
                    
                    date_features = [year, month, day, day_of_week, day_of_year, is_weekend]
                
                processed.append(date_features)
            except:
                # Handle parsing errors with default values
                processed.append([2020, 1, 1, 0, 1, 0])
        
        return np.array(processed, dtype=np.float32)

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        # - load the image
        image = Image.open(
            f"{self.dataset_path}/{self.split}/{self.ids[idx]}.jpg"
        ).convert("RGB")
        image = self.transforms(image)
        
        # Get channel index
        if self.channels is not None:
            channel = self.channels[idx]
            channel_idx = self.channel_to_idx.get(channel, self.channel_to_idx['<UNK>'])
        else:
            channel_idx = 0  # Default unknown channel
        
        # Get date features
        date_features = torch.tensor(self.processed_dates[idx], dtype=torch.float32)
        
        value = {
            "id": self.ids[idx],
            "image": image,
            "text": self.text[idx],
            "channel_idx": torch.tensor(channel_idx, dtype=torch.long),
            "date_features": date_features,
        }
        
        # - don't have the target for test
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return value