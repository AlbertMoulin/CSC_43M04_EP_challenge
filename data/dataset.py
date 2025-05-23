import torch
import pandas as pd
from PIL import Image
from datetime import datetime
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata, date_column=None):
        self.dataset_path = dataset_path
        self.split = split
        self.date_column = date_column
        
        # Read the info csvs
        print(f"{dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        info["description"] = info["description"].fillna("")
        info["meta"] = info[metadata].agg(" + ".join, axis=1)
        
        if "views" in info.columns:
            self.targets = info["views"].values

        # IDs and text
        self.ids = info["id"].values
        self.text = info["meta"].values
        
        # Date processing
        if date_column and date_column in info.columns:
            self.has_dates = True
            # Convert date column to datetime and then to timestamp
            try:
                # Parse ISO format dates with timezone (like "2024-12-19 03:30:00+00:00")
                dates = pd.to_datetime(info[date_column], errors='coerce')
                
                # Convert to timestamp for consistent handling
                self.dates = dates.astype(np.int64) / 10**9  # Convert to seconds
                
                # Fill any NaN dates with median date
                median_date = np.nanmedian(self.dates)
                self.dates = np.where(np.isnan(self.dates), median_date, self.dates)
                
                print(f"Successfully parsed {len(self.dates)} dates from column '{date_column}'")
                print(f"Date range: {dates.min()} to {dates.max()}")
                
            except Exception as e:
                print(f"Warning: Could not parse date column {date_column}: {e}")
                print("Disabling date features for this dataset")
                self.has_dates = False
        else:
            self.has_dates = False
        
        # Transforms
        self.transforms = transforms

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        # Load the image
        image = Image.open(
            f"{self.dataset_path}/{self.split}/{self.ids[idx]}.jpg"
        ).convert("RGB")
        image = self.transforms(image)
        
        value = {
            "id": self.ids[idx],
            "image": image,
            "text": self.text[idx],
        }
        
        # Add date if available
        if self.has_dates:
            value["date"] = torch.tensor(self.dates[idx], dtype=torch.float32)
        
        # Don't have the target for test
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return value