import torch
import pandas as pd
from PIL import Image
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata=None):
        if metadata is None:
            metadata = ["title"]
            
        self.dataset_path = dataset_path
        self.split = split
        
        # Read the CSV file with video information
        print(f"{dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        
        # Fill missing descriptions with empty string
        info["description"] = info["description"].fillna("")
        
        # Combine the specified metadata fields
        info["meta"] = info[metadata].agg(" + ".join, axis=1)
        
        # Store targets if available (not for test set)
        if "views" in info.columns:
            self.targets = info["views"].values

        # Store IDs, text, dates and channel information
        self.ids = info["id"].values
        self.text = info["meta"].values
        
        # Process date information if available
        if "date" in info.columns:
            self.dates = info["date"].values
        else:
            # If not available, use a default date
            self.dates = np.array(["2023-01-01"] * len(self.ids))
            
        # Process channel information if available
        if "channel_id" in info.columns:
            # Get unique channel IDs and create a mapping dictionary
            unique_channels = info["channel_id"].unique()
            self.channel_mapping = {channel: idx for idx, channel in enumerate(unique_channels)}
            # Convert channel IDs to indices
            self.channel_indices = np.array([self.channel_mapping[channel] for channel in info["channel_id"]])
            # Store number of unique channels for embedding layer initialization
            self.num_channels = len(unique_channels)
        else:
            # If not available, use a default channel index
            self.channel_indices = np.zeros(len(self.ids), dtype=int)
            self.num_channels = 1
            
        # Image transforms
        self.transforms = transforms

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        # Load the image
        image = Image.open(
            f"{self.dataset_path}/{self.split}/{self.ids[idx]}.jpg"
        ).convert("RGB")
        image = self.transforms(image)
        
        # Create return dictionary with all features
        value = {
            "id": self.ids[idx],
            "image": image,
            "text": self.text[idx],
            "date": self.dates[idx],
            "channel_id": torch.tensor(self.channel_indices[idx], dtype=torch.long),
        }
        
        # Add target if available (training/validation)
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
            
        return value
    
    def get_num_channels(self):
        """Return the number of unique channels for embedding initialization"""
        return self.num_channels