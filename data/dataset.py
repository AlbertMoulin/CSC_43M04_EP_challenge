import torch
import pandas as pd
from PIL import Image
import numpy as np
import pickle
import os

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata=None, global_channel_mapping=None):
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
            
        # Process channel information with global mapping
        if "channel_id" in info.columns:
            self.channel_mapping, self.num_channels = self._setup_channel_mapping(
                info, global_channel_mapping
            )
            # Convert channel IDs to indices using the mapping
            self.channel_indices = np.array([
                self.channel_mapping.get(channel, 0) for channel in info["channel_id"]
            ])
        else:
            # If not available, use a default channel index
            self.channel_indices = np.zeros(len(self.ids), dtype=int)
            self.num_channels = 1
            self.channel_mapping = {}
            
        # Image transforms
        self.transforms = transforms

    def _setup_channel_mapping(self, info, global_channel_mapping):
        """
        Setup channel mapping consistently across all splits.
        If global_channel_mapping is provided, use it. Otherwise, create and save it.
        """
        mapping_path = os.path.join(self.dataset_path, "channel_mapping.pkl")
        
        if global_channel_mapping is not None:
            # Use provided mapping
            return global_channel_mapping, len(global_channel_mapping)
        
        elif os.path.exists(mapping_path):
            # Load existing mapping
            with open(mapping_path, 'rb') as f:
                channel_mapping = pickle.load(f)
            print(f"Loaded existing channel mapping with {len(channel_mapping)} channels")
            return channel_mapping, len(channel_mapping)
        
        else:
            # Create new mapping from training data
            if self.split == "train_val":
                # Create mapping from all channels in training data
                unique_channels = info["channel_id"].unique()
                channel_mapping = {channel: idx for idx, channel in enumerate(unique_channels)}
                
                # Save the mapping for future use
                with open(mapping_path, 'wb') as f:
                    pickle.dump(channel_mapping, f)
                
                print(f"Created new channel mapping with {len(channel_mapping)} channels")
                return channel_mapping, len(channel_mapping)
            else:
                # For test split, we must have the mapping already
                raise ValueError(f"No channel mapping found for {self.split} split. "
                               "Please run with train_val split first to create the mapping.")

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
    
    def get_channel_mapping(self):
        """Return the channel mapping for use in other splits"""
        return self.channel_mapping