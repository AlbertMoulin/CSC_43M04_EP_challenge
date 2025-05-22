import torch
import pandas as pd
from PIL import Image
import numpy as np
import os 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata=None):
        if metadata is None:
            metadata = ["title"]
            
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        
        # Read the CSV file for the current split
        info = pd.read_csv(os.path.join(dataset_path, f"{split}.csv"))
        
        # Fill missing descriptions with empty string
        info["description"] = info["description"].fillna("")
        
        # Text (title) for CLIP text input
        if "title" in info.columns:
            self.text = info["title"].fillna("").values
        else:
            self.text = np.array([""] * len(info)) 

        # Store targets if available (not for test set)
        if "views" in info.columns:
            self.targets = info["views"].values

        # Store IDs and date information
        self.ids = info["id"].values
        # Store a reference to the full info DataFrame for easy lookup by index in __getitem__
        self.info_df = info 
        
        # Process date information if available
        if "date" in info.columns:
            self.dates = info["date"].values
        else:
            self.dates = np.array(["2000-01-01 00:00:00"] * len(self.ids))
            
        # --- Channel ID Processing (CRITICAL: Needs to be consistent across ALL splits) ---
        # This part is good as it is, as it covers all channels globally.
        if "channel_id" in info.columns:
            # Load all unique channel IDs from both train_val and test CSVs to build a global mapping
            all_channel_ids_train_val = pd.read_csv(os.path.join(dataset_path, "train_val.csv"))['channel_id']
            all_channel_ids_test = pd.read_csv(os.path.join(dataset_path, "test.csv"))['channel_id']
            all_unique_channels_globally = pd.concat([all_channel_ids_train_val, all_channel_ids_test]).unique()
            
            # Create a consistent mapping from original channel_id to integer index
            # This mapping is crucial and must be built once for the entire dataset.
            self.channel_mapping = {channel: idx for idx, channel in enumerate(all_unique_channels_globally)}
            
            # Convert channel_ids in the current split to their integer indices
            # Use .get() with a default index (len(all_unique_channels_globally)) for robustness
            # in case a channel_id in the split is not in the global mapping (shouldn't happen if global mapping is correct)
            self.channel_indices = np.array([self.channel_mapping.get(c_id, len(all_unique_channels_globally)) 
                                             for c_id in info["channel_id"]])
            
            # num_channels for nn.Embedding: total unique channels + 1 for 'unknown' or default
            self.num_channels = len(all_unique_channels_globally) + 1 
        else:
            print("Warning: 'channel_id' column not found in main CSV. Channel embedding will use default index 0.")
            self.channel_indices = np.zeros(len(self.ids), dtype=int)
            self.num_channels = 1 

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        # Load the image
        image_path = os.path.join(self.dataset_path, self.split, f"{self.ids[idx]}.jpg")
        image = Image.open(image_path).convert("RGB")
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
            views = torch.tensor(self.targets[idx], dtype=torch.float32)
            # Apply log(views + 1) transformation to the target
            value["views"] = torch.log1p(views) 
            
        return value
    
    def get_num_channels(self):
        """Return the number of unique channels for embedding initialization"""
        return self.num_channels