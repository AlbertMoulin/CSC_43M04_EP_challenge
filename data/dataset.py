import torch
import pandas as pd
from PIL import Image
from datetime import datetime
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, channel_to_int=None):
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = transforms
        self.channel_to_int = channel_to_int # Accept mapping from DataModule

        print(f"Loading data from {dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        info["description"] = info["description"].fillna("")

        required_cols = ["id", "channel", "title", "date"]
        if not all(col in info.columns for col in required_cols):
            raise ValueError(f"CSV file '{split}.csv' must contain columns: {required_cols}")

        self.ids = info["id"].values
        self.titles = info["title"].values
        self.channels = info["channel"].values
        self.dates = info["date"].values


        if "views" in info.columns:
            self.targets = info["views"].values
        else:
            self.targets = None # Handle test set where views are not available


        # --- Channel Processing ---
        # If mapping is not provided (e.g., for the split used to build the map in DataModule)
        if self.channel_to_int is None:
             # Build map (this should ideally happen only once in DataModule on train data)
            unique_channels = info["channel"].unique().tolist()
            # Add a special ID for unknown channels if not already present
            if '<unk>' not in unique_channels:
                 unique_channels.append('<unk>')
            self.channel_to_int = {channel: i for i, channel in enumerate(unique_channels)}

        # Convert channel names to integer IDs using the provided/built map
        # Corrected dtype from np.long to np.int64
        self.channel_ids = np.array([self.channel_to_int.get(channel, self.channel_to_int.get('<unk>', -1))
                                     for channel in self.channels], dtype=np.int64) # Changed np.long to np.int64


        # --- Date Processing (using timestamp) ---
        self.date_timestamps = []
        for date_str in self.dates:
            try:
                # Assuming the format is 'YYYY-MM-DD HH:MM:SS+00:00'
                dt_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                self.date_timestamps.append(dt_obj.timestamp())
            except (ValueError, TypeError):
                # Handle cases where date parsing fails
                self.date_timestamps.append(0.0) # Using 0.0 as a default for invalid dates
        self.date_timestamps = np.array(self.date_timestamps, dtype=np.float32)


    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        # - load the image
        image = Image.open(
            f"{self.dataset_path}/{self.split}/{self.ids[idx]}.jpg"
        ).convert("RGB")
        image = self.transforms(image)

        value = {
            "id": self.ids[idx],
            "image": image,
            "title": str(self.titles[idx]) if not pd.isna(self.titles[idx]) else "", # Ensure title is string, handle NaN
            "channel_id": torch.tensor(self.channel_ids[idx], dtype=torch.long), # Still use torch.long for the tensor
            "date_timestamp": torch.tensor(self.date_timestamps[idx], dtype=torch.float32)
        }
        # - don't have the target for test
        if self.targets is not None:
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return value

    @property
    def channel_mapping(self):
        return self.channel_to_int

    @property
    def num_channels(self):
         if self.channel_to_int:
             return len(self.channel_to_int)
         return 0