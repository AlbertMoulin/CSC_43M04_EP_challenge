import torch
import pandas as pd
from PIL import Image


class EnhancedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata):
        self.dataset_path = dataset_path
        self.split = split
        # - read the info csvs
        print(f"{dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        info["description"] = info["description"].fillna("")
        
        # Only use title for text (not description since it's repetitive)
        info["meta"] = info["title"]  # Just use title, ignore description
        
        if "views" in info.columns:
            self.targets = info["views"].values

        # - ids
        self.ids = info["id"].values
        # - text (title only)
        self.text = info["meta"].values
        # - channel information
        self.channels = info["channel"].values
        # - date information  
        self.dates = info["date"].values

        # - transforms
        self.transforms = transforms

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
            "text": self.text[idx],
            "channel": self.channels[idx],
            "date": self.dates[idx],
        }
        
        # - don't have the target for test
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return value