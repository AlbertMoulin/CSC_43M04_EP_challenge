import torch
import pandas as pd
from PIL import Image
from transformers import BertTokenizer

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata, max_length=128):
        self.dataset_path = dataset_path
        self.split = split
        # - read the info csvs
        print(f"{dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        info["description"] = info["description"].fillna("")
        info["meta"] = info[metadata].agg(" [SEP] ".join, axis=1)
        if "views" in info.columns:
            self.targets = info["views"].values

        # - ids
        self.ids = info["id"].values
        # - text
        self.text = info["meta"].values

        # - transforms
        self.transforms = transforms

        # - token for BERT
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        # - load the image
        image = Image.open(
            f"{self.dataset_path}/{self.split}/{self.ids[idx]}.jpg"
        ).convert("RGB")
        image = self.transforms(image)
        text = self.text[idx]

        # Tokenize le texte pour BERT
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        value = {
            "id": self.ids[idx],
            "image": image,
            "text": text,
            "input_ids": encoding["input_ids"].squeeze(0),        # [max_length]
            "attention_mask": encoding["attention_mask"].squeeze(0), # [max_length]
        }
        # - don't have the target for test
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return value
