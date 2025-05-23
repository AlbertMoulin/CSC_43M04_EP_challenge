import torch
import pandas as pd
from PIL import Image
from transformers import BertTokenizer
import string
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata, max_length=128, binary_bow=False):
        self.dataset_path = dataset_path
        self.split = split
        # - read the info csvs
        print(f"{dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        info["description"] = info["description"].fillna("")

        self.date_features=None
        if "date" in metadata:
            metadata.remove("date")
            info["date"] = pd.to_datetime(info["date"], utc=True, errors="coerce")
            info["year"] = info["date"].dt.year.fillna(2018).clip(2011, 2025).astype(int)  # gestion des années hors bornes
            info["month"] = info["date"].dt.month.fillna(1).astype(int)
            info["day_of_week"] = info["date"].dt.weekday.fillna(0).astype(int)

            # Encodage cyclique
            info["month_sin"] = np.sin(2 * np.pi * info["month"] / 12)
            info["month_cos"] = np.cos(2 * np.pi * info["month"] / 12)
            info["day_sin"] = np.sin(2 * np.pi * info["day_of_week"] / 7)
            info["day_cos"] = np.cos(2 * np.pi * info["day_of_week"] / 7)

            # Normalisation de l'année entre 0 et 1
            info["year_norm"] = (info["year"] - 2011) / (2025 - 2011)

            # Final tensor
            self.date_features = torch.tensor(
                info[["year_norm", "month_sin", "month_cos", "day_sin", "day_cos"]].values,
                dtype=torch.float32
            )

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

        # - vectorize the text (BOW)
        self.binary_bow = binary_bow
        self.vocab = self.build_vocab(self.text)

    @staticmethod
    def build_vocab(texts, min_freq=1):
        from collections import Counter
        counter = Counter()
        for text in texts:
            tokens = Dataset.tokenize(text)
            counter.update(tokens)
        # On garde les mots qui apparaissent au moins min_freq fois
        vocab = {word: idx for idx, (word, count) in enumerate(counter.items()) if count >= min_freq}
        return vocab

    @staticmethod
    def tokenize(text):
        # Simple tokenisation : minuscule, retrait ponctuation, split espace
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def vectorize(self, texts):
        # Encode une liste de textes en tenseur BoW [batch, vocab_size]
        vectors = torch.zeros((len(texts), len(self.vocab)), dtype=torch.float32)
        for i, text in enumerate(texts):
            tokens = self.tokenize(text)
            for token in tokens:
                idx = self.vocab.get(token)
                if idx is not None:
                    if self.binary_bow:
                        vectors[i, idx] = 1
                    else:
                        vectors[i, idx] += 1
        return vectors

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

        # - vectorize the text (BOW)
        bow_vec = self.vectorize([text])[0]

        value = {
            "id": self.ids[idx],
            "image": image,
            "text": text,
            "input_ids": encoding["input_ids"].squeeze(0),        # [max_length]
            "attention_mask": encoding["attention_mask"].squeeze(0), # [max_length]
            "vectorized_text": bow_vec,
            "date": self.date_features
        }
        # - don't have the target for test
        if hasattr(self, "targets"):
            value["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return value
