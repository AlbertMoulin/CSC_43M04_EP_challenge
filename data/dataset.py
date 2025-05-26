import torch
import pandas as pd
from PIL import Image
import string
import numpy as np

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, split, transforms, metadata, max_length=128, binary_bow=False, vocab = None, max_vocab_size=10000):
        self.dataset_path = dataset_path
        self.split = split
        # - read the info csvs
        print(f"{dataset_path}/{split}.csv")
        info = pd.read_csv(f"{dataset_path}/{split}.csv")
        info["description"] = info["description"].fillna("")

        self.date_features = torch.zeros((len(info), 5), dtype=torch.float32)
        if "date" in metadata:
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
            self.year_features =  torch.tensor(info["year_norm"].values, dtype=torch.float32)

        metadata_no_date = [col for col in metadata if col != "date"]
        info["meta"] = info[metadata_no_date].agg(" [SEP] ".join, axis=1)
        if "views" in info.columns:
            self.log_targets = np.log1p(info["views"].values) # Appliquer la transformation log1p pour forcer la distribution

        # - ids
        self.ids = info["id"].values
        # - text
        self.text = info["meta"].values

        self.channel = info["channel"].values

        # - transforms
        self.transforms = transforms

        # - vectorize the text (BOW)
        self.binary_bow = binary_bow
        if vocab is None:
            self.vocab = self.build_vocab(self.text, max_vocab_size)
        else:
            self.vocab = vocab


    @staticmethod
    def build_vocab(texts, max_vocab_size=10000):
        from collections import Counter
        counter = Counter()
        for text in texts:
            tokens = Dataset.tokenize(text)
            counter.update(tokens)

        # Top 9999 + 1 token UNK
        most_common = counter.most_common(max_vocab_size - 1)
        vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
        vocab["<UNK>"] = len(vocab)  # index pour les mots hors vocab
        return vocab

    @staticmethod
    def tokenize(text):
        # Simple tokenisation : minuscule, retrait ponctuation, split espace
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def vectorize(self, texts):
        # Encode une liste de textes en tenseur BoW [batch, vocab_size]
        vectors = torch.zeros((len(texts), len(self.vocab)), dtype=torch.float32)
        unk_idx = self.vocab["<UNK>"]
        for i, text in enumerate(texts):
            tokens = self.tokenize(text)
            for token in tokens:
                idx = self.vocab.get(token, unk_idx)
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


        # - vectorize the text (BOW)
        bow_vec = self.vectorize([text])[0]

        value = {
            "id": self.ids[idx],
            "image": image,
            "text": text,
            "vectorized_text": bow_vec,
            "date": self.date_features[idx],
            "channel": self.channel[idx],
            "year_norm": self.year_features[idx]
        }
        # - don't have the target for test
        if hasattr(self, "log_targets"):
            value["log_target"] = torch.tensor(self.log_targets[idx], dtype=torch.float32)
        return value
