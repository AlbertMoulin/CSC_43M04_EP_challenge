from torch.utils.data import DataLoader, random_split
import torch
import pandas as pd
from data.dataset import Dataset


class DataModule:
    def __init__(
        self,
        dataset_path,
        train_transform,
        test_transform,
        batch_size,
        num_workers,
        metadata=None,
        validation_set_type="random",
        val_split=0.2,
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata if metadata is not None else ["title"]
        self.val_split = val_split
        self.random_seed = 42  # Pour la reproductibilité
        self._setup_indices(validation_set_type)
        self.vocab = None
        self.setup_vocab()

        self.unique_channels = self._get_unique_channels()

    def _get_unique_channels(self):
        """Get all unique channels from train_val data"""
        train_val_info = pd.read_csv(f"{self.dataset_path}/train_val.csv")
        unique_channels = sorted(train_val_info["channel"].unique().tolist())
        print(f"Found {len(unique_channels)} unique channels")
        return unique_channels

    def get_unique_channels(self):
        """Return unique channels for model initialization"""
        return self.unique_channels
    
    def setup_vocab(self):
        # Crée un Dataset complet train_val juste pour construire vocab
        full_train_set = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.train_transform,
            metadata=self.metadata,
            vocab=None,
        )
        texts = [full_train_set.text[i] for i in self.train_indices]
        self.vocab = full_train_set.build_vocab(texts)

    def _setup_indices(self,validation_set_type="newest"):
        """Prépare les indices pour les ensembles train et validation."""
        # Charger les données pour obtenir le nombre total d'échantillons
        df = pd.read_csv(f"{self.dataset_path}/train_val.csv")
        dataset_size = len(df)
        
        # Calculer la taille de l'ensemble de validation
        val_size = int(self.val_split * dataset_size)
        train_size = dataset_size - val_size
        
        if validation_set_type == "random":
            # Créer un générateur avec une graine fixe pour la reproductibilité
            generator = torch.Generator().manual_seed(self.random_seed)
            
            # Générer les indices pour train et val
            self.train_indices, self.val_indices = random_split(
                range(dataset_size), 
                [train_size, val_size],
                generator=generator
            )
        elif validation_set_type=="oldest":
            df["date"] = pd.to_datetime(df["date"])
            sorted_indices = df.sort_values("date").index.tolist()
            self.val_indices = sorted_indices[:val_size]
            self.train_indices = sorted_indices[val_size:]
        
        elif validation_set_type=="newest":
            df["date"] = pd.to_datetime(df["date"])
            sorted_indices = df.sort_values("date").index.tolist()
            self.train_indices = sorted_indices[:train_size]
            self.val_indices = sorted_indices[train_size:]

    def train_dataloader(self):
        """Train dataloader."""
        train_set = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.train_transform,
            metadata=self.metadata,
            vocab = self.vocab,
        )

        train_dataset = torch.utils.data.Subset(train_set, self.train_indices)

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Implement a strategy to create a validation set from the train set.
        """
        val_set = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.test_transform,
            metadata=self.metadata,
            vocab = self.vocab,
        )

        validation_dataset = torch.utils.data.Subset(val_set, self.val_indices)
        print(list(set(self.val_indices)&set(self.train_indices)))

        return DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        """Test dataloader."""
        dataset = Dataset(
            self.dataset_path,
            "test",
            transforms=self.test_transform,
            metadata=self.metadata,
            vocab = self.vocab,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )