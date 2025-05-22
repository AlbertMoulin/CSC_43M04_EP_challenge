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
        metadata=["title"],
        validation_set_type="random",
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        self.val_split = 0.2
        self.random_seed = 42  # Pour la reproductibilité
        self._setup_indices(validation_set_type)

    def _setup_indices(self,validation_set_type="random"):
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
        )

        validation_dataset = torch.utils.data.Subset(val_set, self.val_indices)

        return DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        """Test dataloader."""
        dataset = Dataset(
            self.dataset_path,
            "test",
            transforms=self.test_transform,
            metadata=self.metadata,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )