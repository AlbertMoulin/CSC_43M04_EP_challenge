from torch.utils.data import DataLoader
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
        val_split=0.2,
        random_seed=42,
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        self.val_split = val_split
        self.random_seed = random_seed
        
        # Create train/val split indices
        self._create_train_val_split()

    def _create_train_val_split(self):
        """Create train/validation split from the train_val dataset."""
        # Read the full training dataset info
        info = pd.read_csv(f"{self.dataset_path}/train_val.csv")
        
        # Set random seed for reproducibility
        torch.manual_seed(self.random_seed)
        
        # Create random indices
        dataset_size = len(info)
        indices = torch.randperm(dataset_size).tolist()
        
        # Calculate split point
        val_size = int(self.val_split * dataset_size)
        
        # Split indices
        self.train_indices = indices[val_size:]
        self.val_indices = indices[:val_size]
        
        print(f"Dataset split: {len(self.train_indices)} train, {len(self.val_indices)} validation")

    def train_dataloader(self):
        """Train dataloader with subset of data."""
        train_set = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.train_transform,
            metadata=self.metadata,
            indices=self.train_indices,
        )
        return DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Validation dataloader with subset of data."""
        val_set = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.test_transform,  # Use test transforms for validation
            metadata=self.metadata,
            indices=self.val_indices,
        )
        return DataLoader(
            val_set,
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
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )