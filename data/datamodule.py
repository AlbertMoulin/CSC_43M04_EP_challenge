from torch.utils.data import DataLoader, Subset
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
        val_split=0.2,  # 20% for validation
        date_column=None,  # New parameter for date column
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        self.val_split = val_split
        self.date_column = date_column

    def train_dataloader(self):
        """Train dataloader with subset of indices."""
        full_dataset = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.train_transform,
            metadata=self.metadata,
            date_column=self.date_column,
        )
        
        # Calculate train indices (first 80%)
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size
        train_indices = list(range(train_size))
        
        train_subset = Subset(full_dataset, train_indices)
        
        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Validation dataloader with subset of indices."""
        full_dataset = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.test_transform,  # Use test transforms for validation
            metadata=self.metadata,
            date_column=self.date_column,
        )
        
        # Calculate val indices (last 20%)
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size
        val_indices = list(range(train_size, total_size))
        
        val_subset = Subset(full_dataset, val_indices)
        
        return DataLoader(
            val_subset,
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
            date_column=self.date_column,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )