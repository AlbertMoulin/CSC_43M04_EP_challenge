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
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        self.val_split = val_split

    def _get_temporal_split_indices(self):
        """Get train/val indices based on temporal split (most recent 20% for validation)"""
        # Read the CSV to get dates
        train_val_info = pd.read_csv(f"{self.dataset_path}/train_val.csv")
        
        # Convert date strings to datetime objects for sorting
        train_val_info['datetime'] = pd.to_datetime(train_val_info['date'])
        
        # Sort by date to get temporal order
        train_val_info_sorted = train_val_info.sort_values('datetime').reset_index(drop=True)
        
        # Calculate split point (most recent 20% for validation)
        total_size = len(train_val_info_sorted)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size
        
        # Get the original indices after sorting
        train_indices = train_val_info_sorted.iloc[:train_size].index.tolist()
        val_indices = train_val_info_sorted.iloc[train_size:].index.tolist()
        
        # Print some info about the split
        train_date_range = (
            train_val_info_sorted.iloc[0]['datetime'].strftime('%Y-%m-%d'),
            train_val_info_sorted.iloc[train_size-1]['datetime'].strftime('%Y-%m-%d')
        )
        val_date_range = (
            train_val_info_sorted.iloc[train_size]['datetime'].strftime('%Y-%m-%d'),
            train_val_info_sorted.iloc[-1]['datetime'].strftime('%Y-%m-%d')
        )
        
        print(f"Temporal split:")
        print(f"  Train: {len(train_indices)} samples from {train_date_range[0]} to {train_date_range[1]}")
        print(f"  Val: {len(val_indices)} samples from {val_date_range[0]} to {val_date_range[1]}")
        
        return train_indices, val_indices

    def train_dataloader(self):
        """Train dataloader with temporal split (older 80% of videos)."""
        full_dataset = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.train_transform,
            metadata=self.metadata,
        )
        
        # Get temporal split indices
        train_indices, _ = self._get_temporal_split_indices()
        train_subset = Subset(full_dataset, train_indices)
        
        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Validation dataloader with temporal split (most recent 20% of videos)."""
        full_dataset = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.test_transform,  # Use test transforms for validation
            metadata=self.metadata,
        )
        
        # Get temporal split indices
        _, val_indices = self._get_temporal_split_indices()
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
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )