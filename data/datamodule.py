import pandas as pd
import torch
import os
from torch.utils.data import DataLoader, random_split # Keep random_split import for now, but we won't use it in _setup_indices anymore

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
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform  
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.metadata = metadata
        self.random_seed = 42
        self.val_split = 0.2 # Percentage for validation set
        
        # We still need to determine num_channels before setting up indices,
        # as the Dataset requires the global channel mapping.
        # Create a dummy dataset for this purpose, just to get the total unique channels.
        # This will load train_val.csv and test.csv internally within Dataset to build mapping.
        dummy_dataset = Dataset(
            self.dataset_path,
            "train_val", # Split can be "train_val" or "test" here, it just needs to load data
            transforms=None, # No transforms needed
            metadata=self.metadata
        )
        self.num_channels = dummy_dataset.get_num_channels()
        del dummy_dataset # Clean up the dummy instance

        self._setup_indices()

    def _setup_indices(self):
        """Prépare les indices pour les ensembles train et validation en utilisant une division temporelle."""
        # Load the full train_val data to sort by date
        df_full = pd.read_csv(os.path.join(self.dataset_path, "train_val.csv"))
        
        # Ensure 'date' column exists and convert to datetime objects
        if 'date' not in df_full.columns:
            raise ValueError("The 'train_val.csv' must contain a 'date' column for time-based splitting.")
        
        # Attempt to parse date strings
        # We need a robust way to handle date parsing here, similar to what's in _parse_date
        # in CLIPBranchModel. For simplicity, we'll try common formats.
        # You might need to adjust this based on your exact date string format.
        df_full['parsed_date'] = pd.to_datetime(df_full['date'], errors='coerce', format='ISO8601')
        
        # Handle rows where date parsing failed (should ideally be cleaned in preprocessing)
        if df_full['parsed_date'].isnull().any():
            print("Warning: Some dates could not be parsed and will be excluded from time-based split.")
            df_full.dropna(subset=['parsed_date'], inplace=True)

        # Sort the DataFrame by date
        df_full = df_full.sort_values(by='parsed_date').reset_index(drop=True)
        
        dataset_size = len(df_full)
        
        # Calculate split sizes
        val_size = int(self.val_split * dataset_size)
        train_size = dataset_size - val_size
        
        # Get indices: train will be older data, validation will be newer data
        self.train_indices = df_full.iloc[:train_size].index.tolist()
        self.val_indices = df_full.iloc[train_size:].index.tolist()

        print(f"Dataset size: {dataset_size}")
        print(f"Train size (older data): {len(self.train_indices)}")
        print(f"Validation size (newer data): {len(self.val_indices)}")


    def train_dataloader(self):
        """Train dataloader."""
        # Dataset instance needs to know its path, split, transforms, and metadata.
        # The key here is that the Dataset needs to load the ENTIRE train_val.csv
        # but then the DataLoader will be restricted to self.train_indices.
        train_set = Dataset(
            self.dataset_path,
            "train_val", # Still load from train_val.csv
            transforms=self.train_transform,
            metadata=self.metadata,
        )

        # Use Subset to apply the pre-calculated indices
        train_dataset = torch.utils.data.Subset(train_set, self.train_indices)

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True, # Shuffle training data
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        """Validation dataloader."""
        val_set = Dataset(
            self.dataset_path,
            "train_val", # Still load from train_val.csv
            transforms=self.test_transform, # Use test_transform for validation
            metadata=self.metadata,
        )

        # Use Subset to apply the pre-calculated indices
        validation_dataset = torch.utils.data.Subset(val_set, self.val_indices)

        return DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False, # Do not shuffle validation data
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Test dataloader."""
        # Test set remains unchanged, always loads from 'test.csv'
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
            pin_memory=True
        )