import pandas as pd
import torch

from torch.utils.data import DataLoader, random_split

# Assuming dataset.py is in a 'data' subdirectory
from data.dataset import Dataset


class DataModule:
    def __init__(
        self,
        dataset_path,
        train_transform,
        test_transform,
        batch_size,
        num_workers,
        val_split=0.2, # Added val_split as parameter
        random_seed=42 # Added random_seed as parameter
    ):
        self.dataset_path = dataset_path
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.random_seed = random_seed

        # --- Determine Channel Mapping and Number of Channels ---
        # Load the full training data temporarily to build the channel map
        try:
            train_val_df = pd.read_csv(f"{self.dataset_path}/train_val.csv")
            unique_channels = train_val_df["channel"].unique().tolist()
            # Add a special ID for unknown channels
            if '<unk>' not in unique_channels:
                 unique_channels.append('<unk>')
            self._channel_to_int = {channel: i for i, channel in enumerate(unique_channels)}
            self._num_channels = len(unique_channels)
            print(f"Found {self._num_channels} unique channels (including <unk>).")
        except FileNotFoundError:
            print(f"Warning: train_val.csv not found at {self.dataset_path}. Cannot build channel map.")
            self._channel_to_int = {}
            self._num_channels = 0


        self._setup_indices()

    def _setup_indices(self):
        """Prépare les indices pour les ensembles train et validation."""
        # Load data just to get the size for splitting
        try:
            df = pd.read_csv(f"{self.dataset_path}/train_val.csv")
            dataset_size = len(df)

            val_size = int(self.val_split * dataset_size)
            train_size = dataset_size - val_size

            generator = torch.Generator().manual_seed(self.random_seed)

            self.train_indices, self.val_indices = random_split(
                range(dataset_size),
                [train_size, val_size],
                generator=generator
            )
            print(f"Dataset split into {train_size} training and {val_size} validation samples.")

        except FileNotFoundError:
            print(f"Error: train_val.csv not found at {self.dataset_path}. Cannot setup data indices.")
            self.train_indices = []
            self.val_indices = []


    def train_dataloader(self):
        """Train dataloader."""
        # Pass the channel_to_int mapping to the Dataset
        train_set = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.train_transform,
            channel_to_int=self._channel_to_int # Pass the mapping
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
        # Pass the channel_to_int mapping to the Dataset
        val_set = Dataset(
            self.dataset_path,
            "train_val",
            transforms=self.test_transform,
             channel_to_int=self._channel_to_int # Pass the mapping
        )

        validation_dataset = torch.utils.data.Subset(val_set, self.val_indices)

        return DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Test dataloader."""
        # Pass the channel_to_int mapping to the Dataset
        # Note: For a real test set, you must use the mapping learned from the training data.
        # This simplified example re-uses the map or creates a new one if train_val was not found.
        test_set = Dataset(
            self.dataset_path,
            "test",
            transforms=self.test_transform,
            channel_to_int=self._channel_to_int # Pass the mapping
        )
        return DataLoader(
            test_set, # Changed from dataset to test_set
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    @property
    def num_channels(self):
        """Returns the number of unique channels in the training data."""
        return self._num_channels

    @property
    def channel_to_int(self):
        """Returns the channel name to integer ID mapping."""
        return self._channel_to_int