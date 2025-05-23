import hydra
from torch.utils.data import DataLoader
import pandas as pd
import torch

from data.enhanced_dataset import EnhancedDataset  # Use enhanced dataset


@hydra.main(config_path="configs", config_name="train")
def create_submission(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize datamodule to get unique channels (needed for channel embeddings)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    # Create test loader using enhanced dataset
    test_loader = DataLoader(
        EnhancedDataset(
            cfg.datamodule.dataset_path,
            "test",
            transforms=hydra.utils.instantiate(cfg.datamodule.test_transform),
            metadata=cfg.datamodule.metadata,
        ),
        batch_size=cfg.datamodule.batch_size,
        shuffle=False,
        num_workers=cfg.datamodule.num_workers,
    )
    
    # Load model
    model = hydra.utils.instantiate(cfg.model.instance)
    
    # Initialize channel embeddings if needed (before loading checkpoint)
    if hasattr(model, 'initialize_channel_embedding'):
        unique_channels = datamodule.get_unique_channels()
        model.initialize_channel_embedding(unique_channels)
        print(f"Initialized channel embeddings for {len(unique_channels)} channels")
    
    # Move model to device
    model = model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded successfully")

    # Create submission.csv
    submission = pd.DataFrame(columns=["ID", "views"])

    print("Generating predictions...")
    for i, batch in enumerate(test_loader):
        batch["image"] = batch["image"].to(device)
        
        with torch.no_grad():
            preds = model(batch).squeeze().cpu().numpy()
        
        # Handle single prediction case
        if preds.ndim == 0:
            preds = [preds.item()]
        
        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"ID": batch["id"], "views": preds}),
            ],
            ignore_index=True
        )
        
        if (i + 1) % 10 == 0:
            print(f"Processed {(i + 1) * cfg.datamodule.batch_size} samples")
    
    # Save submission
    submission_path = f"{cfg.root_dir}/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Total predictions: {len(submission)}")


if __name__ == "__main__":
    create_submission()