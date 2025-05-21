import hydra
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.serialization
import omegaconf.listconfig
import typing
import os

from data.dataset import Dataset

@hydra.main(config_path="configs", config_name="train")
def create_submission(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create test dataset
    test_dataset = Dataset(
        cfg.datamodule.dataset_path,
        "test",
        transforms=hydra.utils.instantiate(cfg.datamodule.test_transform),
        metadata=cfg.datamodule.metadata,
    )
    
    # Update model config with number of channels from dataset
    if 'num_channels' in cfg.model.instance:
        cfg.model.instance.num_channels = test_dataset.get_num_channels()
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.datamodule.batch_size,
        shuffle=False,
        num_workers=cfg.datamodule.num_workers,
    )
    
    # Load model with updated config
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    
    # Get checkpoint path
    if hasattr(cfg, 'checkpoint_path') and os.path.exists(cfg.checkpoint_path):
        checkpoint_path = cfg.checkpoint_path
    else:
        # If checkpoint_path doesn't exist, look for best model
        checkpoint_path = cfg.checkpoint_path.replace(".pt", "_best.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Could not find checkpoint at {checkpoint_path}")
    
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        print(f"Loaded checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation loss: {checkpoint.get('val_loss', 'unknown')}")
    else:
        model_state_dict = checkpoint
    
    model.load_state_dict(model_state_dict)
    print("Model loaded successfully")

    # Create submission.csv
    submission = pd.DataFrame(columns=["ID", "views"])

    # Generate predictions
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # Move image to device (other tensors will be moved in model's forward)
            batch["image"] = batch["image"].to(device)
            
            preds = model(batch).squeeze().cpu().numpy()
            
            submission = pd.concat(
                [
                    submission,
                    pd.DataFrame({"ID": batch["id"], "views": preds}),
                ]
            )
        
    # Save submission file
    submission.to_csv(f"{cfg.root_dir}/submission.csv", index=False)
    print(f"Submission file created at: {cfg.root_dir}/submission.csv")


if __name__ == "__main__":
    create_submission()