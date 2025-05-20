import hydra
from torch.utils.data import DataLoader
import pandas as pd
import torch
import torch.serialization
import omegaconf.listconfig
import typing

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
    
    # Load checkpoint
    checkpoint = torch.load(r'checkpoints/EnhancedCombinedModel_2025-05-16_best.pt', weights_only=False)
    model_state_dict = checkpoint['model_state_dict']
    print(f"Loading model from checkpoint")
    model.load_state_dict(model_state_dict)
    print("Model loaded")

    # Create submission.csv
    submission = pd.DataFrame(columns=["ID", "views"])

    # Generate predictions
    model.eval()
    for i, batch in enumerate(test_loader):
        # Move image to device (other tensors will be moved in model's forward)
        batch["image"] = batch["image"].to(device)
        
        with torch.no_grad():
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