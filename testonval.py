import hydra
from torch.utils.data import DataLoader
import pandas as pd
import torch

from data.dataset import Dataset

torch.cuda.empty_cache()  # Clear cache to avoid memory issues


@hydra.main(config_path="configs", config_name="train")
def create_submission(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    # - Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    checkpoint = torch.load(r'checkpoints/SIMPLE_MULTIMODAL_2025-05-27_21-40-46_best_val_loss.pt')
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    model.eval()
    print("Model loaded")

    # - Create submission.csv
    submission = pd.DataFrame(columns=["ID", "views"])
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch["image"] = batch["image"].to(device)
            
            preds = model(batch).squeeze().cpu().numpy()
            submission = pd.concat(
                [
                    submission,
                    pd.DataFrame({"ID": batch["id"], "views": preds}),
                ]
            )
    submission.to_csv(f"{cfg.root_dir}/submission.csv", index=False)


if __name__ == "__main__":
    create_submission()