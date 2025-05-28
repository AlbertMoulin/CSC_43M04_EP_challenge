import hydra
from torch.utils.data import DataLoader
import pandas as pd
import torch
import numpy as np

from data.dataset import Dataset


@hydra.main(config_path="configs", config_name="train")
def create_submission(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize datamodule to get unique channels (needed for channel embeddings)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    test_loader = DataLoader(
        Dataset(
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

    # - Load checkpoint
    checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    print("Model loaded")
    model.eval()

    # - Create submission.csv
    submission = pd.DataFrame(columns=["ID", "views"])

    for i, batch in enumerate(test_loader):
        batch["image"] = batch["image"].to(device)
        batch["date"] = batch["date"].to(device)
        batch["vectorized_text"] = batch["vectorized_text"].to(device)
        batch["year_norm"] = batch["year_norm"].to(device)
        batch["is_train_major_channel"]= batch["is_train_major_channel"].to(device)

        with torch.no_grad():
            preds_log = model(batch)
            preds = np.expm1(preds_log.detach().cpu().numpy())   # Revenir à l’échelle originale
        submission = pd.concat(
            [
                submission,
                pd.DataFrame({"ID": batch["id"], "views": preds}),
            ]
        )
    submission.to_csv(f"{cfg.root_dir}/submission.csv", index=False)
    print("created")


if __name__ == "__main__":
    create_submission()
