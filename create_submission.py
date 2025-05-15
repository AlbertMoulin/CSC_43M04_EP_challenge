import hydra
from torch.utils.data import DataLoader
import pandas as pd
import torch

from data.dataset import Dataset


@hydra.main(config_path="configs", config_name="train")
def create_submission(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # - Load model and checkpoint
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    encoder = hydra.utils.instantiate(cfg.encoder.instance).to(device)
    fusion = hydra.utils.instantiate(cfg.fusion.instance)
    checkpoint = torch.load(cfg.checkpoint_path)
    print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
    model.load_state_dict(checkpoint)
    print("Model loaded")

    # - Create submission.csv
    submission = pd.DataFrame(columns=["ID", "views"])

    for i, batch in enumerate(test_loader):
        batch["image"] = batch["image"].to(device)
        batch["input_ids"] = batch["input_ids"].to(device)
        batch["attention_mask"] = batch["attention_mask"].to(device)

        image_embed,tex_embed = encoder(batch["image"],batch["text"])
        fused = fusion(image_embed,tex_embed)
        with torch.no_grad():
            preds = model(fused).squeeze().cpu().numpy()
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
