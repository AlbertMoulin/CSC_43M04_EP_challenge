import torch
import wandb
import hydra
from tqdm import tqdm

from utils.sanity import show_images
from omegaconf import DictConfig, OmegaConf # Import DictConfig

@hydra.main(config_path="configs", config_name="train")
def train(cfg: DictConfig): # Add type hint for cfg
    logger = (
        wandb.init(project="challenge_CSC_43M04_EP", name=cfg.experiment_name)
        if cfg.log
        else None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Instantiate datamodule FIRST to calculate num_channels
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # 2. Get the calculated num_channels value
    num_channels = datamodule.num_channels
    print(f"Number of channels obtained from DataModule: {num_channels}")

    # 3. Manually update the model's configuration with the actual num_channels value
    # This overwrites the interpolation in combined_model.yaml before instantiation
    # Ensure the path matches the structure in combined_model.yaml under 'instance'
    if 'model' in cfg and 'instance' in cfg.model:
         cfg.model.instance.num_channels = num_channels
         print(f"Updated model config with num_channels: {cfg.model.instance.num_channels}")
    else:
         raise KeyError("Model instance configuration not found in config.")


    # 4. Now instantiate the model using the updated configuration
    model = hydra.utils.instantiate(cfg.model.instance).to(device)

    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # --- Rest of your training loop remains the same ---
    train_sanity = show_images(train_loader, name="assets/sanity/train_images")
    (
        logger.log({"sanity_checks/train_images": wandb.Image(train_sanity)})
        if logger is not None
        else None
    )
    if val_loader is not None:
        val_sanity = show_images(val_loader, name="assets/sanity/val_images")
        logger.log(
            {"sanity_checks/val_images": wandb.Image(val_sanity)}
        ) if logger is not None else None

    start_epoch = 0
    if cfg.resume_from_checkpoint:
        print(f"Loading checkpoint from {cfg.resume_from_checkpoint}")
        checkpoint = torch.load(cfg.resume_from_checkpoint)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
        else:
            model.load_state_dict(checkpoint)


    best_val_loss = float('inf')
    best_epoch = -1
    for epoch in tqdm(range(start_epoch,cfg.epochs), desc="Epochs"):
        model.train()
        epoch_train_loss = 0
        num_samples_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for i, batch in enumerate(pbar):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            if "target" in batch:
                 batch["target"] = batch["target"].squeeze()

            preds = model(batch).squeeze()
            loss = loss_fn(preds, batch["target"])
            (
                logger.log({"loss": loss.detach().cpu().numpy()})
                if logger is not None
                else None
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.detach().cpu().numpy() * (len(batch["image"]) if "image" in batch else 1)
            num_samples_train += (len(batch["image"]) if "image" in batch else 1)
            pbar.set_postfix({"train/loss_step": loss.detach().cpu().numpy()})
        epoch_train_loss /= num_samples_train
        (
            logger.log(
                {
                    "epoch": epoch,
                    "train/loss_epoch": epoch_train_loss,
                }
            )
            if logger is not None
            else None
        )

        val_metrics = {}
        epoch_val_loss = 0
        num_samples_val = 0
        model.eval()
        if val_loader is not None:
            for _, batch in enumerate(val_loader):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                if "target" in batch:
                    batch["target"] = batch["target"].squeeze()

                with torch.no_grad():
                    preds = model(batch).squeeze()
                loss = loss_fn(preds, batch["target"])
                epoch_val_loss += loss.detach().cpu().numpy() * (len(batch["image"]) if "image" in batch else 1)
                num_samples_val += (len(batch["image"]) if "image" in batch else 1)
            epoch_val_loss /= num_samples_val
            val_metrics["val/loss_epoch"] = epoch_val_loss

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch

                best_model_path = cfg.checkpoint_path.replace('.pt', '_best.pt')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'train_loss': epoch_train_loss,
                }, best_model_path)

                print(f"\n==> Nouveau meilleur modèle sauvegardé (epoch {epoch}) avec val_loss: {best_val_loss:.4f}")


            (
                logger.log(
                    {
                        "epoch": epoch,
                        **val_metrics,
                    }
                )
                if logger is not None
                else None
            )

    print(
        f"""Epoch {epoch}:
        Training metrics:
        - Train Loss: {epoch_train_loss:.4f},
        Validation metrics:
        - Val Loss: {epoch_val_loss:.4f}

        Best model saved at epoch {best_epoch} with val_loss: {best_val_loss:.4f}
"""
    )

    if cfg.log:
        logger.finish()

    final_val_loss = epoch_val_loss if val_loader is not None else None

    torch.save({
        'epoch': cfg.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_train_loss,
        'val_loss': final_val_loss,
    }, cfg.checkpoint_path)


if __name__ == "__main__":
    train()