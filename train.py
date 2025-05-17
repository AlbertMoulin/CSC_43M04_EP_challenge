import torch
import wandb
import hydra
from tqdm import tqdm

from utils.sanity import show_images

@hydra.main(config_path="configs", config_name="train")
def train(cfg):
    logger = (
        wandb.init(project="challenge_CSC_43M04_EP", name=cfg.experiment_name)
        if cfg.log
        else None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize datamodule first
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    # Get number of channels from the dataset
    train_dataset = datamodule.train_dataloader().dataset
    if hasattr(train_dataset, 'dataset'):  # Handle case of Subset wrapping
        num_channels = train_dataset.dataset.get_num_channels()
    else:
        num_channels = train_dataset.get_num_channels()
    
    # Update model config with actual number of channels
    if 'num_channels' in cfg.model.instance:
        cfg.model.instance.num_channels = num_channels
    
    # Now initialize the model with the updated config
    model = hydra.utils.instantiate(cfg.model.instance).to(device)
    
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    # Sanity checks
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
        
        # If you saved more than just model weights (like optimizer state)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
        else:
            # If you saved only model weights
            model.load_state_dict(checkpoint)

    best_val_loss = float('inf')
    best_epoch = -1
    
    # -- loop over epochs
    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="Epochs"):
        # -- Training loop
        model.train()
        epoch_train_loss = 0
        num_samples_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        
        for i, batch in enumerate(pbar):
            # Move all tensor data to device
            batch["image"] = batch["image"].to(device)
            batch["target"] = batch["target"].to(device).squeeze()
            
            # Forward pass
            preds = model(batch).squeeze()
            loss = loss_fn(preds, batch["target"])
            
            # Log step loss
            (
                logger.log({"loss": loss.detach().cpu().numpy()})
                if logger is not None
                else None
            )
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])
            num_samples_train += len(batch["image"])
            pbar.set_postfix({"train/loss_step": loss.detach().cpu().numpy()})
            
        # Calculate epoch loss
        epoch_train_loss /= num_samples_train
        
        # Log epoch metrics
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

        # -- Validation loop
        val_metrics = {}
        epoch_val_loss = 0
        num_samples_val = 0
        model.eval()
        
        if val_loader is not None:
            for _, batch in enumerate(val_loader):
                batch["image"] = batch["image"].to(device)
                batch["target"] = batch["target"].to(device).squeeze()
                
                with torch.no_grad():
                    preds = model(batch).squeeze()
                    
                loss = loss_fn(preds, batch["target"])
                epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                num_samples_val += len(batch["image"])
                
            epoch_val_loss /= num_samples_val
            val_metrics["val/loss_epoch"] = epoch_val_loss

            # Check for best model and save
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                
                # Path for best model
                best_model_path = cfg.checkpoint_path.replace('.pt', '_best.pt')
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'train_loss': epoch_train_loss,
                }, best_model_path)
                
                print(f"\n==> New best model saved (epoch {epoch}) with val_loss: {best_val_loss:.4f}")
            
            # Log validation metrics
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

    # Print final results
    print(
        f"""Epoch {epoch}: 
        Training metrics:
        - Train Loss: {epoch_train_loss:.4f},
        Validation metrics: 
        - Val Loss: {epoch_val_loss:.4f}

        Best model saved at epoch {best_epoch} with val_loss: {best_val_loss:.4f}
"""
    )

    # Finish logging
    if cfg.log:
        logger.finish()

    # Save final model
    torch.save({
        'epoch': cfg.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_train_loss,
        'val_loss': epoch_val_loss if val_loader is not None else None,
    }, cfg.checkpoint_path)


if __name__ == "__main__":
    train()