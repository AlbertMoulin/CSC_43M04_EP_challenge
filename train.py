import torch
import wandb
import hydra
from tqdm import tqdm
import os

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
    
    # Get data loaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    
    # Get number of channels from the dataset
    train_dataset = train_loader.dataset
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
    
    # Initialize learning rate scheduler if specified
    scheduler = None
    if hasattr(cfg, 'lr_scheduler'):
        scheduler_config = cfg.lr_scheduler
        
        # For OneCycleLR we need to calculate total steps
        if hasattr(scheduler_config, '_target_') and 'OneCycleLR' in scheduler_config._target_:
            # Calculate total steps as (epochs * batches per epoch)
            steps_per_epoch = len(train_loader) 
            total_steps = cfg.epochs * steps_per_epoch
            print(f"Calculated {total_steps} total steps for OneCycleLR ({steps_per_epoch} steps per epoch × {cfg.epochs} epochs)")
            scheduler_config.total_steps = total_steps
            
        scheduler = hydra.utils.instantiate(scheduler_config, optimizer=optimizer)
    
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
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
            
            # Compute loss
            loss = loss_fn(preds, batch["target"])
            
            # Log step loss
            if logger is not None:
                logger.log({"train/loss_step": loss.detach().cpu().numpy()})
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update learning rate if using OneCycleLR (updates each step)
            if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                # For OneCycleLR, only step if we haven't reached total steps
                # This prevents the "Tried to step X times" error
                if (epoch * len(train_loader) + i) < scheduler.total_steps:
                    scheduler.step()
            
            # Accumulate loss
            epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])
            num_samples_train += len(batch["image"])
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                "loss": loss.detach().cpu().numpy(),
                "lr": current_lr
            })
            
        # Calculate epoch loss
        epoch_train_loss /= num_samples_train
        
        # Update learning rate if using epoch-based scheduler (not OneCycleLR)
        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # Log epoch metrics
        if logger is not None:
            logger.log({
                "epoch": epoch,
                "train/loss_epoch": epoch_train_loss,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

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

            # Check for best model based on validation loss
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                
                # Path for best model
                best_model_path = cfg.checkpoint_path.replace('.pt', '_best.pt')
                
                # Save best model
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'train_loss': epoch_train_loss,
                }
                
                # Add scheduler state if it exists
                if scheduler is not None:
                    save_dict['scheduler_state_dict'] = scheduler.state_dict()
                    
                torch.save(save_dict, best_model_path)
                
                print(f"\n==> New best model saved (epoch {epoch}) with val_loss: {best_val_loss:.4f}")
            
            # Log validation metrics
            if logger is not None:
                logger.log({
                    "epoch": epoch,
                    **val_metrics,
                })

    # Print final results
    print(
        f"""Epoch {epoch}: 
        Training metrics:
        - Train Loss: {epoch_train_loss:.4f}
        Validation metrics: 
        - Val Loss: {epoch_val_loss:.4f}

        Best model saved at epoch {best_epoch} with val_loss: {best_val_loss:.4f}
"""
    )

    # Finish logging
    if cfg.log:
        logger.finish()

    # Save final model
    save_dict = {
        'epoch': cfg.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': epoch_train_loss,
        'val_loss': epoch_val_loss if val_loader is not None else None,
    }
    
    # Add scheduler state if it exists
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
        
    torch.save(save_dict, cfg.checkpoint_path)


if __name__ == "__main__":
    train()