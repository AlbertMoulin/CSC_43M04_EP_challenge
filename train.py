import torch
import wandb
import hydra
from tqdm import tqdm
import os
import torch.nn as nn

from utils.sanity import show_images
from utils.custom_loss import MSLELoss

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
    
    # Pure MSLE loss for evaluation
    pure_msle_loss = MSLELoss()
    
    # Initialize learning rate scheduler if specified
    scheduler = None
    if hasattr(cfg, 'lr_scheduler'):
        scheduler = hydra.utils.instantiate(cfg.lr_scheduler, optimizer=optimizer)
    
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
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            # If you saved only model weights
            model.load_state_dict(checkpoint)

    best_val_loss = float('inf')
    best_epoch = -1
    best_val_msle = float('inf')  # Track best MSLE separately
    
    # Determine number of epochs for initial training vs fine-tuning
    total_epochs = cfg.epochs
    
    # Default to 80% initial training, 20% fine-tuning if fine-tuning is enabled
    fine_tuning_start = int(total_epochs * 0.8)
    
    # Check if fine-tuning configuration is specified
    fine_tuning_enabled = hasattr(cfg, 'fine_tuning') and cfg.fine_tuning.get('enabled', False)
    
    if fine_tuning_enabled:
        # Use explicitly configured fine-tuning point if provided
        if 'start_epoch' in cfg.fine_tuning:
            fine_tuning_start = cfg.fine_tuning.start_epoch
        
        print(f"Fine-tuning phase will start at epoch {fine_tuning_start}")
        
        # Initialize fine-tuning loss function if specified
        fine_tuning_loss_fn = None
        if 'loss_fn' in cfg.fine_tuning:
            fine_tuning_loss_fn = hydra.utils.instantiate(cfg.fine_tuning.loss_fn)
            print(f"Will switch to fine-tuning loss at epoch {fine_tuning_start}")
    
    # -- loop over epochs
    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="Epochs"):
        # Check if we should switch to fine-tuning phase
        if fine_tuning_enabled and epoch == fine_tuning_start:
            print("\n==> Switching to fine-tuning phase")
            
            # Switch to fine-tuning loss if specified
            if fine_tuning_loss_fn is not None:
                loss_fn = fine_tuning_loss_fn
                print("Switched to fine-tuning loss function")
            
            # Optionally adjust learning rate for fine-tuning
            if 'lr' in cfg.fine_tuning:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg.fine_tuning.lr
                print(f"Adjusted learning rate to {cfg.fine_tuning.lr}")
        
        # -- Training loop
        model.train()
        epoch_train_loss = 0
        epoch_train_msle = 0  # Track MSLE separately
        num_samples_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        
        for i, batch in enumerate(pbar):
            # Move all tensor data to device
            batch["image"] = batch["image"].to(device)
            batch["target"] = batch["target"].to(device).squeeze()
            
            # Forward pass
            preds = model(batch).squeeze()
            
            # Compute loss with L2 regularization if supported
            if hasattr(loss_fn, 'forward') and 'model' in loss_fn.forward.__code__.co_varnames:
                if hasattr(loss_fn, 'return_separate') and loss_fn.forward.__code__.co_varnames[-1] == 'return_separate':
                    combined_loss, _, msle_loss = loss_fn(preds, batch["target"], model, return_separate=True)
                    loss = combined_loss
                else:
                    loss = loss_fn(preds, batch["target"], model)
                    # Calculate MSLE separately for tracking
                    msle_loss = pure_msle_loss(preds, batch["target"])
            else:
                loss = loss_fn(preds, batch["target"])
                # Calculate MSLE separately for tracking
                msle_loss = pure_msle_loss(preds, batch["target"])
            
            # Log step loss
            if logger is not None:
                logger.log({
                    "train/loss_step": loss.detach().cpu().numpy(),
                    "train/msle_step": msle_loss.detach().cpu().numpy()
                })
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update learning rate if using a step-based scheduler
            if scheduler is not None and hasattr(scheduler, 'step_size'):
                scheduler.step()
            
            # Accumulate loss
            epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])
            epoch_train_msle += msle_loss.detach().cpu().numpy() * len(batch["image"])
            num_samples_train += len(batch["image"])
            
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                "loss": loss.detach().cpu().numpy(), 
                "msle": msle_loss.detach().cpu().numpy(),
                "lr": current_lr
            })
            
        # Calculate epoch loss
        epoch_train_loss /= num_samples_train
        epoch_train_msle /= num_samples_train
        
        # Update learning rate if using an epoch-based scheduler
        if scheduler is not None and not hasattr(scheduler, 'step_size'):
            scheduler.step()
        
        # Log epoch metrics
        if logger is not None:
            logger.log({
                "epoch": epoch,
                "train/loss_epoch": epoch_train_loss,
                "train/msle_epoch": epoch_train_msle,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        # -- Validation loop
        val_metrics = {}
        epoch_val_loss = 0
        epoch_val_msle = 0  # Track MSLE separately
        num_samples_val = 0
        model.eval()
        
        if val_loader is not None:
            for _, batch in enumerate(val_loader):
                batch["image"] = batch["image"].to(device)
                batch["target"] = batch["target"].to(device).squeeze()
                
                with torch.no_grad():
                    preds = model(batch).squeeze()
                    
                    # Calculate main loss
                    if hasattr(loss_fn, 'return_separate'):
                        _, _, msle_loss = loss_fn(preds, batch["target"], return_separate=True)
                    else:
                        # Use the main loss function
                        loss = loss_fn(preds, batch["target"])
                        # Always calculate MSLE for competition metric
                        msle_loss = pure_msle_loss(preds, batch["target"])
                
                epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                epoch_val_msle += msle_loss.detach().cpu().numpy() * len(batch["image"])
                num_samples_val += len(batch["image"])
                
            epoch_val_loss /= num_samples_val
            epoch_val_msle /= num_samples_val
            
            val_metrics["val/loss_epoch"] = epoch_val_loss
            val_metrics["val/msle_epoch"] = epoch_val_msle

            # Check for best model based on validation MSLE (competition metric)
            if epoch_val_msle < best_val_msle:
                best_val_msle = epoch_val_msle
                best_epoch = epoch
                
                # Path for best MSLE model
                best_msle_path = cfg.checkpoint_path.replace('.pt', '_best_msle.pt')
                
                # Save best MSLE model
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': epoch_val_loss,
                    'val_msle': best_val_msle,
                    'train_loss': epoch_train_loss,
                    'train_msle': epoch_train_msle,
                }
                
                # Add scheduler state if it exists
                if scheduler is not None:
                    save_dict['scheduler_state_dict'] = scheduler.state_dict()
                    
                torch.save(save_dict, best_msle_path)
                
                print(f"\n==> New best MSLE model saved (epoch {epoch}) with val_msle: {best_val_msle:.4f}")
            
            # Also keep track of best model by validation loss
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                
                # Path for best loss model
                best_loss_path = cfg.checkpoint_path.replace('.pt', '_best_loss.pt')
                
                # Save best loss model
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_msle': epoch_val_msle,
                    'train_loss': epoch_train_loss,
                    'train_msle': epoch_train_msle,
                }
                
                # Add scheduler state if it exists
                if scheduler is not None:
                    save_dict['scheduler_state_dict'] = scheduler.state_dict()
                    
                torch.save(save_dict, best_loss_path)
                
                print(f"\n==> New best loss model saved (epoch {epoch}) with val_loss: {best_val_loss:.4f}")
            
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
        - Train MSLE: {epoch_train_msle:.4f}
        Validation metrics: 
        - Val Loss: {epoch_val_loss:.4f}
        - Val MSLE: {epoch_val_msle:.4f}

        Best model saved at epoch {best_epoch} with val_msle: {best_val_msle:.4f}
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
        'train_msle': epoch_train_msle,
        'val_loss': epoch_val_loss if val_loader is not None else None,
        'val_msle': epoch_val_msle if val_loader is not None else None,
    }
    
    # Add scheduler state if it exists
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
        
    torch.save(save_dict, cfg.checkpoint_path)


if __name__ == "__main__":
    train()