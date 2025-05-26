import torch
import wandb
import hydra
from tqdm import tqdm
import numpy as np

from utils.sanity import show_images
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@hydra.main(config_path="configs", config_name="train")
def train(cfg):
    config = {"model": cfg.model.name, "metadata":cfg.dataset.metadata, "epochs":cfg.epochs, "batch_size":cfg.dataset.batch_size, "lr":cfg.optim.lr, "dropout":cfg.model.instance.dropout_rate, "hidden_dim":cfg.model.instance.final_mlp_hidden_dim, "validation_split":cfg.dataset.val_split, "validation_set_type":cfg.dataset.validation_set_type, "img_proportion":cfg.model.instance.img_proportion, "text_proportion":cfg.model.instance.text_proportion, 
              "date_proportion":cfg.model.instance.date_proportion, "channel_proportion":cfg.model.instance.channel_proportion, "year_proportion":cfg.model.instance.year_proportion}
    logger = (
        wandb.init(project="challenge_CSC_43M04_EP", name=cfg.experiment_name,config=config)
        if cfg.log
        else None
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize datamodule first to get unique channels
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Initialize model
    model = hydra.utils.instantiate(cfg.model.instance)
    
    # Initialize channel embeddings before moving to device
    if hasattr(model, 'initialize_channel_embedding'):
        unique_channels = datamodule.get_unique_channels()
        model.initialize_channel_embedding(unique_channels)
        print(f"Initialized channel embeddings for {len(unique_channels)} channels")
    
    model = model.to(device)

    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

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

    if logger is not None:
        residuals_table = wandb.Table(columns=["true_value", "residual", "epoch"])

    # -- loop over epochs
    for epoch in tqdm(range(cfg.epochs), desc="Epochs"):
        # -- loop over training batches
        model.train()
        epoch_train_loss = 0
        num_samples_train = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        
        for i, batch in enumerate(pbar):
            batch["image"] = batch["image"].to(device)
            batch["log_target"] = batch["log_target"].to(device).squeeze()
            batch["date"] = batch["date"].to(device)
            batch["vectorized_text"] = batch["vectorized_text"].to(device)
            batch["year_norm"] = batch["year_norm"].to(device)

            preds_log = model(batch)
            loss = loss_fn(preds_log, batch["log_target"])
            
            (
                logger.log({"loss": loss.detach().cpu().numpy()})
                if logger is not None
                else None
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.detach().cpu().numpy() * len(batch["image"])
            num_samples_train += len(batch["image"])
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

        # -- validation loop
        val_metrics = {}
        epoch_val_loss = 0
        num_samples_val = 0
        model.eval()
        
        if val_loader is not None: 
            y_true = [] 
            y_pred = []    
            for _, batch in enumerate(val_loader):
                batch["image"] = batch["image"].to(device)
                batch["log_target"] = batch["log_target"].to(device).squeeze()
                batch["date"] = batch["date"].to(device)
                batch["vectorized_text"] = batch["vectorized_text"].to(device)
                batch["year_norm"] = batch["year_norm"].to(device)

                with torch.no_grad():
                    preds_log = model(batch)
                    preds = np.expm1(preds_log.detach().cpu().numpy())  # Revenir à l’échelle originale
                loss = loss_fn(preds_log, batch["log_target"])
                y_pred.append(preds)      # pour histogramme
                y_true.append(np.expm1(batch["log_target"].detach().cpu().numpy()))
            
                epoch_val_loss += loss.detach().cpu().numpy() * len(batch["image"])
                num_samples_val += len(batch["image"])
                
            epoch_val_loss /= num_samples_val
            val_metrics["val/loss_epoch"] = epoch_val_loss
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
            if logger is not None:
                y_pred = np.concatenate(y_pred)
                y_true = np.concatenate(y_true)
                data_true = [[v] for v in y_true if v <= 200000]
                data_pred = [[v] for v in y_pred]
                table_pred = wandb.Table(data=data_pred, columns=["pred_views"])
                table_true = wandb.Table(data=data_true, columns=["true_views"])

                errors = y_pred - y_true
                # Log the residuals
                for t, r in zip(y_true, errors):
                    residuals_table.add_data(t, r, epoch)
                
                
                # Histogrammes 
                logger.log({
                    "hist_pred_views": wandb.plot.histogram(table_pred, "pred_views", title="Distribution des vues prédites")
                })
                logger.log({
                    "hist_true_views": wandb.plot.histogram(table_true, "true_views", title="Distribution des vues réelles")
                })
                logger.log({
                    "residuals": wandb.plot.scatter(
                        residuals_table, "true_value", "residual", title="Résidu des prédictions"
                    )
                })
    

    print(
        f"""Epoch {epoch}: 
        Training metrics:
        - Train Loss: {epoch_train_loss:.4f},
        Validation metrics: 
        - Val Loss: {epoch_val_loss:.4f}"""
    )

    if cfg.log:
        logger.finish()

    torch.save(model.state_dict(), cfg.checkpoint_path)


if __name__ == "__main__":
    train()
