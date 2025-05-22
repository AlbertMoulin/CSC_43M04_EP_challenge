import torch
import torch.nn.functional as F


def log_modality_weights(model, logger=None, epoch=None):
    """
    Log the learned modality weights to see how the model balances different inputs.
    
    Args:
        model: The multimodal model
        logger: wandb logger (optional)
        epoch: Current epoch (optional)
    """
    if hasattr(model, 'modality_weights'):
        # Get normalized weights
        weights = F.softmax(model.modality_weights, dim=0).detach().cpu().numpy()
        
        weight_info = {
            'image_weight': weights[0],
            'text_weight': weights[1], 
            'metadata_weight': weights[2]
        }
        
        print(f"Modality weights: Image={weights[0]:.3f}, Text={weights[1]:.3f}, Metadata={weights[2]:.3f}")
        
        if logger is not None:
            log_dict = {f"weights/{k}": v for k, v in weight_info.items()}
            if epoch is not None:
                log_dict['epoch'] = epoch
            logger.log(log_dict)
        
        return weight_info
    else:
        print("Model doesn't have modality_weights attribute")
        return None


def analyze_individual_predictions(model, batch, device):
    """
    Analyze individual modality contributions for debugging.
    
    Args:
        model: The multimodal model
        batch: A batch of data
        device: torch device
    
    Returns:
        Dictionary with individual predictions and weights
    """
    model.eval()
    with torch.no_grad():
        # Move batch to device
        batch_device = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch_device[key] = value.to(device)
            else:
                batch_device[key] = value
        
        # Get individual features
        image_features = model.image_backbone(batch_device["image"])
        text_features = model._encode_text(batch_device["text"])
        metadata_features = model._encode_metadata(batch_device)
        
        # Get individual predictions
        image_pred = model.image_head(image_features)
        text_pred = model.text_head(text_features)
        metadata_pred = model.metadata_head(metadata_features)
        
        # Get weights
        weights = F.softmax(model.modality_weights, dim=0)
        
        # Final prediction
        final_pred = (
            weights[0] * image_pred +
            weights[1] * text_pred +
            weights[2] * metadata_pred
        )
        
        return {
            'image_predictions': image_pred.cpu().numpy(),
            'text_predictions': text_pred.cpu().numpy(),
            'metadata_predictions': metadata_pred.cpu().numpy(),
            'final_predictions': final_pred.cpu().numpy(),
            'weights': weights.cpu().numpy()
        }