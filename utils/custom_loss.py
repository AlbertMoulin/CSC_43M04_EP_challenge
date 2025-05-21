import torch
from torch import nn

class MSLELoss(nn.Module):
    """
    Mean Squared Logarithmic Error loss.
    
    This is the competition metric and handles well the large variations
    in view counts typical of YouTube videos.
    """
    def __init__(self):
        super(MSLELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Ensure the predictions and targets are non-negative
        y_pred = torch.clamp(y_pred, min=0)
        y_true = torch.clamp(y_true, min=0)

        # Compute the MSLE
        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)
        loss = torch.mean((log_pred - log_true) ** 2)

        return loss

class HuberMSLELoss(nn.Module):
    """
    Combined loss function that uses Huber loss with L2 regularization for training
    while also computing MSLE for validation and competition metric tracking.
    
    Parameters:
    -----------
    delta : float
        Threshold for Huber loss.
    l2_lambda : float
        L2 regularization coefficient.
    alpha : float
        Weight for blending Huber and MSLE losses (0-1).
        0 = pure Huber, 1 = pure MSLE.
    """
    def __init__(self, delta=1.0, l2_lambda=0.01, alpha=0.0):
        super(HuberMSLELoss, self).__init__()
        self.delta = delta
        self.l2_lambda = l2_lambda
        self.alpha = alpha
        self.huber = nn.HuberLoss(delta=delta, reduction='mean')
        self.msle = MSLELoss()
        
    def forward(self, y_pred, y_true, model=None, return_separate=False):
        """
        Compute the combined loss.
        
        Parameters:
        -----------
        y_pred : torch.Tensor
            Predicted values
        y_true : torch.Tensor
            Target values
        model : nn.Module, optional
            Model for which to compute L2 regularization.
        return_separate : bool
            If True, returns (combined_loss, huber_loss, msle_loss)
        """
        # Compute Huber loss
        huber_loss = self.huber(y_pred, y_true)
        
        # Compute MSLE loss
        msle_loss = self.msle(y_pred, y_true)
        
        # Add L2 regularization if model is provided
        l2_reg = 0.0
        if model is not None and self.l2_lambda > 0:
            for param in model.parameters():
                if param.requires_grad:
                    l2_reg += torch.sum(param ** 2)
        
        # Combine losses with weighting
        combined_loss = (1 - self.alpha) * huber_loss + self.alpha * msle_loss
        
        # Add L2 regularization
        if model is not None and self.l2_lambda > 0:
            combined_loss += self.l2_lambda * l2_reg
        
        if return_separate:
            return combined_loss, huber_loss, msle_loss
        
        return combined_loss

    def get_msle(self, y_pred, y_true):
        """Compute just the MSLE component for evaluation."""
        return self.msle(y_pred, y_true)