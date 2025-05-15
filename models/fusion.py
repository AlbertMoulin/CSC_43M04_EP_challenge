import torch
import torch.nn as nn

class ConcatFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_embed, text_embed):
        """
        Args:
            image_embed: Tensor (B, D)
            text_embed: Tensor (B, D)

        Returns:
            fused: Tensor (B, 2D)
        """
        return torch.cat([image_embed, text_embed], dim=1)