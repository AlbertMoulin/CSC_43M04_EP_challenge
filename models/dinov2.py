import torch
import torch.nn as nn


class ImprovedHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Ajouter plusieurs couches cachées avec non-linéarités
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # Normalisation
            layers.append(nn.GELU())  # Non-linéarité GELU (meilleure que ReLU)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Couche de sortie finale
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)


class DinoV2Finetune(nn.Module):
    def __init__(self, frozen=False):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.dim = self.backbone.norm.normalized_shape[0]
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.regression_head = ImprovedHead(
            input_dim=self.dim,
            hidden_dims=[1024, 512, 256],
            output_dim=1,
            dropout_rate=0.2
        )

    def forward(self, x):
        x = self.backbone(x["image"])
        x = self.regression_head(x)
        return x
