import torch
import torch.nn as nn

class MLPRegressor(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Tensor (B, input_dim)

        Returns:
            y: Tensor (B,) — nombre de vues prédites
        """
        return self.net(x).squeeze(1)