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
    


class FeatureGate(nn.Module):
    #pour adapter dynamiquement l’importance de chaque feature
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.gate(x)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.act = nn.ReLU()
    def forward(self, x):
        out = self.linear(x)
        out = self.act(out)
        return out + x  #residual connection

class ImprovedMLPRegressor(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.feature_gate = FeatureGate(hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.feature_gate(x)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x)
        return self.output_layer(x).squeeze(1)