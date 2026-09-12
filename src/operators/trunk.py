import torch
import torch.nn as nn

class TrunkNetwork(nn.Module):
    """
    Trunk Net: Encodes query evaluation domain locations y (e.g., continuous time t)
    into latent operator domain basis functions t(y) in R^p.
    """
    def __init__(self, in_dim: int = 1, hidden_dim: int = 256, latent_dim: int = 128):
        super(TrunkNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y shape: (N, 1) or (B, N, 1) continuous domain query points
        returns: (N, p) or (B, N, p) latent trunk embeddings
        """
        return self.net(y)