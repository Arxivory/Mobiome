import torch
import torch.nn as nn

class BranchNetwork(nn.Module):
    """
    Branch Net: Encodes input trajectory sensors u(t) = [q(t), q_dot(t), q_ddot(t)]
    into latent operator basis coefficients b(u) in R^p.
    """
    def __init__(self, in_features: int, hidden_dim: int = 256, latent_dim: int = 128):
        super(BranchNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u shape: (B, T * D * 3) or (B, input_sensor_dim)
        returns: (B, p) latent branch embeddings
        """
        return self.net(u)