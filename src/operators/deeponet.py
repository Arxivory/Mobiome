import torch
import torch.nn as nn
from src.operators.branch import BranchNetwork
from src.operators.trunk import TrunkNetwork

class DeepONetOperator(nn.Module):
    """
    Unstacked DeepONet for Learning Biomechanical Operators:
    Maps continuous trajectories u(t) to joint reaction torque functions tau(t).
    """
    def __init__(self, sensor_count: int, num_outputs: int = 4, hidden_dim: int = 256, latent_dim: int = 128):
        super(DeepONetOperator, self).__init__()
        
        self.num_outputs = num_outputs  # Number of joint torques (D=4)
        self.latent_dim = latent_dim
        
        # Instantiate Branch and Trunk for each target output dimension (or shared)
        self.branch = BranchNetwork(in_features=sensor_count, hidden_dim=hidden_dim, latent_dim=latent_dim * num_outputs)
        self.trunk = TrunkNetwork(in_dim=1, hidden_dim=hidden_dim, latent_dim=latent_dim * num_outputs)
        
        # Trainable bias per joint output
        self.bias = nn.Parameter(torch.zeros(num_outputs))

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        u: (B, Sensor_Dim) Input trajectory function discretization
        y: (B, N, 1) or (N, 1) Evaluation time points
        returns: (B, N, D) Predicted continuous torque profiles tau(t)
        """
        B = u.shape[0]
        
        # 1. Branch Embedding: (B, D * p)
        b_out = self.branch(u).view(B, self.num_outputs, self.latent_dim)
        
        # 2. Trunk Embedding: (B, N, D * p) if query points differ per batch sample
        if y.ndim == 3:
            N = y.shape[1]
            t_out = self.trunk(y.view(B * N, 1)).view(B, N, self.num_outputs, self.latent_dim)
            # Dot product over latent dimension p: sum(b * t) -> (B, N, D)
            tau_pred = torch.einsum('bdp,bndp->bnd', b_out, t_out) + self.bias
        else:
            N = y.shape[0]
            t_out = self.trunk(y).view(N, self.num_outputs, self.latent_dim)
            # Dot product over latent dimension p: sum(b * t) -> (B, N, D)
            tau_pred = torch.einsum('bdp,ndp->bnd', b_out, t_out) + self.bias

        return tau_pred