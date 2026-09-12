import torch
import torch.nn as nn

class PoseTrajectoryEncoder(nn.Module):
    """
    Lightweight MLP/Temporal Convolution Network mapping 2D keypoint/features 
    into smooth 3D Cartesian coordinates p_3d (T, K, 3).
    """
    def __init__(self, num_joints: int = 17, in_features: int = 2, hidden_dim: int = 256):
        super(PoseTrajectoryEncoder, self).__init__()
        self.num_joints = num_joints
        
        self.net = nn.Sequential(
            nn.Linear(num_joints * in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_joints * 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, K * 2) 2D Keypoint sequences
        returns: (B, T, K, 3) 3D keypoint predictions
        """
        B, T, _ = x.shape
        x_flat = x.view(B * T, -1)
        out_flat = self.net(x_flat)
        out_3d = out_flat.view(B, T, self.num_joints, 3)
        return out_3d