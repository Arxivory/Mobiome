import torch
import torch.nn as nn

class BiomechanicalPoseLoss(nn.Module):
    """
    Combines spatial Euclidean distance (MPJPE), rigid-body bone length 
    conservation, and high-order temporal jerk penalties.
    """
    def __init__(self, bone_pairs: list, lambda_bone: float = 0.5, lambda_smooth: float = 0.1, fps: float = 60.0):
        super(BiomechanicalPoseLoss, self).__init__()
        self.bone_pairs = bone_pairs
        self.lambda_bone = lambda_bone
        self.lambda_smooth = lambda_smooth
        self.dt = 1.0 / fps
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_p3d: torch.Tensor, gt_p3d: torch.Tensor) -> torch.Tensor:
        """
        pred_p3d: (B, T, K, 3) predicted 3D Cartesian coordinates
        gt_p3d:   (B, T, K, 3) ground truth 3D Cartesian coordinates
        """
        # 1. Mean Per Joint Position Error (L_MPJPE)
        l_mpjpe = torch.mean(torch.norm(pred_p3d - gt_p3d, dim=-1))

        # 2. Rigid-Body Bone Length Conservation Loss (L_bone)
        l_bone = self._compute_bone_length_loss(pred_p3d, gt_p3d)

        # 3. Temporal Jerk Smoothness Loss (L_smooth)
        l_smooth = self._compute_jerk_loss(pred_p3d)

        # Total Weighted Loss
        total_loss = l_mpjpe + (self.lambda_bone * l_bone) + (self.lambda_smooth * l_smooth)
        return total_loss, l_mpjpe, l_bone, l_smooth

    def _compute_bone_length_loss(self, pred_p3d: torch.Tensor, gt_p3d: torch.Tensor) -> torch.Tensor:
        """Penalizes bone stretching/shrinking relative to reference skeleton."""
        pred_lengths = []
        gt_lengths = []

        for parent, child in self.bone_pairs:
            p_len = torch.norm(pred_p3d[:, :, parent, :] - pred_p3d[:, :, child, :], dim=-1)
            g_len = torch.norm(gt_p3d[:, :, parent, :] - gt_p3d[:, :, child, :], dim=-1)
            pred_lengths.append(p_len)
            gt_lengths.append(g_len)

        pred_bones = torch.stack(pred_lengths, dim=-1)
        gt_bones = torch.stack(gt_lengths, dim=-1)

        return self.mse_loss(pred_bones, gt_bones)

    def _compute_jerk_loss(self, pred_p3d: torch.Tensor) -> torch.Tensor:
        """Penalizes high-frequency temporal acceleration changes (jerk)."""
        if pred_p3d.shape[1] < 4:
            return torch.tensor(0.0, device=pred_p3d.device)

        # 3rd-order finite difference approximation along time dimension (dim=1)
        jerk = (pred_p3d[:, 3:, :, :] - 3 * pred_p3d[:, 2:-1, :, :] + 
                3 * pred_p3d[:, 1:-2, :, :] - pred_p3d[:, :-3, :, :]) / (self.dt ** 3)

        return torch.mean(torch.norm(jerk, dim=-1))