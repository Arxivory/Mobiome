import torch
from src.vision.backbone import PoseTrajectoryEncoder
from src.vision.losses import BiomechanicalPoseLoss

def run_phase2_validation():
    # Standard H36M 17-joint skeleton bone connections
    h36m_bone_pairs = [
        (0, 1), (1, 2), (2, 3),    # Right Leg
        (0, 4), (4, 5), (5, 6),    # Left Leg
        (0, 7), (7, 8), (8, 9),    # Spine/Head
        (8, 11), (11, 12), (12, 13), # Left Arm
        (8, 14), (14, 15), (15, 16)  # Right Arm
    ]

    model = PoseTrajectoryEncoder(num_joints=17)
    criterion = BiomechanicalPoseLoss(bone_pairs=h36m_bone_pairs, lambda_bone=0.5, lambda_smooth=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Simulated batch: 2 batches, 30 frames, 17 joints
    dummy_2d = torch.randn(2, 30, 17 * 2)
    dummy_3d_gt = torch.randn(2, 30, 17, 3)

    print("--- Phase 2: Testing Biomechanical Loss & Vision Backbone ---")
    
    optimizer.zero_grad()
    pred_3d = model(dummy_2d)
    total_loss, l_mpjpe, l_bone, l_smooth = criterion(pred_3d, dummy_3d_gt)
    
    total_loss.backward()
    optimizer.step()

    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"  ├─ L_MPJPE:  {l_mpjpe.item():.4f}")
    print(f"  ├─ L_bone:   {l_bone.item():.4f}")
    print(f"  └─ L_smooth: {l_smooth.item():.4f}")

if __name__ == "__main__":
    run_phase2_validation()