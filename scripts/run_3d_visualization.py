import cv2
import torch
import numpy as np
import mediapipe as mp
from scipy.signal import savgol_filter
from src.operators.deeponet import DeepONetOperator
from src.physics.opensim_bridge import BiomechanicalInverseDynamics
from src.utils.derivatives import compute_kinematic_derivatives
from src.visualization.viewer_3d import Biomechanical3DViewer
from src.analytics.metrics import compute_biomechanical_metrics

def map_mediapipe_world_to_h36m(landmarks) -> np.ndarray:
    """
    Re-map MediaPipe world landmarks to the 17-joint Human3.6M topology.

    MediaPipe world landmarks are already 3D coordinates in meters. Keep that
    metric scale and center the pose at the pelvis so the physics and viewer
    receive a coherent body-centered skeleton.
    """
    lm = np.array([[l.x, l.y, l.z] for l in landmarks], dtype=np.float32)

    # MediaPipe Indices:
    # 0: Nose, 11: L_Shoulder, 12: R_Shoulder, 13: L_Elbow, 14: R_Elbow,
    # 15: L_Wrist, 16: R_Wrist, 23: L_Hip, 24: R_Hip, 25: L_Knee, 26: R_Knee, 27: L_Ankle, 28: R_Ankle

    pelvis = (lm[23] + lm[24]) / 2.0
    neck = (lm[11] + lm[12]) / 2.0
    spine = (pelvis + neck) / 2.0
    head = lm[0]
    site = head + (head - neck) * 0.5

    # Reconstruct exact 17-joint Human3.6M array order
    h36m_kpts = np.array([
        pelvis,     # 0: Pelvis
        lm[24],     # 1: R_Hip
        lm[26],     # 2: R_Knee
        lm[28],     # 3: R_Ankle
        lm[23],     # 4: L_Hip
        lm[25],     # 5: L_Knee
        lm[27],     # 6: L_Ankle
        spine,      # 7: Spine
        neck,       # 8: Neck
        head,       # 9: Head
        site,       # 10: Head Top (Site)
        lm[11],     # 11: L_Shoulder
        lm[13],     # 12: L_Elbow
        lm[15],     # 13: L_Wrist
        lm[12],     # 14: R_Shoulder
        lm[14],     # 15: R_Elbow
        lm[16]      # 16: R_Wrist
    ])

    return h36m_kpts - pelvis

def analyze_video_and_benchmark(video_path: str = "sample_input.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    operator_model = DeepONetOperator(sensor_count=720, num_outputs=4).to(device)
    operator_model.load_state_dict(torch.load("checkpoints/deeponet_fp32.pth", map_location=device))
    operator_model.eval()

    norm_stats = torch.load("checkpoints/norm_stats.pt", map_location=device)
    u_mean, u_std = norm_stats["mean"].to(device), norm_stats["std"].to(device)

    dynamics_bridge = BiomechanicalInverseDynamics()

    # Process Video Frames
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    keypoints_3d_list = []

    print(f"--- Extracting MediaPipe world landmarks for {video_path} ---")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_world_landmarks:
            kpts_h36m = map_mediapipe_world_to_h36m(res.pose_world_landmarks.landmark)
            keypoints_3d_list.append(kpts_h36m)

    cap.release()

    if len(keypoints_3d_list) < 60:
        raise ValueError("Video must contain at least 60 valid pose frames.")

    p_3d = np.asarray(keypoints_3d_list[:60], dtype=np.float32)
    if p_3d.shape != (60, 17, 3) or not np.isfinite(p_3d).all():
        raise ValueError(f"Unexpected world-landmark shape or values: {p_3d.shape}")
    p_3d = savgol_filter(p_3d, window_length=7, polyorder=2, axis=0)

    # Dynamics & Kinematics
    q = dynamics_bridge.cartesian_to_generalized_coordinates(p_3d)
    if q.shape[0] >= 7:
        q = savgol_filter(q, window_length=7, polyorder=2, axis=0)
    q, q_dot, q_ddot, _ = compute_kinematic_derivatives(q, fps=fps)

    # RNEA Solver Ground Truth
    tau_solver_gt = dynamics_bridge.solve_rnea(q, q_dot, q_ddot)

    # DeepONet Inference
    u_input = np.concatenate([q, q_dot, q_ddot], axis=-1).flatten()
    u_tensor = torch.tensor(u_input, dtype=torch.float32).unsqueeze(0).to(device)
    u_norm = (u_tensor - u_mean) / u_std
    y_query = torch.linspace(0, 1, 60).unsqueeze(-1).to(device)

    with torch.no_grad():
        tau_pred = operator_model(u_norm, y_query).cpu().numpy().squeeze(0)

    # Display Metrics Table
    joint_names = ["R-Knee", "L-Knee", "R-Elbow", "L-Elbow"]
    print("\n=================== JOINT TORQUE BENCHMARK (REAL SOLVED VS PREDICTED) ===================")
    print(f"{'Joint Name':<12} | {'Real Solver Mean (N·m)':<23} | {'Predicted Mean (N·m)':<22} | {'RMSE (N·m)':<10} | {'Pearson (r)':<10}")
    print("-" * 82)

    metrics = compute_biomechanical_metrics(tau_pred, tau_solver_gt)
    for i, name in enumerate(joint_names):
        real_mean = np.mean(np.abs(tau_solver_gt[:, i]))
        pred_mean = np.mean(np.abs(tau_pred[:, i]))
        rmse_val = metrics['rmse'][i]
        corr_val = metrics['correlation'][i]
        print(f"{name:<12} | {real_mean:<23.4f} | {pred_mean:<22.4f} | {rmse_val:<10.4f} | {corr_val:<10.4f}")
    print("=========================================================================================\n")

    # Launch PyVista Visualizer
    viewer = Biomechanical3DViewer(p_3d, tau_pred, fps=fps)
    viewer.show()

if __name__ == "__main__":
    analyze_video_and_benchmark("sample_input.mp4")