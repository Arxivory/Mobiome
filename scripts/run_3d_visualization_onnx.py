import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import onnxruntime as ort
import torch
import mediapipe as mp
from scipy.signal import savgol_filter

from src.physics.opensim_bridge import BiomechanicalInverseDynamics
from src.utils.derivatives import compute_kinematic_derivatives
from src.visualization.viewer_3d import Biomechanical3DViewer
from src.analytics.metrics import compute_biomechanical_metrics


def map_mediapipe_world_to_h36m(landmarks) -> np.ndarray:
    """
    Re-map MediaPipe world landmarks to the 17-joint Human3.6M topology.
    """
    lm = np.array([[-l.x, -l.y, l.z] for l in landmarks], dtype=np.float32)

    pelvis = (lm[23] + lm[24]) / 2.0
    neck = (lm[11] + lm[12]) / 2.0
    spine = (pelvis + neck) / 2.0
    head = lm[0]
    site = head + (head - neck) * 0.5

    h36m_kpts = np.array([
        pelvis,
        lm[24],
        lm[26],
        lm[28],
        lm[23],
        lm[25],
        lm[27],
        spine,
        neck,
        head,
        site,
        lm[11],
        lm[13],
        lm[15],
        lm[12],
        lm[14],
        lm[16],
    ])
    return h36m_kpts - pelvis


def analyze_video_and_benchmark_onnx(video_path: str = "sample_input.mp4", onnx_path: str = "checkpoints/deeponet_fp32.onnx"):
    """Run the same 3D visualization pipeline using the exported ONNX DeepONet model."""
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)

    norm_stats = torch.load("checkpoints/norm_stats.pt", map_location="cpu")
    u_mean = norm_stats["mean"]
    u_std = norm_stats["std"]

    dynamics_bridge = BiomechanicalInverseDynamics()
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
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

    p0 = keypoints_3d_list[0]
    print("=== COORDINATE SPACE SANITY CHECK ===")
    print(f"Pelvis (Root Joint 0): {p0[0]}")
    print(f"Head Top (Joint 10)  : {p0[10]}")
    print(f"R-Ankle  (Joint 3)   : {p0[3]}")
    print(f"L-Ankle  (Joint 6)   : {p0[6]}")
    print(f"R-Shoulder (Joint 14): {p0[14]}")
    print(f"L-Shoulder (Joint 11): {p0[11]}")
    print("=====================================")

    p_3d = np.asarray(keypoints_3d_list[:60], dtype=np.float32)
    if p_3d.shape != (60, 17, 3) or not np.isfinite(p_3d).all():
        raise ValueError(f"Unexpected world-landmark shape or values: {p_3d.shape}")
    p_3d = savgol_filter(p_3d, window_length=7, polyorder=2, axis=0)

    q = dynamics_bridge.cartesian_to_generalized_coordinates(p_3d)
    if q.shape[0] >= 7:
        q = savgol_filter(q, window_length=7, polyorder=2, axis=0)
    q, q_dot, q_ddot, _ = compute_kinematic_derivatives(q, fps=fps)

    tau_solver_gt = dynamics_bridge.solve_rnea(q, q_dot, q_ddot)

    u_input = np.concatenate([q, q_dot, q_ddot], axis=-1).flatten()
    u_tensor = torch.tensor(u_input, dtype=torch.float32).unsqueeze(0)
    u_norm = (u_tensor - u_mean) / u_std
    y_query = torch.linspace(0, 1, 60).unsqueeze(-1)

    ort_inputs = {
        "trajectory_sensors_u": u_norm.numpy().astype(np.float32),
        "query_locations_y": y_query.numpy().astype(np.float32),
    }
    tau_pred = session.run(None, ort_inputs)[0].squeeze(0)

    joint_names = ["R-Knee", "L-Knee", "R-Elbow", "L-Elbow"]
    print("\n=================== JOINT TORQUE BENCHMARK (REAL SOLVED VS ONNX) ===================")
    print(f"{'Joint Name':<12} | {'Real Solver Mean (N·m)':<23} | {'ONNX Mean (N·m)':<20} | {'RMSE (N·m)':<10} | {'Pearson (r)':<10}")
    print("-" * 82)

    metrics = compute_biomechanical_metrics(tau_pred, tau_solver_gt)
    for i, name in enumerate(joint_names):
        real_mean = np.mean(np.abs(tau_solver_gt[:, i]))
        pred_mean = np.mean(np.abs(tau_pred[:, i]))
        rmse_val = metrics["rmse"][i]
        corr_val = metrics["correlation"][i]
        print(f"{name:<12} | {real_mean:<23.4f} | {pred_mean:<20.4f} | {rmse_val:<10.4f} | {corr_val:<10.4f}")
    print("========================================================================================\n")

    viewer = Biomechanical3DViewer(p_3d, tau_pred, fps=fps)
    viewer.show()


if __name__ == "__main__":
    analyze_video_and_benchmark_onnx("sample_input.mp4", "checkpoints/deeponet_fp32.onnx")
