import cv2
import torch
import numpy as np
import os
import mediapipe as mp
from scipy.signal import savgol_filter
from src.vision.backbone import PoseTrajectoryEncoder
from src.operators.deeponet import DeepONetOperator
from src.physics.opensim_bridge import BiomechanicalInverseDynamics
from src.utils.derivatives import compute_kinematic_derivatives

def run_video_inference(video_path: str, vision_ckpt: str, operator_ckpt: str, output_path: str = "output_demo.mp4"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vision_model = PoseTrajectoryEncoder(num_joints=17).to(device)
    if os.path.exists(vision_ckpt):
        vision_model.load_state_dict(torch.load(vision_ckpt, map_location=device))
    vision_model.eval()

    operator_model = DeepONetOperator(sensor_count=720, num_outputs=4).to(device)
    operator_model.load_state_dict(torch.load(operator_ckpt, map_location=device))
    operator_model.eval()

    # Load Normalization Stats
    norm_stats = torch.load("checkpoints/norm_stats.pt", map_location=device)
    u_mean = norm_stats["mean"].to(device)
    u_std = norm_stats["std"].to(device)

    dynamics_bridge = BiomechanicalInverseDynamics()

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    keypoints_2d_buffer = []
    torque_history = []  # Buffer for temporal torque smoothing

    print("--- Running Video Inference with Normalized Input & Exponential Smoothing ---")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            lm = results.pose_landmarks.landmark
            h36m_mp_indices = [0, 11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28, 1, 2, 3, 4]
            kpts_2d = np.array([[lm[i].x, lm[i].y] for i in h36m_mp_indices]).flatten()
            keypoints_2d_buffer.append(kpts_2d)
        else:
            keypoints_2d_buffer.append(np.zeros(17 * 2))

        if len(keypoints_2d_buffer) >= 60:
            seq_2d = torch.tensor(np.array(keypoints_2d_buffer[-60:]), dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                pred_p3d = vision_model(seq_2d).cpu().numpy().squeeze(0)

            q = dynamics_bridge.cartesian_to_generalized_coordinates(pred_p3d)
            
            if q.shape[0] >= 7:
                q = savgol_filter(q, window_length=7, polyorder=2, axis=0)

            q, q_dot, q_ddot, _ = compute_kinematic_derivatives(q, fps=fps)

            u_input = np.concatenate([q, q_dot, q_ddot], axis=-1).flatten()
            u_tensor = torch.tensor(u_input, dtype=torch.float32).unsqueeze(0).to(device)
            
            u_norm = (u_tensor - u_mean) / u_std
            y_query = torch.linspace(0, 1, 60).unsqueeze(-1).to(device)

            with torch.no_grad():
                pred_tau = operator_model(u_norm, y_query).cpu().numpy().squeeze(0)

            raw_tau = pred_tau[-1]

            torque_history.append(raw_tau)
            if len(torque_history) > 5:
                torque_history.pop(0)
            smooth_tau = np.mean(torque_history, axis=0)

            cv2.rectangle(frame, (20, 15), (320, 150), (0, 0, 0), -1)
            cv2.putText(frame, f"R-Knee Torque:  {smooth_tau[0]:.2f} N m", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"L-Knee Torque:  {smooth_tau[1]:.2f} N m", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"R-Elbow Torque: {smooth_tau[2]:.2f} N m", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"L-Elbow Torque: {smooth_tau[3]:.2f} N m", (30, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print("Inference completed successfully.")

if __name__ == "__main__":
    run_video_inference("sample_input.mp4", "checkpoints/vision_fp32.pth", "checkpoints/deeponet_fp32.pth")