import argparse
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import pyvista as pv
import torch
from scipy.signal import savgol_filter

from src.utils.derivatives import compute_kinematic_derivatives


SKELETON_BONES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16),
]
JOINT_NAMES = ["R-Knee", "L-Knee", "R-Elbow", "L-Elbow"]
WINDOW_SIZE = 60


def map_mediapipe_world_to_h36m(landmarks) -> np.ndarray:
    lm = np.array([[-landmark.x, -landmark.y, landmark.z] for landmark in landmarks], dtype=np.float32)
    pelvis = (lm[23] + lm[24]) / 2.0
    neck = (lm[11] + lm[12]) / 2.0
    spine = (pelvis + neck) / 2.0
    head = lm[0]
    head_top = head + (head - neck) * 0.5

    h36m = np.array([
        pelvis,
        lm[24], lm[26], lm[28],
        lm[23], lm[25], lm[27],
        spine, neck, head, head_top,
        lm[11], lm[13], lm[15],
        lm[12], lm[14], lm[16],
    ])
    return h36m - pelvis


def make_session(model_path: Path, provider: str, device_id: int) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if provider == "cuda" and "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable. Install a compatible onnxruntime-gpu "
            f"build; available providers: {available}"
        )

    use_cuda = provider == "cuda" or (provider == "auto" and "CUDAExecutionProvider" in available)
    providers = (
        [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        if use_cuda else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(model_path), providers=providers)
    print(f"ONNX Runtime providers: {session.get_providers()}")
    if use_cuda:
        print("CUDA is preferred; unsupported INT8 operators can still fall back to CPU.")
    else:
        print("Running ONNX inference on CPU.")
    return session


def set_overlay_text(actor, text: str) -> None:
    if hasattr(actor, "SetInput"):
        actor.SetInput(text)
    else:
        actor.SetText(2, text)


def build_viewer(frame_width: int, frame_height: int):
    plotter = pv.Plotter(shape=(1, 2), window_size=(1280, 720), title="Live INT8 Pose and Torque")
    plotter.set_background("#171d21")

    plotter.subplot(0, 0)
    image = pv.ImageData(dimensions=(frame_width, frame_height, 1))
    initial_rgb = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    image.point_data["rgb"] = np.flipud(initial_rgb).reshape(-1, 3)
    plotter.add_mesh(image, scalars="rgb", rgb=True, lighting=False)
    plotter.add_text("Camera", position="upper_left", font_size=12, color="white")
    plotter.camera_position = "xy"
    plotter.camera.parallel_projection = True
    plotter.reset_camera()

    plotter.subplot(0, 1)
    initial_points = np.zeros((17, 3), dtype=np.float32)
    skeleton = pv.PolyData(initial_points)
    skeleton.lines = np.asarray(
        [value for parent, child in SKELETON_BONES for value in (2, parent, child)], dtype=np.int64
    )
    skeleton_actor = plotter.add_mesh(skeleton, color="#35c6c8", line_width=4)
    joint_cloud = pv.PolyData(initial_points)
    plotter.add_points(joint_cloud, color="#f0a34a", point_size=12, render_points_as_spheres=True)
    torque_label = plotter.add_text("Waiting for pose...", position="upper_left", font_size=11, color="white")
    plotter.add_text("3D Pose / INT8 Torque", position="lower_left", font_size=12, color="white")
    plotter.camera_position = [(0, 0.6, -2.5), (0, 0, 0), (0, 1, 0)]
    plotter.add_axes()
    return plotter, image, skeleton, joint_cloud, torque_label


def run_realtime(args) -> None:
    model_path = Path(args.onnx)
    stats_path = Path(args.norm_stats)
    if not model_path.exists():
        raise FileNotFoundError(f"INT8 ONNX model not found: {model_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization statistics not found: {stats_path}")

    session = make_session(model_path, args.provider, args.device_id)
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    u_mean = stats["mean"].cpu().numpy().astype(np.float32)
    u_std = stats["std"].cpu().numpy().astype(np.float32)
    y_query = np.linspace(0.0, 1.0, WINDOW_SIZE, dtype=np.float32).reshape(-1, 1)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = args.fps or capture.get(cv2.CAP_PROP_FPS) or 30.0
    frames = deque(maxlen=WINDOW_SIZE)
    running = [True]
    viewer = None
    pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=args.pose_complexity,
    )
    previous_time = time.perf_counter()

    try:
        while running[0]:
            ret, frame = capture.read()
            if not ret:
                raise RuntimeError("Camera stopped returning frames.")

            if viewer is None:
                viewer = build_viewer(width, height)
                plotter, image, skeleton, joint_cloud, torque_label = viewer
                plotter.add_key_event("q", lambda: running.__setitem__(0, False))
                plotter.show(interactive_update=True, auto_close=False)

            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_world_landmarks:
                frames.append(map_mediapipe_world_to_h36m(result.pose_world_landmarks.landmark))
            elif frames:
                frames.append(frames[-1].copy())

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.point_data["rgb"] = np.flipud(image_rgb).reshape(-1, 3)

            if len(frames) == WINDOW_SIZE:
                p_3d = np.asarray(frames, dtype=np.float32)
                p_3d = savgol_filter(p_3d, window_length=7, polyorder=2, axis=0)
                q = np.empty((WINDOW_SIZE, 4), dtype=np.float32)
                joint_chains = ((2, 1, 3), (5, 4, 6), (15, 14, 16), (12, 11, 13))
                for index, (joint, parent, child) in enumerate(joint_chains):
                    first = p_3d[:, parent] - p_3d[:, joint]
                    second = p_3d[:, child] - p_3d[:, joint]
                    denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
                    cosine = np.divide(
                        np.sum(first * second, axis=1),
                        denominator,
                        out=np.zeros_like(denominator),
                        where=denominator > 1e-6,
                    )
                    q[:, index] = np.arccos(np.clip(cosine, -1.0, 1.0))

                q = savgol_filter(q, window_length=7, polyorder=2, axis=0)
                q, q_dot, q_ddot, _ = compute_kinematic_derivatives(q, fps=fps)
                u = np.concatenate((q, q_dot, q_ddot), axis=-1).reshape(1, -1)
                u_norm = (u - u_mean) / np.maximum(u_std, 1e-8)
                prediction = session.run(None, {
                    "trajectory_sensors_u": u_norm.astype(np.float32),
                    "query_locations_y": y_query,
                })[0].squeeze(0)
                if prediction.shape != (WINDOW_SIZE, len(JOINT_NAMES)):
                    raise ValueError(f"Unexpected model output shape: {prediction.shape}")

                current_pose = p_3d[-1]
                skeleton.points = current_pose
                joint_cloud.points = current_pose
                torque = prediction[-1]
                set_overlay_text(torque_label, "\n".join(
                    f"{name}: {value:+.2f} N.m" for name, value in zip(JOINT_NAMES, torque)
                ))

            elapsed = time.perf_counter() - previous_time
            previous_time = time.perf_counter()
            plotter.update(stime=1, force_redraw=True)
            if args.verbose and elapsed > 0:
                print(f"Loop: {1.0 / elapsed:.1f} FPS | pose window: {len(frames)}/{WINDOW_SIZE}", end="\r")
    finally:
        capture.release()
        pose.close()
        if viewer is not None:
            viewer[0].close()


def parse_args():
    parser = argparse.ArgumentParser(description="Live webcam pose and INT8 ONNX torque inference.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--onnx", default="checkpoints/deeponet_int8.onnx", help="INT8 ONNX model path.")
    parser.add_argument("--norm-stats", default="checkpoints/norm_stats.pt", help="Normalization stats path.")
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device index.")
    parser.add_argument("--fps", type=float, default=None, help="Override camera FPS used for derivatives.")
    parser.add_argument("--pose-complexity", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--verbose", action="store_true", help="Print loop FPS and pose window fill.")
    return parser.parse_args()


if __name__ == "__main__":
    run_realtime(parse_args())