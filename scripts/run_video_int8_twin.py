import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import mediapipe as mp
import numpy as np
import pyvista as pv
import torch
from scipy.signal import savgol_filter

from scripts.run_realtime_int8_camera import (
    JOINT_NAMES,
    SKELETON_BONES,
    WINDOW_SIZE,
    make_session,
    map_mediapipe_world_to_h36m,
    set_overlay_text,
)
from src.utils.derivatives import compute_kinematic_derivatives


JOINT_CHAINS = ((2, 1, 3), (5, 4, 6), (15, 14, 16), (12, 11, 13))


def extract_pose_track(video_path: Path, pose_complexity: int):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    pose_track = []
    valid_frames = []
    pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=pose_complexity,
    )

    try:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.pose_world_landmarks:
                pose_track.append(map_mediapipe_world_to_h36m(result.pose_world_landmarks.landmark))
                valid_frames.append(frame_index)
            else:
                pose_track.append(None)
            frame_index += 1
    finally:
        pose.close()

    if not pose_track:
        capture.release()
        raise ValueError("The input video contains no readable frames.")
    if not valid_frames:
        capture.release()
        raise ValueError("MediaPipe could not detect a pose in any video frame.")
    if len(pose_track) < WINDOW_SIZE:
        capture.release()
        raise ValueError(f"Video must contain at least {WINDOW_SIZE} frames.")

    track = np.full((len(pose_track), 17, 3), np.nan, dtype=np.float32)
    for index, points in enumerate(pose_track):
        if points is not None:
            track[index] = points
    frame_indices = np.arange(len(track))
    for joint in range(track.shape[1]):
        for coordinate in range(track.shape[2]):
            values = track[:, joint, coordinate]
            track[:, joint, coordinate] = np.interp(
                frame_indices, valid_frames, values[valid_frames]
            )

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return capture, track, fps, width, height


def joint_angles(pose_track: np.ndarray) -> np.ndarray:
    angles = np.empty((pose_track.shape[0], len(JOINT_CHAINS)), dtype=np.float32)
    for index, (joint, parent, child) in enumerate(JOINT_CHAINS):
        first = pose_track[:, parent] - pose_track[:, joint]
        second = pose_track[:, child] - pose_track[:, joint]
        denominator = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        cosine = np.divide(
            np.sum(first * second, axis=1),
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 1e-6,
        )
        angles[:, index] = np.arccos(np.clip(cosine, -1.0, 1.0))
    return angles


def infer_torque_track(pose_track, fps, session, u_mean, u_std, verbose=False):
    frame_count = len(pose_track)
    torques = np.zeros((frame_count, len(JOINT_NAMES)), dtype=np.float32)
    y_query = np.linspace(0.0, 1.0, WINDOW_SIZE, dtype=np.float32).reshape(-1, 1)

    for end_frame in range(WINDOW_SIZE - 1, frame_count):
        window = pose_track[end_frame - WINDOW_SIZE + 1:end_frame + 1]
        window = savgol_filter(window, window_length=7, polyorder=2, axis=0)
        q = joint_angles(window)
        q = savgol_filter(q, window_length=7, polyorder=2, axis=0)
        q, q_dot, q_ddot, _ = compute_kinematic_derivatives(q, fps=fps)
        sensor_input = np.concatenate((q, q_dot, q_ddot), axis=-1).reshape(1, -1)
        sensor_input = (sensor_input - u_mean) / np.maximum(u_std, 1e-8)
        prediction = session.run(None, {
            "trajectory_sensors_u": sensor_input.astype(np.float32),
            "query_locations_y": y_query,
        })[0].squeeze(0)
        if prediction.shape != (WINDOW_SIZE, len(JOINT_NAMES)):
            raise ValueError(f"Unexpected model output shape: {prediction.shape}")
        torques[end_frame] = prediction[-1]
        if verbose and (end_frame + 1) % 100 == 0:
            print(f"Prepared frame {end_frame + 1}/{frame_count}")

    return torques


def create_viewer(capture, pose_track, torques, fps, width, height):
    frame_count = len(pose_track)
    plotter = pv.Plotter(shape=(1, 2), window_size=(1440, 820), title="MP4 INT8 Pose and 3D Twin")
    plotter.set_background("#171d21")

    plotter.subplot(0, 0)
    initial_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    image = pv.ImageData(dimensions=(width, height, 1))
    image.point_data["rgb"] = np.flipud(initial_rgb).reshape(-1, 3)
    plotter.add_mesh(image, scalars="rgb", rgb=True, lighting=False)
    plotter.add_text("MP4 Video", position="upper_left", font_size=12, color="white")
    plotter.camera_position = "xy"
    plotter.camera.parallel_projection = True
    plotter.reset_camera()

    plotter.subplot(0, 1)
    initial_points = pose_track[0]
    skeleton = pv.PolyData(initial_points.copy())
    skeleton.lines = np.asarray(
        [value for parent, child in SKELETON_BONES for value in (2, parent, child)],
        dtype=np.int64,
    )
    plotter.add_mesh(skeleton, color="#35c6c8", line_width=4)
    joints = pv.PolyData(initial_points.copy())
    plotter.add_points(joints, color="#f0a34a", point_size=12, render_points_as_spheres=True)
    torque_label = plotter.add_text("Waiting for 60-frame window", position="upper_left", font_size=11, color="white")
    frame_label = plotter.add_text("", position="upper_right", font_size=10, color="white")
    plotter.add_text("3D Digital Twin", position="lower_left", font_size=12, color="white")
    plotter.camera_position = [(0, 0.6, -2.5), (0, 0, 0), (0, 1, 0)]
    plotter.add_axes()

    state = {"frame": 0, "playing": False}

    def display_frame(frame_index):
        frame_index = int(np.clip(frame_index, 0, frame_count - 1))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.point_data["rgb"] = np.flipud(rgb).reshape(-1, 3)
        points = pose_track[frame_index]
        skeleton.points = points
        joints.points = points
        state["frame"] = frame_index
        timestamp = frame_index / fps
        set_overlay_text(frame_label, f"Frame {frame_index + 1}/{frame_count}  |  {timestamp:0.2f}s")
        if frame_index >= WINDOW_SIZE - 1:
            set_overlay_text(torque_label, "\n".join(
                f"{name}: {value:+.2f} N.m"
                for name, value in zip(JOINT_NAMES, torques[frame_index])
            ))
        else:
            set_overlay_text(torque_label, "Torque available after first 60 frames")
        plotter.render()

    plotter.subplot(0, 1)
    slider = plotter.add_slider_widget(
        callback=lambda value: display_frame(round(value)),
        rng=(0, frame_count - 1),
        value=0,
        title="Video frame",
        pointa=(0.12, 0.08),
        pointb=(0.88, 0.08),
        interaction_event="always",
        fmt="%0.0f",
    )

    def toggle_playback():
        state["playing"] = not state["playing"]
        if state["playing"] and state["frame"] >= frame_count - 1:
            state["frame"] = 0
            slider.GetRepresentation().SetValue(0)
            display_frame(0)

    def advance_frame(_step):
        if not state["playing"]:
            return
        next_frame = state["frame"] + 1
        if next_frame >= frame_count:
            state["playing"] = False
            return
        slider.GetRepresentation().SetValue(next_frame)
        display_frame(next_frame)

    plotter.add_key_event("space", toggle_playback)
    plotter.add_key_event("Right", lambda: display_frame(state["frame"] + 1))
    plotter.add_key_event("Left", lambda: display_frame(state["frame"] - 1))
    plotter.subplot(0, 0)
    plotter.add_text(
        "Space: play/pause   Left/Right: step",
        position="lower_left",
        font_size=10,
        color="white",
    )
    plotter.add_timer_event(
        max_steps=2_000_000_000,
        duration=max(1, round(1000 / fps)),
        callback=advance_frame,
    )

    display_frame(0)
    plotter.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play an MP4 beside its synchronized 3D pose and INT8 torque twin."
    )
    parser.add_argument("video", type=Path, default=Path("sample_input.mp4"), help="Path to the input MP4 video file.")
    parser.add_argument("--onnx", type=Path, default=Path("checkpoints/deeponet_int8.onnx"))
    parser.add_argument("--norm-stats", type=Path, default=Path("checkpoints/norm_stats.pt"))
    parser.add_argument("--provider", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--pose-complexity", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run_video(args):
    video_path = args.video
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not args.onnx.exists():
        raise FileNotFoundError(f"INT8 ONNX model not found: {args.onnx}")
    if not args.norm_stats.exists():
        raise FileNotFoundError(f"Normalization statistics not found: {args.norm_stats}")

    capture, pose_track, fps, width, height = extract_pose_track(video_path, args.pose_complexity)
    try:
        print(f"Extracted poses from {len(pose_track)} frames at {fps:.2f} FPS.")
        session = make_session(args.onnx, args.provider, args.device_id)
        stats = torch.load(args.norm_stats, map_location="cpu", weights_only=False)
        u_mean = stats["mean"].cpu().numpy().astype(np.float32)
        u_std = stats["std"].cpu().numpy().astype(np.float32)
        torques = infer_torque_track(pose_track, fps, session, u_mean, u_std, args.verbose)
        create_viewer(capture, pose_track, torques, fps, width, height)
    finally:
        capture.release()


if __name__ == "__main__":
    run_video(parse_args())