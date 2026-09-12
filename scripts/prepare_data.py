import os
import glob
import numpy as np
from parser import H36MKeypointLoader
from src.utils.derivatives import compute_kinematic_derivatives
from src.physics.opensim_bridge import BiomechanicalInverseDynamics

def process_h36m_subjects(raw_dir: str, output_dir: str, synthetic_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(synthetic_dir, exist_ok=True)

    loader = H36MKeypointLoader()
    dynamics_bridge = BiomechanicalInverseDynamics()
    subjects = ['S1', 'S5', 'S9', 'S11']

    print("--- Processing Human3.6M Kinematics & True Inverse Dynamics ---")

    for sub in subjects:
        sub_path = os.path.join(raw_dir, sub)
        if not os.path.exists(sub_path):
            print(f"Skipping {sub} (directory not found at {sub_path})")
            continue

        file_candidates = glob.glob(os.path.join(sub_path, "**/*.h5"), recursive=True)

        for file_path in file_candidates:
            # 1. Load Cartesian 3D Keypoints (T, 17, 3)
            p_3d = loader.load_from_h5(file_path)

            # 2. Convert 3D Cartesian coordinates to Generalized Angles q (T, D)
            q = dynamics_bridge.cartesian_to_generalized_coordinates(p_3d)

            # 3. Compute Derivatives over Generalized Coordinate Trajectories
            q, q_dot, q_ddot, q_dddot = compute_kinematic_derivatives(q, fps=60.0)

            # 4. Compute True Generalized Joint Torques τ (T, D) in N·m
            tau_ground_truth = dynamics_bridge.solve_rnea(q, q_dot, q_ddot)

            # Save processed files
            activity_name = os.path.splitext(os.path.basename(file_path))[0]
            np.save(
                os.path.join(output_dir, f"{sub}_{activity_name}_kinematics.npy"),
                {
                    "p_3d": p_3d,          # Cartesian 3D Pose
                    "q": q,                # Generalized Angles (Radians)
                    "q_dot": q_dot,        # Angular Velocities (rad/s)
                    "q_ddot": q_ddot,      # Angular Accelerations (rad/s²)
                    "q_dddot": q_dddot     # Angular Jerk (rad/s³)
                },
            )
            np.save(
                os.path.join(synthetic_dir, f"{sub}_{activity_name}_torques.npy"),
                tau_ground_truth,
            )

            print(f"Processed {activity_name}: Angles shape {q.shape}, Torques shape {tau_ground_truth.shape} (N·m)")

if __name__ == "__main__":
    process_h36m_subjects("data/raw", "data/processed", "data/synthetic")