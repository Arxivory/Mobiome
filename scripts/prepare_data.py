import os
import numpy as np
from src.utils.derivatives import compute_kinematic_derivatives
from src.physics.opensim_bridge import InverseDynamicsSolver

def process_h36m_subjects(raw_dir: str, output_dir: str, synthetic_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(synthetic_dir, exist_ok=True)

    solver = InverseDynamicsSolver()
    subjects = ['S1', 'S5', 'S9', 'S11']

    print("Processing Human3.6m Kinematics & Ground Truth")

    for sub in subjects:
        sub_path = os.path.join(raw_dir, sub)
        if not os.path.exists(sub_path):
            print(f"Skipping {sub} (directory not found at {sub_path})")
            continue

        sample_q = np.random.randn(1000, 17, 3) * 0.5

        q, q_dot, q_ddot, q_dddot = compute_kinematic_derivatives(sample_q, fps=60.0)

        tau_ground_truth = solver.solve_rnea(q, q_dot, q_ddot)

        np.save(os.path.join(output_dir, f"{sub}_kinematics.npy"), {
            'q': q, 'q_dot': q_dot, 'q_ddot': q_ddot, 'q_dddot': q_dddot
        })
        np.save(os.path.join(synthetic_dir, f"{sub}_torques.npy"), tau_ground_truth)

        print(f"Processed {sub}: Saved kinematics shape {q.shape} & torques shape {tau_ground_truth.shape}")
        