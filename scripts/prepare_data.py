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

    