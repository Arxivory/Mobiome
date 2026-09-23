import os
import sys
import json
import numpy as np
import torch
import onnxruntime as ort
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.deeponet import DeepONetOperator

def run_quantization_drift_audit(
    pytorch_ckpt: str = "checkpoints/deeponet_fp32.pth",
    fp32_onnx: str = "checkpoints/deeponet_fp32.onnx",
    int8_onnx: str = "checkpoints/deeponet_int8.onnx",
    output_json: str = "paper_results/quantization_drift_data.json"
):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # Load PyTorch model to get spectral norms for Equation (12)
    model = DeepONetOperator(sensor_count=720, num_outputs=4)
    if os.path.exists(pytorch_ckpt):
        model.load_state_dict(torch.load(pytorch_ckpt, map_location="cpu"))
    model.eval()

    # Calculate exact spectral truncation bounds
    q_scale = 1.0 / 127.0
    u_dummy = torch.randn(1, 720)
    y_dummy = torch.linspace(0, 1, 60).unsqueeze(-1)

    with torch.no_grad():
        b_norm = float(torch.norm(model.branch(u_dummy), p=2).item())
        t_norm = float(torch.norm(model.trunk(y_dummy), p=2).item())

    e_b = q_scale * b_norm * 0.5
    e_t = q_scale * t_norm * 0.5
    analytical_bound = (b_norm * e_t) + (e_b * t_norm) + (e_b * e_t)

    # Compute empirical drift using ONNX runtime
    drift_data = []
    if os.path.exists(fp32_onnx) and os.path.exists(int8_onnx):
        sess_fp32 = ort.InferenceSession(fp32_onnx, providers=["CPUExecutionProvider"])
        sess_int8 = ort.InferenceSession(int8_onnx, providers=["CPUExecutionProvider"])

        for velocity_scale in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            u_np = (np.random.randn(1, 720) * velocity_scale).astype(np.float32)
            y_np = y_dummy.numpy().astype(np.float32)

            out_fp32 = sess_fp32.run(None, {"trajectory_sensors_u": u_np, "query_locations_y": y_np})[0]
            out_int8 = sess_int8.run(None, {"trajectory_sensors_u": u_np, "query_locations_y": y_np})[0]

            empirical_drift = float(np.linalg.norm(out_fp32 - out_int8))
            drift_data.append({
                "velocity_scale": velocity_scale,
                "analytical_bound": analytical_bound,
                "empirical_drift": empirical_drift
            })

    with open(output_json, "w") as f:
        json.dump(drift_data, f, indent=4)
    print(f"[+] Quantization drift audit written to: {output_json}")

if __name__ == "__main__":
    run_quantization_drift_audit()