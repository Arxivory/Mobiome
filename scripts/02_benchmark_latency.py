import os
import sys
import time
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.physics.opensim_bridge import BiomechanicalInverseDynamics

def benchmark_solvers_and_onnx(
    fp32_onnx_path: str = "checkpoints/deeponet_fp32.onnx",
    int8_onnx_path: str = "checkpoints/deeponet_int8.onnx",
    output_json: str = "paper_results/latency_benchmark_data.json",
    num_runs: int = 100
):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # 1. Classical OpenSim / RNEA Latency
    bridge = BiomechanicalInverseDynamics()
    q = np.random.randn(60, 4)
    q_dot = np.random.randn(60, 4)
    q_ddot = np.random.randn(60, 4)

    rnea_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = bridge.solve_rnea(q, q_dot, q_ddot)
        rnea_times.append((time.perf_counter() - t0) * 1000.0)

    # 2. ONNX Inference Sessions
    u_dummy = np.random.randn(1, 720).astype(np.float32)
    y_dummy = np.linspace(0, 1, 60).reshape(60, 1).astype(np.float32)

    onnx_fp32_times = []
    if os.path.exists(fp32_onnx_path):
        sess_fp32 = ort.InferenceSession(fp32_onnx_path, providers=["CPUExecutionProvider"])
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = sess_fp32.run(None, {"trajectory_sensors_u": u_dummy, "query_locations_y": y_dummy})
            onnx_fp32_times.append((time.perf_counter() - t0) * 1000.0)

    onnx_int8_times = []
    if os.path.exists(int8_onnx_path):
        sess_int8 = ort.InferenceSession(int8_onnx_path, providers=["CPUExecutionProvider"])
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = sess_int8.run(None, {"trajectory_sensors_u": u_dummy, "query_locations_y": y_dummy})
            onnx_int8_times.append((time.perf_counter() - t0) * 1000.0)

    bench_data = {
        "OpenSim_RNEA": {"mean_ms": float(np.mean(rnea_times)), "std_ms": float(np.std(rnea_times)), "fps": float(1000.0 / np.mean(rnea_times))},
        "DeepONet_FP32": {"mean_ms": float(np.mean(onnx_fp32_times)) if onnx_fp32_times else 0.0, "fps": float(1000.0 / np.mean(onnx_fp32_times)) if onnx_fp32_times else 0.0},
        "DeepONet_INT8": {"mean_ms": float(np.mean(onnx_int8_times)) if onnx_int8_times else 0.0, "fps": float(1000.0 / np.mean(onnx_int8_times)) if onnx_int8_times else 0.0}
    }

    with open(output_json, "w") as f:
        json.dump(bench_data, f, indent=4)
    print(f"[+] Benchmark performance data written to: {output_json}")

if __name__ == "__main__":
    benchmark_solvers_and_onnx()