import numpy as np
import torch
import onnxruntime as ort

from src.operators.deeponet import DeepONetOperator


def test_onnx_vs_pytorch(
    pytorch_ckpt: str = "checkpoints/deeponet_fp32.pth",
    onnx_path: str = "checkpoints/deeponet_fp32.onnx",
    sensor_dim: int = 720,
    seq_len: int = 60,
    tolerance: float = 1e-3,
):
    device = torch.device("cpu")

    # 1) Load reference PyTorch model
    model = DeepONetOperator(sensor_count=sensor_dim, num_outputs=4).to(device)
    model.load_state_dict(torch.load(pytorch_ckpt, map_location=device))
    model.eval()

    norm_stats = torch.load("checkpoints/norm_stats.pt", map_location=device)
    u_mean = norm_stats["mean"].to(device)
    u_std = norm_stats["std"].to(device)

    # 2) Sample input matching the project workflow
    u = torch.randn(1, sensor_dim, dtype=torch.float32)
    y = torch.linspace(0.0, 1.0, seq_len, dtype=torch.float32).unsqueeze(-1)
    u_norm = (u - u_mean) / u_std

    with torch.no_grad():
        ref_out = model(u_norm, y).cpu().numpy()

    # 3) Load ONNX model
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    onnx_inputs = {
        "trajectory_sensors_u": u_norm.cpu().numpy().astype(np.float32),
        "query_locations_y": y.cpu().numpy().astype(np.float32),
    }

    onnx_out = sess.run(None, onnx_inputs)[0]

    # 4) Compare outputs
    print("PyTorch output shape:", ref_out.shape)
    print("ONNX output shape:", onnx_out.shape)
    np.testing.assert_allclose(ref_out, onnx_out, rtol=1e-3, atol=tolerance)
    print("[OK] ONNX output matches PyTorch output within tolerance.")

    return ref_out, onnx_out


if __name__ == "__main__":
    test_onnx_vs_pytorch()
