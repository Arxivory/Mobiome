import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.ao.quantization as quantization
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.operators.deeponet import DeepONetOperator

try:
    from scripts.train import BiomechanicsOperatorDataset
except ImportError:
    from train import BiomechanicsOperatorDataset


# ---------------------------------------------------------------------------
# 1. Quantized Model Wrapper (QAT & PTQ Support)
# ---------------------------------------------------------------------------
class QuantizableDeepONet(nn.Module):
    """
    Quantization-compatible wrapper for DeepONetOperator.
    Inserts Quant/DeQuant stubs at domain boundaries to enable INT8 lowerings.
    """
    def __init__(self, model: DeepONetOperator):
        super(QuantizableDeepONet, self).__init__()
        self.quant_u = quantization.QuantStub()
        self.quant_y = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        self.model = model

    @staticmethod
    def _ensure_batch_dim(x: torch.Tensor, expected_ndim: int = 2) -> torch.Tensor:
        if x.dim() == expected_ndim - 1:
            x = x.unsqueeze(0)
        return x

    def forward(self, u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u = self._ensure_batch_dim(u, expected_ndim=2)
        y = self._ensure_batch_dim(y, expected_ndim=2)

        # Quantize inputs
        u_q = self.quant_u(u)
        y_q = self.quant_y(y)

        # Execute operator forward pass
        tau_pred = self.model(u_q, y_q)

        # Dequantize predictions to float space
        return self.dequant(tau_pred)


def make_y_query(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Create a shared query grid for the DeepONet trunk input."""
    return torch.linspace(0.0, 1.0, steps=seq_len, device=device, dtype=torch.float32).unsqueeze(-1)


# ---------------------------------------------------------------------------
# 2. Calibration & Dataset Helper
# ---------------------------------------------------------------------------
def build_data_loader(
    kinematics_dir: str = "data/processed",
    torques_dir: str = "data/synthetic",
    batch_size: int = 16,
    seq_len: int = 60
) -> DataLoader:
    """
    Builds DataLoader utilizing BiomechanicsOperatorDataset from train.py.
    """
    dataset = BiomechanicsOperatorDataset(
        kinematics_dir=kinematics_dir,
        torques_dir=torques_dir,
        seq_len=seq_len
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


# ---------------------------------------------------------------------------
# 3. Post-Training Quantization (PTQ) Workflow
# ---------------------------------------------------------------------------
def run_post_training_quantization(
    model: DeepONetOperator,
    calib_loader: DataLoader,
    seq_len: int = 60
) -> nn.Module:
    """
    Executes Static Post-Training Quantization (PTQ) using fbgemm observer engine.
    """
    print("\n" + "=" * 50)
    print("      RUNNING POST-TRAINING QUANTIZATION (PTQ)      ")
    print("=" * 50)

    quantized_wrapper = QuantizableDeepONet(model)
    quantized_wrapper.eval()

    # Define Quantization Engine Config
    quantized_wrapper.qconfig = quantization.get_default_qconfig('fbgemm')
    
    # Prepare model observers
    model_prepared = quantization.prepare(quantized_wrapper, inplace=False)

    # Observer Calibration Pass
    print("[*] Calibrating dynamic range observers...")
    y_query = make_y_query(seq_len)

    with torch.no_grad():
        for batch in calib_loader:
            if isinstance(batch, (list, tuple)):
                u_batch = batch[0]
            else:
                u_batch = batch
            if u_batch.dim() == 1:
                u_batch = u_batch.unsqueeze(0)
            model_prepared(u_batch, y_query)

    # Convert to INT8 Model Representation
    model_quantized = quantization.convert(model_prepared, inplace=False)
    print("[+] Static PTQ conversion completed.")
    return model_quantized


# ---------------------------------------------------------------------------
# 4. Quantization-Aware Training (QAT) Workflow
# ---------------------------------------------------------------------------
def run_quantization_aware_training(
    model: DeepONetOperator,
    train_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-4,
    seq_len: int = 60
) -> nn.Module:
    """
    Performs QAT fine-tuning with Straight-Through Estimators (STE) on simulated fake-quantized operations.
    """
    print("\n" + "=" * 50)
    print("    RUNNING QUANTIZATION-AWARE TRAINING (QAT)      ")
    print("=" * 50)

    quantized_wrapper = QuantizableDeepONet(model)
    quantized_wrapper.train()

    # Configure QAT specifications
    quantized_wrapper.qconfig = quantization.get_default_qat_qconfig('fbgemm')
    model_qat = quantization.prepare_qat(quantized_wrapper, inplace=False)

    optimizer = torch.optim.Adam(model_qat.parameters(), lr=lr)
    criterion = nn.MSELoss()
    y_query = make_y_query(seq_len)

    print(f"[*] Beginning QAT fine-tuning loop for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches = 0
        
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                u_batch, tau_gt = batch[0], batch[1]
            else:
                u_batch, tau_gt = batch, torch.randn(batch.shape[0], seq_len, 4)

            optimizer.zero_grad()
            tau_pred = model_qat(u_batch, y_query)
            loss = criterion(tau_pred, tau_gt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(batches, 1)
        print(f"    Epoch {epoch}/{epochs} | QAT MSE Loss: {avg_loss:.6f}")

    # Freeze observers and convert to INT8
    model_qat.eval()
    model_quantized = quantization.convert(model_qat, inplace=False)
    print("[+] QAT fine-tuning and conversion completed.")
    return model_quantized


# ---------------------------------------------------------------------------
# 5. ONNX Export Engine
# ---------------------------------------------------------------------------
def export_to_onnx(
    model: nn.Module,
    output_path: str = "checkpoints/quantized_deeponet.onnx",
    sensor_dim: int = 720,
    seq_len: int = 60
):
    """
    Exports floating-point or fake-quantized state graphs to ONNX runtime representation.
    """
    try:
        import onnxscript  # noqa: F401
    except ModuleNotFoundError:
        print("[!] ONNX export skipped: install 'onnxscript' and 'onnx' to export the graph.")
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.eval()

    dummy_u = torch.randn(1, sensor_dim, dtype=torch.float32)
    dummy_y = make_y_query(seq_len)

    print(f"[*] Exporting graph to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (dummy_u, dummy_y),
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['trajectory_sensors_u', 'query_locations_y'],
        output_names=['predicted_torques_tau'],
        dynamic_axes={
            'trajectory_sensors_u': {0: 'batch_size'},
            'predicted_torques_tau': {0: 'batch_size'}
        }
    )
    print(f"[+] Successfully exported ONNX artifact to {output_path}")
    return True


# ---------------------------------------------------------------------------
# Main Execution Strategy
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SEQ_LEN = 60
    NUM_JOINTS = 4  # Matches num_outputs default in DeepONetOperator
    SENSOR_DIM = SEQ_LEN * NUM_JOINTS * 3  # 720 features (60 timesteps * 4 joints * [q, q_dot, q_ddot])

    # 1. Instantiate FP32 Ground Model
    base_model = DeepONetOperator(
        sensor_count=SENSOR_DIM,
        num_outputs=NUM_JOINTS,
        hidden_dim=256,
        latent_dim=128
    )

    # Load FP32 pre-trained weights if available
    ckpt_path = "checkpoints/deeponet_fp32.pth"
    if os.path.exists(ckpt_path):
        base_model.load_state_dict(torch.load(ckpt_path))
        print(f"[*] Loaded trained FP32 weights from {ckpt_path}")
    else:
        print("[!] Pre-trained weights not found at checkpoints/deeponet_fp32.pth. Executing with initialized weights.")

    # 2. Build Data Loader
    data_loader = build_data_loader(
        kinematics_dir="data/processed",
        torques_dir="data/synthetic",
        batch_size=16,
        seq_len=SEQ_LEN
    )

    # 3. Execute PTQ Strategy
    ptq_model = run_post_training_quantization(base_model, data_loader, seq_len=SEQ_LEN)

    # 4. Execute QAT Strategy
    qat_model = run_quantization_aware_training(base_model, data_loader, epochs=3, lr=1e-4, seq_len=SEQ_LEN)

    # 5. Export Baseline Graph Artifact
    export_to_onnx(base_model, output_path="checkpoints/deeponet_fp32.onnx", sensor_dim=SENSOR_DIM, seq_len=SEQ_LEN)