import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.operators.deeponet import DeepONetOperator

class BiomechanicsOperatorDataset(Dataset):
    def __init__(self, kinematics_dir: str, torques_dir: str, seq_len: int = 60):
        self.seq_len = seq_len
        self.inputs = []
        self.targets = []

        kin_files = sorted(glob.glob(os.path.join(kinematics_dir, "*.npy")))
        for kin_file in kin_files:
            base_name = os.path.basename(kin_file).replace("_kinematics.npy", "_torques.npy")
            trq_file = os.path.join(torques_dir, base_name)
            
            if not os.path.exists(trq_file):
                continue
                
            kin_data = np.load(kin_file, allow_pickle=True).item()
            tau_data = np.load(trq_file)

            # Concatenate [q, q_dot, q_ddot] -> shape (T, D * 3)
            u_seq = np.concatenate([kin_data['q'], kin_data['q_dot'], kin_data['q_ddot']], axis=-1)
            T, D3 = u_seq.shape

            # Slice continuous sequences into fixed windows
            for i in range(0, T - seq_len, seq_len // 2):
                self.inputs.append(u_seq[i:i + seq_len].flatten()) # Flattened trajectory sensor array
                self.targets.append(tau_data[i:i + seq_len])      # Target torque profile (seq_len, D)

        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def train_phase3():
    dataset = BiomechanicsOperatorDataset("data/processed", "data/synthetic", seq_len=60)
    if len(dataset) == 0:
        print("No processed data samples found. Run Phase 1 data script first.")
        return

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    sensor_dim = dataset.inputs.shape[1]
    num_joints = dataset.targets.shape[-1]

    # Domain query array (t normalized from 0 to 1 over frame sequence)
    y_query = torch.linspace(0, 1, 60).unsqueeze(-1) # (60, 1)

    model = DeepONetOperator(sensor_count=sensor_dim, num_outputs=num_joints)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    print(f"--- Phase 3: Training DeepONet Operator ({len(dataset)} samples) ---")
    
    model.train()
    for epoch in range(1, 6):
        epoch_loss = 0.0
        for u_batch, tau_gt in loader:
            optimizer.zero_grad()
            tau_pred = model(u_batch, y_query)
            loss = criterion(tau_pred, tau_gt)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch}/5 | MSE Operator Loss: {epoch_loss / len(loader):.6f}")

    # Save trained PyTorch operator for Phase 4 export
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/deeponet_fp32.pth")
    print("Successfully saved FP32 DeepONet model to checkpoints/deeponet_fp32.pth")

if __name__ == "__main__":
    train_phase3()