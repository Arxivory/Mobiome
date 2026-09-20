import torch

# Load normalization stats
stats = torch.load("checkpoints/norm_stats.pt")
u_mean = stats["mean"]  # Expected shape: [1, 720] or [720]
u_std = stats["std"]    # Expected shape: [1, 720] or [720]

# Flatten to 1D array if batched [1, 720] -> [720]
mean_flat = u_mean.squeeze()
std_flat = u_std.squeeze()

# Reshape into (60 frames, 12 channels per frame)
# Channel layout per frame: [0:4] = q, [4:8] = q_dot, [8:12] = q_ddot
mean_2d = mean_flat.view(60, 12)
std_2d = std_flat.view(60, 12)

# Slice channels across all 60 frames
q_mean = mean_2d[:, 0:4]
q_dot_mean = mean_2d[:, 4:8]
q_ddot_mean = mean_2d[:, 8:12]

q_std = std_2d[:, 0:4]
q_dot_std = std_2d[:, 4:8]
q_ddot_std = std_2d[:, 8:12]

print("=== CORRECTED CHANNEL-WISE STATS SUMMARY ===")
print(f"q      (Pos)   - Mean range: [{q_mean.min():.4f}, {q_mean.max():.4f}] | Std avg: {q_std.mean():.4f} (range: [{q_std.min():.4f}, {q_std.max():.4f}])")
print(f"q_dot  (Vel)   - Mean range: [{q_dot_mean.min():.4f}, {q_dot_mean.max():.4f}] | Std avg: {q_dot_std.mean():.4f} (range: [{q_dot_std.min():.4f}, {q_dot_std.max():.4f}])")
print(f"q_ddot (Accel) - Mean range: [{q_ddot_mean.min():.4f}, {q_ddot_mean.max():.4f}] | Std avg: {q_ddot_std.mean():.4f} (range: [{q_ddot_std.min() if 'q_ddot_min' in locals() else q_ddot_mean.min():.4f}, {q_ddot_mean.max():.4f}]) | Std avg: {q_ddot_std.mean():.4f} (range: [{q_ddot_std.min():.4f}, {q_ddot_std.max():.4f}])")