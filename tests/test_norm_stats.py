import torch

stats_path = "checkpoints/norm_stats.pt"
norm_stats = torch.load(stats_path, map_location="cpu")

print("=== NORM STATS DICTIONARY KEYS ===")
print(list(norm_stats.keys()))

u_mean = norm_stats.get("mean")
u_std = norm_stats.get("std")

print("\n=== SHAPE ANALYSIS ===")
print(f"u_mean shape: {u_mean.shape if u_mean is not None else 'None'}")
print(f"u_std shape : {u_std.shape if u_std is not None else 'None'}")

if u_mean is not None:
    print("\n=== MEAN VALUES PER CHANNEL / FEATURE ===")
    print("u_mean values:\n", u_mean)
    print("\nu_std values:\n", u_std)

if u_mean.shape[-1] == 720:
    q_mean = u_mean[..., :51]
    q_dot_mean = u_mean[..., 51:102]
    q_ddot_mean = u_mean[..., 102:153]

    q_std = u_std[..., :51]
    q_dot_std = u_std[..., 51:102]
    q_ddot_std = u_std[..., 102:153]

    print("\n=== CHANNEL-WISE STATS SUMMARY ===")
    print(f"q      (Pos)  - Mean range: [{q_mean.min():.4f}, {q_mean.max():.4f}] | Std avg: {q_std.mean():.4f}")
    print(f"q_dot  (Vel)  - Mean range: [{q_dot_mean.min():.4f}, {q_dot_mean.max():.4f}] | Std avg: {q_dot_std.mean():.4f}")
    print(f"q_ddot (Accel)- Mean range: [{q_ddot_mean.min():.4f}, {q_ddot_mean.max():.4f}] | Std avg: {q_ddot_std.mean():.4f}")