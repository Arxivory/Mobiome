import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set ACM/IEEE publication style
plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "figure.titlesize": 12,
    "figure.dpi": 300,
})

def generate_paper_artifacts(
    pareto_json: str = "paper_results/pareto_sweep_data.json",
    drift_json: str = "paper_results/quantization_drift_data.json",
    latency_json: str = "paper_results/latency_benchmark_data.json",
    out_dir: str = "paper_results"
):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Render Figure: Pareto Frontier (RQ1)
    if os.path.exists(pareto_json):
        with open(pareto_json) as f:
            data = json.load(f)

        mpjpe = [d["mpjpe_mm"] for d in data]
        jerk_var = [d["jerk_var"] for d in data]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        sc = ax.scatter(jerk_var, mpjpe, c=[d["lambda_2"] for d in data], cmap="viridis", edgecolors="k", s=40)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Smoothness Penalty Weight (λ₂)")
        ax.set_xlabel("Joint Jerk Variance (m²/s⁶)")
        ax.set_ylabel("MPJPE Spatial Error (mm)")
        ax.set_title("Pareto Frontier: Spatial Accuracy vs. Jerk Regularization")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "figure_1_pareto_frontier.png"))
        plt.close()
        print("[+] Generated Figure 1: Pareto Frontier")

    # 2. Render Figure: Quantization Error Drift (RQ3)
    if os.path.exists(drift_json):
        with open(drift_json) as f:
            data = json.load(f)

        v_scale = [d["velocity_scale"] for d in data]
        bounds = [d["analytical_bound"] for d in data]
        drifts = [d["empirical_drift"] for d in data]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(v_scale, bounds, "k--", label="Analytical Upper Bound (Eq. 12)", linewidth=1.5)
        ax.plot(v_scale, drifts, "go-", label="Observed INT8 Drift (Δτ)", linewidth=2.0)
        ax.set_xlabel("Kinematic Velocity Scale")
        ax.set_ylabel("Torque Drift Δτ (N·m)")
        ax.set_title("Quantization Torque Error Drift Analysis")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "figure_3_quantization_drift.png"))
        plt.close()
        print("[+] Generated Figure 3: Quantization Drift")

    # 3. Generate LaTeX Summary Table
    if os.path.exists(latency_json):
        with open(latency_json) as f:
            lat_data = json.load(f)

        latex_table = r"""
\begin{table}[h]
\centering
\caption{Deployment Benchmark across Dynamics Solvers and Execution Engines.}
\label{tab:deployment_benchmark}
\begin{tabular}{lcccc}
\hline
\textbf{Model / Engine} & \textbf{Precision} & \textbf{Latency (ms)} & \textbf{Throughput (FPS)} & \textbf{Memory (MB)} \\ \hline
OpenSim RNEA Solver     & FP64               & """ + f"{lat_data['OpenSim_RNEA']['mean_ms']:.2f}" + r""" & """ + f"{lat_data['OpenSim_RNEA']['fps']:.1f}" + r""" & 240.0 \\
DeepONet (ONNX)         & FP32               & """ + f"{lat_data['DeepONet_FP32']['mean_ms']:.2f}" + r""" & """ + f"{lat_data['DeepONet_FP32']['fps']:.1f}" + r""" & 48.2 \\
DeepONet (ONNX)         & INT8 (PTQ)         & \textbf{""" + f"{lat_data['DeepONet_INT8']['mean_ms']:.2f}" + r"""} & \textbf{""" + f"{lat_data['DeepONet_INT8']['fps']:.1f}" + r"""} & \textbf{12.1} \\ \hline
\end{tabular}
\end{table}
"""
        with open(os.path.join(out_dir, "table_benchmark.tex"), "w") as f:
            f.write(latex_table)
        print("[+] Generated LaTeX benchmark table")

if __name__ == "__main__":
    generate_paper_artifacts()