import numpy as np

def compute_biomechanical_metrics(tau_pred: np.ndarray, tau_gt: np.ndarray) -> dict:
    """
    Evaluates empirical accuracy across joint torque profile trajectories:
      - RMSE (Root Mean Square Error in N·m)
      - MAE (Mean Absolute Error in N·m)
      - Pearson Correlation Coefficient (r)
      - Peak Torque Error (%)
    """
    # 1. Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean((tau_pred - tau_gt) ** 2, axis=0))

    # 2. Mean Absolute Error (MAE)
    mae = np.mean(np.abs(tau_pred - tau_gt), axis=0)

    # 3. Pearson Correlation (r) per joint profile
    correlation = []
    for d in range(tau_gt.shape[-1]):
        r = np.corrcoef(tau_pred[:, d], tau_gt[:, d])[0, 1]
        correlation.append(r if not np.isnan(r) else 0.0)

    # 4. Peak Torque Error (%)
    peak_gt = np.max(np.abs(tau_gt), axis=0)
    peak_pred = np.max(np.abs(tau_pred), axis=0)
    peak_error_pct = (np.abs(peak_gt - peak_pred) / (peak_gt + 1e-6)) * 100.0

    return {
        "rmse": rmse,
        "mae": mae,
        "correlation": np.array(correlation),
        "peak_error_pct": peak_error_pct
    }