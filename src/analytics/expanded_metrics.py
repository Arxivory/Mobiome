import numpy as np
from scipy.signal import welch

def compute_mpjpe(p_pred: np.ndarray, p_gt: np.ndarray) -> float:
    """Mean Per Joint Position Error in millimeters (mm)."""
    # Shapes: (T, K, 3)
    errors = np.linalg.norm(p_pred - p_gt, axis=-1)
    return float(np.mean(errors) * 1000.0)

def compute_bone_length_variance(p_3d: np.ndarray) -> float:
    """Computes mean bone length variance across skeleton connectivity graph (mm^2)."""
    # Standard Human3.6M bone pairs
    bone_pairs = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
        (12, 13), (8, 14), (14, 15), (15, 16)
    ]
    variances = []
    for u, v in bone_pairs:
        lengths = np.linalg.norm(p_3d[:, u, :] - p_3d[:, v, :], axis=-1) * 1000.0  # in mm
        variances.append(np.var(lengths))
    return float(np.mean(variances))

def compute_sparc(movement: np.ndarray, fs: float = 60.0, padlevel: int = 4, fc: float = 10.0) -> float:
    """
    Computes Spectral Arc Length (SPARC) measure of movement smoothness.
    Ref: Balasubramanian et al., 2015.
    """
    if movement.ndim > 1:
        movement = np.linalg.norm(movement, axis=-1)
    
    # Zero-pad signal
    n = len(movement)
    n_fft = int(2 ** (np.ceil(np.log2(n)) + padlevel))
    
    # Compute Normalized Magnitude Spectrum
    freqs, psd = welch(movement, fs=fs, nperseg=n, nfft=n_fft)
    spectrum = np.sqrt(psd)
    max_val = np.max(spectrum) + 1e-9
    spectrum = spectrum / max_val
    
    # Select frequency band [0, fc]
    fc_idx = np.where(freqs <= fc)[0]
    f_sub = freqs[fc_idx]
    s_sub = spectrum[fc_idx]
    
    # Arc length calculation
    df = f_sub[1] - f_sub[0]
    ds_df = np.gradient(s_sub, df)
    arc_length = -np.sum(np.sqrt((1.0 / fc) ** 2 + ds_df ** 2)) * df
    return float(arc_length)

def compute_biomechanical_metrics(tau_pred: np.ndarray, tau_gt: np.ndarray) -> dict:
    """Calculates MAE, RMSE, and Pearson correlation coefficient across target joints."""
    mae = np.mean(np.abs(tau_pred - tau_gt), axis=0)
    rmse = np.sqrt(np.mean((tau_pred - tau_gt) ** 2, axis=0))
    
    correlations = []
    for i in range(tau_gt.shape[-1]):
        if np.std(tau_pred[:, i]) < 1e-6 or np.std(tau_gt[:, i]) < 1e-6:
            correlations.append(0.0)
        else:
            r = np.corrcoef(tau_pred[:, i], tau_gt[:, i])[0, 1]
            correlations.append(r)
            
    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": np.array(correlations),
        "mean_mae": float(np.mean(mae)),
        "mean_rmse": float(np.mean(rmse)),
        "mean_corr": float(np.mean(correlations))
    }