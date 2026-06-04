import numpy as np

def compute_channel_stats(eeg: np.ndarray) -> dict:
    """
    Compute global per-channel mean and std.
    
    Args:
        eeg: np.ndarray of shape (T, C) or (N, T, C)
        
    Returns:
        dict with 'mean' and 'std' arrays of shape (C,)
    """
    if eeg.ndim == 3:
        mean = eeg.mean(axis=(0, 1))
        std = eeg.std(axis=(0, 1))
    elif eeg.ndim == 2:
        mean = eeg.mean(axis=0)
        std = eeg.std(axis=0)
    else:
        raise ValueError(f"EEG must be 2D or 3D, got {eeg.shape}")

    # Numerical safety
    std = np.maximum(std, 1e-6)
    return {"mean": mean, "std": std}

def apply_channel_norm(eeg: np.ndarray, stats: dict) -> np.ndarray:
    """
    Apply Z-score normalization per channel.
    """
    return (eeg - stats["mean"]) / stats["std"]

def compute_global_minmax(eeg: np.ndarray, clip_percentile: float = 99.0) -> dict:
    """
    Compute global min and max per channel using percentiles.
    
    Args:
        eeg: np.ndarray of shape (T, C) or (N, T, C)
        clip_percentile: percentile to clip outliers (e.g. 99.0 means 1st and 99th percentile)
        
    Returns:
        dict with 'min' and 'max' arrays of shape (C,)
    """
    if eeg.ndim == 3:
        eeg_flat = eeg.reshape(-1, eeg.shape[-1])
    elif eeg.ndim == 2:
        eeg_flat = eeg
    else:
        raise ValueError(f"EEG must be 2D or 3D, got {eeg.shape}")
        
    low = np.percentile(eeg_flat, 100 - clip_percentile, axis=0)
    high = np.percentile(eeg_flat, clip_percentile, axis=0)
    return {"min": low, "max": high}

def apply_global_minmax(eeg: np.ndarray, stats: dict) -> np.ndarray:
    """
    Apply Min-Max scaling to bounded range [0, 1] using precomputed stats.
    """
    low = stats["min"]
    high = stats["max"]
    
    eeg_clipped = np.clip(eeg, low, high)
    
    denom = high - low
    denom[denom == 0] = 1e-6
    return (eeg_clipped - low) / denom

def denormalize_eeg(eeg_scaled: np.ndarray, minmax_stats: dict, z_stats: dict) -> np.ndarray:
    """
    Reverse the 2-stage normalization:
    1) Undo Min-Max (back to Z-score)
    2) Undo Z-score (back to original physical units)
    """
    # 1. Reverse Min-Max
    low = minmax_stats["min"]
    high = minmax_stats["max"]
    eeg_z = eeg_scaled * (high - low) + low
    
    # 2. Reverse Z-score
    mean = z_stats["mean"]
    std = z_stats["std"]
    eeg_orig = eeg_z * std + mean
    
    return eeg_orig

def save_stats(stats: dict, path: str) -> None:
    """
    Save stats to a .npz file.
    """
    np.savez(path, **stats)

def load_stats(path: str) -> dict:
    """
    Load stats from a .npz file.
    """
    data = np.load(path)
    return {k: data[k] for k in data.files}