import numpy as np

def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Normalize volume to uint8"""
    
    v_min = volume.min()
    v_max = volume.max()

    if v_max == v_min:
        return np.zeros_like(volume, dtype=np.uint8)

    normalized = (volume - v_min) / (v_max - v_min) * 255
    return normalized.astype(np.uint8)