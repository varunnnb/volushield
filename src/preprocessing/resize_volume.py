import numpy as np

def center_crop(volume: np.ndarray, target_size=(64,64,64)) -> np.ndarray:
    """
    Crop the center of a 3D volume
    """

    d, h, w = volume.shape
    td, th, tw = target_size

    start_d = (d - td) // 2
    start_h = (h - th) // 2
    start_w = (w - tw) // 2

    cropped = volume[
        start_d:start_d+td,
        start_h:start_h+th,
        start_w:start_w+tw
    ]

    return cropped