import numpy as np


def permute_volume(volume: np.ndarray, perm_indices: np.ndarray) -> np.ndarray:
    """
    Apply voxel permutation encryption
    """

    flat = volume.flatten()

    permuted = flat[perm_indices]

    return permuted.reshape(volume.shape)