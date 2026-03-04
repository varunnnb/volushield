import numpy as np


def chaotic_to_permutation_indices(seq: np.ndarray, n: int) -> np.ndarray:
    """
    Convert chaotic sequence → permutation indices
    """

    return np.argsort(seq[:n]).astype(np.int64)


def chaotic_to_keystream(seq: np.ndarray, n: int) -> np.ndarray:
    """
    Convert chaotic sequence → uint8 keystream
    """

    ks = (np.abs(seq[:n]) * 1e14).astype(np.uint64) % 256

    return ks.astype(np.uint8)