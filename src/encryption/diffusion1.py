import numpy as np


def diffuse_volume(volume: np.ndarray, keystream: np.ndarray) -> np.ndarray:
    """
    XOR diffusion encryption
    """

    flat = volume.flatten()

    encrypted = np.bitwise_xor(flat, keystream[:len(flat)])

    return encrypted.reshape(volume.shape)