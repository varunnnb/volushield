import numpy as np

from src.chaos.hyperchaotic import MemristorHyperchaos
from src.chaos.sequence_gen import (
    chaotic_to_permutation_indices,
    chaotic_to_keystream
)

from src.encryption.permutation1 import permute_volume
from src.encryption.diffusion import diffuse_volume


def encrypt_volume(volume: np.ndarray):

    n_voxels = volume.size

    chaos = MemristorHyperchaos()

    x,y,z,w = chaos.generate_sequence(
        0.1,0.2,0.3,0.4,
        n_samples = 2*n_voxels
    )

    perm_indices = chaotic_to_permutation_indices(x, n_voxels)

    permuted = permute_volume(volume, perm_indices)

    keystream = chaotic_to_keystream(y, n_voxels)

    encrypted = diffuse_volume(permuted, keystream)

    return encrypted