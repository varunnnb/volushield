import numpy as np
from cryptography.hazmat.primitives import hashes

def compute_volume_hash(volume: np.ndarray) -> bytes:

    digest = hashes.Hash(hashes.SHA512())

    digest.update(volume.tobytes())

    return digest.finalize()