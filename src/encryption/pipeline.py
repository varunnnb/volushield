import numpy as np

from src.crypto.dilithium import DilithiumCrypto
from src.crypto.seed_derivation import derive_seeds_from_signature
from src.crypto.hash_utils import compute_volume_hash

from src.chaos.hyperchaotic import MemristorHyperchaos
from src.encryption.permutation import permute_volume_3d
from src.encryption.diffusion import encrypt_diffusion


def encrypt_volume(volume: np.ndarray,
                   sk_override: bytes | None = None):

    # Step 1 — hash volume
    volume_hash = compute_volume_hash(volume)

    # Step 2 — Dilithium keys
    dc = DilithiumCrypto()

    if sk_override is None:
        public_key, secret_key = dc.generate_keypair()
    else:
        secret_key = sk_override
        public_key, _ = dc.generate_keypair()

    # Step 3 — sign hash
    signature = dc.sign_volume_hash(volume_hash)

    # Step 4 — derive seeds
    x0,y0,z0,w0 = derive_seeds_from_signature(signature, volume_hash)

    # Step 5 — generate chaotic sequences
    chaos = MemristorHyperchaos()

    D,H,W = volume.shape
    total = D*H*W

    seq_x,seq_y,seq_z,seq_w = chaos.generate_sequence(
        x0,y0,z0,w0,
        n_samples=total,
        transient=5000
    )

    # Step 6 — permutation
    permuted = permute_volume_3d(volume, seq_x, seq_y)

    # Step 7 — IV from hash
    iv = int(volume_hash[0])

    # Step 8 — diffusion
    encrypted = encrypt_diffusion(permuted, seq_z, seq_w, iv)

    return encrypted, signature, public_key, secret_key, volume_hash