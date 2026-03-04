# src/decryption/pipeline.py

import numpy as np
from src.crypto.dilithium import DilithiumCrypto
from src.crypto.seed_derivation import derive_seeds_from_signature
from src.crypto.hash_utils import compute_volume_hash
from src.chaos.hyperchaotic import MemristorHyperchaos
from src.decryption.inv_diffusion import (inverse_forward_diffusion, 
                                           inverse_backward_diffusion)
from src.decryption.inv_permutation import inverse_permute_volume_3d
from src.chaos.sequence_gen import chaotic_to_keystream

def decrypt_volume(encrypted_volume: np.ndarray,
                   signature: bytes,
                   public_key: bytes,
                   secret_key: bytes,
                   original_hash: bytes) -> np.ndarray:
    """Full decryption pipeline."""
    
    # Step 1: Verify signature
    dc = DilithiumCrypto()
    is_valid = dc.verify_signature(original_hash, signature, public_key)
    if not is_valid:
        raise ValueError("Signature verification FAILED — volume may be tampered!")
    print("✓ Signature verified successfully")
    
    # Step 2: Regenerate seeds (must use same inputs as encryption)
    x0, y0, z0, w0 = derive_seeds_from_signature(signature, original_hash)
    
    # Step 3: Regenerate chaotic sequences
    chaos = MemristorHyperchaos()
    D, H, W = encrypted_volume.shape
    total = D * H * W
    seq_x, seq_y, seq_z, seq_w = chaos.generate_sequence(
        x0, y0, z0, w0, n_samples=total, transient=5000
    )
    
    # Step 4: Compute IV from original hash
    iv = int(original_hash[0])
    
    # Step 5: Inverse backward diffusion (reverse second pass)
    flat_enc = encrypted_volume.flatten()
    ks_backward = chaotic_to_keystream(seq_w, total)
    after_inv_backward = inverse_backward_diffusion(flat_enc, ks_backward, iv)
    
    # Step 6: Inverse forward diffusion (reverse first pass)
    ks_forward = chaotic_to_keystream(seq_z, total)
    after_inv_forward = inverse_forward_diffusion(after_inv_backward, ks_forward, iv)
    after_inv_forward = after_inv_forward.reshape(D, H, W)
    
    # Step 7: Inverse permutation
    recovered = inverse_permute_volume_3d(after_inv_forward, seq_x, seq_y)
    
    return recovered
