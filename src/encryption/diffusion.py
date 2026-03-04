# src/encryption/diffusion.py

import numpy as np

def forward_diffusion(permuted_volume: np.ndarray,
                       keystream: np.ndarray,
                       iv: int = 0) -> np.ndarray:
    """
    Forward diffusion: encrypt permuted volume using chaotic keystream.
    
    C[0] = permuted[0] XOR keystream[0] XOR iv
    C[i] = permuted[i] XOR C[i-1] XOR keystream[i]
    
    This creates chain dependency: each ciphertext voxel depends
    on all previous ciphertext voxels.
    
    Args:
        permuted_volume: uint8 ndarray (D, H, W)
        keystream: uint8 array of length D*H*W
        iv: initialization vector (single byte, derived from volume hash)
    
    Returns:
        Encrypted volume (D, H, W) as uint8
    """
    D, H, W = permuted_volume.shape
    total = D * H * W
    
    flat = permuted_volume.flatten()
    cipher = np.zeros(total, dtype=np.uint8)
    
    # First voxel
    cipher[0] = flat[0] ^ keystream[0] ^ iv
    
    # Remaining voxels (chain dependency)
    for i in range(1, total):
        cipher[i] = flat[i] ^ cipher[i-1] ^ keystream[i]
    
    return cipher.reshape(D, H, W)

def backward_diffusion(cipher_volume: np.ndarray,
                        keystream: np.ndarray,
                        iv: int = 0) -> np.ndarray:
    """
    Second diffusion pass: process volume in REVERSE order.
    This ensures the last voxel depends on all others — 
    improves NPCR/UACI scores significantly.
    """
    D, H, W = cipher_volume.shape
    total = D * H * W
    
    flat = cipher_volume.flatten()
    cipher2 = np.zeros(total, dtype=np.uint8)
    
    # Process in reverse
    cipher2[-1] = flat[-1] ^ keystream[-1] ^ iv
    
    for i in range(total - 2, -1, -1):
        cipher2[i] = flat[i] ^ cipher2[i+1] ^ keystream[i]
    
    return cipher2.reshape(D, H, W)

def encrypt_diffusion(permuted_volume: np.ndarray,
                       seq_z: np.ndarray,
                       seq_w: np.ndarray,
                       iv: int = 0) -> np.ndarray:
    """
    Full diffusion encryption: forward pass followed by backward pass.
    Uses z-component for forward keystream, w-component for backward.
    """
    from src.chaos.sequence_gen import chaotic_to_keystream
    
    D, H, W = permuted_volume.shape
    total = D * H * W
    
    ks_forward = chaotic_to_keystream(seq_z, total)
    ks_backward = chaotic_to_keystream(seq_w, total)
    
    # Forward pass
    after_forward = forward_diffusion(permuted_volume, ks_forward, iv)
    
    # Backward pass (on the already forward-diffused volume)
    encrypted = backward_diffusion(after_forward, ks_backward, iv)
    
    return encrypted