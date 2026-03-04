# src/crypto/seed_derivation.py

import hashlib
import numpy as np
import struct
from typing import Tuple

def derive_chaotic_seeds(secret_key: bytes, 
                          volume_hash: bytes) -> Tuple[float, float, float, float]:
    """
    Derive four chaotic initial conditions from Dilithium secret key 
    and volume hash using SHA-512.
    
    Why SHA-512: produces 64 bytes = 512 bits, which we split into
    four 128-bit segments, each converted to a float64 initial condition.
    
    The volume hash ensures that seeds are image-specific (different image
    → different seeds even with the same key).
    """
    # Concatenate secret key bytes with volume hash
    combined = secret_key + volume_hash
    
    # Derive 64 bytes using SHA-512
    hash_bytes = hashlib.sha512(combined).digest()  # 64 bytes
    
    # Split into four 16-byte segments
    seg0 = hash_bytes[0:16]
    seg1 = hash_bytes[16:32]
    seg2 = hash_bytes[32:48]
    seg3 = hash_bytes[48:64]
    
    # Convert each segment to a float64 in a controlled range
    # Method: interpret as uint128, normalize to [0.1, 0.9]
    def bytes_to_seed(b: bytes) -> float:
        # Take first 8 bytes as uint64
        val = struct.unpack('>Q', b[:8])[0]
        # Normalize to (0, 1) range, then shift to (0.1, 0.9)
        normalized = val / (2**64 - 1)
        return 0.1 + normalized * 0.8  # Ensures non-trivial initial condition
    
    x0 = bytes_to_seed(seg0)
    y0 = bytes_to_seed(seg1)
    z0 = bytes_to_seed(seg2)
    w0 = bytes_to_seed(seg3)
    
    return x0, y0, z0, w0

def derive_seeds_from_signature(signature: bytes,
                                 volume_hash: bytes) -> Tuple[float, float, float, float]:
    """
    Alternative: derive seeds from signature rather than secret key.
    Useful when you want seeds to depend on BOTH the key and the signed content.
    This is your primary method — document this in the paper.
    """
    combined = hashlib.sha512(signature + volume_hash).digest()
    
    def bytes_to_seed(b: bytes) -> float:
        val = struct.unpack('>Q', b[:8])[0]
        return 0.1 + (val / (2**64 - 1)) * 0.8
    
    return (bytes_to_seed(combined[0:16]),
            bytes_to_seed(combined[16:32]),
            bytes_to_seed(combined[32:48]),
            bytes_to_seed(combined[48:64]))