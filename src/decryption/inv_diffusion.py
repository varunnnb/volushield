# src/decryption/inv_diffusion.py

import numpy as np

def inverse_forward_diffusion(cipher: np.ndarray,
                               keystream: np.ndarray,
                               iv: int = 0) -> np.ndarray:
    """
    Reverse the forward diffusion.
    Given: C[i] = V[i] XOR C[i-1] XOR K[i]
    Therefore: V[i] = C[i] XOR C[i-1] XOR K[i]
    
    Note: This CAN be vectorized because C[i-1] is known ciphertext!
    """
    n = len(cipher)
    plain = np.zeros(n, dtype=np.uint8)
    
    plain[0] = cipher[0] ^ keystream[0] ^ np.uint8(iv)
    
    # Vectorized for i > 0: plain[i] = cipher[i] XOR cipher[i-1] XOR keystream[i]
    plain[1:] = cipher[1:] ^ cipher[:-1] ^ keystream[1:]
    
    return plain

def inverse_backward_diffusion(cipher: np.ndarray,
                                 keystream: np.ndarray,
                                 iv: int = 0) -> np.ndarray:
    """
    Reverse the backward diffusion.
    Given: C2[i] = C1[i] XOR C2[i+1] XOR K[i]
    Therefore: C1[i] = C2[i] XOR C2[i+1] XOR K[i]
    """
    n = len(cipher)
    result = np.zeros(n, dtype=np.uint8)
    
    result[-1] = cipher[-1] ^ keystream[-1] ^ np.uint8(iv)
    
    # Vectorized for i < n-1
    result[:-1] = cipher[:-1] ^ cipher[1:] ^ keystream[:-1]
    
    return result