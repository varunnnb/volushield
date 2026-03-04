# src/encryption/permutation.py

import numpy as np
from src.chaos.sequence_gen import chaotic_to_permutation_indices

def permute_volume_3d(volume: np.ndarray, 
                       seq_x: np.ndarray,
                       seq_y: np.ndarray) -> np.ndarray:
    """
    3D voxel permutation using chaotic sequences.
    
    Strategy:
    1. Flatten 3D volume to 1D array
    2. Generate permutation indices from chaotic sequence (argsort method)
    3. Apply permutation
    4. Reshape back to 3D
    
    Args:
        volume: uint8 ndarray of shape (D, H, W)
        seq_x: chaotic sequence for permutation (from x-component)
        seq_y: chaotic sequence for secondary permutation (y-component)
    
    Returns:
        Permuted volume of same shape
    """
    D, H, W = volume.shape
    total_voxels = D * H * W
    
    # Step 1: Flatten to 1D
    flat = volume.flatten()  # Shape: (total_voxels,)
    
    # Step 2: Generate permutation indices using combined chaotic sequence
    # Combine two sequences for higher security
    combined_seq = seq_x[:total_voxels] + seq_y[:total_voxels]
    perm_indices = chaotic_to_permutation_indices(combined_seq, total_voxels)
    
    # Step 3: Apply permutation
    permuted_flat = flat[perm_indices]
    
    # Step 4: Reshape back to 3D
    return permuted_flat.reshape(D, H, W)

def inverse_permute_volume_3d(permuted_volume: np.ndarray,
                               seq_x: np.ndarray,
                               seq_y: np.ndarray) -> np.ndarray:
    """
    Inverse 3D permutation for decryption.
    
    Key insight: if forward permutation is P, inverse is P^(-1).
    Given perm_indices where permuted[i] = original[perm_indices[i]],
    inverse is: original[perm_indices[i]] = permuted[i]
    → Use np.argsort(perm_indices) to get inverse permutation.
    """
    D, H, W = permuted_volume.shape
    total_voxels = D * H * W
    
    flat = permuted_volume.flatten()
    
    combined_seq = seq_x[:total_voxels] + seq_y[:total_voxels]
    perm_indices = chaotic_to_permutation_indices(combined_seq, total_voxels)
    
    # Inverse permutation
    inv_perm = np.argsort(perm_indices)
    restored_flat = flat[inv_perm]
    
    return restored_flat.reshape(D, H, W)