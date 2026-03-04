# src/metrics/security.py

import numpy as np

def compute_npcr(cipher1: np.ndarray, cipher2: np.ndarray) -> float:
    """
    Compute NPCR between two ciphertexts encrypted from volumes
    differing by one voxel.
    
    Args:
        cipher1: Encrypted volume from original plaintext
        cipher2: Encrypted volume from plaintext with one voxel changed
    
    Returns:
        NPCR percentage [0, 100]
    """
    assert cipher1.shape == cipher2.shape
    diff = (cipher1 != cipher2).astype(np.float64)
    npcr = (diff.sum() / diff.size) * 100.0
    return npcr


def compute_uaci(cipher1: np.ndarray, cipher2: np.ndarray) -> float:
    """
    Compute UACI between two ciphertexts.
    Returns UACI percentage [0, 100].
    """
    assert cipher1.shape == cipher2.shape
    diff = np.abs(cipher1.astype(np.float64) - cipher2.astype(np.float64))
    uaci = (diff.sum() / (diff.size * 255.0)) * 100.0
    return uaci


def compute_entropy(volume: np.ndarray) -> float:
    """
    Compute Shannon entropy of a volume.
    For perfectly encrypted uint8 data, expected value ≈ 7.999.
    """
    flat = volume.flatten()
    # Count frequency of each intensity value
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    # Convert to probabilities
    probs = hist / flat.size
    # Remove zeros (log2(0) is undefined)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def compute_correlation(volume: np.ndarray, 
                         direction: str = 'horizontal') -> float:
    """
    Compute correlation coefficient between adjacent voxels.
    
    Args:
        direction: 'horizontal', 'vertical', or 'diagonal'
    
    Returns:
        Pearson correlation coefficient in [-1, 1]
    """
    flat = volume.flatten().astype(np.float64)
    
    if direction == 'horizontal':
        x = flat[:-1]
        y = flat[1:]
    elif direction == 'vertical':
        # Reshape and take vertically adjacent pairs
        d, h, w = volume.shape
        x = volume[:, :-1, :].flatten().astype(np.float64)
        y = volume[:, 1:, :].flatten().astype(np.float64)
    elif direction == 'diagonal':
        d, h, w = volume.shape
        x = volume[:, :-1, :-1].flatten().astype(np.float64)
        y = volume[:, 1:, 1:].flatten().astype(np.float64)
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    # Pearson correlation
    x_mean, y_mean = x.mean(), y.mean()
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    PSNR between original and reconstructed volume.
    For perfect decryption: returns np.inf (MSE = 0).
    For plaintext vs ciphertext: expect very low values (~8-12 dB).
    """
    mse = np.mean((original.astype(np.float64) - 
                   reconstructed.astype(np.float64))**2)
    if mse == 0:
        return np.inf
    return 10 * np.log10(255**2 / mse)


from skimage.metrics import structural_similarity as ssim_skimage

def compute_ssim_3d(original: np.ndarray, encrypted: np.ndarray) -> float:
    """
    Compute average SSIM across all slices of a 3D volume.
    For encrypted vs original: expect SSIM ≈ 0 (no structural similarity).
    For decrypted vs original: expect SSIM = 1.0 (identical).
    """
    ssim_values = []
    for d in range(original.shape[0]):
        s = ssim_skimage(
            original[d].astype(np.float64), 
            encrypted[d].astype(np.float64),
            data_range=255.0
        )
        ssim_values.append(s)
    return float(np.mean(ssim_values))


def key_sensitivity_test(volume: np.ndarray, 
                           encrypt_func, 
                           n_bit_flips: int = 5) -> dict:
    """
    Test if flipping a single bit in the key produces completely different ciphertext.
    A secure system: NPCR ≈ 99.6%, UACI ≈ 33.4% even for 1-bit key change.
    """
    # Encrypt with original key
    cipher_original, sig_orig, pk, sk, _ = encrypt_func(volume)
    
    results = []
    for _ in range(n_bit_flips):
        # Flip one random bit in the secret key
        sk_modified = bytearray(sk)
        byte_idx = np.random.randint(0, len(sk_modified))
        bit_idx = np.random.randint(0, 8)
        sk_modified[byte_idx] ^= (1 << bit_idx)
        
        # Encrypt with modified key
        cipher_modified, _, _, _, _ = encrypt_func(volume, sk_override=bytes(sk_modified))
        
        npcr = compute_npcr(cipher_original, cipher_modified)
        uaci = compute_uaci(cipher_original, cipher_modified)
        results.append({'npcr': npcr, 'uaci': uaci})
    
    return {
        'mean_npcr': np.mean([r['npcr'] for r in results]),
        'mean_uaci': np.mean([r['uaci'] for r in results]),
        'std_npcr': np.std([r['npcr'] for r in results])
    }


def compute_histogram_uniformity(volume: np.ndarray) -> float:
    """
    Chi-squared test for histogram uniformity.
    For truly uniform distribution (perfect encryption), chi-sq → 0.
    """
    from scipy.stats import chisquare
    hist, _ = np.histogram(volume.flatten(), bins=256, range=(0, 256))
    expected = np.full(256, volume.size / 256)
    chi2, p_value = chisquare(hist, expected)
    return {'chi2': chi2, 'p_value': p_value}