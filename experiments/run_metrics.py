import numpy as np

from src.encryption.pipeline import encrypt_volume

from src.metrics.security import (
    compute_entropy,
    compute_npcr,
    compute_uaci,
    compute_correlation,
    key_sensitivity_test,
    compute_histogram_uniformity
)

volume = np.load("data/processed/brats_vol64.npy")

cipher1, sig, pk, sk, hash1 = encrypt_volume(volume)

volume2 = volume.copy()
volume2[0,0,0] ^= 1

cipher2, _, _, _, _ = encrypt_volume(volume2)

print("Entropy:", compute_entropy(cipher1))

print("NPCR:", compute_npcr(cipher1,cipher2))

print("UACI:", compute_uaci(cipher1,cipher2))

print("Correlation H:", compute_correlation(cipher1,"horizontal"))

print("Correlation V:", compute_correlation(cipher1,"vertical"))

print(key_sensitivity_test(volume, encrypt_volume))

print(compute_histogram_uniformity(cipher1))