import numpy as np

enc = np.load("data/encrypted/encrypted_vol.npy")

print("Entropy:", -np.sum(
    np.bincount(enc.flatten(), minlength=256) / enc.size *
    np.log2(np.bincount(enc.flatten(), minlength=256) / enc.size + 1e-12)
))