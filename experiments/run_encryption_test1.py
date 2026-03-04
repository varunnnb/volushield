import numpy as np

from src.encryption.pipeline1 import encrypt_volume


volume = np.load("data/processed/brats_vol64.npy")

encrypted = encrypt_volume(volume)

print("Original mean:", volume.mean())
print("Encrypted mean:", encrypted.mean())

np.save("data/encrypted/encrypted_vol.npy", encrypted)