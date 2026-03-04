import numpy as np
import matplotlib.pyplot as plt

from src.encryption.pipeline import encrypt_volume


volume = np.load("data/processed/brats_vol64.npy")

cipher,_,_,_,_ = encrypt_volume(volume)

plt.figure(figsize=(8,5))

plt.hist(cipher.flatten(),bins=256)

plt.title("Ciphertext Histogram")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.show()