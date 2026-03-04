import numpy as np
import matplotlib.pyplot as plt

vol = np.load("data/processed/brats_vol64.npy")
enc = np.load("data/encrypted/encrypted_vol.npy")

slice_idx = 32

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(vol[slice_idx], cmap="gray")

plt.subplot(1,2,2)
plt.title("Encrypted")
plt.imshow(enc[slice_idx], cmap="gray")

plt.show()