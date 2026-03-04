import numpy as np
import matplotlib.pyplot as plt

enc = np.load("data/encrypted/encrypted_vol.npy")

plt.hist(enc.flatten(), bins=256)
plt.title("Ciphertext Histogram")
plt.show()