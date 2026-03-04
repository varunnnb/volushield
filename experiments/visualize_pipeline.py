import numpy as np
import matplotlib.pyplot as plt

from src.encryption.pipeline import encrypt_volume
from src.decryption.pipeline import decrypt_volume


volume = np.load("data/processed/brats_vol64.npy")

cipher,signature,pk,sk,hashv = encrypt_volume(volume)

recovered = decrypt_volume(
    cipher,
    signature,
    pk,
    sk,
    hashv
)

slice_idx = volume.shape[0]//2

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(volume[slice_idx],cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cipher[slice_idx],cmap="gray")
plt.title("Encrypted")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(recovered[slice_idx],cmap="gray")
plt.title("Decrypted")
plt.axis("off")

plt.tight_layout()
plt.show()