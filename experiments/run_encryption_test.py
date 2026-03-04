import numpy as np

from src.encryption.pipeline import encrypt_volume
from src.decryption.pipeline import decrypt_volume
from src.metrics.security import compute_psnr

INPUT = "data/processed/brats_vol64.npy"

def main():

    print("Loading volume...")
    volume = np.load(INPUT)

    print("Encrypting...")

    cipher, signature, pk, sk, volume_hash = encrypt_volume(volume)

    print("Decrypting...")

    recovered = decrypt_volume(
        cipher,
        signature,
        pk,
        sk,
        volume_hash
    )

    print("Checking equality...")

    if np.array_equal(volume, recovered):
        print("✓ Perfect recovery")
    else:
        print("❌ Decryption mismatch")

    psnr = compute_psnr(volume, recovered)

    print("PSNR:", psnr)

if __name__ == "__main__":
    main()