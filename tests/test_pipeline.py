import numpy as np

from src.encryption.pipeline import encrypt_volume
from src.decryption.pipeline import decrypt_volume


def test_full_pipeline():

    volume = np.random.randint(
        0,256,(32,32,32),dtype=np.uint8
    )

    cipher,signature,pk,sk,hashv = encrypt_volume(volume)

    recovered = decrypt_volume(
        cipher,
        signature,
        pk,
        sk,
        hashv
    )

    assert np.array_equal(volume,recovered)