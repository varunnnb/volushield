import numpy as np

from src.encryption.diffusion import encrypt_diffusion
from src.decryption.inv_diffusion import (
    inverse_forward_diffusion,
    inverse_backward_diffusion
)

from src.chaos.hyperchaotic import MemristorHyperchaos
from src.chaos.sequence_gen import chaotic_to_keystream


def test_diffusion_inverse():

    volume = np.random.randint(
        0,256,(16,16,16),dtype=np.uint8
    )

    chaos = MemristorHyperchaos()

    _,_,seq_z,seq_w = chaos.generate_sequence(
        0.2,0.3,0.4,0.5,
        n_samples=volume.size
    )

    iv = 123

    encrypted = encrypt_diffusion(
        volume,
        seq_z,
        seq_w,
        iv
    )

    flat = encrypted.flatten()

    ks_backward = chaotic_to_keystream(
        seq_w,
        flat.size
    )

    ks_forward = chaotic_to_keystream(
        seq_z,
        flat.size
    )

    after_inv_back = inverse_backward_diffusion(
        flat,
        ks_backward,
        iv
    )

    restored = inverse_forward_diffusion(
        after_inv_back,
        ks_forward,
        iv
    )

    restored = restored.reshape(volume.shape)

    assert np.array_equal(volume,restored)