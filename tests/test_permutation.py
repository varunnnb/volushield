import numpy as np

from src.encryption.permutation import (
    permute_volume_3d,
    inverse_permute_volume_3d
)

from src.chaos.hyperchaotic import MemristorHyperchaos


def test_permutation_inverse():

    volume = np.random.randint(
        0,256,(32,32,32),dtype=np.uint8
    )

    chaos = MemristorHyperchaos()

    seq_x,seq_y,_,_ = chaos.generate_sequence(
        0.2,0.3,0.4,0.5,
        n_samples=volume.size
    )

    permuted = permute_volume_3d(volume,seq_x,seq_y)

    restored = inverse_permute_volume_3d(
        permuted,
        seq_x,
        seq_y
    )

    assert np.array_equal(volume,restored)