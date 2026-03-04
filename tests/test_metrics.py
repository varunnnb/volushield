import numpy as np

from src.metrics.security import compute_entropy


def test_entropy_range():

    volume = np.random.randint(
        0,256,(32,32,32),dtype=np.uint8
    )

    entropy = compute_entropy(volume)

    assert 7 <= entropy <= 8