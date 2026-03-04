import numpy as np
from src.chaos.hyperchaotic import MemristorHyperchaos


def test_chaos_determinism():

    chaos = MemristorHyperchaos()

    seq1 = chaos.generate_sequence(
        0.2,0.3,0.4,0.5,
        n_samples=1000
    )

    seq2 = chaos.generate_sequence(
        0.2,0.3,0.4,0.5,
        n_samples=1000
    )

    for a,b in zip(seq1,seq2):
        assert np.allclose(a,b)