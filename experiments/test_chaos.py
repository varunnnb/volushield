import numpy as np

from src.chaos.hyperchaotic import MemristorHyperchaos


N = 64*64*64

chaos = MemristorHyperchaos()

x, y, z, w = chaos.generate_sequence(
    x0=0.1,
    y0=0.2,
    z0=0.3,
    w0=0.4,
    n_samples=2*N
)

print("Generated samples:", len(x))
print("Example values:", x[:5])