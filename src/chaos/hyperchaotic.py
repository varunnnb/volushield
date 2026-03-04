import numpy as np
from typing import Tuple


class MemristorHyperchaos:
    """
    4D memristor hyperchaotic system.
    """

    def __init__(self,
                 a: float = 35.0,
                 b: float = 3.0,
                 c: float = 12.0,
                 d: float = 7.0,
                 k: float = 0.5,
                 e: float = 0.5):

        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.k = k
        self.e = e


    def derivatives(self, state: np.ndarray) -> np.ndarray:
        """
        Compute dx/dt, dy/dt, dz/dt, dw/dt
        """

        x, y, z, w = state

        dx = self.a * (y - x) + self.d * x * z
        dy = self.k * x - x * z + self.c * y
        dz = x * y - self.b * z
        dw = -x + self.e * w

        return np.array([dx, dy, dz, dw], dtype=np.float64)


    def rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """
        One RK4 integration step
        """

        k1 = self.derivatives(state)
        k2 = self.derivatives(state + 0.5 * dt * k1)
        k3 = self.derivatives(state + 0.5 * dt * k2)
        k4 = self.derivatives(state + dt * k3)

        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


    def generate_sequence(self,
                          x0: float,
                          y0: float,
                          z0: float,
                          w0: float,
                          n_samples: int,
                          dt: float = 0.0001,
                          transient: int = 5000) -> Tuple[np.ndarray, ...]:
        """
        Generate chaotic sequences
        """

        state = np.array([x0, y0, z0, w0], dtype=np.float64)

        # discard transient
        for _ in range(transient):
            state = self.rk4_step(state, dt)

        sequences = np.zeros((n_samples, 4), dtype=np.float64)

        for i in range(n_samples):
            state = self.rk4_step(state, dt)
            sequences[i] = state

        return (
            sequences[:,0],
            sequences[:,1],
            sequences[:,2],
            sequences[:,3]
        )
    

