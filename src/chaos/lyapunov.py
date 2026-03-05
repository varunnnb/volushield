import numpy as np
from src.chaos.hyperchaotic import MemristorHyperchaos


def compute_lyapunov(system=None,
                     steps=50000,
                     dt=0.0001):

    if system is None:
        system = MemristorHyperchaos()

    # initial state
    state = np.array([0.2,0.3,0.4,0.5],dtype=np.float64)

    eps = 1e-8

    state_perturbed = state + eps

    sum_log = 0.0

    for _ in range(steps):

        state = system.rk4_step(state,dt)
        state_perturbed = system.rk4_step(state_perturbed,dt)

        delta = np.linalg.norm(state_perturbed-state)

        if delta == 0:
            continue

        sum_log += np.log(abs(delta/eps))

        state_perturbed = state + eps*(state_perturbed-state)/delta

    lyapunov = sum_log/(steps*dt)

    return lyapunov