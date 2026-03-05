import numpy as np

from src.encryption.pipeline import encrypt_volume
from src.metrics.performance import (
    measure_encryption_time,
    measure_memory_usage
)

volume = np.load("data/processed/brats_vol64.npy")

print("Running performance benchmark...")

time_result = measure_encryption_time(
    encrypt_volume,
    volume,
    n_runs=5
)

print("\nEncryption timing:")
print(time_result)

mem_result = measure_memory_usage(
    encrypt_volume,
    volume
)

print("\nMemory usage:")
print(mem_result)