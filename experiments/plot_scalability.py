import matplotlib.pyplot as plt
import numpy as np

from src.encryption.pipeline import encrypt_volume
from src.metrics.performance import scalability_test


results = scalability_test(
    encrypt_volume,
    sizes=[
        (64,64,64),
        (96,96,96),
        (128,128,128)
    ]
)

voxels = [r["total_voxels"] for r in results]
times = [r["mean_seconds"] for r in results]

plt.figure(figsize=(6,4))

plt.plot(voxels,times,marker="o")

plt.xlabel("Total Voxels")
plt.ylabel("Encryption Time (seconds)")
plt.title("Scalability of Volume Encryption")

plt.grid(True)

plt.show()