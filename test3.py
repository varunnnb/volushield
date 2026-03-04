import numpy as np

v = np.load("data/processed/brats_vol64.npy")

print(v.shape)
print(v.dtype)
print(v.min(), v.max())