import pandas as pd


# src/metrics/performance.py

import time
import psutil
import os
import numpy as np
from typing import Callable

def measure_encryption_time(encrypt_func: Callable, 
                              volume: np.ndarray,
                              n_runs: int = 5) -> dict:
    """
    Measure encryption time over multiple runs.
    Reports mean, std, and throughput (voxels/second).
    """
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        encrypt_func(volume)
        end = time.perf_counter()
        times.append(end - start)
    
    mean_time = np.mean(times)
    throughput = volume.size / mean_time  # voxels per second
    
    return {
        'mean_seconds': mean_time,
        'std_seconds': np.std(times),
        'throughput_voxels_per_sec': throughput,
        'volume_mb': volume.nbytes / (1024**2)
    }

def measure_memory_usage(func: Callable, *args) -> dict:
    """
    Measure peak memory usage during function execution.
    """
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024**2)  # MB
    
    result = func(*args)
    
    mem_after = process.memory_info().rss / (1024**2)  # MB
    
    return {
        'result': result,
        'memory_before_mb': mem_before,
        'memory_after_mb': mem_after,
        'peak_increase_mb': mem_after - mem_before
    }

def scalability_test(encrypt_func: Callable,
                      sizes: list = [(64,64,64), (128,128,128), 
                                     (256,256,128), (256,256,256)]) -> list:
    """
    Test encryption time vs volume size.
    Results go into a scalability table in the paper.
    """
    results = []
    for size in sizes:
        volume = np.random.randint(0, 256, size=size, dtype=np.uint8)
        timing = measure_encryption_time(encrypt_func, volume, n_runs=3)
        timing['size'] = size
        timing['total_voxels'] = np.prod(size)
        results.append(timing)
        print(f"Size {size}: {timing['mean_seconds']:.3f}s ± {timing['std_seconds']:.3f}s")
    return results



def save_performance_table(results, filepath="results/tables/performance.csv"):

    df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    df.to_csv(filepath,index=False)

    print("Saved performance results to",filepath)