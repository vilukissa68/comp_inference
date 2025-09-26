#!/usr/bin/env python3

import numpy as np
import time

# Import your module - adjust the import path as needed
try:
    from comp_inference import _core
except ImportError:
    print("Error: Could not import _core module.")
    print("Make sure the module is in your Python path or in the current directory.")
    exit(1)


def test_vector_add():
    """Test the CUDA vector addition function."""
    # Create sample input vectors
    size = 1000000
    a = np.random.rand(size).astype(np.float32)
    b = np.random.rand(size).astype(np.float32)

    # Expected result (CPU computation)
    expected = a + b

    # Call CUDA implementation
    start_time = time.time()
    result = _core.vector_add(a, b)
    cuda_time = time.time() - start_time

    # Verify results
    is_correct = np.allclose(result, expected, rtol=1e-5, atol=1e-5)

    # Print results
    print(f"Test {'PASSED' if is_correct else 'FAILED'}")
    print(f"CUDA computation time: {cuda_time:.6f} seconds")
    print(f"Input size: {size} elements")

    # Additional validation
    if not is_correct:
        max_diff = np.max(np.abs(result - expected))
        print(f"Maximum difference: {max_diff}")
        print(f"First few results: {result[:5]} vs expected: {expected[:5]}")

    return is_correct


def benchmark_comparison():
    """Compare CUDA vs CPU performance for different sizes."""
    sizes = [10000, 100000, 1000000, 10000000]

    print("\nBenchmarking CUDA vs CPU performance:")
    print(f"{'Size':<12} {'CUDA (ms)':<12} {'CPU (ms)':<12} {'Speedup':<10}")
    print("-" * 45)

    for size in sizes:
        a = np.random.rand(size).astype(np.float32)
        b = np.random.rand(size).astype(np.float32)

        # CPU timing
        start = time.time()
        cpu_result = a + b
        cpu_time = (time.time() - start) * 1000  # ms

        # GPU timing
        start = time.time()
        gpu_result = _core.vector_add(a, b)
        gpu_time = (time.time() - start) * 1000  # ms

        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")

        # Print results
        print(f"{size:<12} {gpu_time:<12.2f} {cpu_time:<12.2f} {speedup:<10.2f}x")


if __name__ == "__main__":
    # Run basic test
    test_result = test_vector_add()

    if test_result:
        # If basic test passes, run benchmark
        benchmark_comparison()
    else:
        print("\nBasic test failed. Skipping benchmarks.")
