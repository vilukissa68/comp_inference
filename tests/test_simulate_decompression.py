#!/usr/bin/env python3

import numpy as np
import time

from comp_inference import _core

GIGS = 5
SIZE = int(GIGS * 1024**3)
R = 0.66


def generate_array_uint8(size):
    """Generate a random array of uint8."""
    return (np.random.rand(size) * 127).astype(np.uint8)


def test_simulate_decompression():
    """Test the simulate_decompression function."""
    # Create sample input data
    start_time = time.time()
    size_reduced = int(SIZE * R)
    input_data = generate_array_uint8(size_reduced)
    input_data = input_data[:size_reduced]
    print(f"Data generation time: {time.time() - start_time:.2f} seconds")

    # Call the simulate_decompression function
    output_data = _core.simulate_decompression(input_data, SIZE)
    print(f"First 10 elements of input data: {input_data[:10]}")
    print(f"Last 10 elements of input data: {input_data[-10:]}")
    print(f"First 10 elements of output data: {output_data[:10]}")
    print(f"Last 10 elements of output data: {output_data[-10:]}")
    print(f"Output data size: {output_data.size}")

    return True


if __name__ == "__main__":
    if test_simulate_decompression():
        print("test_simulate_decompression completed successfully.")
