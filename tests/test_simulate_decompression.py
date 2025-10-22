#!/usr/bin/env python3

import numpy as np
import time

from comp_inference import _core

GIGS = 5.0
SIZE = int(GIGS * 1024**3)  # Bytes
R = 0.66
# Approximate memory usage during operation
APPROXIMATE_MEMORY = int(SIZE * (2 + R))
MAX_MEMORY = 11200 * 1024**2  # 11.2 GB
REPEAT = 3
CLOCK_RATE = 1480 * 1e6  # 1480 MHz in Hz
# if APPROXIMATE_MEMORY > MAX_MEMORY:
#    raise MemoryError(
#        f"Approximate memory usage {APPROXIMATE_MEMORY / 1024**3:.2f} GB exceeds the limit of {MAX_MEMORY / 1024**3:.2f} GB."
#    )


def generate_array_uint8(size):
    """Generate a random array of uint8."""
    return (np.random.rand(size) * 127).astype(np.uint8)


def test_simulate_decompression():
    """Test the simulate_decompression function."""
    decomp_times = []
    for _ in range(REPEAT):
        pair = []
        for r in [R, 1.0]:
            start_time = time.time()
            size_reduced = int(SIZE * r)
            input_data = generate_array_uint8(size_reduced)
            input_data = input_data[:size_reduced]
            print(f"Data generation time: {time.time() - start_time:.2f} seconds")

            # Call the simulate_decompression function
            output_data, decomp_time_ms = _core.simulate_decompression(input_data, SIZE)
            # print(f"First 10 elements of input data: {input_data[:10]}")
            # print(f"Last 10 elements of input data: {input_data[-10:]}")
            # print(f"First 10 elements of output data: {output_data[:10]}")
            # print(f"Last 10 elements of output data: {output_data[-10:]}")
            # print(f"Output data size: {output_data.size}")
            print(f"Decompression time: {decomp_time_ms / 1000:.4f} seconds")
            print("-" * 80)
            pair.append(decomp_time_ms)

        decomp_times.append(pair)

    reduced_times = [t[0] for t in decomp_times]
    full_times = [t[1] for t in decomp_times]

    avg_reduced_time = np.mean(reduced_times)
    avg_full_time = np.mean(full_times)
    print(f"Average time for reduced data: {avg_reduced_time / 1000:.4f} seconds")
    print(f"Average time for full data: {avg_full_time / 1000:.4f} seconds")

    difference = avg_full_time - avg_reduced_time
    print(f"Difference in average times: {difference / 1000:.4f} seconds")

    cycles_saved = difference * CLOCK_RATE / 1000
    print(f"Estimated cycles saved: {cycles_saved:.2f} cycles")

    return True


if __name__ == "__main__":
    if test_simulate_decompression():
        print("test_simulate_decompression completed successfully.")
