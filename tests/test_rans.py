import numpy as np
import time
import sys
import os
import torch

import comp_inference.ccore as ccore

def print_pass(msg):
    print(f"\033[92m[PASS] {msg}\033[0m")

def print_fail(msg):
    print(f"\033[91m[FAIL] {msg}\033[0m")


# Generate probabilities 1/n^skew
def generate_zipf_data(size, skew=2.0):
    print(f"Generating {size/1024/1024:.2f} MB of data (skew={skew})...")
    x = np.arange(1, 257).astype(np.float64)
    weights = 1.0 / (x ** skew)
    weights /= weights.sum()
    data = np.random.choice(np.arange(256), size=size, p=weights)
    return data.astype(np.uint8)

def prepare_rans_tables(data):
    counts = np.bincount(data, minlength=256)
    
    target_sum = 4096
    freqs = np.floor(counts / counts.sum() * target_sum).astype(np.int32)
    freqs[freqs == 0] = 1
    
    current_sum = freqs.sum()
    diff = target_sum - current_sum
    
    if diff > 0:
        freqs[np.argmax(freqs)] += diff
    elif diff < 0:
        while diff < 0:
            idx = np.argmax(freqs)
            amount = min(freqs[idx] - 1, abs(diff))
            freqs[idx] -= amount
            diff += amount

    freqs = freqs.astype(np.uint16)
    assert freqs.sum() == 4096

    cdf = np.zeros(256, dtype=np.uint16)
    running_sum = 0
    for i in range(256):
        cdf[i] = running_sum
        running_sum += freqs[i]

    return freqs, cdf

def benchmark(size_mb=256, iterations=5):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    size_bytes = size_mb * 1024 * 1024
    
    data = generate_zipf_data(size_bytes, skew=4.0)
    freqs, cdf = prepare_rans_tables(data)

    # Setup pinned memory
    print("Setting up pinned memory...")
    pinned_input = ccore.allocate_pinned_memory(size_bytes)
    pinned_output = ccore.allocate_pinned_memory(size_bytes)
    pinned_stream_staging = ccore.allocate_pinned_memory(int(len(data) * 1.5))

    np.copyto(pinned_input, data)

    print("-" * 50)
    print(f"Testing CUDA rANS Extension")
    print(f"Input Size: {size_mb} MB")
    print("-" * 50)

    try:
        manager = ccore.RansManager(size_bytes)
        print("RansManager initialized (GPU Workspace allocated).")
    except Exception as e:
        print_fail(f"Failed to init manager: {e}")
        return

    print("Warming up GPU...")
    res = manager.compress(pinned_input, freqs, cdf)
    pinned_stream = ccore.allocate_pinned_memory(res.stream.size)

    print("Compression warm-up done.")
    _ = manager.decompress(pinned_stream, res.states, res.output_sizes, 
                           res.num_streams, len(data), freqs, cdf)

    print("Successfully completed warm-up runs.")
    print("Starting benchmark...")
    compress_times = []
    decompress_times = []
    kernel_times = []
    
    encoded_result = None
    decoded_data = None

    for i in range(iterations):
        t0 = time.time()
        encoded_result = manager.compress(pinned_input, freqs, cdf)
        t1 = time.time()
        compress_times.append((t1 - t0) * 1000)

        stream_len = len(encoded_result.stream)
        # Create a slice view of our staging buffer
        current_stream_view = pinned_stream_staging[:stream_len]
        np.copyto(current_stream_view, encoded_result.stream)

        t0 = time.time()
        
        kernel_ms = manager.decompress_into(
            current_stream_view,         
            encoded_result.states,
            encoded_result.output_sizes,
            encoded_result.num_streams,
            freqs,
            cdf,
            pinned_output               
        )
        
        t1 = time.time()
        decompress_times.append((t1 - t0) * 1000)
        kernel_times.append(kernel_ms)
        
        print(f"Iter {i+1}: Enc={compress_times[-1]:.2f}ms, Dec={decompress_times[-1]:.2f}ms (Kernel: {kernel_ms:.2f}ms)")

    # Verification
    if np.array_equal(data, pinned_output):
        print_pass("Verification Successful")
    else:
        print_fail("Verification FAILED")
        return

    avg_enc_time = np.mean(compress_times)
    avg_dec_time = np.mean(decompress_times)
    avg_kernel_time = np.mean(kernel_times)
    compressed_size = len(encoded_result.stream)
    ratio = size_bytes / compressed_size
    enc_throughput = (size_bytes / 1024**3) / (avg_enc_time / 1000)
    dec_throughput = (size_bytes / 1024**3) / (avg_dec_time / 1000)
    kernel_throughput = (size_bytes / 1024**3) / (avg_kernel_time / 1000)

    print("-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"Ratio: {ratio:.2f}:1")
    print(f"Enc Throughput: {enc_throughput:.2f} GiB/s")
    print(f"Dec Throughput: {dec_throughput:.2f} GiB/s")
    print(f"Kernel Throughput: {kernel_throughput:.2f} GiB/s")
    print("-" * 50)

if __name__ == "__main__":
    benchmark(size_mb=256, iterations=10)
