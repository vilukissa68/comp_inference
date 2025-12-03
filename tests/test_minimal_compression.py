#!/usr/bin/env python3
import torch
from nvidia import nvcomp
import cupy as cp
import numpy as np
import argparse


codec = nvcomp.Codec(algorithm="Bitcomp")  # Use ZSTD, not ANS
x_rand = torch.randint(0, 256, (8 * 1024 * 1024,), dtype=torch.uint8, device="cuda")
x_zero = torch.zeros_like(x_rand)


for name, x in [("random", x_rand), ("zeros", x_zero)]:
    comp = codec.encode(nvcomp.as_array(x))
    y = torch.utils.dlpack.from_dlpack(comp.to_dlpack()).to("cpu")
    ratio = y.numel() / x.numel()
    print(f"{name}: ratio={ratio:.4f}")


def compress_and_report(codec_name, np_array):
    codec = nvcomp.Codec(algorithm=codec_name, uncomp_chunk_size=np_array.nbytes)
    print(f"\nTesting codec: {codec_name}")
    uncompressed_size_gb = np_array.nbytes / (1024**3)
    print(
        f"Input shape: {np_array.shape}, dtype: {np_array.dtype}, size: {uncompressed_size_gb:.3f} GB"
    )

    # Convert NumPy array to nvcomp array (this transfers to GPU internally)
    nvarr_h = nvcomp.as_array(np_array.tobytes())
    nvarr_d = nvarr_h.cuda()

    # Warm-up
    for _ in range(5):
        _ = codec.encode(nvarr_d)

    # Encode
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()
    compressed = codec.encode(nvarr_d)
    end_event.record()
    end_event.synchronize()
    compression_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    compression_time_s = compression_time_ms / 1000.0
    compression_bw = np_array.nbytes / compression_time_s / 1e9  # GB/s

    ratio = compressed.buffer_size / np_array.nbytes
    print(f"Compressed size: {compressed.buffer_size / (1024**2):.2f} MB")
    print(f"Compression ratio: {ratio:.4f}")
    print(f"Compression time: {compression_time_ms:.2f} ms")
    print(f"Compression Bandwidth: {compression_bw:.2f} GB/s")

    # Decode
    # Warm-up
    for _ in range(5):
        _ = codec.decode(compressed)

    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()
    start_event.record()
    decoded = codec.decode(compressed)
    end_event.record()
    end_event.synchronize()
    decompression_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    decompression_time_s = decompression_time_ms / 1000.0
    decompression_bw = np_array.nbytes / decompression_time_s / 1e9  # GB/s
    print(f"Decompression time: {decompression_time_ms:.2f} ms")
    print(f"Decompression Bandwidth: {decompression_bw:.2f} GB/s")

    # Move decoded data back to host
    decoded_h = decoded.cpu()

    # Convert to a NumPy array from the raw bytes
    decoded_bytes = bytes(decoded_h)  # nvcomp.Array supports buffer protocol
    decoded_np = np.frombuffer(decoded_bytes, dtype=np.uint8)

    # Reinterpret dtype and shape
    decoded_np = decoded_np.view(np_array.dtype).reshape(np_array.shape)

    # Verify correctness
    ok = np.array_equal(np_array.view(np.uint8), decoded_np.view(np.uint8))
    print("Decoded matches original:", ok)

    return ratio


def parse_size(size_str: str) -> int:
    """Parse a size string (e.g., '10MB', '1GB') into bytes."""
    size_str = size_str.strip().upper()
    if size_str.endswith("GB"):
        return int(size_str[:-2]) * (1024**3)
    if size_str.endswith("MB"):
        return int(size_str[:-2]) * (1024**2)
    if size_str.endswith("KB"):
        return int(size_str[:-2]) * 1024
    if size_str.endswith("B"):
        return int(size_str[:-1])
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size format: {size_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Test nvcomp compression with variable data sizes."
    )
    parser.add_argument(
        "--size",
        type=str,
        default="1MB",
        help="Size of the data to compress (e.g., '10MB', '1GB'). Default is 1MB.",
    )
    args = parser.parse_args()

    try:
        data_size_bytes = parse_size(args.size)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Prepare data on CPU
    print(f"Generating data of size: {args.size} ({data_size_bytes} bytes)")
    random_data = np.random.randint(0, 256, data_size_bytes, dtype=np.uint8)
    low_entropy_data = np.random.randint(0, 4, data_size_bytes, dtype=np.uint8)

    compress_and_report("ANS", random_data)
    compress_and_report("ANS", low_entropy_data)


if __name__ == "__main__":
    main()
