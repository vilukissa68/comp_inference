#!/usr/bin/env python3
import torch
from nvidia import nvcomp
import cupy as cp
import numpy as np


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
    print(
        f"Input shape: {np_array.shape}, dtype: {np_array.dtype}, size: {np_array.nbytes / 1024:.1f} KB"
    )

    # Convert NumPy array to nvcomp array (this transfers to GPU internally)
    nvarr_h = nvcomp.as_array(np_array.tobytes())
    nvarr_d = nvarr_h.cuda()

    # Encode
    compressed = codec.encode(nvarr_d)

    ratio = compressed.buffer_size / np_array.nbytes
    print(f"Compressed size: {compressed.buffer_size / 1024:.1f} KB")
    print(f"Compression ratio: {ratio:.4f}")

    # Decode
    decoded = codec.decode(compressed)

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


def main():
    # Prepare data on CPU
    random_data = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)
    low_entropy_data = np.random.randint(0, 4, (1024, 1024), dtype=np.uint8)

    compress_and_report("ANS", random_data)
    compress_and_report("ANS", low_entropy_data)


if __name__ == "__main__":
    main()
