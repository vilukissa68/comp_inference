#!/usr/bin/env python3
import torch
import numpy as np
import struct

SYMBOL_DTYPE = np.uint8
VOCAB_SIZE = 256

PROB_BITS = 12  # 2^12 = 4096
PROB_SCALE = 1 << PROB_BITS
MASK = PROB_SCALE - 1

# IO / Renormalization Emission
# - 8  = Emit Bytes (Standard). Stream is uint8[].
# - 16 = Emit Shorts. Stream is uint16[]. Requires larger RANS_L.
IO_BITS = 8
IO_MASK = (1 << IO_BITS) - 1

# Renormalization range (Lower bound = 2^16, Upper bound = 2^24 approx)
RANS_L = 1 << 16

# Check for invalid config immediately
if RANS_L < (1 << IO_BITS):
    raise ValueError(f"RANS_L ({RANS_L}) is too small for IO_BITS ({IO_BITS})")


def generate_tables(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy().flatten().astype(np.uint8)
    else:
        data = np.array(data).flatten().astype(np.uint8)

    counts = np.bincount(data.flatten(), minlength=VOCAB_SIZE)
    total_count = counts.sum()

    freqs = np.zeros(256, dtype=np.int32)
    scale = PROB_SCALE / total_count

    # Fill frequencies with floor of 1
    for i in range(VOCAB_SIZE):
        if counts[i] > 0:
            val = int(counts[i] * scale)
            freqs[i] = max(1, val)  # Floor to 1

    # Scale frequencies to exactly PROB_SCALE
    unbalanced_sum = freqs.sum()
    diff = PROB_SCALE - unbalanced_sum

    if diff > 0:
        # Sum is less than PROB_SCALE, add to the largest freq
        largest_idx = np.argmax(freqs)
        freqs[largest_idx] += diff
    elif diff < 0:
        # Sum is more than PROB_SCALE, subtract from the largest freq
        sorted_idxs = np.argsort(freqs)[::-1]
        to_subtract = -diff

        for i in sorted_idxs:
            if to_subtract == 0:
                break
            if freqs[i] > 1:
                # Ensure floor of 1
                can_take = freqs[i] - 1
                take = min(can_take, to_subtract)
                freqs[i] -= take
                to_subtract -= take
        if to_subtract > 0:
            raise ValueError(
                "Could not balance frequencies to PROB_SCALE, not enough room to subtract."
            )

    assert (
        freqs.sum() == PROB_SCALE
    ), "Frequencies do not sum to PROB_SCALE, even after balancing."

    # Build CDF
    cdf = np.zeros(VOCAB_SIZE + 1, dtype=np.int32)
    np.cumsum(freqs, out=cdf[1:])

    # Build inverse mapping symbol table
    slot_to_sym = np.zeros(PROB_SCALE, dtype=np.uint8)
    for s in range(VOCAB_SIZE):
        start = cdf[s]
        end = cdf[s + 1]
        slot_to_sym[start:end] = s

    return {
        "cdf": cdf.astype(np.uint16),  # [VOCAB_SIZE + 1]
        "freqs": freqs.astype(np.uint16),  # [VOCAB_SIZE]
        "symbol": slot_to_sym,  # [PROB_SCALE]
    }


def rans_compress(data, tables):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    data = data.flatten().astype(SYMBOL_DTYPE)

    stream = []
    state = RANS_L

    cdf = tables["cdf"]
    freqs = tables["freqs"]

    for symbol in reversed(data):
        start = cdf[symbol]
        freq = freqs[symbol]

        x = state

        # Renormalization
        x_max = ((RANS_L >> PROB_BITS) << IO_BITS) * freq
        if x >= x_max:
            stream.append(x & IO_MASK)
            x >>= IO_BITS

        # Update state / encode symbol
        state = ((x // freq) << PROB_BITS) + (x % freq) + start

    stream = list(reversed(stream))
    return stream, state


def rans_decompress(state, stream, tables, num_symbols):
    cdf = tables["cdf"]
    freqs = tables["freqs"]
    symbol_table = tables["symbol"]

    decompressed = []

    for _ in range(num_symbols):
        slot = state & MASK
        symbol = symbol_table[slot]
        decompressed.append(symbol)

        f = freqs[symbol]
        start = cdf[symbol]

        state = f * (state >> PROB_BITS) + (slot - start)

        # Renormalization
        while state < RANS_L and len(stream) > 0:
            val = stream.pop(0)
            state = (state << IO_BITS) | val

    return decompressed


if __name__ == "__main__":
    N = 128 * 1024  # Number of symbols
    original_tensor = torch.randint(0, 256, (N,), dtype=torch.uint8)
    original_tensor = original_tensor // 4
    print("Original tensor:", original_tensor)

    # Make data slightly compressible (skewed distribution)
    # e.g., zero out upper bits to lower entropy
    # original_tensor = original_tensor // 2

    print(f"Original size: {original_tensor.numel()} bytes")

    # Construct CDF and frequencies
    tables = generate_tables(original_tensor)

    # Compress
    stream, state = rans_compress(original_tensor, tables)
    stream_bytes = len(stream) * 2  # Each entry is 2 bytes
    stream_with_tables = (
        stream_bytes
        + 4
        + len(tables["cdf"]) * 2
        + len(tables["freqs"]) * 2
        + len(tables["symbol"])
    )

    print(f"Compressed size: {stream_bytes + 4} bytes (stream + final state)")
    print(f"Compression ratio: {original_tensor.numel() / (stream_bytes + 4):.2f}")
    print(f"Total size with tables: {stream_with_tables} bytes")
    print(
        f"Total compression ratio with tables: {original_tensor.numel() / stream_with_tables:.2f}"
    )

    # Decompress
    decompressed_tensor = rans_decompress(state, stream, tables, N)

    # Verify
    for n in range(original_tensor.numel()):
        if original_tensor[n].item() != decompressed_tensor[n].item():
            print(
                f"Mismatch at index {n}: original {original_tensor[n].item()} vs decoded {decompressed_tensor[n].item()}"
            )
    print("Original:", original_tensor[:10])
    print("Decoded :", decompressed_tensor[:10])
