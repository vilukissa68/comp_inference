import torch
import torch.nn as nn
import numpy as np
from . import ccore
from typing import Tuple


def extract_exp_and_mantissa(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract unbiased exponent and mantissa+sign bits from float16/32/bfloat16 tensor.
    Mantissa is raw IEEE mantissa bits; sign is in its own bit.
    """

    if tensor.dtype != torch.bfloat16:
        raise ValueError(
            "Only bfloat16 dtype is supported for exponent/mantissa extraction."
        )

    exp_bits, mantissa_bits, bias, storage_bits = (8, 7, 127, 16)

    raw = tensor.view(torch.uint16).to(torch.int32)

    # Extract fields
    sign = (raw >> (exp_bits + mantissa_bits)) & 0x1
    exponent = (raw >> mantissa_bits) & ((1 << exp_bits) - 1)
    mantissa = raw & ((1 << mantissa_bits) - 1)

    exponent = exponent - bias
    mantissa = (sign << mantissa_bits) | mantissa
    return exponent.to(torch.uint8), (mantissa).to(torch.uint8)


@torch.compile(dynamic=True)  # Use torch.compile to fuse this into one kernel!
def reconstruct_from_exp_and_mantissa(
    exponent, mantissa_and_sign, dtype=torch.bfloat16
):
    # Ensure inputs are int32 for bitwise ops
    exponent = exponent.to(torch.int32)
    mantissa_and_sign = mantissa_and_sign.to(torch.int32)

    # Hardcoded for BF16 (8 exp, 7 man)
    # BF16 Layout: [1 Sign | 8 Exp | 7 Man]

    # Extract sign from your packed format [Sign | Mantissa]
    sign = (mantissa_and_sign >> 7) & 0x1
    mantissa = mantissa_and_sign & 0x7F

    # Re-bias exponent (You subtracted bias during compression)
    # BF16 bias is 127
    exponent_raw = exponent + 127

    # Assemble bits: S << 15 | E << 7 | M
    bits = (sign << 15) | (exponent_raw << 7) | mantissa

    # View as BF16
    return bits.to(torch.uint16).view(dtype)


def get_rans_lut(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates Frequency and CDF tables for rANS (12-bit precision).
    Input: uint8 tensor
    Output: (freqs, cdf) as int32 tensors
    """
    # 1. Histogram
    data_np = data.cpu().numpy().flatten()
    counts = np.bincount(data_np, minlength=256)

    # 2. Normalize to sum exactly to 4096 (12 bits)
    TARGET_SUM = 4096

    # Safe normalization avoiding zero frequencies (rANS breaks with freq=0)
    # We add 1 to everything first to ensure non-zero, then normalize rest
    # Or simple approach: standard float norm -> floor -> fix zeros -> fix sum

    freqs = np.floor(counts / counts.sum() * TARGET_SUM).astype(np.int32)
    freqs[freqs == 0] = 1  # Prevent invalid rANS state

    # 3. Fix Rounding Errors to ensure sum == 4096
    current_sum = freqs.sum()
    diff = TARGET_SUM - current_sum

    if diff > 0:
        # Add to most frequent symbol
        freqs[np.argmax(freqs)] += diff
    elif diff < 0:
        # Remove from most frequent symbols (carefully)
        while diff < 0:
            idx = np.argmax(freqs)
            amount = min(freqs[idx] - 1, abs(diff))  # Keep at least 1
            freqs[idx] -= amount
            diff += amount

    assert freqs.sum() == TARGET_SUM

    # 4. Compute CDF
    cdf = np.zeros(256, dtype=np.int32)
    running_sum = 0
    for i in range(256):
        cdf[i] = running_sum
        running_sum += freqs[i]

    return torch.from_numpy(freqs), torch.from_numpy(cdf)


def rans_compress_module_weight_bf16(module: nn.Module) -> None:
    if not hasattr(module, "weight"):
        return
    if module.weight.dtype != torch.bfloat16:
        print(
            f"Module weight dtype is {module.weight.dtype}, expected bfloat16. Can't compress with bf16 method. Skipping."
        )
        return

    exponent, mantissa = extract_exp_and_mantissa(module.weight)
    print("Extracted exponent and mantissa from bf16 weights.")

    module.exponent_freqs, module.exponent_cdf = get_rans_lut(exponent.to(torch.uint8))
    print("Computed rANS LUTs for exponent.")
    module.mantissa_freqs, module.mantissa_cdf = get_rans_lut(mantissa.to(torch.uint8))
    print("Computed rANS LUTs for mantissa.")

    print(f"== Freqs ==")
    print(module.exponent_freqs)
    print(module.mantissa_freqs)
    print(f"== CDFs ==")
    print(module.exponent_cdf)
    print(module.mantissa_cdf)



    bytes_to_allocate = module.weight.numel() * 10
    print(f"Allocating RANS memory manager with {bytes_to_allocate} bytes.")
    exp_memory_manager = ccore.RansManager(bytes_to_allocate)
    mantissa_memory_manager = ccore.RansManager(bytes_to_allocate)
    print("RANS memory manager allocated.")

    print("Starting mantissa compression...")
    mantissa_compression = mantissa_memory_manager.compress(
        mantissa.cpu().numpy().astype(np.uint8),
        module.mantissa_cdf.numpy(),
        module.mantissa_freqs.numpy(),
    )
    print("Mantissa compression completed.")

    print("Starting exponent compression...")
    exponent_compression = exp_memory_manager.compress(
        exponent.cpu().numpy().astype(np.uint8),
        module.exponent_cdf.numpy(),
        module.exponent_freqs.numpy(),
    )
    print("Exponent compression completed.")

    if exponent_compression.success:
        print(
            f"Exponent compression successful: Compressed {module.weight.numel()*1} bytes to {len(exponent_compression.stream)} bytes."
        )
    else:
        print("Exponent compression failed.")

    if mantissa_compression.success:
        print(
            f"Mantissa compression successful: Compressed {module.weight.numel()*1} bytes to {len(mantissa_compression.stream)} bytes."
        )
    else:
        print("Mantissa compression failed.")

    module.compressed = "rans_bfloat16"

    module.exponent_compressed_weight = torch.from_numpy(exponent_compression.stream)
    module.exponent_states = exponent_compression.states
    module.exponent_num_streams = exponent_compression.num_streams
    module.exponent_stream_size = len(exponent_compression.stream)
    module.exponent_output_size = exponent_compression.output_size

    module.mantissa_compressed_weight = torch.from_numpy(mantissa_compression.stream)
    module.mantissa_states = mantissa_compression.states
    module.mantissa_num_streams = mantissa_compression.num_streams
    module.mantissa_stream_size = len(mantissa_compression.stream)
    module.mantissa_output_size = mantissa_compression.output_size

    module.expanded_size = module.weight.numel()

    # Delete original weight to save memory
    del module.weight
    return


def rans_compress_module_weight_int8(module: nn.Module) -> None:
    """
    Compress the module's weight using RANS and store it in compressed_weight.
    """
    if not hasattr(module, "weight"):
        return

    # Check weight dtype is uint8/int8 if not split mantissa and exponent
    if module.weight.dtype not in [torch.uint8, torch.int8]:
        if not hasattr(module, "cdf") or not hasattr(module, "freqs"):
            print("Module to compress does not have cdf or freqs. Computing them now.")
            cdf, freqs = get_rans_lut(module.weight.to(torch.uint8))
            module.cdf = cdf
            module.freqs = freqs

    bytes_to_allocate = module.weight.numel()
    memory_manager = ccore.RansManager(bytes_to_allocate)

    weight_np = module.weight.cpu().numpy()
    compression_result = memory_manager.compress(
        module.weight.numpy(), module.cdf.numpy(), module.freqs.numpy()
    )

    stream_size = len(compression_result.stream)
    compressed_weight = torch.from_numpy(compression_result.stream)
    num_streams = compression_result.num_streams
    starting_states = compression_result.states

    # Assign compressed attributes to module
    module.compressed = "rans_int8"
    module.compressed_weight = compressed_weight
    module.states = starting_states
    module.num_streams = num_streams
    module.stream_size = stream_size
    module.output_size = compression_result.output_size
    module.expanded_size = module.weight.numel()

    # Delete original weight to save memory
    del module.weight

    print("Module weight compressed using RANS.")


def rans_compress_module_weight(module: nn.Module) -> None:
    if not hasattr(module, "weight"):
        return
    if module.weight.dtype == torch.bfloat16:
        rans_compress_module_weight_bf16(module)
    elif module.weight.dtype in [torch.uint8, torch.int8]:
        rans_compress_module_weight_int8(module)
    else:
        print(
        f"Module weight dtype is {module.weight.dtype}, not supported for RANS compression. Skipping."
        )

def rans_decompress_module_weight_bf16(module: nn.Module) -> None:
    if not hasattr(module, "compressed"):
        print("Module is not compressed. Skipping decompression.")
        return
    if module.compressed != "rans_bfloat16":
        print(
            f"Module is compressed with {module.compressed}, not rans_bfloat16. Skipping."
        )
        return

    # Manager handles the decoding logic
    manager_exp = ccore.RansManager(module.exponent_stream_size)

    raw_exponent = ccore.allocate_pinned_memory(module.expanded_size)

    _ = manager_exp.decompress(
        raw_exponent,  # Output buffer (Destination)
        module.exponent_states,  # Initial states for parallel streams
        module.exponent_output_size,
        module.exponent_num_streams,
        module.expanded_size,
        module.exponent_freqs,
        module.exponent_cdf,
    )

    manager_man = ccore.RansManager(module.mantissa_stream_size)

    # Output buffer for mantissas
    raw_mantissa = ccore.allocate_pinned_memory(module.expanded_size)

    _ = manager_man.decompress(
        raw_mantissa,
        module.mantissa_states,
        module.mantissa_output_size,
        module.mantissa_num_streams,
        module.expanded_size,
        module.mantissa_freqs,
        module.mantissa_cdf,
    )

    decompressed_flat = reconstruct_from_exp_and_mantissa(
        raw_exponent, raw_mantissa, dtype=torch.bfloat16
    )

    if hasattr(module, "input_shape"):
        module.weight = decompressed_flat.reshape(module.input_shape)
    else:
        print("Warning: 'input_shape' not found in module. Weight will be flat.")
        module.weight = decompressed_flat

    print(f"Success: Decompressed bf16 weights. Shape: {module.weight.shape}")


def rans_decompress_module_weight_int8(module: nn.Module) -> None:
    """Decompress the module's weight using RANS and restore it to weight."""

    if not hasattr(module, "compressed"):
        print("Module is not compressed. Skipping decompression.")
        return

    # Check that module has necessary attributes
    if not all(
        hasattr(module, attr)
        for attr in ["weight", "states", "num_streams", "stream_size", "cdf", "freqs"]
    ):
        print("Module missing attributes for decompression. Skipping.")
        return

    # Reserve space for decompressed weight
    manager = ccore.RansManager(module.stream_size)
    # Allocate pinned memory for decompressed weight
    pinned_stream = ccore.allocate_pinned_memory(module.stream_size)

    _ = manager.decompress(
        pinned_stream,
        module.states,
        module.output_size,
        module.num_streams,
        module.expanded_size,
        module.freqs,
        module.cdf,
    )

    # Restore weight
    decompressed_weight = pinned_stream[: module.expanded_size].reshape(
        module.weight.shape
    )
    module.weight = decompressed_weight

    print("Module weight decompressed using RANS.")

def rans_decompress_module_weight(module: nn.Module) -> None:
    """
    Dispatcher: Checks the compression type and calls the correct decompressor.
    """
    if not hasattr(module, "compressed"):
        # This is normal for layers like LayerNorm or activation functions
        # that don't have weights or weren't compressed.
        return

    # Dispatch based on the flag set during compression
    if module.compressed == "rans_bfloat16":
        rans_decompress_module_weight_bf16(module)
    elif module.compressed == "rans_int8":
        rans_decompress_module_weight_int8(module)
    else:
        print(f"Unknown compression type: {module.compressed}")
        return

    # CRITICAL STEP FOR HUGGING FACE INFERENCE:
    # The decompression puts a raw Tensor into module.weight.
    # We must wrap it in nn.Parameter so the model treats it as a trainable/model weight
    # and not just a buffer.
    if not isinstance(module.weight, nn.Parameter):
        module.weight = nn.Parameter(module.weight, requires_grad=False)


