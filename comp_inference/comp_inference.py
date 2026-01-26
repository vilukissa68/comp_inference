import torch
import torch.nn as nn
import numpy as np
import os
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

    exponent = exponent# - bias
    mantissa = (sign << mantissa_bits) | mantissa
    return exponent.to(torch.uint8), (mantissa).to(torch.uint8)


@torch.compile(dynamic=True)  # Use torch.compile to fuse this into one kernel!
def reconstruct_from_exp_and_mantissa(
    exponent, mantissa_and_sign, dtype=torch.bfloat16
):
    # Ensure inputs are int32 for bitwise ops
    exponent = exponent.to(torch.int32)
    exponent = exponent
    mantissa_and_sign = mantissa_and_sign.to(torch.int32)

    # Hardcoded for BF16 (8 exp, 7 man)
    # BF16 Layout: [1 Sign | 8 Exp | 7 Man]

    # Extract sign from your packed format [Sign | Mantissa]
    sign = (mantissa_and_sign >> 7) & 0x1
    mantissa = mantissa_and_sign & 0x7F

    # Re-bias exponent (You subtracted bias during compression)
    # BF16 bias is 127
    exponent_raw = exponent

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

    return torch.from_numpy(freqs).to(torch.uint16), torch.from_numpy(cdf).to(torch.uint16)


def rans_compress_module_weight_bf16(module: nn.Module, skip_mantissa=True) -> None:
    if not hasattr(module, "weight"):
        return
    if module.weight.dtype != torch.bfloat16:
        print(f"Skipping {module.weight.dtype}, expected bfloat16.")
        return

    # 1. Save original metadata needed for reconstruction
    # We save this BEFORE deleting the weight
    module.input_shape = list(module.weight.shape)
    module.expanded_size = module.weight.numel()

    # 2. Extract
    exponent, mantissa = extract_exp_and_mantissa(module.weight)

    # 3. Compute LUTs (Assuming get_rans_lut returns Tensors)
    module.exponent_freqs, module.exponent_cdf = get_rans_lut(exponent.to(torch.uint8))
    module.mantissa_freqs, module.mantissa_cdf = get_rans_lut(mantissa.to(torch.uint8))

    # 4. Allocate Managers
    # Note: Allocating full size is safe but conservative.
    bytes_to_allocate = module.expanded_size
    exp_memory_manager = ccore.RansManager(bytes_to_allocate)
    
    # --- EXPONENT COMPRESSION ---
    print("Compressing Exponent...")
    exponent_compression = exp_memory_manager.compress(
        exponent, # Tensor (uint8)
        module.exponent_freqs,
        module.exponent_cdf
    )   

    if exponent_compression.success:
        # Direct assignment (Assuming C++ returns a Tensor now)
        module.exponent_compressed_weight = exponent_compression.stream
        module.exponent_states = exponent_compression.states
        module.exponent_output_sizes = exponent_compression.output_sizes
        module.exponent_num_streams = exponent_compression.num_streams
        # We don't strictly need stream_size, we can use len(tensor)
    else:
        # Fallback: Store raw
        module.exponent_raw = exponent 
        print("Exponent compression failed. Storing raw.")

    # --- MANTISSA COMPRESSION ---
    mantissa_compression = None
    if not skip_mantissa:
        print("Compressing Mantissa...")
        mantissa_memory_manager = ccore.RansManager(bytes_to_allocate)
        mantissa_compression = mantissa_memory_manager.compress(
            mantissa, # Tensor (uint8)
            module.mantissa_freqs,
            module.mantissa_cdf,
        )

    if mantissa_compression and mantissa_compression.success:
        module.mantissa_compressed_weight = mantissa_compression.stream
        module.mantissa_states = mantissa_compression.states
        module.mantissa_output_sizes = mantissa_compression.output_sizes
        module.mantissa_num_streams = mantissa_compression.num_streams
    else:
        # Fallback: Store raw
        # If skip_mantissa=True, we end up here too.
        module.mantissa_raw = mantissa 

    # Cleanup
    module.compressed = "rans_bfloat16"
    del module.weight
    print(f"Compression complete. Original: {module.expanded_size*2} bytes.")

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
        print(f"Skipping {module.compressed}, expected rans_bfloat16.")
        return

    if hasattr(module, "exponent_compressed_weight"):
        # Create output buffer
        raw_exponent = ccore.allocate_pinned_tensor(module.expanded_size)
        
        # Instantiate Manager
        manager_exp = ccore.RansManager(len(module.exponent_compressed_weight))

        # Decompress
        _ = manager_exp.decompress_into(
            module.exponent_compressed_weight, # Stream
            module.exponent_states.to(torch.int32), # Ensure Int32 for C++
            module.exponent_output_sizes.to(torch.int32),
            module.exponent_num_streams,
            module.exponent_freqs,
            module.exponent_cdf,
            raw_exponent # Destination
        )
    else:
        # Fallback: Check for raw attribute
        if hasattr(module, "exponent_raw"):
            raw_exponent = module.exponent_raw.flatten()
        elif hasattr(module, "exponent"): # Legacy check
            raw_exponent = module.exponent.flatten()
        else:
            raise RuntimeError(f"Missing exponent data for {module}")

    if hasattr(module, "mantissa_compressed_weight"):
        # Create output buffer
        raw_mantissa = ccore.allocate_pinned_tensor(module.expanded_size)
        
        # Instantiate Manager
        manager_man = ccore.RansManager(module.mantissa_stream_size)

        # Decompress
        _ = manager_man.decompress_into(
            module.mantissa_compressed_weight,
            module.mantissa_states.to(torch.int32),
            module.mantissa_output_sizes.to(torch.int32),
            module.mantissa_num_streams,
            module.mantissa_freqs,
            module.mantissa_cdf,
            raw_mantissa # Destination
        )
    else:
        # Fallback
        if hasattr(module, "mantissa_raw"):
            raw_mantissa = module.mantissa_raw.flatten()
        elif hasattr(module, "mantissa"):
            raw_mantissa = module.mantissa.flatten()
        else:
            raise RuntimeError(f"Missing mantissa data for {module}")

    # Reconstruct tensor (Exp Tensor + Mantissa Tensor -> Tensor)
    decompressed_flat = reconstruct_from_exp_and_mantissa(
        raw_exponent, 
        raw_mantissa, 
        dtype=torch.bfloat16
    )

    # Determine Shape
    if hasattr(module, "input_shape"):
        shape = module.input_shape
    elif hasattr(module, "exponent_shape"):
        shape = module.exponent_shape
    else:
        # Dangerous fallback, but prevents crash if shape lost
        shape = (module.expanded_size,)
        print(f"Warning: No shape found for {module}, keeping flat.")

    # Assign final weight
    module.weight = decompressed_flat.reshape(shape)

    print(f"Success: Decompressed. Shape: {module.weight.shape}")

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
