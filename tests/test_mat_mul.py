#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from comp_inference import (
    fused_rans_linear_triton,
    rans_compress_module_weight,
)
from comp_inference.comp_inference import (
    reconstruct_from_exp_and_mantissa,
    uninterleave_mantissas,
)
from comp_inference.rans_triton import rans_decomp_triton_tiled


def create_coordinate_weights(K, N):
    # We use small integers that are exact in BF16
    # Value = RowIndex + (ColIndex / 1000)
    # e.g., Row 5, Col 32 = 5.032
    k_indices = torch.arange(K, device="cuda").view(K, 1)
    n_indices = torch.arange(N, device="cuda").view(1, N)
    weights = k_indices + (n_indices / 1000.0)
    return weights.to(torch.bfloat16)


# def test_module_integration():
#     # K = Input features, N = Output features
#     K, N = 2048, 1024
#     M = 8192  # Batch size

#     # 1. Weights and Bias Initialization
#     # weights_golden = create_coordinate_weights(K, N)
#     weights_golden = torch.randn((K, N), dtype=torch.bfloat16, device="cuda")
#     weights_golden_copy = weights_golden.clone()

#     # Generate random bias for the N output features
#     bias_golden = torch.randn(N, dtype=torch.bfloat16, device="cuda")
#     bias_golden_copy = bias_golden.clone()
#     bias_golden = None

#     x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")

#     # 2. Golden Reference: F.linear(input, weight, bias)
#     # Note: weight is expected as [N, K], hence the transpose
#     result_golden = F.linear(x, weights_golden.t(), bias=None)
#     result_functional = torch.nn.functional.linear(x, weights_golden.t(), bias=None)

#     # 3. Compression
#     module = torch.nn.Module()
#     module.weight = torch.nn.Parameter(weights_golden_copy)
#     module.bias = torch.nn.Parameter(bias_golden_copy)

#     # This call populates the module with compressed metadata
#     rans_compress_module_weight(module)

#     # 4. Fused Triton Execution
#     # Ensure your fused_rans_linear_triton wrapper is updated to handle 'bias'
#     result_compressed = fused_rans_linear_triton(
#         x=x.cuda(),
#         compressed_data=module.exponent_compressed_weight.cuda(),
#         tile_offsets=module.exponent_tile_offsets.cuda(),
#         tile_max_lens=module.exponent_tile_max_lens.cuda(),
#         initial_states=module.exponent_states.cuda(),
#         mantissas=module.mantissa_raw.cuda(),
#         slot_map=module.exponent_slot_map.cuda(),
#         tables=module.exponent_tables.cuda(),
#         K=K,
#         N=N,
#         bias=None,  # Set to None if your Triton kernel doesn't support bias yet
#     )

#     # 5. Result Verification
#     print(f"M={M}, K={K}, N={N}")
#     print("Result Golden Sample:", result_golden[0, :5])
#     print("Result functional Sample:", result_functional[0, :5])
#     print("Result Compressed Sample:", result_compressed[0, :5])

#     print("Result Golden:", result_golden)
#     print("Result Functional:", result_functional)
#     print("Result Compressed:", result_compressed)

#     print("Result Golden Shape:", result_golden.shape)
#     print("Result Functional Shape:", result_functional.shape)
#     print("Result Compressed Shape:", result_compressed.shape)

#     # Tolerance check (f32 accumulation vs bf16 native)
#     if torch.allclose(result_golden, result_compressed, atol=1e-2, rtol=1e-2):
#         print("\033[92mTest passed: Results are equal (within tolerance).\033[0m")
#     else:
#         print("\033[91mTest failed: Results are NOT equal.\033[0m")
#         diff = result_golden - result_compressed
#         print("Max Absolute Difference:", torch.max(torch.abs(diff)).item())
#         print("Mean Absolute Difference:", torch.mean(torch.abs(diff)).item())

#     if torch.allclose(result_functional, result_compressed, atol=1e-2, rtol=1e-2):
#         print(
#             "\033[92mTest passed: Functional and Compressed results are equal (within tolerance).\033[0m"
#         )
#     else:
#         print(
#             "\033[91mTest failed: Functional and Compressed results are NOT equal.\033[0m"
#         )
#         diff = result_functional - result_compressed
#         print("Max Absolute Difference:", torch.max(torch.abs(diff)).item())
#         print("Mean Absolute Difference:", torch.mean(torch.abs(diff)).item())


def test_module_integration(tile_height=1024, tile_width=32):
    # K = Input features (reduction dim), N = Output features (parallel dim)
    # We use 2048 to test the "Height Stacked" (multiple K-tiles) logic
    K, N = 3072, 1024
    M = 8192  # Batch size

    # 1. Weights and Bias Initialization
    # We generate weights in (K, N) shape because that is the "Logical"
    # layout the Triton kernel reconstructs into w_tile [BLOCK_K, TILE_N]
    weights_kn = torch.randn((K, N), dtype=torch.bfloat16, device="cuda")
    bias_golden = torch.randn(N, dtype=torch.bfloat16, device="cuda")
    bias_golden = None

    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")

    # Ensure it's in (K, N) format for the reference calculation
    # 2. Reference Calculation
    # Since weights_kn is (K, N), we do standard matmul: (M, K) @ (K, N)
    # This is equivalent to F.linear(x, weights_kn.t(), bias)
    if bias_golden is not None:
        result_golden = torch.matmul(x, weights_kn) + bias_golden
    else:
        result_golden = torch.matmul(x, weights_kn)

    # 3. Compression Setup
    module = torch.nn.Module()
    # IMPORTANT: Assign weights in (K, N) format.
    # The rans_compress_module_weight utility will tile this into [Tiles_K, Tiles_N]
    module.weight = torch.nn.Parameter(weights_kn.clone().contiguous())
    if bias_golden is not None:
        module.bias = torch.nn.Parameter(bias_golden.clone()).cuda()
    else:
        module.bias = None

    # This call populates the module with:
    # .exponent_compressed_weight, .exponent_tile_offsets, .mantissa_raw, etc.
    rans_compress_module_weight(
        module, tile_height=tile_height, tile_width=tile_width, transpose_weight=False
    )

    # 4. Fused Triton Execution
    # Use the cleaned API. We pass K (features) and N (neurons).
    result_compressed = fused_rans_linear_triton(
        x=x,
        compressed_data=module.exponent_compressed_weight.cuda(),
        initial_states=module.exponent_states.cuda(),
        tables=module.exponent_tables.cuda(),
        slot_map=module.exponent_slot_map.cuda(),
        weight_shape=(K, N),  # Inform the kernel of the original weight shape
        tile_offsets=module.exponent_tile_offsets.cuda(),
        tile_max_lens=module.exponent_tile_max_lens.cuda(),
        tile_k=tile_height,  # Ensure this matches your tiling logic
        tile_n=tile_width,  # Ensure this matches your tiling logic
        mantissas=module.mantissa_raw.cuda(),
        bias=module.bias,
        accum_block_size=tile_width,
    )

    # Normal decomp
    decomp_exp = rans_decomp_triton_tiled(
        compressed_streams=module.exponent_compressed_weight.cuda(),
        initial_states=module.exponent_states.cuda(),
        tables=module.exponent_tables.cuda(),
        slot_map=module.exponent_slot_map.cuda(),
        output_shape=(K, N),
        tile_offsets=module.exponent_tile_offsets.cuda(),
        tile_max_lens=module.exponent_tile_max_lens.cuda(),
        tile_k=tile_height,
        tile_n=tile_width,
    )

    decomp_man = uninterleave_mantissas(
        module.mantissa_raw.cuda(), K, N, TILE_K=tile_height, TILE_N=tile_width
    )

    reassembled_weight = reconstruct_from_exp_and_mantissa(
        decomp_exp, decomp_man, dtype=torch.bfloat16
    )

    if bias_golden is not None:
        result_decomp = torch.matmul(x, reassembled_weight) + bias_golden
    else:
        result_decomp = torch.matmul(x, reassembled_weight)

    print("Result Golden Sample:", result_golden)
    print("Result Compressed Sample:", result_compressed)
    print("Result Decompressed Sample:", result_decomp)

    # 5. Verification
    print(f"\n[TEST] M={M}, K={K}, N={N}")

    # Check shape
    if result_golden.shape != result_compressed.shape:
        print(
            f"❌ Shape Mismatch: Golden {result_golden.shape} vs Compressed {result_compressed.shape}"
        )
        return

    if result_golden.shape != result_decomp.shape:
        print(
            f"❌ Shape Mismatch: Golden {result_golden.shape} vs Decompressed {result_decomp.shape}"
        )
        return

    # Accuracy check
    diff = torch.abs(result_golden - result_compressed)
    max_diff_fused = diff.max().item()
    mean_diff_fused = diff.mean().item()

    decomp_success = False
    fused_success = False

    if torch.allclose(result_golden, result_compressed, atol=1e-2, rtol=1e-2):
        print(
            f"✅ \033[92mFUSED PASSED\033[0m | Max Diff: {max_diff_fused:.6f} | Mean Diff: {mean_diff_fused:.6f}"
        )
        fused_success = True
    else:
        print(
            f"❌ \033[91mFUSED FAILED\033[0m | Max Diff: {max_diff_fused:.6f} | Mean Diff: {mean_diff_fused:.6f}"
        )

        # Debugging clues
        print("\nGolden Sample (Row 0):", result_golden[0, :5])
        print("Kernel Sample (Row 0):", result_compressed[0, :5])

    diff_decomp = torch.abs(result_golden - result_decomp)
    max_diff = diff_decomp.max().item()
    mean_diff = diff_decomp.mean().item()

    if torch.allclose(result_golden, result_decomp, atol=1e-2, rtol=1e-2):
        print(
            f"✅ \033[92mDECOMP PASSED\033[0m | Max Diff: {max_diff:.6f} | Mean Diff: {mean_diff:.6f}"
        )
        decomp_success = True
    else:
        print(
            f"❌ \033[91mDECOMP FAILED\033[0m | Max Diff: {max_diff:.6f} | Mean Diff: {mean_diff:.6f}"
        )

        # Debugging clues
        print("\nGolden Sample (Row 0):", result_golden[0, :5])
        print("Decomp Sample (Row 0):", result_decomp[0, :5])

    return {
        "decomp_max_diff": max_diff,
        "fused_max_diff": max_diff_fused,
        "decomp_mean_diff": mean_diff,
        "fused_mean_diff": mean_diff_fused,
        "decomp_success": decomp_success,
        "fused_success": fused_success,
    }


if __name__ == "__main__":
    heights = [32, 64, 128, 256, 512, 1024]
    widths = [16, 32, 64, 128]
    results = []
    for h in heights:
        for w in widths:
            print(f"\n=== Testing Tile Height: {h}, Tile Width: {w} ===")
            res = test_module_integration(tile_height=h, tile_width=w)
            results.append({"tile_height": h, "tile_width": w, **res})

    print("\n=== SUMMARY OF RESULTS ===")
    for r in results:
        print(
            f"Tile Height: {r['tile_height']}, Tile Width: {r['tile_width']} | "
            f"Fused Max Diff: {r['fused_max_diff']:.6f} | Decomp Max Diff: {r['decomp_max_diff']:.6f} | "
            f"Fused Mean Diff: {r['fused_mean_diff']:.6f} | Decomp Mean Diff: {r['decomp_mean_diff']:.6f} | "
            f"Fused Success: {r['fused_success']} | Decomp Success: {r['decomp_success']}"
        )
