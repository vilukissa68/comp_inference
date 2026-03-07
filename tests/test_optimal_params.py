#!/usr/bin/env python3

# import torch
# import triton
# import os
# from safetensors import safe_open
# from huggingface_hub import snapshot_download

# from comp_inference import (
#     load_compressed_model_with_auto_model,
#     pack_and_save_tensors,
#     save_rans_model_package,
#     save_rans_model_gguf,
#     rans_compress_qkv_fused,
#     rans_compress_gate_up_fused,
#     rans_compress_module_weight,
#     rans_decompress_module_weight,
#     reconstruct_from_exp_and_mantissa,
#     fused_rans_linear_triton,
#     fused_rans_linear_transposed_triton,
#     rans_decomp_triton,
#     uninterleave_mantissas,
#     rans_decomp_triton_tiled,
#     fused_rans_embedding_triton,
#     triton_matmul,
# )

# WARMUP_RUNS = 100
# EVAL_RUNS = 50


# def get_real_layer_weight(repo_id, target_tensor="model.layers.0.mlp.down_proj.weight"):
#     """Downloads the model (cached) and surgically extracts a single tensor to prevent RAM OOM."""
#     print(f"📥 Downloading/Verifying full model '{repo_id}' via HuggingFace Cache...")
#     print("   (Don't worry, it only downloads once and caches locally!)")

#     # Download all safetensor shards (ignores index files entirely to prevent 404s)
#     try:
#         model_path = snapshot_download(
#             repo_id=repo_id, allow_patterns=["*.safetensors"], local_files_only=False
#         )
#     except Exception as e:
#         raise RuntimeError(
#             f"Failed to download model. If the model is gated, run 'huggingface-cli login' in your terminal. Error: {e}"
#         )

#     print(f"🔍 Scanning downloaded shards in {model_path} for {target_tensor}...")

#     # Scan all downloaded safetensor shards
#     for filename in os.listdir(model_path):
#         if filename.endswith(".safetensors"):
#             filepath = os.path.join(model_path, filename)

#             # safe_open allows us to read the keys WITHOUT loading the 30GB file into RAM
#             with safe_open(filepath, framework="pt", device="cpu") as f:
#                 if target_tensor in f.keys():
#                     # We found the shard! Extract only this tensor directly to the GPU
#                     weight = f.get_tensor(target_tensor).cuda().to(torch.bfloat16)
#                     print(f"✅ Successfully extracted {target_tensor} from {filename}!")

#                     # Transpose from HF's [Out, In] to Triton's [In, Out]
#                     return weight.t().contiguous()

#     raise ValueError(
#         f"❌ Tensor '{target_tensor}' not found in any of the downloaded shards."
#     )


# # 6. Academic Microbenchmarking (Explicit CUDA Events)
# def profile_function(func):
#     # 1. Warmup (Wake up GPU clocks and fill L2 Cache)
#     for _ in range(WARMUP_RUNS):
#         func()
#     torch.cuda.synchronize()

#     # 2. Setup precise hardware timers
#     start_events = [torch.cuda.Event(enable_timing=True) for _ in range(EVAL_RUNS)]
#     end_events = [torch.cuda.Event(enable_timing=True) for _ in range(EVAL_RUNS)]

#     # 3. Execution Loop
#     func_result = None
#     for i in range(EVAL_RUNS):
#         start_events[i].record()
#         func_result = func()
#         end_events[i].record()

#     # 4. Wait for the GPU to finish all queued operations
#     torch.cuda.synchronize()

#     # 5. Extract times and calculate the average
#     times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
#     avg_time_ms = sum(times_ms) / EVAL_RUNS
#     print(func_result)
#     return avg_time_ms, func_result


# def test_module_integration(model_path, tile_height=1024, tile_width=32, batch_size=1):
#     # 1. Load Real Weights
#     try:
#         weights_kn = get_real_layer_weight(model_path)
#     except Exception as e:
#         print(f"⚠️ Using synthetic data (Fallback): {e}")
#         weights_kn = torch.randn((3072, 1024), dtype=torch.bfloat16, device="cuda")

#     K, N = weights_kn.shape
#     M = batch_size
#     x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")

#     # 2. Compression Setup
#     module = torch.nn.Module()
#     module.weight = torch.nn.Parameter(weights_kn.clone().contiguous())
#     module.bias = None

#     rans_compress_module_weight(
#         module, tile_height=tile_height, tile_width=tile_width, transpose_weight=False
#     )

#     module.cuda()
#     module.exponent_compressed_weight = module.exponent_compressed_weight.cuda()
#     module.mantissa_raw = module.mantissa_raw.cuda()
#     module.exponent_states = module.exponent_states.cuda()
#     module.exponent_tables = module.exponent_tables.cuda()
#     module.exponent_slot_map = module.exponent_slot_map.cuda()
#     module.exponent_tile_offsets = module.exponent_tile_offsets.cuda()
#     module.exponent_tile_max_lens = module.exponent_tile_max_lens.cuda()

#     # 3. Calculate Compression Ratios
#     total_stream_bytes = module.exponent_compressed_weight.numel()
#     nonzero_in_stream = torch.count_nonzero(module.exponent_compressed_weight).item()

#     # Note: This is an estimation. Some valid rANS compressed bytes might naturally be 0,
#     # but the vast majority of zeros here will be tiling padding.
#     padding_overhead = total_stream_bytes - nonzero_in_stream

#     original_bytes = weights_kn.numel() * 2  # bfloat16 = 2 bytes
#     compressed_bytes = (
#         total_stream_bytes * 1  # uint8
#         + module.mantissa_raw.numel() * 1  # uint8
#         + module.exponent_states.numel() * 4  # uint32
#         + module.exponent_tables.numel() * 4  # uint32
#         + module.exponent_slot_map.numel() * 1  # uint32
#         + module.exponent_tile_offsets.numel() * 4  # uint32
#         + module.exponent_tile_max_lens.numel() * 4  # uint32
#     )

#     # Reverted to: compressed / original (e.g., 0.65x)
#     compression_ratio = compressed_bytes / original_bytes

#     # Ideal ratio if all padding zeros were removed
#     ideal_compressed_bytes = compressed_bytes - padding_overhead
#     ideal_ratio = ideal_compressed_bytes / original_bytes

#     # 4. Define Execution Wrappers for Benchmarking
#     def run_baseline():
#         return torch.matmul(x, weights_kn)

#     def run_triton_baseline():
#         return triton_matmul(x, weights_kn)

#     def run_fused():
#         return fused_rans_linear_triton(
#             x=x,
#             compressed_data=module.exponent_compressed_weight,
#             initial_states=module.exponent_states,
#             tables=module.exponent_tables,
#             slot_map=module.exponent_slot_map,
#             weight_shape=(K, N),
#             tile_offsets=module.exponent_tile_offsets,
#             tile_max_lens=module.exponent_tile_max_lens,
#             tile_k=tile_height,
#             tile_n=tile_width,
#             mantissas=module.mantissa_raw,
#             accum_block_size=tile_width,
#         )

#     def run_unfused():
#         decomp_exp = rans_decomp_triton_tiled(
#             compressed_streams=module.exponent_compressed_weight,
#             initial_states=module.exponent_states,
#             tables=module.exponent_tables,
#             slot_map=module.exponent_slot_map,
#             output_shape=(K, N),
#             tile_offsets=module.exponent_tile_offsets,
#             tile_max_lens=module.exponent_tile_max_lens,
#             tile_k=tile_height,
#             tile_n=tile_width,
#         )
#         decomp_man = uninterleave_mantissas(
#             module.mantissa_raw, K, N, TILE_K=tile_height, TILE_N=tile_width
#         )
#         reassembled_weight = reconstruct_from_exp_and_mantissa(
#             decomp_exp, decomp_man, dtype=torch.bfloat16
#         )
#         return torch.matmul(x, reassembled_weight)

#     # 5. Correctness Verification
#     ms_baseline, out_base = profile_function(run_baseline)
#     ms_triton_baseline, out_triton = profile_function(run_triton_baseline)
#     ms_fused, out_fused = profile_function(run_fused)
#     ms_unfused, out_unfused = profile_function(run_unfused)

#     is_triton_valid = torch.allclose(out_base, out_triton, rtol=1e-2, atol=1e-2)
#     is_valid_fused = torch.allclose(out_base, out_fused, rtol=1e-2, atol=1e-2)
#     is_valid_unfused = torch.allclose(out_base, out_unfused, rtol=1e-2, atol=1e-2)
#     max_error_triton = torch.max(torch.abs(out_base - out_triton)).item()
#     max_error_fused = torch.max(torch.abs(out_base - out_fused)).item()
#     max_error_unfused = torch.max(torch.abs(out_base - out_unfused)).item()

#     return {
#         "ratio": compression_ratio,
#         "ideal_ratio": ideal_ratio,
#         "ms_baseline": ms_baseline,
#         "ms_triton_baseline": ms_triton_baseline,
#         "is_triton_valid": is_triton_valid,
#         "is_valid_fused": is_valid_fused,
#         "is_valid_unfused": is_valid_unfused,
#         "ms_fused": ms_fused,
#         "ms_unfused": ms_unfused,
#         "max_error_triton": max_error_triton,
#         "max_error_fused": max_error_fused,
#         "max_error_unfused": max_error_unfused,
#         "orig_mb": original_bytes / (1024**2),
#         "comp_mb": compressed_bytes / (1024**2),
#     }


# if __name__ == "__main__":
#     # Point this to your local Qwen or Llama folder
#     MODEL_PATH = "Qwen/Qwen3-14B"

#     heights = [64, 128, 256, 512, 1024, 2048]
#     widths = [32, 64, 128]

#     # M=1 simulates decoding, M=2048 simulates prefill
#     BATCH_SIZE = 1

#     results = []
#     print(f"\n🚀 Starting rANS Ablation Study | Batch Size (M): {BATCH_SIZE}")
#     print("-" * 125)

#     for h in heights:
#         for w in widths:
#             res = test_module_integration(
#                 MODEL_PATH, tile_height=h, tile_width=w, batch_size=BATCH_SIZE
#             )
#             results.append({"H": h, "W": w, **res})

#             # Print inline so you don't have to wait for the whole suite to finish
#             status = "✅" if res["is_valid_fused"] else "❌"
#             max_error_info = (
#                 f" | Max Error: {res['max_error_fused']:.4f}"
#                 if not res["is_valid_fused"]
#                 else ""
#             )
#             print(
#                 f"{status} Tile [{h:4d}x{w:3d}] | Ratio: {res['ratio']:.2f}x | Ideal: {res['ideal_ratio']:.2f}x | Size: {res['comp_mb']:.1f}MB | Base: {res['ms_baseline']:.3f}ms | Unfused: {res['ms_unfused']:.3f}ms | Fused: {res['ms_fused']:.3f}ms"
#             )

#     # Final Academic Markdown Table
#     print("\n\n📊 FINAL ABLATION RESULTS")
#     print(
#         "| Tile Size (H x W) | Comp. Ratio | Ideal Ratio | Fused Valid | Baseline (ms) | Baseline Triton (ms) | Unfused (ms) | Fused (ms) |"
#     )
#     print(
#         "|-------------------|-------------|-------------|-------------|---------------|----------------------|--------------|------------|"
#     )
#     for r in results:
#         valid = (
#             "Pass"
#             if r["is_valid_fused"]
#             else f"Fail (Max Err: {r['max_error_fused']:.4f})"
#         )
#         print(
#             f"| {r['H']:^4d} x {r['W']:^4d}     | {r['ratio']:^9.2f}x | {r['ideal_ratio']:^9.2f}x | {valid:^11s} | {r['ms_baseline']:^13.3f}| {r['ms_triton_baseline']:^20.3f} | {r['ms_unfused']:^12.3f} | {r['ms_fused']:^10.3f} |"
#         )

import torch
import triton
import os
import matplotlib.pyplot as plt
from safetensors import safe_open
from huggingface_hub import snapshot_download
import numpy as np

from comp_inference import (
    load_compressed_model_with_auto_model,
    pack_and_save_tensors,
    save_rans_model_package,
    save_rans_model_gguf,
    rans_compress_qkv_fused,
    rans_compress_gate_up_fused,
    rans_compress_module_weight,
    rans_decompress_module_weight,
    reconstruct_from_exp_and_mantissa,
    fused_rans_linear_triton,
    fused_rans_linear_transposed_triton,
    rans_decomp_triton,
    uninterleave_mantissas,
    rans_decomp_triton_tiled,
    fused_rans_embedding_triton,
    triton_matmul,
)

# 1. Increased for rigorous stability
WARMUP_RUNS = 100
EVAL_RUNS = 100


def get_real_layer_weight(repo_id, target_tensor="model.layers.0.mlp.down_proj.weight"):
    """Downloads the model (cached) and surgically extracts a single tensor to prevent RAM OOM."""
    print(f"📥 Downloading/Verifying full model '{repo_id}' via HuggingFace Cache...")
    try:
        model_path = snapshot_download(
            repo_id=repo_id, allow_patterns=["*.safetensors"], local_files_only=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")

    for filename in os.listdir(model_path):
        if filename.endswith(".safetensors"):
            filepath = os.path.join(model_path, filename)
            with safe_open(filepath, framework="pt", device="cpu") as f:
                if target_tensor in f.keys():
                    weight = f.get_tensor(target_tensor).cuda().to(torch.bfloat16)
                    return weight.t().contiguous()

    raise ValueError(
        f"ERROR: Tensor '{target_tensor}' not found in any of the downloaded shards."
    )


def profile_function(func):
    """Executes stable microbenchmarking returning the median time to drop OS noise anomalies."""
    # 1. Warmup (Wake up GPU clocks and fill L2 Cache)
    for _ in range(WARMUP_RUNS):
        func()
    torch.cuda.synchronize()

    # 2. Setup precise hardware timers
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(EVAL_RUNS)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(EVAL_RUNS)]

    # 3. Execution Loop
    for i in range(EVAL_RUNS):
        start_events[i].record()
        func()
        end_events[i].record()

    # 4. Wait for the GPU to finish all queued operations
    torch.cuda.synchronize()

    # 5. Extract times and calculate the median (more stable than average)
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    median_time_ms = np.median(times_ms)
    return median_time_ms


def test_module_integration(model_path, tile_height=1024, tile_width=32, batch_size=1):
    # 1. Load Real Weights
    try:
        weights_kn = get_real_layer_weight(model_path)
    except Exception as e:
        weights_kn = torch.randn((3072, 1024), dtype=torch.bfloat16, device="cuda")

    K, N = weights_kn.shape
    M = batch_size
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")

    # 2. Compression Setup
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(weights_kn.clone().contiguous())
    module.bias = None

    rans_compress_module_weight(
        module, tile_height=tile_height, tile_width=tile_width, transpose_weight=False
    )

    # --- MEMORY PREPARATION ---
    # A: ONBOARD (CUDA)
    exp_comp_cuda = module.exponent_compressed_weight.cuda()
    man_raw_cuda = module.mantissa_raw.cuda()
    exp_states_cuda = module.exponent_states.cuda()
    exp_tables_cuda = module.exponent_tables.cuda()
    exp_slot_map_cuda = module.exponent_slot_map.cuda()
    exp_tile_offsets_cuda = module.exponent_tile_offsets.cuda()
    exp_tile_max_lens_cuda = module.exponent_tile_max_lens.cuda()

    # B: OFFLOADED (Pinned CPU Memory for max PCIe bandwidth)
    exp_comp_cpu = module.exponent_compressed_weight.cpu().pin_memory()
    man_raw_cpu = module.mantissa_raw.cpu().pin_memory()
    exp_states_cpu = module.exponent_states.cpu().pin_memory()
    exp_tables_cpu = module.exponent_tables.cpu().pin_memory()
    exp_slot_map_cpu = module.exponent_slot_map.cpu().pin_memory()
    exp_tile_offsets_cpu = module.exponent_tile_offsets.cpu().pin_memory()
    exp_tile_max_lens_cpu = module.exponent_tile_max_lens.cpu().pin_memory()

    # 3. Calculate Compression Ratios
    original_bytes = weights_kn.numel() * 2
    compressed_bytes = (
        exp_comp_cuda.numel() * 1
        + man_raw_cuda.numel() * 1
        + exp_states_cuda.numel() * 4
        + exp_tables_cuda.numel() * 4
        + exp_slot_map_cuda.numel() * 1
        + exp_tile_offsets_cuda.numel() * 4
        + exp_tile_max_lens_cuda.numel() * 4
    )
    compression_ratio = compressed_bytes / original_bytes

    # 4. Correctness Validation (Run once, off the clock)
    out_base = torch.matmul(x, weights_kn)
    out_fused = fused_rans_linear_triton(
        x=x,
        compressed_data=exp_comp_cuda,
        initial_states=exp_states_cuda,
        tables=exp_tables_cuda,
        slot_map=exp_slot_map_cuda,
        weight_shape=(K, N),
        tile_offsets=exp_tile_offsets_cuda,
        tile_max_lens=exp_tile_max_lens_cuda,
        tile_k=tile_height,
        tile_n=tile_width,
        mantissas=man_raw_cuda,
        accum_block_size=tile_width,
    )
    is_valid_fused = torch.allclose(out_base, out_fused, rtol=1e-2, atol=1e-2)
    max_error_fused = torch.max(torch.abs(out_base - out_fused)).item()

    # 5. Define Execution Wrappers for Profiling
    def run_baseline():
        return torch.matmul(x, weights_kn)

    def run_fused_onboard():
        return fused_rans_linear_triton(
            x=x,
            compressed_data=exp_comp_cuda,
            initial_states=exp_states_cuda,
            tables=exp_tables_cuda,
            slot_map=exp_slot_map_cuda,
            weight_shape=(K, N),
            tile_offsets=exp_tile_offsets_cuda,
            tile_max_lens=exp_tile_max_lens_cuda,
            tile_k=tile_height,
            tile_n=tile_width,
            mantissas=man_raw_cuda,
            accum_block_size=tile_width,
        )

    def run_fused_offload():
        # Simulates PCIe transfer bottleneck from pinned memory + execution
        c_comp = exp_comp_cpu.to("cuda", non_blocking=True)
        c_states = exp_states_cpu.to("cuda", non_blocking=True)
        c_tables = exp_tables_cpu.to("cuda", non_blocking=True)
        c_slots = exp_slot_map_cpu.to("cuda", non_blocking=True)
        c_offsets = exp_tile_offsets_cpu.to("cuda", non_blocking=True)
        c_lens = exp_tile_max_lens_cpu.to("cuda", non_blocking=True)
        c_man = man_raw_cpu.to("cuda", non_blocking=True)

        return fused_rans_linear_triton(
            x=x,
            compressed_data=c_comp,
            initial_states=c_states,
            tables=c_tables,
            slot_map=c_slots,
            weight_shape=(K, N),
            tile_offsets=c_offsets,
            tile_max_lens=c_lens,
            tile_k=tile_height,
            tile_n=tile_width,
            mantissas=c_man,
            accum_block_size=tile_width,
        )

    # 6. Execute Profiling
    ms_baseline = profile_function(run_baseline)
    ms_fused_onboard = profile_function(run_fused_onboard)
    ms_fused_offload = profile_function(run_fused_offload)

    return {
        "ratio": compression_ratio,
        "is_valid_fused": is_valid_fused,
        "max_error_fused": max_error_fused,
        "ms_baseline": ms_baseline,
        "ms_fused_onboard": ms_fused_onboard,
        "ms_fused_offload": ms_fused_offload,
    }


def plot_pareto_frontier(results):
    """Plots a Pareto frontier for Compression Ratio vs Latency."""
    # Extract data
    onboard_pts = [
        (r["ratio"], r["ms_fused_onboard"], f"{r['H']}x{r['W']}")
        for r in results
        if r["is_valid_fused"]
    ]
    offload_pts = [
        (r["ratio"], r["ms_fused_offload"], f"{r['H']}x{r['W']}")
        for r in results
        if r["is_valid_fused"]
    ]

    if not onboard_pts:
        print("No valid runs to plot.")
        return

    def get_pareto_optimal(points):
        # Sort by X (Compression Ratio) ascending
        sorted_pts = sorted(points, key=lambda x: x[0])
        pareto_front = []
        min_y = float("inf")
        for pt in sorted_pts:
            if pt[1] < min_y:
                pareto_front.append(pt)
                min_y = pt[1]
        return pareto_front

    pareto_onboard = get_pareto_optimal(onboard_pts)
    pareto_offload = get_pareto_optimal(offload_pts)

    plt.figure(figsize=(10, 6))

    # Plot Scatter
    plt.scatter(
        [p[0] for p in onboard_pts],
        [p[1] for p in onboard_pts],
        color="blue",
        alpha=0.5,
        label="Onboard (VRAM)",
    )
    plt.scatter(
        [p[0] for p in offload_pts],
        [p[1] for p in offload_pts],
        color="red",
        alpha=0.5,
        label="Offload (PCIe -> VRAM)",
    )

    # Plot Pareto Lines
    plt.plot(
        [p[0] for p in pareto_onboard],
        [p[1] for p in pareto_onboard],
        color="blue",
        linestyle="--",
        label="Onboard Pareto Front",
    )
    plt.plot(
        [p[0] for p in pareto_offload],
        [p[1] for p in pareto_offload],
        color="red",
        linestyle="--",
        label="Offload Pareto Front",
    )

    # Annotate Pareto Points
    for pt in pareto_onboard:
        plt.annotate(
            pt[2],
            (pt[0], pt[1]),
            textcoords="offset points",
            xytext=(0, -15),
            ha="center",
            fontsize=8,
            color="blue",
        )
    for pt in pareto_offload:
        plt.annotate(
            pt[2],
            (pt[0], pt[1]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
            color="red",
        )

    # Draw Baseline
    baseline_ms = np.median([r["ms_baseline"] for r in results])
    plt.axhline(
        y=baseline_ms,
        color="green",
        linestyle=":",
        label=f"Baseline PyTorch ({baseline_ms:.3f} ms)",
    )

    plt.title("rANS Microbenchmark: Compression vs. Latency Pareto Frontier")
    plt.xlabel("Compression Ratio (Lower is Better)")
    plt.ylabel("Latency (ms, Lower is Better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("rans_pareto_frontier.png", dpi=300)
    print("\n Saved Pareto Frontier plot to 'rans_pareto_frontier.png'")


if __name__ == "__main__":
    MODEL_PATH = "Qwen/Qwen3-14B"
    heights = [64, 128, 256, 512, 1024, 2048]
    widths = [32, 64, 128]
    BATCH_SIZE = 1

    results = []
    print(f"\n Starting rANS Ablation Study | Batch Size (M): {BATCH_SIZE}")
    print("-" * 135)
    print(
        f"| {'Tile (H x W)':^14} | {'Valid':^7} | {'Ratio':^8} | {'Baseline (ms)':^15} | {'Onboard (ms)':^14} | {'Offloaded (ms)':^16} |"
    )
    print("-" * 135)

    for h in heights:
        for w in widths:
            res = test_module_integration(
                MODEL_PATH, tile_height=h, tile_width=w, batch_size=BATCH_SIZE
            )
            res["H"] = h
            res["W"] = w
            results.append(res)

            status = (
                "OK" if res["is_valid_fused"] else f"FAIL({res['max_error_fused']:.2f})"
            )
            print(
                f"| {h:>4d} x {w:<4d}   | {status:^7} | {res['ratio']:^7.2f}x | {res['ms_baseline']:^15.3f} | {res['ms_fused_onboard']:^14.3f} | {res['ms_fused_offload']:^16.3f} |"
            )

    plot_pareto_frontier(results)
