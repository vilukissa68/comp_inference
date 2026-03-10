#!/usr/bin/env python3
# import torch
# import triton
# import os
# import matplotlib.pyplot as plt
# from safetensors import safe_open
# from huggingface_hub import snapshot_download
# import numpy as np

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

# # 1. Increased for rigorous stability
# WARMUP_RUNS = 100
# EVAL_RUNS = 100


# def get_real_layer_weight(repo_id, target_tensor="model.layers.0.mlp.down_proj.weight"):
#     """Downloads the model (cached) and surgically extracts a single tensor to prevent RAM OOM."""
#     print(f"📥 Downloading/Verifying full model '{repo_id}' via HuggingFace Cache...")
#     try:
#         model_path = snapshot_download(
#             repo_id=repo_id, allow_patterns=["*.safetensors"], local_files_only=False
#         )
#     except Exception as e:
#         raise RuntimeError(f"Failed to download model: {e}")

#     for filename in os.listdir(model_path):
#         if filename.endswith(".safetensors"):
#             filepath = os.path.join(model_path, filename)
#             with safe_open(filepath, framework="pt", device="cpu") as f:
#                 if target_tensor in f.keys():
#                     weight = f.get_tensor(target_tensor).cuda().to(torch.bfloat16)
#                     return weight.t().contiguous()

#     raise ValueError(
#         f"ERROR: Tensor '{target_tensor}' not found in any of the downloaded shards."
#     )


# def profile_function(func):
#     """Executes stable microbenchmarking returning the median time to drop OS noise anomalies."""
#     # 1. Warmup (Wake up GPU clocks and fill L2 Cache)
#     for _ in range(WARMUP_RUNS):
#         func()
#     torch.cuda.synchronize()

#     # 2. Setup precise hardware timers
#     start_events = [torch.cuda.Event(enable_timing=True) for _ in range(EVAL_RUNS)]
#     end_events = [torch.cuda.Event(enable_timing=True) for _ in range(EVAL_RUNS)]

#     # 3. Execution Loop
#     for i in range(EVAL_RUNS):
#         start_events[i].record()
#         func()
#         end_events[i].record()

#     # 4. Wait for the GPU to finish all queued operations
#     torch.cuda.synchronize()

#     # 5. Extract times and calculate the median (more stable than average)
#     times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
#     median_time_ms = np.median(times_ms)
#     return median_time_ms


# def test_module_integration(model_path, tile_height=1024, tile_width=32, batch_size=1):
#     # 1. Load Real Weights
#     try:
#         weights_kn = get_real_layer_weight(model_path)
#     except Exception as e:
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

#     # --- MEMORY PREPARATION ---
#     # A: ONBOARD (CUDA)
#     exp_comp_cuda = module.exponent_compressed_weight.cuda()
#     man_raw_cuda = module.mantissa_raw.cuda()
#     exp_states_cuda = module.exponent_states.cuda()
#     exp_tables_cuda = module.exponent_tables.cuda()
#     exp_slot_map_cuda = module.exponent_slot_map.cuda()
#     exp_tile_offsets_cuda = module.exponent_tile_offsets.cuda()
#     exp_tile_max_lens_cuda = module.exponent_tile_max_lens.cuda()

#     # B: OFFLOADED (Pinned CPU Memory for max PCIe bandwidth)
#     exp_comp_cpu = module.exponent_compressed_weight.cpu().pin_memory()
#     man_raw_cpu = module.mantissa_raw.cpu().pin_memory()
#     exp_states_cpu = module.exponent_states.cpu().pin_memory()
#     exp_tables_cpu = module.exponent_tables.cpu().pin_memory()
#     exp_slot_map_cpu = module.exponent_slot_map.cpu().pin_memory()
#     exp_tile_offsets_cpu = module.exponent_tile_offsets.cpu().pin_memory()
#     exp_tile_max_lens_cpu = module.exponent_tile_max_lens.cpu().pin_memory()

#     # 3. Calculate Compression Ratios
#     original_bytes = weights_kn.numel() * 2
#     compressed_bytes = (
#         exp_comp_cuda.numel() * 1
#         + man_raw_cuda.numel() * 1
#         + exp_states_cuda.numel() * 4
#         + exp_tables_cuda.numel() * 4
#         + exp_slot_map_cuda.numel() * 1
#         + exp_tile_offsets_cuda.numel() * 4
#         + exp_tile_max_lens_cuda.numel() * 4
#     )
#     compression_ratio = compressed_bytes / original_bytes

#     # 4. Correctness Validation (Run once, off the clock)
#     out_base = torch.matmul(x, weights_kn)
#     out_fused = fused_rans_linear_triton(
#         x=x,
#         compressed_data=exp_comp_cuda,
#         initial_states=exp_states_cuda,
#         tables=exp_tables_cuda,
#         slot_map=exp_slot_map_cuda,
#         weight_shape=(K, N),
#         tile_offsets=exp_tile_offsets_cuda,
#         tile_max_lens=exp_tile_max_lens_cuda,
#         tile_k=tile_height,
#         tile_n=tile_width,
#         mantissas=man_raw_cuda,
#         accum_block_size=tile_width,
#     )
#     is_valid_fused = torch.allclose(out_base, out_fused, rtol=1e-2, atol=1e-2)
#     max_error_fused = torch.max(torch.abs(out_base - out_fused)).item()

#     # 5. Define Execution Wrappers for Profiling
#     def run_baseline():
#         return torch.matmul(x, weights_kn)

#     def run_fused_onboard():
#         return fused_rans_linear_triton(
#             x=x,
#             compressed_data=exp_comp_cuda,
#             initial_states=exp_states_cuda,
#             tables=exp_tables_cuda,
#             slot_map=exp_slot_map_cuda,
#             weight_shape=(K, N),
#             tile_offsets=exp_tile_offsets_cuda,
#             tile_max_lens=exp_tile_max_lens_cuda,
#             tile_k=tile_height,
#             tile_n=tile_width,
#             mantissas=man_raw_cuda,
#             accum_block_size=tile_width,
#         )

#     def run_fused_offload():
#         # Simulates PCIe transfer bottleneck from pinned memory + execution
#         c_comp = exp_comp_cpu.to("cuda", non_blocking=True)
#         c_states = exp_states_cpu.to("cuda", non_blocking=True)
#         c_tables = exp_tables_cpu.to("cuda", non_blocking=True)
#         c_slots = exp_slot_map_cpu.to("cuda", non_blocking=True)
#         c_offsets = exp_tile_offsets_cpu.to("cuda", non_blocking=True)
#         c_lens = exp_tile_max_lens_cpu.to("cuda", non_blocking=True)
#         c_man = man_raw_cpu.to("cuda", non_blocking=True)

#         return fused_rans_linear_triton(
#             x=x,
#             compressed_data=c_comp,
#             initial_states=c_states,
#             tables=c_tables,
#             slot_map=c_slots,
#             weight_shape=(K, N),
#             tile_offsets=c_offsets,
#             tile_max_lens=c_lens,
#             tile_k=tile_height,
#             tile_n=tile_width,
#             mantissas=c_man,
#             accum_block_size=tile_width,
#         )

#     # 6. Execute Profiling
#     ms_baseline = profile_function(run_baseline)
#     ms_fused_onboard = profile_function(run_fused_onboard)
#     ms_fused_offload = profile_function(run_fused_offload)

#     return {
#         "ratio": compression_ratio,
#         "is_valid_fused": is_valid_fused,
#         "max_error_fused": max_error_fused,
#         "ms_baseline": ms_baseline,
#         "ms_fused_onboard": ms_fused_onboard,
#         "ms_fused_offload": ms_fused_offload,
#     }


# def plot_pareto_frontier(results):
#     """Plots a Pareto frontier for Compression Ratio vs Latency."""
#     # Extract data
#     onboard_pts = [
#         (r["ratio"], r["ms_fused_onboard"], f"{r['H']}x{r['W']}") for r in results
#     ]
#     offload_pts = [
#         (r["ratio"], r["ms_fused_offload"], f"{r['H']}x{r['W']}") for r in results
#     ]

#     if not onboard_pts:
#         print("No valid runs to plot.")
#         return

#     def get_pareto_optimal(points):
#         # Sort by X (Compression Ratio) ascending
#         sorted_pts = sorted(points, key=lambda x: x[0])
#         pareto_front = []
#         min_y = float("inf")
#         for pt in sorted_pts:
#             if pt[1] < min_y:
#                 pareto_front.append(pt)
#                 min_y = pt[1]
#         return pareto_front

#     pareto_onboard = get_pareto_optimal(onboard_pts)
#     pareto_offload = get_pareto_optimal(offload_pts)

#     plt.figure(figsize=(10, 6))

#     # Plot Scatter
#     plt.scatter(
#         [p[0] for p in onboard_pts],
#         [p[1] for p in onboard_pts],
#         color="blue",
#         alpha=0.5,
#         label="Onboard (VRAM)",
#     )
#     plt.scatter(
#         [p[0] for p in offload_pts],
#         [p[1] for p in offload_pts],
#         color="red",
#         alpha=0.5,
#         label="Offload (PCIe -> VRAM)",
#     )

#     # Plot Pareto Lines
#     plt.plot(
#         [p[0] for p in pareto_onboard],
#         [p[1] for p in pareto_onboard],
#         color="blue",
#         linestyle="--",
#         label="Onboard Pareto Front",
#     )
#     plt.plot(
#         [p[0] for p in pareto_offload],
#         [p[1] for p in pareto_offload],
#         color="red",
#         linestyle="--",
#         label="Offload Pareto Front",
#     )

#     # Annotate Pareto Points
#     for pt in pareto_onboard:
#         plt.annotate(
#             pt[2],
#             (pt[0], pt[1]),
#             textcoords="offset points",
#             xytext=(0, -15),
#             ha="center",
#             fontsize=8,
#             color="blue",
#         )
#     for pt in pareto_offload:
#         plt.annotate(
#             pt[2],
#             (pt[0], pt[1]),
#             textcoords="offset points",
#             xytext=(0, 10),
#             ha="center",
#             fontsize=8,
#             color="red",
#         )

#     # Draw Baseline
#     baseline_ms = np.median([r["ms_baseline"] for r in results])
#     plt.axhline(
#         y=baseline_ms,
#         color="green",
#         linestyle=":",
#         label=f"Baseline PyTorch ({baseline_ms:.3f} ms)",
#     )

#     plt.title("rANS Microbenchmark: Compression vs. Latency Pareto Frontier")
#     plt.xlabel("Compression Ratio (Lower is Better)")
#     plt.ylabel("Latency (ms, Lower is Better)")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("rans_pareto_frontier.png", dpi=300)
#     print("\n Saved Pareto Frontier plot to 'rans_pareto_frontier.png'")


# if __name__ == "__main__":
#     MODEL_PATH = "Qwen/Qwen3-14B"
#     heights = [64, 128, 256, 512, 1024, 2048]
#     widths = [32, 64, 128]
#     BATCH_SIZE = 1

#     results = []
#     print(f"\n Starting rANS Ablation Study | Batch Size (M): {BATCH_SIZE}")
#     print("-" * 135)
#     print(
#         f"| {'Tile (H x W)':^14} | {'Valid':^7} | {'Ratio':^8} | {'Baseline (ms)':^15} | {'Onboard (ms)':^14} | {'Offloaded (ms)':^16} |"
#     )
#     print("-" * 135)

#     for h in heights:
#         for w in widths:
#             res = test_module_integration(
#                 MODEL_PATH, tile_height=h, tile_width=w, batch_size=BATCH_SIZE
#             )
#             res["H"] = h
#             res["W"] = w
#             results.append(res)

#             status = (
#                 "OK" if res["is_valid_fused"] else f"FAIL({res['max_error_fused']:.2f})"
#             )
#             print(
#                 f"| {h:>4d} x {w:<4d}   | {status:^7} | {res['ratio']:^7.2f}x | {res['ms_baseline']:^15.3f} | {res['ms_fused_onboard']:^14.3f} | {res['ms_fused_offload']:^16.3f} |"
#             )

#     plot_pareto_frontier(results)


# def plot_pareto_frontiers(results_dict):
#     """
#     Creates a 2-panel stacked plot (3.5 inch width) showing the pareto
#     frontiers for two different layer shapes independently.
#     """
#     plt.rcParams.update(
#         {
#             "font.family": "serif",
#             "font.size": 8,
#             "axes.labelsize": 8,
#             "legend.fontsize": 6,
#             "xtick.labelsize": 7,
#             "ytick.labelsize": 7,
#         }
#     )

#     # 3.5 inches wide, 4.5 inches tall (perfect for a double-column stack)
#     fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.5), sharex=False)

#     layers = list(results_dict.keys())
#     titles = ["Attention (QKV) Layer", "MLP (Down) Layer"]

#     for ax, layer_name, title in zip(axes, layers, titles):
#         layer_results = results_dict[layer_name]

#         on_pts = [
#             (r["ratio"], r["ms_fused_on"], f"{r['H']}x{r['W']}") for r in layer_results
#         ]
#         off_pts = [
#             (r["ratio"], r["ms_fused_off"], f"{r['H']}x{r['W']}") for r in layer_results
#         ]

#         def get_pareto(points):
#             sorted_pts = sorted(points, key=lambda x: x[0])
#             front = []
#             min_y = float("inf")
#             for pt in sorted_pts:
#                 if pt[1] < min_y:
#                     front.append(pt)
#                     min_y = pt[1]
#             return front

#         pareto_on = get_pareto(on_pts)
#         pareto_off = get_pareto(off_pts)

#         # Scatters
#         ax.scatter(
#             [p[0] for p in on_pts],
#             [p[1] for p in on_pts],
#             color="#2C7BB6",
#             alpha=0.5,
#             s=15,
#             label="rANS Onboard",
#         )
#         ax.scatter(
#             [p[0] for p in off_pts],
#             [p[1] for p in off_pts],
#             color="#D7191C",
#             alpha=0.5,
#             s=15,
#             label="rANS Offload",
#         )

#         # Lines
#         ax.plot(
#             [p[0] for p in pareto_on],
#             [p[1] for p in pareto_on],
#             color="#2C7BB6",
#             linestyle="-",
#             linewidth=1,
#         )
#         ax.plot(
#             [p[0] for p in pareto_off],
#             [p[1] for p in pareto_off],
#             color="#D7191C",
#             linestyle="-",
#             linewidth=1,
#         )

#         # Baselines
#         avg_base_on = np.mean([r["ms_base_on"] for r in layer_results])
#         avg_base_off = np.mean([r["ms_base_off"] for r in layer_results])
#         ax.axhline(
#             avg_base_on,
#             color="#1A9641",
#             linestyle="--",
#             linewidth=1,
#             label=f"Base Onboard",
#         )
#         ax.axhline(
#             avg_base_off,
#             color="#FDAE61",
#             linestyle=":",
#             linewidth=1.5,
#             label=f"Base Offload",
#         )

#         # Label points (staggered to prevent overlap in dense 3.5" space)
#         for i, pt in enumerate(on_pts):
#             y_offset = -5 if i % 2 == 0 else -10
#             ax.annotate(
#                 pt[2],
#                 (pt[0], pt[1]),
#                 textcoords="offset points",
#                 xytext=(0, y_offset),
#                 ha="center",
#                 fontsize=4.5,
#                 color="#2C7BB6",
#             )
#         for i, pt in enumerate(off_pts):
#             y_offset = 4 if i % 2 == 0 else 8
#             ax.annotate(
#                 pt[2],
#                 (pt[0], pt[1]),
#                 textcoords="offset points",
#                 xytext=(0, y_offset),
#                 ha="center",
#                 fontsize=4.5,
#                 color="#D7191C",
#             )

#         ax.set_title(f"{title} Pareto Front", fontweight="bold", fontsize=8)
#         ax.set_ylabel("Latency (ms)", fontweight="bold")
#         ax.grid(True, alpha=0.25, linestyle="--")

#     # Only set X-label for the bottom plot to save space
#     axes[1].set_xlabel("Compression Ratio (Lower is Better)", fontweight="bold")

#     # One unified legend at the very top
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(
#         handles,
#         labels,
#         loc="upper center",
#         bbox_to_anchor=(0.5, 1.05),
#         ncol=2,
#         frameon=False,
#         columnspacing=0.8,
#     )

#     plt.tight_layout()
#     plt.savefig(
#         "rans_pareto_frontiers_ieee.pdf", format="pdf", bbox_inches="tight", dpi=300
#     )
#     print("\n✅ Saved IEEE 2-panel Pareto plot to 'rans_pareto_frontiers_ieee.pdf'")

import torch
import triton
import os
import matplotlib.pyplot as plt
from safetensors import safe_open
from huggingface_hub import snapshot_download
import numpy as np

# --- Import your custom modules here ---
from comp_inference import (
    rans_compress_module_weight,
    fused_rans_linear_triton,
)

WARMUP_RUNS = 50
EVAL_RUNS = 100


def get_compressible_layers(repo_id, num_layers=10):
    """
    Dynamically scans safetensors shards and lazily loads the first N
    weight matrices with dim >= 2. This guarantees we get a diverse
    sample of real entropy regardless of the model architecture.
    """
    print(
        f"📥 Dynamically fetching the first {num_layers} compressible layers from '{repo_id}'..."
    )
    model_path = snapshot_download(
        repo_id=repo_id, allow_patterns=["*.safetensors"], local_files_only=False
    )

    weights = {}
    for filename in sorted(os.listdir(model_path)):
        if not filename.endswith(".safetensors"):
            continue
        filepath = os.path.join(model_path, filename)

        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if len(weights) >= num_layers:
                    return weights

                # Skip embedding and LM_head
                if "embed" in key.lower() or "lm_head" in key.lower():
                    continue
                # Use get_slice to check shape BEFORE loading to save RAM
                shape = f.get_slice(key).get_shape()

                # We only want linear layers (dim >= 2).
                # Skip embeddings if they are massive (e.g., vocab size > 100k) to focus on standard projections
                if len(shape) >= 2 and shape[0] < 100000:
                    w = f.get_tensor(key).cuda().to(torch.bfloat16).t().contiguous()
                    weights[key] = w
                    print(f"   ✅ Loaded {key}: {w.shape}")

    if len(weights) == 0:
        raise ValueError("Failed to find any 2D tensors in the safetensors shards.")

    return weights


def profile_function(func):
    for _ in range(WARMUP_RUNS):
        func()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(EVAL_RUNS)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(EVAL_RUNS)]

    for i in range(EVAL_RUNS):
        start_events[i].record()
        func()
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return np.median(times_ms)


def test_single_matrix(weights_kn, tile_height, tile_width, batch_size):
    K, N = weights_kn.shape
    x = torch.randn((batch_size, K), dtype=torch.bfloat16, device="cuda")
    weights_kn_cpu = weights_kn.cpu().pin_memory()

    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(weights_kn.clone())
    module.bias = None
    rans_compress_module_weight(
        module, tile_height=tile_height, tile_width=tile_width, transpose_weight=False
    )

    exp_comp_cuda = module.exponent_compressed_weight.cuda()
    man_raw_cuda = module.mantissa_raw.cuda()
    exp_states_cuda = module.exponent_states.cuda()
    exp_tables_cuda = module.exponent_tables.cuda()
    exp_slot_map_cuda = module.exponent_slot_map.cuda()
    exp_tile_offsets_cuda = module.exponent_tile_offsets.cuda()
    exp_tile_max_lens_cuda = module.exponent_tile_max_lens.cuda()

    exp_comp_cpu = module.exponent_compressed_weight.cpu().pin_memory()
    man_raw_cpu = module.mantissa_raw.cpu().pin_memory()
    exp_states_cpu = module.exponent_states.cpu().pin_memory()
    exp_tables_cpu = module.exponent_tables.cpu().pin_memory()
    exp_slot_map_cpu = module.exponent_slot_map.cpu().pin_memory()
    exp_tile_offsets_cpu = module.exponent_tile_offsets.cpu().pin_memory()
    exp_tile_max_lens_cpu = module.exponent_tile_max_lens.cpu().pin_memory()

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

    def run_baseline_onboard():
        return torch.matmul(x, weights_kn)

    def run_baseline_offload():
        w_cuda = weights_kn_cpu.to("cuda", non_blocking=True)
        return torch.matmul(x, w_cuda)

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

    return {
        "orig_bytes": original_bytes,
        "comp_bytes": compressed_bytes,
        "ms_base_on": profile_function(run_baseline_onboard),
        "ms_base_off": profile_function(run_baseline_offload),
        "ms_fused_on": profile_function(run_fused_onboard),
        "ms_fused_off": profile_function(run_fused_offload),
    }


# def plot_pareto_frontier(results):
#     """
#     Plots the aggregated 10-layer results into a single IEEE-compliant panel.
#     """
#     plt.rcParams.update(
#         {
#             "font.family": "serif",
#             "font.size": 8,
#             "axes.labelsize": 8,
#             "legend.fontsize": 6,
#             "xtick.labelsize": 7,
#             "ytick.labelsize": 7,
#         }
#     )

#     fig, ax = plt.subplots(figsize=(3.5, 3.0))

#     onboard_pts = [
#         (r["ratio"], r["ms_fused_on"], f"{r['H']}x{r['W']}") for r in results
#     ]
#     offload_pts = [
#         (r["ratio"], r["ms_fused_off"], f"{r['H']}x{r['W']}") for r in results
#     ]

#     def get_pareto(points):
#         sorted_pts = sorted(points, key=lambda x: x[0])
#         front = []
#         min_y = float("inf")
#         for pt in sorted_pts:
#             if pt[1] < min_y:
#                 front.append(pt)
#                 min_y = pt[1]
#         return front

#     pareto_on = get_pareto(onboard_pts)
#     pareto_off = get_pareto(offload_pts)

#     # Scatters
#     ax.scatter(
#         [p[0] for p in onboard_pts],
#         [p[1] for p in onboard_pts],
#         color="#2C7BB6",
#         alpha=0.6,
#         s=15,
#         label="rANS Onboard",
#     )
#     ax.scatter(
#         [p[0] for p in offload_pts],
#         [p[1] for p in offload_pts],
#         color="#D7191C",
#         alpha=0.6,
#         s=15,
#         label="rANS Offloaded",
#     )

#     # Pareto Lines
#     ax.plot(
#         [p[0] for p in pareto_on],
#         [p[1] for p in pareto_on],
#         color="#2C7BB6",
#         linestyle="-",
#         linewidth=1,
#     )
#     ax.plot(
#         [p[0] for p in pareto_off],
#         [p[1] for p in pareto_off],
#         color="#D7191C",
#         linestyle="-",
#         linewidth=1,
#     )

#     # Baselines
#     avg_base_on = np.mean([r["ms_base_on"] for r in results])
#     avg_base_off = np.mean([r["ms_base_off"] for r in results])

#     ax.axhline(
#         avg_base_on,
#         color="#1A9641",
#         linestyle="--",
#         linewidth=1,
#         label=f"Base On ({avg_base_on:.1f}ms)",
#     )
#     ax.axhline(
#         avg_base_off,
#         color="#FDAE61",
#         linestyle=":",
#         linewidth=1.5,
#         label=f"Base Off ({avg_base_off:.1f}ms)",
#     )

#     # Label every point, staggering text to prevent overlaps
#     for i, pt in enumerate(onboard_pts):
#         y_offset = -6 if i % 2 == 0 else -11
#         ax.annotate(
#             pt[2],
#             (pt[0], pt[1]),
#             textcoords="offset points",
#             xytext=(0, y_offset),
#             ha="center",
#             fontsize=4.5,
#             color="#2C7BB6",
#         )

#     for i, pt in enumerate(offload_pts):
#         y_offset = 4 if i % 2 == 0 else 9
#         ax.annotate(
#             pt[2],
#             (pt[0], pt[1]),
#             textcoords="offset points",
#             xytext=(0, y_offset),
#             ha="center",
#             fontsize=4.5,
#             color="#D7191C",
#         )

#     ax.set_xlabel("Aggregated Compression Ratio (Lower is Better)", fontweight="bold")
#     ax.set_ylabel("10-Layer Latency Sum (ms)", fontweight="bold")
#     ax.grid(True, alpha=0.25, linestyle="--")

#     ax.legend(
#         loc="upper center",
#         bbox_to_anchor=(0.5, 1.25),
#         ncol=2,
#         frameon=False,
#         columnspacing=0.8,
#     )
#     plt.subplots_adjust(top=0.78, bottom=0.15, left=0.15, right=0.98)

#     plt.savefig("rans_pareto_10_layers_ieee.pdf", format="pdf", dpi=300)
#     print(
#         "\n✅ Saved IEEE-formatted Aggregate Pareto plot to 'rans_pareto_10_layers_ieee.pdf'"
#     )

from adjustText import adjust_text


def plot_pareto_frontier(results):
    """
    Plots the aggregated 10-layer results into a single IEEE-compliant panel.
    """
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    onboard_pts = [
        (r["ratio"], r["ms_fused_on"], f"{r['H']}x{r['W']}") for r in results
    ]
    offload_pts = [
        (r["ratio"], r["ms_fused_off"], f"{r['H']}x{r['W']}") for r in results
    ]

    def get_pareto(points):
        sorted_pts = sorted(points, key=lambda x: x[0])
        front = []
        min_y = float("inf")
        for pt in sorted_pts:
            if pt[1] < min_y:
                front.append(pt)
                min_y = pt[1]
        return front

    pareto_on = get_pareto(onboard_pts)
    pareto_off = get_pareto(offload_pts)

    # Scatters
    ax.scatter(
        [p[0] for p in onboard_pts],
        [p[1] for p in onboard_pts],
        color="#82C8F0",
        alpha=0.6,
        s=15,
        label="rANS Onboard",
    )
    ax.scatter(
        [p[0] for p in offload_pts],
        [p[1] for p in offload_pts],
        color="#F5A5C8",
        alpha=0.6,
        s=15,
        label="rANS Offloaded",
    )

    # Pareto Lines
    ax.plot(
        [p[0] for p in pareto_on],
        [p[1] for p in pareto_on],
        color="#82C8F0",
        linestyle="-",
        linewidth=1,
    )
    ax.plot(
        [p[0] for p in pareto_off],
        [p[1] for p in pareto_off],
        color="#F5A5C8",
        linestyle="-",
        linewidth=1,
    )

    # Baselines
    avg_base_on = np.mean([r["ms_base_on"] for r in results])
    avg_base_off = np.mean([r["ms_base_off"] for r in results])

    ax.axhline(
        avg_base_on,
        color="#7DCDBE",
        linestyle="--",
        linewidth=1,
        label=f"Base On ({avg_base_on:.1f}ms)",
    )
    ax.axhline(
        avg_base_off,
        color="#FFDCA5",
        linestyle=":",
        linewidth=1.5,
        label=f"Base Off ({avg_base_off:.1f}ms)",
    )

    # --- THE FIX: Collect text objects and use adjustText ---
    texts = []
    for pt in onboard_pts:
        texts.append(
            ax.text(
                pt[0],
                pt[1],
                pt[2],
                fontsize=5.5,
                # color="#82C8F0",
                color="#000000",
                ha="center",
                va="center",
            )
        )

    for pt in offload_pts:
        texts.append(
            ax.text(
                pt[0],
                pt[1],
                pt[2],
                fontsize=5.5,
                # color="#F5A5C8",
                color="#000000",
                ha="center",
                va="center",
            )
        )

    print("Repelling overlapping text labels...")
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.3, alpha=0.6),
        expand_points=(1.2, 1.2),
        force_text=(0.5, 1.0),
    )
    # --------------------------------------------------------

    ax.set_xlabel("Aggregated Compression Ratio (Lower is Better)", fontweight="bold")
    ax.set_ylabel("10-Layer Aggregated Latency (ms)", fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
        frameon=False,
        columnspacing=0.8,
    )
    plt.subplots_adjust(top=0.78, bottom=0.15, left=0.15, right=0.98)

    plt.savefig("rans_pareto_10_layers_ieee.pdf", format="pdf", dpi=300)
    print(
        "\n✅ Saved IEEE-formatted Aggregate Pareto plot to 'rans_pareto_10_layers_ieee.pdf'"
    )


if __name__ == "__main__":
    MODEL_PATH = "Qwen/Qwen3-14B"
    heights = [256, 512, 1024]
    widths = [32, 64, 128, 256]
    BATCH_SIZE = 1
    NUM_LAYERS = 10

    print(f"\nStarting 10-Layer Ablation Study | Batch Size: {BATCH_SIZE}")
    real_weights = get_compressible_layers(MODEL_PATH, num_layers=NUM_LAYERS)

    results = []

    print("-" * 110)
    print(
        f"| {'Tile':^10} | {'Ratio':^7} | {'Base On (ms)':^12} | {'Base Off (ms)':^13} | {'rANS On (ms)':^12} | {'rANS Off (ms)':^13} |"
    )
    print("-" * 110)

    for h in heights:
        for w in widths:
            tot_orig = (
                tot_comp
            ) = tot_base_on = tot_base_off = tot_fused_on = tot_fused_off = 0.0

            for name, weight_tensor in real_weights.items():
                res = test_single_matrix(
                    weight_tensor, tile_height=h, tile_width=w, batch_size=BATCH_SIZE
                )
                tot_orig += res["orig_bytes"]
                tot_comp += res["comp_bytes"]
                tot_base_on += res["ms_base_on"]
                tot_base_off += res["ms_base_off"]
                tot_fused_on += res["ms_fused_on"]
                tot_fused_off += res["ms_fused_off"]

            ratio = tot_comp / tot_orig

            results.append(
                {
                    "H": h,
                    "W": w,
                    "ratio": ratio,
                    "ms_base_on": tot_base_on,
                    "ms_base_off": tot_base_off,
                    "ms_fused_on": tot_fused_on,
                    "ms_fused_off": tot_fused_off,
                }
            )

            print(
                f"| {h:>4}x{w:<4} | {ratio:^7.3f}x | {tot_base_on:^12.2f} | {tot_base_off:^13.2f} | {tot_fused_on:^12.2f} | {tot_fused_off:^13.2f} |"
            )

    plot_pareto_frontier(results)
