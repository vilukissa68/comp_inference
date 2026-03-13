import torch
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from datetime import datetime
from safetensors import safe_open
from huggingface_hub import snapshot_download

# --- Import your custom modules here ---
from comp_inference import (
    rans_compress_module_weight,
    fused_rans_linear_triton,
    fused_rans_linear_triton_uncoalesced,
)

WARMUP_RUNS = 25
EVAL_RUNS = 100

# --- IEEE STYLE GLOBALS ---
FIG_WIDTH = 3.5
COLORS = {
    "baseline": "#7DCDBE",  # Green
    "rans_coal": "#82C8F0",  # Light Blue (Legacy)
    "rans_uncoal": "#FFDCA5",  # Dark Blue (Proposed)
}
LABELS = {
    "baseline": "FP16 Base",
    "rans_coal": "Coalesced",
    "rans_uncoal": "Uncoalesced",
}


def get_compressible_layers(repo_id, num_layers=None):
    if num_layers is None:
        print(f"📥 Dynamically fetching ALL compressible layers from '{repo_id}'...")
    else:
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
                if num_layers is not None and len(weights) >= num_layers:
                    return weights

                if "embed" in key.lower() or "lm_head" in key.lower():
                    continue

                shape = f.get_slice(key).get_shape()
                # Focus on standard projections
                if len(shape) >= 2 and shape[0] < 100000:
                    w = f.get_tensor(key).cuda().to(torch.bfloat16).t().contiguous()
                    weights[key] = w
                    print(f"   ✅ Loaded {key}: {w.shape}")

    if len(weights) == 0:
        raise ValueError("Failed to find any 2D tensors in the safetensors shards.")

    print(f"🎉 Successfully loaded {len(weights)} layers!")
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


def test_split_k_sweep(
    weights_kn, tile_height, tile_width, batch_sizes, split_k_values
):
    K, N = weights_kn.shape
    shape_str = f"{K}x{N}"
    results = []

    print(f"\n⚙️  Compressing {shape_str}...")

    # --- 1. COALESCED SETUP ---
    mod_coal = torch.nn.Module()
    mod_coal.weight = torch.nn.Parameter(weights_kn.clone())
    mod_coal.bias = None
    rans_compress_module_weight(
        mod_coal,
        tile_height=tile_height,
        tile_width=tile_width,
        transpose_weight=False,
        uncoalesced_interleaving=False,
    )

    c_exp = mod_coal.exponent_compressed_weight.cuda()
    c_man = mod_coal.mantissa_raw.cuda()
    c_states = mod_coal.exponent_states.cuda()
    c_tables = mod_coal.exponent_tables.cuda()
    c_slot = mod_coal.exponent_slot_map.cuda()
    c_off = mod_coal.exponent_tile_offsets.cuda()
    c_len = mod_coal.exponent_tile_max_lens.cuda()

    # --- 2. UNCOALESCED SETUP ---
    mod_uncoal = torch.nn.Module()
    mod_uncoal.weight = torch.nn.Parameter(weights_kn.clone())
    mod_uncoal.bias = None
    rans_compress_module_weight(
        mod_uncoal,
        tile_height=tile_height,
        tile_width=tile_width,
        transpose_weight=False,
        uncoalesced_interleaving=True,
    )

    u_exp = mod_uncoal.exponent_compressed_weight.cuda()
    u_man = mod_uncoal.mantissa_raw.cuda()
    u_states = mod_uncoal.exponent_states.cuda()
    u_tables = mod_uncoal.exponent_tables.cuda()
    u_slot = mod_uncoal.exponent_slot_map.cuda()
    u_off = mod_uncoal.exponent_stream_offsets.cuda()

    # Safe uint32 stream size calc
    u_offsets_i64 = u_off.to(torch.int64)
    u_tot_bytes = torch.tensor([u_exp.numel()], dtype=torch.int64, device="cuda")
    u_sizes = (torch.cat([u_offsets_i64[1:], u_tot_bytes]) - u_offsets_i64).to(
        torch.uint32
    )

    for bs in batch_sizes:
        x = torch.randn((bs, K), dtype=torch.bfloat16, device="cuda")

        # Baseline (invariant to SPLIT_K)
        ms_base = profile_function(lambda: torch.matmul(x, weights_kn))

        for sk in split_k_values:
            # We let the Triton wrapper dynamically allocate the workspace for the microbenchmark
            try:
                ms_coal = profile_function(
                    lambda: fused_rans_linear_triton(
                        x=x,
                        compressed_data=c_exp,
                        initial_states=c_states,
                        tables=c_tables,
                        slot_map=c_slot,
                        weight_shape=(K, N),
                        tile_offsets=c_off,
                        tile_max_lens=c_len,
                        tile_k=tile_height,
                        tile_n=tile_width,
                        mantissas=c_man,
                        accum_block_size=tile_width,
                        SPLIT_K=sk,
                    )
                )
            except Exception as e:
                print(f"Coal Failed (BS={bs}, SK={sk}): {e}")
                ms_coal = float("nan")

            try:
                ms_uncoal = profile_function(
                    lambda: fused_rans_linear_triton_uncoalesced(
                        x=x,
                        compressed_data=u_exp,
                        initial_states=u_states,
                        tables=u_tables,
                        slot_map=u_slot,
                        weight_shape=(K, N),
                        stream_offsets=u_off,
                        stream_sizes=u_sizes,
                        tile_k=tile_height,
                        tile_n=tile_width,
                        mantissas=u_man,
                        accum_block_size=tile_width,
                        SPLIT_K=sk,
                    )
                )
            except Exception as e:
                print(f"Uncoal Failed (BS={bs}, SK={sk}): {e}")
                ms_uncoal = float("nan")

            print(
                f"   [BS: {bs:>3} | SK: {sk:>2}] Base: {ms_base:.3f}ms | Coal: {ms_coal:.3f}ms | Uncoal: {ms_uncoal:.3f}ms"
            )

            results.append(
                {
                    "shape": shape_str,
                    "K": K,
                    "N": N,
                    "batch_size": bs,
                    "split_k": sk,
                    "ms_baseline": ms_base,
                    "ms_rans_coal": ms_coal,
                    "ms_rans_uncoal": ms_uncoal,
                }
            )

    # Cleanup VRAM
    del mod_coal, mod_uncoal, x
    torch.cuda.empty_cache()

    return results


def plot_split_k_frontier(results, output_dir, timestamp):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.lines import Line2D

    df = pd.DataFrame(results)

    # --- IEEE 3.5-inch Formatting ---
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "legend.fontsize": 6,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
        }
    )

    # Exact color match to the Pareto plots
    COLORS = {
        "baseline": "#2ca02c",  # Green
        "rans_coal": "#aec7e8",  # Light Blue (Legacy)
        "rans_uncoal": "#1f77b4",  # Dark Blue (Proposed)
    }

    target_bss = [1, 16]

    # STRICT LIMIT: Exactly 2 unique shapes for a 1x2 side-by-side layout
    unique_shapes = df["shape"].unique()
    shapes_to_plot = unique_shapes[:2]

    for bs in target_bss:
        bs_df = df[(df["batch_size"] == bs) & (df["shape"].isin(shapes_to_plot))]
        if bs_df.empty:
            continue

        # Force a 1-row, 2-column layout. Keep it short (1.7 inches) to save space.
        fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.7), sharex=True)

        # Safety catch if the model only had 1 layer type
        if not isinstance(axes, np.ndarray):
            axes = [axes]

        for idx, shape in enumerate(shapes_to_plot):
            ax = axes[idx]
            shape_data = bs_df[bs_df["shape"] == shape]

            # 1. Plot Baseline as a constant horizontal dashed line (No markers needed)
            base_latency = shape_data["ms_baseline"].iloc[0]
            ax.axhline(
                base_latency,
                color=COLORS["baseline"],
                linestyle="--",
                linewidth=1.2,
                zorder=1,
            )

            # 2. Plot Coalesced with Circular Markers ('o')
            ax.plot(
                shape_data["split_k"],
                shape_data["ms_rans_coal"],
                color=COLORS["rans_coal"],
                marker="o",
                markersize=4,
                linewidth=1.2,
                zorder=2,
            )

            # 3. Plot Uncoalesced with Square Markers ('s')
            ax.plot(
                shape_data["split_k"],
                shape_data["ms_rans_uncoal"],
                color=COLORS["rans_uncoal"],
                marker="s",
                markersize=4,
                linewidth=1.2,
                zorder=3, 
            )

            ax.set_title(f"Layer: {shape}", fontsize=7, fontweight="bold", pad=3)

            # Only label the Y-axis on the leftmost plot
            if idx == 0:
                ax.set_ylabel("Latency (ms)")

            ax.set_xlabel("SPLIT_K")

            # Lock the X-ticks to the exact SPLIT_K values tested (e.g., 1, 2, 4, 8, 16)
            ax.set_xticks(sorted(shape_data["split_k"].unique()))
            ax.grid(True, alpha=0.3, linestyle="--")

        # Hide the second axis if the model only yielded 1 unique shape
        if len(shapes_to_plot) == 1 and len(axes) > 1:
            axes[1].set_visible(False)

        # --- THE FIX: Custom Legend with Exact Shape Mapping ---
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=COLORS["baseline"],
                lw=1.2,
                linestyle="--",
                label="Baseline",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["rans_coal"],
                lw=1.2,
                marker="o",
                markersize=4,
                label="Coalesced",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["rans_uncoal"],
                lw=1.2,
                marker="s",
                markersize=4,
                label="Uncoalesced",
            ),            
        ]

        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),  # Push slightly higher to clear titles
            ncol=3,
            frameon=False,
            columnspacing=0.8,
            handletextpad=0.4,
        )

        # Adjust layout tightly for IEEE format
        plt.subplots_adjust(top=0.82, bottom=0.22, left=0.14, right=0.98, wspace=0.25)

        fname = f"split_k_ablation_bs{bs}_{timestamp}.pdf"
        plt.savefig(
            os.path.join(output_dir, fname), format="pdf", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"📊 Saved SPLIT_K Ablation plot for BS={bs} to {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-14B",
        help="HF Repo or local safetensors path",
    )
    parser.add_argument("--tile_h", type=int, default=1024)
    parser.add_argument("--tile_w", type=int, default=128)
    parser.add_argument("--split_ks", type=str, default="1,2,4,8,12,16,32")
    parser.add_argument("--batch_sizes", type=str, default="1")
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="How many unique layers to pull from the model",
    )
    args = parser.parse_args()

    split_ks = [int(x) for x in args.split_ks.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"split_k_sweep_{timestamp}.json"

    print(f"Starting SPLIT_K Sweep | Tile: {args.tile_h}x{args.tile_w}")
    real_weights = get_compressible_layers(args.model, num_layers=args.num_layers)

    all_results = []

    for name, weight_tensor in real_weights.items():
        # Note: We group identical layer shapes in plotting.
        layer_results = test_split_k_sweep(
            weight_tensor,
            tile_height=args.tile_h,
            tile_width=args.tile_w,
            batch_sizes=batch_sizes,
            split_k_values=split_ks,
        )
        all_results.extend(layer_results)

        # Safety save
        with open(json_filename, "w") as f:
            json.dump(all_results, f, indent=4)

    print(f"\nSweep Complete. Saved to {json_filename}")
    plot_split_k_frontier(all_results, output_dir=".", timestamp=timestamp)
