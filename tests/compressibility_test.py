#!/usr/bin/env python3
import os
import gc
import json
import torch
import argparse
import zipnn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

# --- Attempt to import NVIDIA Model Optimizer ---
try:
    import modelopt.torch.quantization as mtq
    import modelopt.torch.opt as mto

    MODELOPT_AVAILABLE = True
except ImportError:
    print("[Warn] nvidia-modelopt not found. FP8/NVFP8 compression will be skipped.")
    MODELOPT_AVAILABLE = False

from comp_inference import (
    rans_compress_module_weight,
    rans_compress_qkv_fused,
    rans_compress_gate_up_fused,
    save_rans_model_package,
)

import heapq
from collections import Counter

# --- CONFIGURATION ---
MODELS_TO_TEST = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "meta-llama/Meta-Llama-3-8B",
    "mistralai/Mixtral-8x7B-v0.1",
]

BASE_DIR = "./compression_benchmarks"
METHODS = ["Baseline (BF16)", "rANS (Ours)", "ZipNN", "FP8*", "NVFP4*"]


def is_tied_lm_head(model):
    tied_lm_head = False
    if getattr(model.config, "tie_word_embeddings", False):
        tied_lm_head = True
    else:
        try:
            in_emb = model.get_input_embeddings().weight
            out_emb = model.get_output_embeddings().weight
            # Check if they share the exact same memory address or exact values
            if in_emb is out_emb or torch.equal(in_emb, out_emb):
                tied_lm_head = True
        except Exception:
            print(
                "Could not verify tied weights, but it may still be present. Proceeding with compression. If you see unexpected errors, try enabling --skip_embedding."
            )
            pass

    if tied_lm_head:
        print("Detected tied weights! lm_head is identical to embed_tokens.")
    return tied_lm_head


# --- HELPER: GET DIR SIZE ---
def get_dir_size_gb(path: str) -> float:
    total_size = 0
    if not os.path.exists(path):
        return 0.0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total_size += os.path.getsize(os.path.join(dirpath, f))
    return total_size / (1024**3)


# --- ZIPNN INTEGRATION ---
def encode_tensor_zipnn(param):
    zpn = zipnn.ZipNN(input_format="torch", threads=1, method="huffman")
    compressed_param = zpn.compress(param)
    return compressed_param


def compress_zipnn(model, out_dir):
    """
    Iterates through the model's state dict, compresses each tensor using ZipNN,
    and saves the raw bytestreams to disk to measure the final payload size.
    """
    print("  -> Running ZipNN compression...")
    os.makedirs(out_dir, exist_ok=True)

    compressed_dict = {}
    for name, tensor in model.state_dict().items():
        # Skip lm_head if tied
        if "lm_head" in name and is_tied_lm_head(model):
            continue

        # ZipNN requires contiguous CPU tensors
        cpu_tensor = tensor.cpu().contiguous()

        # Only compress 2D+ weight matrices (skip 1D biases/layernorms)
        if cpu_tensor.dim() >= 2:
            try:
                # Returns raw bytes
                compressed_bytes = encode_tensor_zipnn(cpu_tensor)

                # Convert bytes to a 1D uint8 tensor so safetensors can save it
                byte_tensor = torch.frombuffer(
                    compressed_bytes, dtype=torch.uint8
                ).clone()
                compressed_dict[name] = byte_tensor
            except Exception as e:
                print(f"     [Warn] ZipNN failed on {name}: {e}. Storing uncompressed.")
                compressed_dict[name] = cpu_tensor
        else:
            compressed_dict[name] = cpu_tensor

    save_file(compressed_dict, os.path.join(out_dir, "model_zipnn.safetensors"))


def calculate_huffman_bits(tensor):
    """
    Calculates the theoretical bit-count of a tensor using Huffman coding.
    Works natively with bfloat16 without Numpy conversion.
    """
    # Flatten and stay on device (or CPU)
    data = tensor.view(-1)
    if data.numel() == 0:
        return 0

    # Stay in PyTorch for frequency analysis to support BFloat16
    # unique() returns the values and their respective counts
    symbols, counts = torch.unique(data, return_counts=True)

    # Convert to Python lists for the Huffman tree building
    # (The tree logic is scalar-heavy, so Python lists are fine here)
    counts_list = counts.tolist()
    symbols_list = symbols.tolist()

    if len(counts_list) <= 1:
        return data.numel()  # 1 bit per element if all symbols are identical

    # Build Huffman Tree using a priority queue (heap)
    # Stores [frequency, [symbol, bit_string]]
    heap = [[c, [s, ""]] for c, s in zip(counts_list, symbols_list)]
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    # Map bit lengths to symbols
    # total bits = sum of (frequency of symbol * length of its Huffman code)
    total_bits = 0
    for pair in heap[0][1:]:
        symbol_freq = dict(zip(symbols_list, counts_list))[pair[0]]
        total_bits += symbol_freq * len(pair[1])

    return total_bits


def compress_huffman(model, out_dir):
    """
    Simulates Huff-LLM by calculating bits for each layer.
    """
    print("  -> Running Huffman (Huff-LLM) frequency analysis...")
    os.makedirs(out_dir, exist_ok=True)

    total_bits = 0
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            # Only run Huffman on weight matrices (2D+)
            if tensor.dim() >= 2:
                total_bits += calculate_huffman_bits(tensor)
            else:
                # Store biases/norms at full BF16 size
                total_bits += tensor.numel() * 16

    total_bytes = (total_bits + 7) // 8

    # Save a dummy file to represent the theoretical compressed size
    with open(os.path.join(out_dir, "huffman_estimate.bin"), "wb") as f:
        # Avoid huge seek if total_bytes is massive
        f.truncate(int(total_bytes))

    print(f"     Huffman theoretical size: {total_bytes / (1024**3):.2f} GB")


# --- RANS INTEGRATION ---
def compress_rans(model, tokenizer, model_name_clean, out_dir):
    """
    Applies your custom rANS compression flow, including structural fusions.
    """
    print("  -> Running rANS compression...")
    os.makedirs(out_dir, exist_ok=True)

    # Mock Args struct to match your logic
    class Args:
        skip_embedding = False
        fuse_qkv = True
        fuse_gate_up = True
        tile_height = 1024
        tile_width = 128

    args = Args()
    layers_compressed = 0

    # Detect tied head
    tied_lm_head = False

    # Check if the config explicitly declares it, or if the tensors match exactly
    if getattr(model.config, "tie_word_embeddings", False):
        tied_lm_head = True
    else:
        try:
            in_emb = model.get_input_embeddings().weight
            out_emb = model.get_output_embeddings().weight
            # Check if they share the exact same memory address or exact values
            if in_emb is out_emb or torch.equal(in_emb, out_emb):
                tied_lm_head = True
        except Exception:
            print(
                "Could not verify tied weights, but it may still be present. Proceeding with compression. If you see unexpected errors, try enabling --skip_embedding."
            )
            pass

    if tied_lm_head:
        print("Detected tied weights! lm_head is identical to embed_tokens.")

    for name, module in model.named_modules():
        # Skip embedding/lm_head if requested
        if (
            name in ["model.embed_tokens", "model.lm_head", "lm_head"]
            and args.skip_embedding
        ):
            print("Skipping compressing embedding layer.")
            continue

        # Skip tied lm_head
        if name in ["model.lm_head", "lm_head"] and tied_lm_head:
            print(f"Skipping {name} compression (weights are tied to embed_tokens).")
            # if hasattr(module, "weight"):
            #     del module.weight
            continue

        if name in ["model.embed_tokens", "model.lm_head", "lm_head"]:
            print(f"Compressing embedding/lm_head layer: {name}")
            rans_compress_module_weight(
                module,
                tile_height=args.tile_height,
                tile_width=args.tile_width,
                transpose_weight=False,
            )
            layers_compressed += 1
            continue

        if (
            args.fuse_qkv
            and hasattr(module, "q_proj")
            and hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
        ):
            rans_compress_qkv_fused(
                module, tile_height=args.tile_height, tile_width=args.tile_width
            )
            layers_compressed += 1
            continue

        if args.fuse_gate_up and (
            (hasattr(module, "gate") and hasattr(module, "up"))
            or (hasattr(module, "gate_proj") and hasattr(module, "up_proj"))
        ):
            rans_compress_gate_up_fused(
                module, tile_height=args.tile_height, tile_width=args.tile_width
            )
            layers_compressed += 1
            continue

        if hasattr(module, "weight") and module.weight is not None:
            if module.weight.ndim < 2:
                continue
            rans_compress_module_weight(
                module, tile_height=args.tile_height, tile_width=args.tile_width
            )
            layers_compressed += 1

    # Save the compressed package to the target directory
    # Note: Ensure your `save_rans_model_package` function writes to `out_dir`
    save_rans_model_package(
        model, tokenizer, out_dir, fuse=True, tied_lm_head=tied_lm_head
    )


# --- NVIDIA FP8 / NVFP8 INTEGRATION ---
def compress_nvidia_fp8(model_id, out_dir, format_type="fp8_e4m3"):
    """Uses nvidia-modelopt to natively quantize the model."""
    if not MODELOPT_AVAILABLE:
        return

    print(f"  -> Running NVIDIA {format_type} compression...")
    os.makedirs(out_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    quant_cfg = {
        "quant_cfg": {
            "*weight_quantizer": {
                "num_bits": 8,
                "default_format": format_type,  # "fp8_e4m3" or "nvfp8"
            }
        }
    }

    try:
        mtq.quantize(model, quant_cfg=quant_cfg)
        model.save_pretrained(out_dir, safe_serialization=True)
    except Exception as e:
        print(f"  [Error] {format_type} failed: {e}")
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


# --- PHYSICAL FP8 / NVFP4 PACKING ---
def compress_nvidia(model_id, out_dir, format_type="FP8"):
    """
    Physically builds the true memory layout of NVIDIA's lossy formats
    (including block scales) and saves them as raw bytes in safetensors,
    bypassing PyTorch's fake-quantization entirely.
    """
    print(f"  -> Physically packing NVIDIA {format_type} layout...")
    os.makedirs(out_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    packed_dict = {}
    for name, tensor in model.state_dict().items():
        # Only compress 2D+ weight matrices (skip 1D biases/layernorms)
        if tensor.dim() >= 2:
            N = tensor.numel()

            if format_type == "NVFP4":
                # NVFP4 Layout:
                # 1. 4-bit weights (2 weights per byte)
                packed_weights = torch.empty(N // 2, dtype=torch.uint8)
                # 2. FP8 block scales (1 byte per 16 elements)
                block_scales = torch.empty(N // 16, dtype=torch.uint8)

                packed_dict[f"{name}.weight_packed"] = packed_weights
                packed_dict[f"{name}.weight_scale_inv"] = block_scales

            elif format_type == "FP8":
                # FP8 Layout:
                # 1. 8-bit weights (1 weight per byte)
                packed_weights = torch.empty(N, dtype=torch.uint8)
                # 2. Per-channel FP32 scale (1 float per output channel)
                channel_scales = torch.empty(tensor.shape[0], dtype=torch.float32)

                packed_dict[f"{name}.weight_packed"] = packed_weights
                packed_dict[f"{name}.weight_scale_inv"] = channel_scales
        else:
            # Leave biases and LayerNorms uncompressed
            packed_dict[name] = tensor.cpu()

    save_file(packed_dict, os.path.join(out_dir, "model.safetensors"))

    del model
    gc.collect()
    torch.cuda.empty_cache()


from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from datasets import Dataset


def compress_llmcompressor_fp8(model_id, out_dir):
    """
    Physically quantizes weights to FP8 (float8_e4m3fn) and saves
    the true compressed safetensors to disk.
    """
    print("  -> Running LLMcompressor FP8 physical packing...")
    os.makedirs(out_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_BLOCK", ignore=["lm_head", "model.lm_head"]
    )

    # Construct a valid Hugging Face Dataset object
    dummy_dict = {
        "input_ids": [torch.randint(0, 1000, (8,)).tolist()],
        "attention_mask": [torch.ones(8, dtype=torch.int64).tolist()],
    }
    dummy_dataset = Dataset.from_dict(dummy_dict)

    # Execute the physical quantization
    # oneshot(model=model, dataset=dummy_dataset, recipe=recipe)
    oneshot(model=model, recipe=recipe)

    if getattr(model.config, "tie_word_embeddings", False) and hasattr(
        model, "lm_head"
    ):
        print("  -> Purging duplicate lm_head before saving...")
        del model.lm_head

    model.save_pretrained(out_dir, save_compressed=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


# WORKS!!!!
def compress_llmcompressor_nvfp4(model_id, out_dir):
    """
    Physically packs weights into NVIDIA NVFP4 format (4-bit floats + FP8 scales).
    """
    print("  -> Running LLMcompressor NVFP4 physical packing...")
    os.makedirs(out_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )

    recipe = QuantizationModifier(
        targets="Linear", scheme="NVFP4", ignore=["lm_head", "model.lm_head"]
    )

    dummy_dict = {
        "input_ids": [torch.randint(0, 1000, (8,)).tolist()],
        "attention_mask": [torch.ones(8, dtype=torch.int64).tolist()],
    }
    dummy_dataset = Dataset.from_dict(dummy_dict)

    oneshot(model=model, dataset=dummy_dataset, recipe=recipe)

    if getattr(model.config, "tie_word_embeddings", False) and hasattr(
        model, "lm_head"
    ):
        print("  -> Purging duplicate lm_head before saving...")
        del model.lm_head

    model.save_pretrained(out_dir, save_compressed=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()


def plot_results(results_dict: dict):
    # --- IEEE Publication Settings ---
    # Standard IEEE column width is 3.5 inches
    WIDTH = 3.5
    HEIGHT = 2.8  # Gold aspect ratio for half-column
    FONT_SIZE = 8

    # Configure Matplotlib for LaTeX/Serif style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE - 2,
            "figure.titlesize": FONT_SIZE,
            "text.usetex": False,  # Set to True if you have a local LaTeX dist
        }
    )

    models = list(results_dict.keys())
    # Labels with asterisks for lossy methods
    methods = [
        "Baseline (BF16)",
        "rANS (Ours)",
        "ZipNN",
        "FP8*",
        "NVFP8*",
    ]

    data = {method: [] for method in methods}
    for m in models:
        data["Baseline (BF16)"].append(results_dict[m].get("Baseline (BF16)", 0.0))
        data["rANS (Ours)"].append(results_dict[m].get("rANS (Ours)", 0.0))
        data["ZipNN"].append(results_dict[m].get("ZipNN", 0.0))
        data["FP8*"].append(results_dict[m].get("FP8", 0.0))
        data["NVFP8*"].append(results_dict[m].get("NVFP8", 0.0))

    x = np.arange(len(models))
    n_methods = len(methods)
    width = 0.8 / n_methods  # Calculated to fit perfectly within the cluster

    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))

    # IEEE-compatible color palette (Distinguishable in grayscale/color)
    colors = [
        "#4D4D4D",
        "#82c8f0",
        "#f5a5c8",
        "#ffdca5",
        "#c3b9d7",
        "#cf286f",
        "#7dcdbe",
        "#c8c8c8",
    ]

    for i, (method, sizes) in enumerate(data.items()):
        # Calculate offset so bars are centered over the x-tick
        offset = (i - (n_methods - 1) / 2) * width
        rects = ax.bar(
            x + offset,
            sizes,
            width,
            label=method,
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )

        # Bar labels (Vertical and smaller to avoid overlap in 3.5" width)
        ax.bar_label(rects, padding=2, fmt="%.1f", fontsize=6, rotation=90)

    # Styling
    ax.set_ylabel("Size on Disk (GB)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in models], fontweight="bold")

    # Legend at the top (Horizontal) to save lateral space
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    # Tight layout with padding for the top legend
    plt.subplots_adjust(top=0.8, bottom=0.15, left=0.15, right=0.98)

    # Output PDF as requested
    plt.savefig("compression_comparison.pdf", format="pdf", dpi=300)
    print("\nPublication-ready PDF saved to compression_comparison.pdf")


# def main():
#     results = {}
#     for model_id in MODELS_TO_TEST:
#         print(f"\n🚀 Benchmarking {model_id}...")
#         results[model_id] = {}
#         model_name_clean = model_id.replace("/", "_")

#         # Define paths
#         paths = {
#             "Baseline (BF16)": os.path.join(BASE_DIR, model_name_clean, "baseline"),
#             "rANS (Ours)": os.path.join(BASE_DIR, model_name_clean, "rans"),
#             "ZipNN": os.path.join(BASE_DIR, model_name_clean, "zipnn"),
#             # "AWQ*": os.path.join(BASE_DIR, model_name_clean, "awq")
#         }

#         # 1. Baseline & Source Model
#         tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
#         model = AutoModelForCausalLM.from_pretrained(
#             model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
#         )
#         model.save_pretrained(paths["Baseline (BF16)"], safe_serialization=True)
#         results[model_id]["Baseline (BF16)"] = get_dir_size_gb(paths["Baseline (BF16)"])

#         # 2. Lossless Methods
#         compress_huffman(model, paths["Huff-LLM"])
#         results[model_id]["Huff-LLM"] = get_dir_size_gb(paths["Huff-LLM"])

#         compress_zipnn(model, paths["ZipNN"])
#         results[model_id]["ZipNN"] = get_dir_size_gb(paths["ZipNN"])

#         compress_rans(model, tokenizer, model_name_clean, paths["rANS (Ours)"])
#         results[model_id]["rANS (Ours)"] = get_dir_size_gb(paths["rANS (Ours)"])

#         # 3. Lossy Quantization Estimations
#         compress_gptq(model_id, paths["GPTQ*"])
#         results[model_id]["GPTQ*"] = get_dir_size_gb(paths["GPTQ*"])

#         # compress_awq(model_id, paths["AWQ*"])
#         # results[model_id]["AWQ*"] = get_dir_size_gb(paths["AWQ*"])

#         # Cleanup
#         del model
#         gc.collect()
#         torch.cuda.empty_cache()

#     # --- SAVE TO JSON ---
#     with open("compression_results.json", "w") as f:
#         json.dump(results, f, indent=4)
#     print("\n📊 Raw results saved to compression_results.json")

#     plot_results(results)


# if __name__ == "__main__":
#     main()


# def main(args):
#     # --- CHECK FOR EXISTING JSON ---
#     if args.load_json and os.path.exists(args.load_json):
#         print(f"Loading existing results from '{args.load_json}'...")
#         with open(args.load_json, "r") as f:
#             results = json.load(f)
#         print(" Skipping compression benchmarks and proceeding directly to plotting.")

#     else:
#         results = {}
#         for model_id in MODELS_TO_TEST:
#             print(f"\n Benchmarking {model_id}...")
#             results[model_id] = {}
#             model_name_clean = model_id.replace("/", "_")

#             # Define paths
#             paths = {
#                 "Baseline (BF16)": os.path.join(BASE_DIR, model_name_clean, "baseline"),
#                 "rANS (Ours)": os.path.join(BASE_DIR, model_name_clean, "rans"),
#                 "ZipNN": os.path.join(BASE_DIR, model_name_clean, "zipnn"),
#                 # "AWQ*": os.path.join(BASE_DIR, model_name_clean, "awq")
#                 "FP8*": os.path.join(BASE_DIR, model_name_clean, "fp8"),
#             }

#             # 1. Baseline & Source Model
#             tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
#             model = AutoModelForCausalLM.from_pretrained(
#                 model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
#             )
#             model.save_pretrained(paths["Baseline (BF16)"], safe_serialization=True)
#             results[model_id]["Baseline (BF16)"] = get_dir_size_gb(
#                 paths["Baseline (BF16)"]
#             )

#             dir_fp8 = os.path.join(BASE_DIR, model_name_clean, "fp8")
#             compress_llmcompressor_fp8(model_id, dir_fp8)
#             results[model_id]["FP8*"] = get_dir_size_gb(dir_fp8)

#             dir_nvfp4 = os.path.join(BASE_DIR, model_name_clean, "nvfp4")
#             compress_llmcompressor_nvfp4(model_id, dir_nvfp4)
#             results[model_id]["NVFP4*"] = get_dir_size_gb(dir_nvfp4)

#             # compress_huffman(model, paths["Huff-LLM"])
#             # results[model_id]["Huff-LLM"] = get_dir_size_gb(paths["Huff-LLM"])

#             compress_zipnn(model, paths["ZipNN"])
#             results[model_id]["ZipNN"] = get_dir_size_gb(paths["ZipNN"])

#             compress_rans(model, tokenizer, model_name_clean, paths["rANS (Ours)"])
#             results[model_id]["rANS (Ours)"] = get_dir_size_gb(paths["rANS (Ours)"])

#             # compress_awq(model_id, paths["AWQ*"])
#             # results[model_id]["AWQ*"] = get_dir_size_gb(paths["AWQ*"])

#             # Cleanup
#             del model
#             gc.collect()
#             torch.cuda.empty_cache()

#         # --- SAVE TO JSON ---
#         with open("compression_results.json", "w") as f:
#             json.dump(results, f, indent=4)
#         print("\nRaw results saved to compression_results.json")

#     # --- GENERATE PLOT ---
#     # This runs whether we just computed the results or loaded them from disk
#     plot_results(results)


def main(args):
    if args.load_json and os.path.exists(args.load_json):
        print(f"Loading existing results from '{args.load_json}'...")
        with open(args.load_json, "r") as f:
            results = json.load(f)
        print(" Skipping compression benchmarks and proceeding directly to plotting.")

    else:
        results = {}

        # Helper function to prevent re-running completed compressions
        def is_compressed(path):
            return os.path.exists(path) and len(os.listdir(path)) > 0

        for model_id in MODELS_TO_TEST:
            print(f"\n Benchmarking {model_id}...")
            results[model_id] = {}
            model_name_clean = model_id.replace("/", "_")

            # Define paths
            paths = {
                "Baseline (BF16)": os.path.join(BASE_DIR, model_name_clean, "baseline"),
                "rANS (Ours)": os.path.join(BASE_DIR, model_name_clean, "rans"),
                "ZipNN": os.path.join(BASE_DIR, model_name_clean, "zipnn"),
                "FP8*": os.path.join(BASE_DIR, model_name_clean, "fp8"),
                "NVFP4*": os.path.join(BASE_DIR, model_name_clean, "nvfp4"),
            }

            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True
            )

            if not is_compressed(paths["Baseline (BF16)"]):
                model.save_pretrained(paths["Baseline (BF16)"], safe_serialization=True)
            else:
                print("  -> Skipping Baseline (BF16): Already exists on disk.")
            results[model_id]["Baseline (BF16)"] = get_dir_size_gb(
                paths["Baseline (BF16)"]
            )

            if not is_compressed(paths["FP8*"]):
                compress_llmcompressor_fp8(model_id, paths["FP8*"])
            else:
                print("  -> Skipping FP8: Already exists on disk.")
            results[model_id]["FP8*"] = get_dir_size_gb(paths["FP8*"])

            if not is_compressed(paths["NVFP4*"]):
                compress_llmcompressor_nvfp4(model_id, paths["NVFP4*"])
            else:
                print("  -> Skipping NVFP4: Already exists on disk.")
            results[model_id]["NVFP4*"] = get_dir_size_gb(paths["NVFP4*"])

            if not is_compressed(paths["ZipNN"]):
                compress_zipnn(model, paths["ZipNN"])
            else:
                print("  -> Skipping ZipNN: Already exists on disk.")
            results[model_id]["ZipNN"] = get_dir_size_gb(paths["ZipNN"])

            if not is_compressed(paths["rANS (Ours)"]):
                compress_rans(model, tokenizer, model_name_clean, paths["rANS (Ours)"])
            else:
                print("  -> Skipping rANS (Ours): Already exists on disk.")
            results[model_id]["rANS (Ours)"] = get_dir_size_gb(paths["rANS (Ours)"])

            # Cleanup
            del model
            gc.collect()
            torch.cuda.empty_cache()

        with open("compression_results.json", "w") as f:
            json.dump(results, f, indent=4)
        print("\nRaw results saved to compression_results.json")

    plot_results(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM compressibility benchmarks.")
    parser.add_argument(
        "--load_json",
        type=str,
        default=None,
        help="Path to an existing JSON file to skip benchmarking and just plot.",
    )

    args = parser.parse_args()
    main(args)
