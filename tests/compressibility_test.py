#!/usr/bin/env python3
import os
import gc
import json
import torch
import zipnn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

# --- Attempt to import NVIDIA Model Optimizer ---
try:
    import modelopt.torch.quantization as mtq

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

# --- CONFIGURATION ---
MODELS_TO_TEST = [
    "Qwen/Qwen3-14B",
    # "meta-llama/Meta-Llama-3-8B",
]

BASE_DIR = "./compression_benchmarks"


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
    zpn = zipnn.ZipNN(input_format="torch", threads=1)
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
        tile_height = 64
        tile_width = 64

    args = Args()
    layers_compressed = 0

    for name, module in model.named_modules():
        if (
            name in ["model.embed_tokens", "model.lm_head", "lm_head"]
        ) and args.skip_embedding:
            continue

        if name in ["model.embed_tokens", "model.lm_head", "lm_head"]:
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
    save_rans_model_package(model, tokenizer, out_dir)


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


# --- PLOTTING ---
def plot_results(results_dict: dict):
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.family": "serif", "font.size": 12})

    models = list(results_dict.keys())
    methods = ["Baseline (BF16)", "rANS (Ours)", "ZipNN", "FP8", "NVFP8"]

    data = {method: [] for method in methods}
    for m in models:
        for method in methods:
            data[method].append(results_dict[m].get(method, 0.0))

    x = np.arange(len(models))
    width = 0.15
    multiplier = 0

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#cccccc", "#7DCDBE", "#FFA07A", "#82c8f0", "#9370DB"]

    for (method, sizes), color in zip(data.items(), colors):
        offset = width * multiplier
        rects = ax.bar(x + offset, sizes, width, label=method, color=color)
        ax.bar_label(rects, padding=3, fmt="%.1f GB", fontsize=9)
        multiplier += 1

    ax.set_ylabel("Model Size on Disk (GB)", fontweight="bold")
    ax.set_title("Compression Efficiency Comparison", fontweight="bold", y=1.05)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([m.split("/")[-1] for m in models])
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig("compression_comparison.svg", format="svg", bbox_inches="tight")
    print("\n✅ Graph saved to compression_comparison.svg")


# --- MAIN PIPELINE ---
def main():
    results = {}

    for model_id in MODELS_TO_TEST:
        print(f"\n🚀 Benchmarking {model_id}...")
        results[model_id] = {}
        model_name_clean = model_id.replace("/", "_")

        dir_base = os.path.join(BASE_DIR, model_name_clean, "baseline")
        dir_zipnn = os.path.join(BASE_DIR, model_name_clean, "zipnn")
        dir_rans = os.path.join(BASE_DIR, model_name_clean, "rans")
        dir_fp8 = os.path.join(BASE_DIR, model_name_clean, "fp8")
        dir_nvfp8 = os.path.join(BASE_DIR, model_name_clean, "nvfp8")

        # 1. Load Baseline to measure size and act as source
        print("  -> Loading baseline model...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )

        model.save_pretrained(dir_base, safe_serialization=True)
        results[model_id]["Baseline (BF16)"] = get_dir_size_gb(dir_base)

        # 2. Run Python-native ZipNN Compression
        compress_zipnn(model, dir_zipnn)
        results[model_id]["ZipNN"] = get_dir_size_gb(dir_zipnn)

        # 3. Run Custom rANS Compression
        # We pass the in-memory model directly to your functions
        compress_rans(model, tokenizer, model_name_clean, dir_rans)
        results[model_id]["rANS (Ours)"] = get_dir_size_gb(dir_rans)

        # Free the baseline memory before running NVIDIA ModelOpt (which loads its own instance)
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # 4. Run NVIDIA ModelOpt FP8/NVFP8
        compress_nvidia_fp8(model_id, dir_fp8, format_type="fp8_e4m3")
        results[model_id]["FP8"] = get_dir_size_gb(dir_fp8)

        compress_nvidia_fp8(model_id, dir_nvfp8, format_type="nvfp8")
        results[model_id]["NVFP8"] = get_dir_size_gb(dir_nvfp8)

    # Output and Graph
    with open("compression_results.json", "w") as f:
        json.dump(results, f, indent=4)

    plot_results(results)


if __name__ == "__main__":
    main()
