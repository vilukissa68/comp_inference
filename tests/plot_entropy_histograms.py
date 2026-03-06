#!/usr/bin/env python3

#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM

# --- PUBLICATION STYLE CONFIGURATION ---
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }
)


def calc_entropy(counts: torch.Tensor) -> float:
    """Calculates Shannon Entropy in bits from a tensor of bin counts."""
    p = counts.float() / counts.sum()
    p = p[p > 0]  # Drop zero probabilities to avoid log2(0) NaN
    return -(p * torch.log2(p)).sum().item()


def split_bfloat16(tensor: torch.Tensor):
    """
    Splits a bfloat16 tensor into two uint8 streams:
    1. Exponent (8 bits): The isolated IEEE 754 exponent.
    2. Sign+Mantissa (8 bits): 1 Sign bit + 7 Mantissa bits.
    """
    # View memory as 16-bit integers to allow bitwise operations
    int16_view = tensor.contiguous().view(torch.int16)

    # Extract the 8-bit exponent (Bits 7 to 14)
    # Right shift by 7, then mask with 0xFF (255) to strip the sign bit
    exponent = (int16_view >> 7) & 0xFF

    # Extract the 7-bit mantissa (Bits 0 to 6)
    mantissa = int16_view & 0x7F

    # Extract the 1-bit sign (Bit 15) and move it to the 8th bit position
    sign = (int16_view >> 15) & 0x1

    # Pack Sign + Mantissa into a single byte
    sign_mantissa = (sign << 7) | mantissa

    return exponent.to(torch.uint8), sign_mantissa.to(torch.uint8)


def fuse_projections(state_dict: dict) -> dict:
    """
    Identifies fragmented Q/K/V and Gate/Up projections and fuses them
    along the output dimension (dim 0). Safely ignores and restores
    incomplete groups without dropping parameters.
    """
    print("Fusing QKV and Gate+Up projections...")
    new_state_dict = {}
    fused_groups = {}

    for key, tensor in state_dict.items():
        if "q_proj" in key or "k_proj" in key or "v_proj" in key:
            base = (
                key.replace("q_proj", "qkv_proj")
                .replace("k_proj", "qkv_proj")
                .replace("v_proj", "qkv_proj")
            )
            if base not in fused_groups:
                fused_groups[base] = {"q": None, "k": None, "v": None, "orig_keys": []}

            if "q_proj" in key:
                fused_groups[base]["q"] = tensor
            if "k_proj" in key:
                fused_groups[base]["k"] = tensor
            if "v_proj" in key:
                fused_groups[base]["v"] = tensor

            # Cache original key/tensor pair in case fusion fails
            fused_groups[base]["orig_keys"].append((key, tensor))

        elif "gate_proj" in key or "up_proj" in key:
            base = key.replace("gate_proj", "gate_up_proj").replace(
                "up_proj", "gate_up_proj"
            )
            if base not in fused_groups:
                fused_groups[base] = {"gate": None, "up": None, "orig_keys": []}

            if "gate_proj" in key:
                fused_groups[base]["gate"] = tensor
            if "up_proj" in key:
                fused_groups[base]["up"] = tensor

            fused_groups[base]["orig_keys"].append((key, tensor))
        else:
            new_state_dict[key] = tensor

    # Perform the concatenation
    for base_key, parts in fused_groups.items():
        # Check if it's a valid QKV group
        if (
            "q" in parts
            and parts["q"] is not None
            and parts["k"] is not None
            and parts["v"] is not None
        ):
            new_state_dict[base_key] = torch.cat(
                [parts["q"], parts["k"], parts["v"]], dim=0
            )

        # Check if it's a valid Gate+Up group
        elif "gate" in parts and parts["gate"] is not None and parts["up"] is not None:
            new_state_dict[base_key] = torch.cat([parts["gate"], parts["up"]], dim=0)

        # If incomplete (e.g. Q has a bias, but K/V don't), restore the original tensors unharmed
        else:
            for orig_key, orig_tensor in parts["orig_keys"]:
                new_state_dict[orig_key] = orig_tensor

    print(
        f"Reduced tensor count from {len(state_dict)} to {len(new_state_dict)} via fusion."
    )
    return new_state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Analyze bfloat16 compressibility via Exponent/Mantissa histograms."
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HF Model ID or local path"
    )
    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Specific layer name to analyze (e.g., 'model.layers.0.mlp.gate_up_proj.weight'). If None, aggregates all linear layers.",
    )
    parser.add_argument(
        "--fuse",
        action="store_true",
        help="Fuse QKV and Gate+Up projections before analysis.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="entropy_analysis.pdf",
        help="Output plot filename.",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",  # Keep on CPU to avoid massive VRAM usage during global aggregation
    )

    state_dict = model.state_dict()

    if args.fuse:
        state_dict = fuse_projections(state_dict)

    # Accumulators for the 256-bin histograms
    global_exp_counts = torch.zeros(256, dtype=torch.long)
    global_man_counts = torch.zeros(256, dtype=torch.long)

    total_params = 0
    analyzed_layers = 0

    print("Extracting and Binning Streams...")
    for name, tensor in state_dict.items():
        # Only analyze weights (skip layernorms, biases, and non-2D tensors)
        if "weight" not in name or tensor.dim() < 2:
            continue

        # If a specific layer is requested, skip everything else
        if args.layer and args.layer != name:
            continue

        exp_uint8, man_uint8 = split_bfloat16(tensor)

        # Accumulate bincounts (minlength=256 ensures identical array sizes)
        global_exp_counts += torch.bincount(exp_uint8.flatten().long(), minlength=256)
        global_man_counts += torch.bincount(man_uint8.flatten().long(), minlength=256)

        total_params += tensor.numel()
        analyzed_layers += 1

    if analyzed_layers == 0:
        print("No layers matched the criteria. Check your --layer argument.")
        return

    print(f"Analyzed {total_params:,} parameters across {analyzed_layers} layers.")

    # Calculate Shannon Entropy
    exp_entropy = calc_entropy(global_exp_counts)
    man_entropy = calc_entropy(global_man_counts)

    print("-" * 40)
    print(f"Exponent Entropy:       {exp_entropy:.3f} bits / symbol (Max: 8.0)")
    print(f"Sign+Mantissa Entropy:  {man_entropy:.3f} bits / symbol (Max: 8.0)")
    total_entropy = exp_entropy + man_entropy
    print(f"Theoretical File Size:  {total_entropy / 16.0:.3f}x Compression Ratio")
    print("-" * 40)

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Exponent Histogram
    axes[0].bar(range(256), global_exp_counts.numpy(), color="#82c8f0", width=1.0)
    axes[0].set_title(
        f"Exponent Distribution\nEntropy: {exp_entropy:.2f} bits", fontweight="bold"
    )
    axes[0].set_xlabel("8-bit Exponent Value")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0, 255)

    # Mantissa Histogram
    axes[1].bar(range(256), global_man_counts.numpy(), color="#FFDCA5", width=1.0)
    axes[1].set_title(
        f"Sign+Mantissa Distribution\nEntropy: {man_entropy:.2f} bits",
        fontweight="bold",
    )
    axes[1].set_xlabel("8-bit Sign+Mantissa Value")
    axes[1].set_xlim(0, 255)

    suptitle = f"Compressibility Analysis: {args.model}"
    if args.fuse:
        suptitle += " (Fused Projections)"
    if args.layer:
        suptitle += f"\nLayer: {args.layer}"

    fig.suptitle(suptitle, y=1.05, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    plt.savefig(
        "entropy_histograms.svg", format="svg", transparent=True, bbox_inches="tight"
    )
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
