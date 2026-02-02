#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse

from comp_inference import (
    load_compressed_model_with_auto_model,
    pack_and_save_tensors,
    save_rans_model_package,
    save_rans_model_gguf,
    rans_compress_qkv_fused,
    rans_compress_gate_up_fused,
    rans_compress_module_weight,
    rans_decompress_module_weight,
)


def successful_verification(name):
    print(f"\033[92m[PASS] Module {name} passed weight verification.\033[0m")


def failed_verification(name):
    print(f"\033[91m[FAIL] Weight mismatch in module: {name}\033[0m")


def main():
    parser = argparse.ArgumentParser(description="Full Model rANS Compression Test")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Name of the Hugging Face model",
    )
    parser.add_argument("--fuse_qkv", action="store_true", help="Enable QKV fusion")
    parser.add_argument(
        "--fuse_gate_up", action="store_true", help="Enable Gate+Up fusion"
    )
    parser.add_argument(
        "--skip_embedding", action="store_true", help="Skip embedding layer compression"
    )
    args = parser.parse_args()

    print(f"Loading {args.model_name} in Bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
    )
    model.to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    succesfuls = 0
    failures = 0

    previous_slot = torch.empty(0)
    previous_table = torch.empty(0)

    for name, module in model.named_modules():
        print(f"Processing module: {name} ({type(module)})")

        if hasattr(module, "weight"):
            # Skip small layers
            if module.weight.ndim < 2:
                continue

            original_weight = module.weight.clone()

            # Compress
            rans_compress_module_weight(module)

            current_slot = module.exponent_slot_map
            current_table = module.exponent_tables
            # Check if previous slot and table can be reused
            if torch.equal(current_slot, previous_slot):
                print(
                    "WARNING: Previous exponent slot map is same as current. Reusing previous table."
                )
            if torch.equal(current_table, previous_table):
                print(
                    "WARNING: Previous exponent table is same as current. Reusing previous table."
                )

            previous_slot = current_slot
            previous_table = current_table

            # Decompress
            rans_decompress_module_weight(module)

            # Verify
            if not torch.allclose(original_weight, module.weight, atol=1e-3):
                failed_verification(name)
                print(f"Orginal Weight Sample: {original_weight.view(-1)[:5]}")
                print(f"Decompressed Weight Sample: {module.weight.view(-1)[:5]}")
                failures += 1
            else:
                successful_verification(name)
                print(f"Orginal Weight Sample: {original_weight.view(-1)[:5]}")
                print(f"Decompressed Weight Sample: {module.weight.view(-1)[:5]}")
                succesfuls += 1
            print("-" * 80)
        print(f"Verification completed: {succesfuls}/{succesfuls + failures} passed.")


if __name__ == "__main__":
    main()
