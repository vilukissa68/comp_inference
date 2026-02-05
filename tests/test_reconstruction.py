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
        print("-" * 80)
        print(f"Processing module: {name} ({type(module)})")

        if hasattr(module, "weight"):
            # Skip small layers
            if module.weight.ndim < 2:
                continue

            module.weight = torch.nn.Parameter(
                module.weight.t().contiguous()
            )  # Transpose to [K, N] for compression
            original_weight = module.weight.clone()

            # Compress
            rans_compress_module_weight(module, block_size=512, transpose_weight=False)

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
            print("Decompressed weight: ", module.weight.shape)
            # ----------------------------------------------------------------
            # DRIFT ANALYZER: Pinpoint the exact (row, col) of divergence
            # ----------------------------------------------------------------
            TILE_K = 1024
            TILE_N = 32

            # We analyze the RAW DECOMPRESSED EXPONENTS (uint8)
            # because reconstruction (bf16) can hide the exact point of failure
            H, W = original_weight.shape  # Assuming you have the original uint8s
            num_tiles_k = (H + TILE_K - 1) // TILE_K
            num_tiles_n = (W + TILE_N - 1) // TILE_N

            print(f"Analyzing drift for {num_tiles_k}x{num_tiles_n} tiles...")

            drift_stats = []

            for tk in range(num_tiles_k):
                for tn in range(num_tiles_n):
                    r_start = tk * TILE_K
                    r_end = min(r_start + TILE_K, H)
                    c_start = tn * TILE_N
                    c_end = min(c_start + TILE_N, W)

                    orig_tile = original_weight[r_start:r_end, c_start:c_end]
                    deco_tile = module.weight[r_start:r_end, c_start:c_end]

                    # Find the first row in this tile that has a mismatch
                    mismatch_mask = orig_tile != deco_tile

                    if not torch.any(mismatch_mask):
                        continue  # Tile is perfect

                    # Find the index of the first row with a mismatch
                    # mismatch_mask.any(dim=1) gives a boolean vector of size [rows_in_tile]
                    rows_with_mismatch = torch.any(mismatch_mask, dim=1)
                    first_bad_row_local = torch.where(rows_with_mismatch)[0][0].item()

                    # Find which column in that row failed first
                    first_bad_col_local = torch.where(
                        mismatch_mask[first_bad_row_local]
                    )[0][0].item()

                    drift_stats.append(
                        {
                            "tile": (tk, tn),
                            "first_bad_row": first_bad_row_local,
                            "first_bad_col": first_bad_col_local,
                            "orig_val": orig_tile[
                                first_bad_row_local, first_bad_col_local
                            ].item(),
                            "deco_val": deco_tile[
                                first_bad_row_local, first_bad_col_local
                            ].item(),
                        }
                    )

            # --- ANALYSIS REPORT ---
            if not drift_stats:
                print("SUCCESS: No drift detected in any tile!")
            else:
                print(
                    f"\n[DRIFT REPORT] Total Corrupted Tiles: {len(drift_stats)} / {num_tiles_k * num_tiles_n}"
                )

                # Analyze the row of failure
                first_rows = [d["first_bad_row"] for d in drift_stats]
                avg_fail_row = sum(first_rows) / len(first_rows)
                min_fail_row = min(first_rows)

                print(f"Average Row of Failure: {avg_fail_row:.2f}")
                print(f"Earliest Row of Failure: {min_fail_row}")

                print("\n[DETAILED DRIFT SAMPLES]")
                for i, d in enumerate(drift_stats[:10]):
                    print(
                        f"Tile {d['tile']}: Fails at Row {d['first_bad_row']}, Col {d['first_bad_col']}. "
                        f"Expected {d['orig_val']}, Got {d['deco_val']}"
                    )

            print("-" * 80)
            # Verify
            if not torch.allclose(original_weight, module.weight, atol=1e-8):
                failed_verification(name)
                print(
                    f"Orginal Weight Sample: {original_weight.view(-1)[:5]} ... {original_weight.view(-1)[-5:]}"
                )
                print(
                    f"Decompressed Weight Sample: {module.weight.view(-1)[:5]} ... {module.weight.view(-1)[-5:]}"
                )
                failures += 1
            else:
                successful_verification(name)
                print(
                    f"Orginal Weight Sample: {original_weight.view(-1)[:5]} ... {original_weight.view(-1)[-5:]}"
                )
                print(
                    f"Decompressed Weight Sample: {module.weight.view(-1)[:5]} ... {module.weight.view(-1)[-5:]}"
                )
                succesfuls += 1
            print("-" * 80)
        print(f"Verification completed: {succesfuls}/{succesfuls + failures} passed.")


if __name__ == "__main__":
    main()
