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
    parser.add_argument(
        "--tile_height", type=int, default=1024, help="rANS tile height"
    )
    parser.add_argument("--tile_width", type=int, default=32, help="rANS tile width")
    parser.add_argument(
        "--no_transpose",
        action="store_true",
        default=False,
        help="Tranpose standard linear layers",
    )

    args = parser.parse_args()

    print("Full Model Round-Trip rANS Compression Test")
    model_name = args.model_name

    print(f"Loading {model_name} in Bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    )
    model.to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_text = "What is the capital of France?"
    inputs = tokenizer(input_text, return_tensors="pt")

    # 1. Baseline Run
    print("Running baseline inference...")
    with torch.no_grad():
        original_output = model(**inputs).logits

    # 2. Compress & Decompress Loop
    print("Starting Round-Trip (Compress -> Decompress)...")

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

    compressed_count = 0
    total_params = 0
    layers_compressed = 0

    for name, module in model.named_modules():
        print("-" * 80)
        print(f"Processing module: {name} ({type(module)})")

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

        # QKV fusion
        if (
            args.fuse_qkv
            and hasattr(module, "q_proj")
            and hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
        ):
            print(f"Fusing QKV for {name}")
            rans_compress_qkv_fused(
                module, tile_height=args.tile_height, tile_width=args.tile_width
            )
            layers_compressed += 1
            continue

        # Gate up fusion
        if args.fuse_gate_up and (
            hasattr(module, "gate")
            and hasattr(module, "up")
            or hasattr(module, "gate_proj")
            and hasattr(module, "up_proj")
        ):
            print(f"Fusing gate and up for {name}")
            rans_compress_gate_up_fused(
                module, tile_height=args.tile_height, tile_width=args.tile_width
            )
            layers_compressed += 1
            continue

        # Only target modules with weights
        if hasattr(module, "weight") and module.weight is not None:
            # Skip 1D tensors (LayerNorm)
            if module.weight.ndim < 2:
                continue

            rans_compress_module_weight(
                module,
                tile_height=args.tile_height,
                tile_width=args.tile_width,
                transpose_weight=not args.no_transpose,
            )
            layers_compressed += 1
            if not hasattr(module, "compressed"):
                print(f"Warning: Compression failed for {name}")
    print("Compression phase completed.")
    fuse = args.fuse_qkv or args.fuse_gate_up
    # pack_and_save_tensors(model, "compressed_model.safetensors", fuse=fuse)

    model_name_clean = model_name.replace("/", "_")
    save_rans_model_package(
        model, tokenizer, model_name_clean, fuse=fuse, tied_lm_head=tied_lm_head
    )
    save_rans_model_gguf(model, tokenizer, "compressed_model.gguf", model_name)
    del model  # Free memory

    # Read back the compressed model
    model = load_compressed_model_with_auto_model(
        model_name,
        "compressed_model.safetensors",
        device="cpu",
    )

    print("Model reloaded from compressed file.")

    for name, module in model.named_modules():
        # Check if compression actually happened
        if hasattr(module, "compressed"):
            compressed_count += 1

            # B. Decompress
            # This reads compressed_weight + LUTs and recreates module.weight
            rans_decompress_module_weight(module)

            # Verify shape restoration
            assert hasattr(module, "weight"), f"Decompression failed for {name}"

    print(f"Processed {compressed_count} layers.")

    # 3. Verification Run
    print("Running verification inference...")
    with torch.no_grad():
        decompressed_output = model(**inputs).logits

    # 4. Compare
    # rANS is a lossless entropy coding. If your bit-packing logic is correct,
    # the error should be 0.0 (identical).
    # However, float/bf16 casting sometimes introduces tiny bit differences if not strictly cast.
    # Additionally parameter fusion can't be tested here as the model is reloaded from disk

    if not args.fuse_qkv and not args.fuse_gate_up:
        if torch.equal(original_output, decompressed_output):
            print("\nSUCCESS: Outputs are BIT-EXACT matches.")
        elif torch.allclose(original_output, decompressed_output, atol=1e-5):
            print("\nSUCCESS (Approximate): Outputs match within tolerance.")
            diff = (original_output - decompressed_output).abs().max()
            print(f"Max difference: {diff}")
        else:
            print("\nFAILURE: Outputs do not match.")
            diff = (original_output - decompressed_output).abs().max()
            print(f"Max difference: {diff}")
    else:
        print("Weight fusion enable, can't verify parameter correctness")

    # Convert output to text
    original_text = tokenizer.decode(torch.argmax(original_output, dim=-1)[0])
    decompressed_text = tokenizer.decode(torch.argmax(decompressed_output, dim=-1)[0])
    print(f"\nOriginal Output Text: {original_text}")
    print(f"Decompressed Output Text: {decompressed_text}")

    # Reload the model and verify all the parameters are back to original
    print("\nReloading model to verify parameter integrity...")
    reloaded_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    )
    reloaded_model.to("cpu")
    mismatch_count = 0
    for (name1, param1), (name2, param2) in zip(
        model.named_parameters(), reloaded_model.named_parameters()
    ):
        if not torch.equal(param1, param2):
            mismatch_count += 1
            print(f"Parameter mismatch: {name1} vs {name2}")
            print(f"Original param: {param1}")
            print(f"Reloaded param: {param2}")
    if mismatch_count == 0:
        print("All parameters match the reloaded model.")
    else:
        print(f"{mismatch_count} parameters did not match the reloaded model.")


if __name__ == "__main__":
    main()
