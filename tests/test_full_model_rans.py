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
    args = parser.parse_args()

    print("Full Model Round-Trip rANS Compression Test")
    model_name = "Qwen/Qwen3-0.6B"

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

    compressed_count = 0
    total_params = 0

    for name, module in model.named_modules():
        print(f"Processing module: {name} ({type(module)})")
        if name == "model.embed_tokens" and args.skip_embedding:
            print("Skipping compressing embedding layer.")
            continue

        # QKV fusion
        if (
            args.fuse_qkv
            and hasattr(module, "q_proj")
            and hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
        ):
            print(f"Fusing QKV for {name}")
            rans_compress_qkv_fused(module)
            continue

        # Gate up fusion
        if args.fuse_gate_up and (
            hasattr(module, "gate")
            and hasattr(module, "up")
            or hasattr(module, "gate_proj")
            and hasattr(module, "up_proj")
        ):
            print(f"Fusing gate and up for {name}")
            rans_compress_gate_up_fused(module)
            continue

        # Only target modules with weights
        if hasattr(module, "weight") and module.weight is not None:
            # Skip 1D tensors (LayerNorm)
            if module.weight.ndim < 2:
                continue

            rans_compress_module_weight(module)
            if not hasattr(module, "compressed"):
                print(f"Warning: Compression failed for {name}")
    print("Compression phase completed.")
    pack_and_save_tensors(model, "compressed_model.safetensors")
    save_rans_model_package(model, tokenizer, "model")
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
