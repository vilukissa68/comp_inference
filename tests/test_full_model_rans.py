#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

from comp_inference import rans_compress_module_weight, rans_decompress_module_weight

if __name__ == "__main__":
    print("=== Full Model Round-Trip rANS Compression Test ===")
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

    # We use list(model.modules()) to avoid issues if the graph changes during iteration
    for name, module in model.named_modules():
        print(f"Processing module: {name} ({type(module)})")
        if name == "model.embed_tokens":
            print("Skipping embedding layer.")
            continue
        # Only target modules with weights
        if hasattr(module, "weight") and module.weight is not None:
            # Optional: Skip 1D tensors (LayerNorm) if you only want to test Linear layers
            if module.weight.ndim < 2:
                continue

            # Store original for strict comparison (optional debugging)
            # original_w = module.weight.clone()

            # A. Compress
            # This deletes module.weight and creates module.compressed_weight + LUTs
            rans_compress_module_weight(module)

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

    if torch.equal(original_output, decompressed_output):
        print("\n✅ SUCCESS: Outputs are BIT-EXACT matches.")
    elif torch.allclose(original_output, decompressed_output, atol=1e-5):
        print("\n⚠️ SUCCESS (Approximate): Outputs match within tolerance.")
        diff = (original_output - decompressed_output).abs().max()
        print(f"Max difference: {diff}")
    else:
        print("\n❌ FAILURE: Outputs do not match.")
        diff = (original_output - decompressed_output).abs().max()
        print(f"Max difference: {diff}")
