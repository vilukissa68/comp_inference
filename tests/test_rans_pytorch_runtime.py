#!/usr/bin/env python3

#!/usr/bin/env python3
import argparse
import time
import os
import torch
from transformers import AutoTokenizer

# Import our custom loader that hooks the Triton kernels
from comp_inference import load_compressed_model_with_auto_model


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a rANS compressed LLM."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the directory containing config.json, tokenizer, and model.safetensors",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="In a strictly academic sense, the fundamental advantage of entropy coding in neural networks is",
        help="The input text to feed the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (TorchInductor) for faster execution.",
    )

    args = parser.parse_args()

    safetensors_path = os.path.join(args.model, "model.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(
            f"Error: Could not find {safetensors_path}. Did you pack the model correctly?"
        )

    # 1. Load Tokenizer
    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load the Hooked rANS Model
    model = load_compressed_model_with_auto_model(
        model_id=args.model,
        safetensors_path=safetensors_path,
        split_k=1,
        compile_model=args.compile,
    )
    model.eval()

    # 3. Prepare Inputs
    print("Tokenizing prompt...")
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[1]

    print("\n" + "=" * 60)
    print(f"PROMPT: {args.prompt}")
    print("=" * 60)

    # 4. Generate!
    print(f"Generating up to {args.max_tokens} tokens (Compile={args.compile})...")

    # Warmup run (highly recommended if using torch.compile so you don't benchmark the compilation time)
    if args.compile:
        print("   (Running warmup pass for TorchInductor compilation...)")
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        print("   (Warmup complete!)")

    # Start the actual timed generation
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,  # Greedy decoding for stable benchmarking
            use_cache=True,  # Enable KV cache
            pad_token_id=tokenizer.pad_token_id,
        )

    torch.cuda.synchronize()
    end_time = time.perf_counter()

    # 5. Decode and Calculate Metrics
    generated_ids = output_ids[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    num_generated_tokens = len(generated_ids)
    time_taken = end_time - start_time
    tps = num_generated_tokens / time_taken if time_taken > 0 else 0.0

    print(f"\n{args.prompt}{generated_text}")
    print("\n" + "=" * 60)
    print(f"Generation Complete!")
    print(f"Time taken: {time_taken:.3f} seconds")
    print(f"Speed:      {tps:.2f} tokens/second")
    print("=" * 60)


if __name__ == "__main__":
    main()
