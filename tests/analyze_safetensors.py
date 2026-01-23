#!/usr/bin/env python3

import sys
from safetensors.torch import safe_open

def inspect(path):
    print(f"\n📦 Inspecting: {path}")
    total_size_bytes = 0
    
    metadata_size = 0

    with safe_open(path, framework="pt", device="cpu") as f:
        # 1. Print Metadata
        print(f"Metadata: {f.metadata()}")
        metadata_size = len(str(f.metadata()))
        
        
        # 2. Iterate Keys
        keys = f.keys()
        print(f"Total Tensors: {len(keys)}")
        print("-" * 80)
        print(f"{'Name':<50} | {'Shape':<20} | {'Dtype':<10} | {'Size (MB)':<10}")
        print("-" * 80)

        for key in keys:
            tensor = f.get_tensor(key)
            
            # Calculate size
            numel = tensor.numel()
            element_size = tensor.element_size() # bytes per element
            total_bytes = numel * element_size
            total_size_bytes += total_bytes
            
            size_mb = total_bytes / (1024 * 1024)
            dtype_str = str(tensor.dtype).replace("torch.", "")

            # Filter mostly for compressed layers to reduce noise
            # Remove this 'if' to see everything
            if "rans" in key or "expert" in key:
                print(f"{key:<50} | {str(list(tensor.shape)):<20} | {dtype_str:<10} | {size_mb:.4f}")

    print("-" * 80)
    print(f" Total Tensors Processed: {len(keys)}")
    print(f" Metadata Size: {metadata_size / 1024:.4f} KB")
    print(f"Total Content Size: {total_size_bytes / (1024**3):.4f} GB")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_safetensors.py <path_to_file>")
    else:
        inspect(sys.argv[1])
