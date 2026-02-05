#!/usr/bin/env python3

import torch

# Load both
truth = torch.load("../compression_truth.pt")["weight_kn"]  # [K, N]
recon = torch.load("../../vllm-entropy/debug_dump.pt")["reconstructed_weight"]  # [K, N]

# 1. Check for absolute parity
diff = torch.abs(truth - recon)
print(f"Max Difference: {diff.max().item()}")

# 2. THE TRANSPOSE TEST
# If the error is high, check if you accidentally compared [K, N] vs [N, K]
diff_t = torch.abs(truth - recon.t()) if truth.shape != recon.shape else None
if diff_t is not None:
    print(f"Max Diff if Transposed: {diff_t.max().item()}")

# 3. THE MANTISSA SIGN-BIT TEST
# If Max Diff is exactly 0.03125 or 0.0625, it's a bit-shift error.
# Let's look at the raw bits of a failing element
mask = diff > 0
if mask.any():
    idx = mask.nonzero()[0]
    t_val = truth[idx[0], idx[1]]
    r_val = recon[idx[0], idx[1]]
    print(f"\nFirst Mismatch at {idx.tolist()}:")
    print(f"Truth: {t_val.item():.6f} | Bits: {bin(t_val.view(torch.uint16).item())}")
    print(f"Recon: {r_val.item():.6f} | Bits: {bin(r_val.view(torch.uint16).item())}")
