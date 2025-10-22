#!/usr/bin/env python3
# Implements nn.Linear with weight compression and decompression
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CompressedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.in_features = in_features
        self.out_features = out_features

        # Save compressed data
        self.register_buffer("compressed_weight", None)
        self.register_buffer("compressed_bias", None)

    def compress(self):
        """Compress weights and bias"""
        with torch.no_grad():
            w = self.weight
            scale = w.abs().max() / 127
            q_w = torch.clamp((w / scale).round(), -128, 127).to(torch.int8)

            self.compressed_weight = q_w
            self.scale.data = torch.tensor(scale, device=w.device)

            if self.bias is not None:
                self.compressed_bias = self.bias.to(torch.float16)

            # Remove original weights to save memory
            del self._parameters["weight"]
            if self.bias is not None:
                del self._parameters["bias"]

    def decompress(self):
        """Restore the full-precision weights from the compressed form."""
        with torch.no_grad():
            if self.compressed_weight is None:
                raise RuntimeError("Layer not compressed!")

            w = self.compressed_weight.float() * self.scale
            self.weight = nn.Parameter(w)
            if self.compressed_bias is not None:
                self.bias = nn.Parameter(self.compressed_bias.float())

            # Free compressed form if not needed
            self.compressed_weight = None
            self.compressed_bias = None

    def forward(self, x: torch.Tensor):
        if self.compressed_weight is not None:
            self.decompress()
        return F.linear(x, self.weight, self.bias)
