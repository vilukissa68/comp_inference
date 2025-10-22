#!/usr/bin/env python3
# Implements nn.Linear with weight compression and decompression
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..encoders.ans import ANSCompressor


class CompressedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.in_features = in_features
        self.out_features = out_features

        # Save compressed data
        # self.register_buffer("compressed_weight", None)
        self.register_buffer("compressed_bias", None)
        self.compressed_weight = None
        self.encoder = ANSCompressor()

    def compress(self):
        """Compress weights and bias"""
        with torch.no_grad():
            self.compressed_weight = self.encoder.encode(self.weight)

            if self.bias is not None:
                self.compressed_bias = self.encoder.encode(self.bias)

            # Remove original weights to save memory
            del self._parameters["weight"]
            if self.bias is not None:
                del self._parameters["bias"]

    def decompress(self):
        """Restore the full-precision weights from the compressed form."""
        with torch.no_grad():
            if self.compressed_weight is None:
                raise RuntimeError("Layer not compressed!")

            w = self.encoder.decode(
                self.compressed_weight,
                dtype=torch.float32,  # TODO: make dtype configurable
                shape=(self.out_features, self.in_features),
            )
            self.weight = nn.Parameter(w)
            if self.compressed_bias is not None:
                self.bias = nn.Parameter(
                    self.encoder.decode(
                        self.compressed_bias,
                        dtype=torch.float32,
                        shape=(self.out_features,),
                    )
                )

            # Free compressed form if not needed
            self.compressed_weight = None
            self.compressed_bias = None

    def forward(self, x: torch.Tensor):
        if self.compressed_weight is not None:
            self.decompress()
        assert x.device == self.weight.device and (
            self.bias is None or x.device == self.bias.device
        ), "Device mismatch in CompressedLinear forward"
        return F.linear(x, self.weight, self.bias)
