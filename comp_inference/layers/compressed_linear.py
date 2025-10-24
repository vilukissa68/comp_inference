#!/usr/bin/env python3
# Implements nn.Linear with weight compression and decompression
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..encoders.ans import ANSCompressor
from ..utils.dtype_enum import DTypeEnum, DTYPE_TO_ENUM, ENUM_TO_DTYPE


class CompressedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype=torch.float32,
        bias_dtype=None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if bias_dtype is None:
            bias_dtype = dtype
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=bias_dtype)) if bias else None
        )
        self.in_features = in_features
        self.out_features = out_features

        # Save compressed data
        self.register_buffer("compressed_weight", None)
        self.register_buffer("compressed_bias", None)

        # Buffers to track original dtype
        self.register_buffer(
            "weight_dtype",
            torch.tensor(DTYPE_TO_ENUM[self.weight.dtype], dtype=torch.int8),
            persistent=True,
        )
        if bias is not None:
            self.register_buffer(
                "bias_dtype",
                torch.tensor(DTYPE_TO_ENUM[self.bias.dtype], dtype=torch.int8),
                persistent=True,
            )
        self.encoder = ANSCompressor()

    def compress(self):
        """Compress weights and bias"""
        with torch.no_grad():
            self.compressed_weight = self.encoder.encode(self.weight)
            original_size = self.weight.element_size() * self.weight.nelement()
            compressed_size = (
                self.compressed_weight.element_size()
                * self.compressed_weight.nelement()
            )
            print(
                "[CompressedLinear] Compressed weight from {:.2f} MB to {:.2f} MB. Compression ratio: {}".format(
                    original_size / (1024 * 1024),
                    compressed_size / (1024 * 1024),
                    compressed_size / original_size,
                )
            )

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
                dtype=ENUM_TO_DTYPE[self.weight_dtype.item()],
                shape=(self.out_features, self.in_features),
            )
            self.weight = nn.Parameter(w)
            if self.compressed_bias is not None:
                self.bias = nn.Parameter(
                    self.encoder.decode(
                        self.compressed_bias,
                        dtype=ENUM_TO_DTYPE[self.bias_dtype.item()],
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
        ), "Device mismatch in CompressedLinear forward. x.device: {}, weight.device: {}, bias.device: {}".format(
            x.device,
            self.weight.device,
            None if self.bias is None else self.bias.device,
        )
        return F.linear(x, self.weight, self.bias)
