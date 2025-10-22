#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Optional
from ..encoders.ans import ANSCompressor


class CompressedLayerNorm(nn.Module):
    """
    Drop-in replacement for nn.LayerNorm with support for quantized/compressed parameters.
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else tuple(normalized_shape)
        )
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("compressed_weight", None)
        self.register_buffer("compressed_bias", None)
        self.encoder = ANSCompressor()

    def compress(self):
        """Compress weights and biases with chosen encoder"""
        if not self.elementwise_affine:
            return

        with torch.no_grad():
            self.compressed_weight = self.encoder.encode(self.weight)

            if self.bias is not None:
                self.compressed_bias = self.encoder.encode(self.bias)

            del self._parameters["weight"]
            if self.bias is not None:
                del self._parameters["bias"]

    def decompress(self):
        """Decompress weights and biases with chosen encoder"""
        if self.compressed_weight is None:
            return

        with torch.no_grad():
            w = self.encoder.decode(
                self.compressed_weight,
                dtype=torch.float32,  # TODO: make dtype configurable
                shape=self.normalized_shape,
            )
            self.weight = nn.Parameter(w)
            if self.compressed_bias is not None:
                self.bias = self.encoder.decode(
                    self.compressed_bias,
                    dtype=torch.float32,  # TODO: make dtype configurable
                    shape=self.normalized_shape,
                )

            self.compressed_weight = None
            self.compressed_bias = None

    def forward(self, x: torch.Tensor):
        if self.compressed_weight is not None:
            self.decompress()
        return nn.functional.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )
