#!/usr/bin/env python3
import torch
import torch.nn as nn
from typing import Optional


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
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def compress(self):
        """Quantize weight and bias to int8 / float16"""
        if not self.elementwise_affine:
            return

        with torch.no_grad():
            w = self.weight
            scale = w.abs().max() / 127
            q_w = torch.clamp((w / scale).round(), -128, 127).to(torch.int8)

            self.compressed_weight = q_w
            self.scale.data = torch.tensor(scale, device=w.device)

            if self.bias is not None:
                self.compressed_bias = self.bias.to(torch.float16)

            del self._parameters["weight"]
            if self.bias is not None:
                del self._parameters["bias"]

    def decompress(self):
        """Dequantize back to float32"""
        if self.compressed_weight is None:
            return

        with torch.no_grad():
            w = self.compressed_weight.float() * self.scale
            self.weight = nn.Parameter(w)
            if self.compressed_bias is not None:
                self.bias = nn.Parameter(self.compressed_bias.float())

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
