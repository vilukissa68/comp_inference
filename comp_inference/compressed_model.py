#!/usr/bin/env python3

import torch
import torch.nn as nn
from comp_inference.layers import compressed_layer_norm
from comp_inference.layers.compressed_linear import CompressedLinear
from comp_inference.layers.compressed_embedding import CompressedEmbedding
from comp_inference.layers.compressed_layer_norm import CompressedLayerNorm
from typing import Optional, Dict


class CompressedModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        compress_linear=True,
        compress_embedding=True,
        compress_layer_norm=True,
    ):
        """
        Wraps a model and automatically replaces Linear, Embedding, and LayerNorm layers
        with their compressed versions.
        """
        super().__init__()
        self.model = model
        self.compress_linear = compress_linear
        self.compress_embedding = compress_embedding
        self.compress_layer_norm = compress_layer_norm
        self._replace_all_layers(self.model)

    def _replace_all_layers(self, module: nn.Module):
        for name, child in module.named_children():
            # Replace Linear
            if isinstance(child, nn.Linear) and self.compress_linear:
                new_layer = CompressedLinear(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None),
                    dtype=child.weight.dtype,
                )
                new_layer.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.data.copy_(child.bias.data)
                setattr(module, name, new_layer)

            # Replace Embedding
            elif isinstance(child, nn.Embedding) and self.compress_embedding:
                new_layer = CompressedEmbedding(
                    num_embeddings=child.num_embeddings,
                    embedding_dim=child.embedding_dim,
                    padding_idx=child.padding_idx,
                    # max_norm=child.max_norm,
                    # norm_type=child.norm_type,
                    # scale_grad_by_freq=child.scale_grad_by_freq,
                    # sparse=child.sparse,
                )
                new_layer.weight.data.copy_(child.weight.data)
                setattr(module, name, new_layer)

            # Replace LayerNorm
            elif isinstance(child, nn.LayerNorm) and self.compress_layer_norm:
                new_layer = CompressedLayerNorm(
                    normalized_shape=child.normalized_shape,
                    eps=child.eps,
                    elementwise_affine=child.elementwise_affine,
                )
                if child.elementwise_affine:
                    new_layer.weight.data.copy_(child.weight.data)
                    new_layer.bias.data.copy_(child.bias.data)
                setattr(module, name, new_layer)

            else:
                self._replace_all_layers(child)

    def compress(self):
        """Compress all supported layers."""
        for layer in self.model.modules():
            if hasattr(layer, "compress"):
                layer.compress()

    def decompress(self):
        """Decompress all supported layers."""
        for layer in self.model.modules():
            if hasattr(layer, "decompress"):
                layer.decompress()

    def forward(self, *args, **kwargs):
        """
        Forward pass: decompress layers on the fly if compressed.
        """
        # Decompress layers temporarily if needed
        for layer in self.model.modules():
            if (
                hasattr(layer, "compressed_weight")
                and layer.compressed_weight is not None
            ):
                layer.decompress()

        return self.model(*args, **kwargs)

    def state_dict(self):
        state = {}
        for name, layer in self.model.named_modules():
            # skip containers
            if len(list(layer.children())) > 0:
                continue

            # Save compressed buffers if present
            if getattr(layer, "compressed_weight", None) is not None:
                state[f"{name}.compressed_weight"] = layer.compressed_weight
                state[f"{name}.weight_dtype"] = layer.weight_dtype
            elif hasattr(layer, "weight"):
                state[f"{name}.weight"] = layer.weight.data

            if getattr(layer, "compressed_bias", None) is not None:
                state[f"{name}.compressed_bias"] = layer.compressed_bias
                state[f"{name}.bias_dtype"] = layer.bias_dtype
            elif hasattr(layer, "bias") and layer.bias is not None:
                state[f"{name}.bias"] = layer.bias.data

        return state

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        for name, layer in self.model.named_modules():
            if len(list(layer.children())) > 0:
                continue

            # Load compressed buffers first
            if f"{name}.compressed_weight" in state_dict:
                layer.compressed_weight = state_dict[f"{name}.compressed_weight"]
                layer.weight_dtype = state_dict[f"{name}.weight_dtype"]
            elif f"{name}.weight" in state_dict:
                layer.weight.data.copy_(state_dict[f"{name}.weight"])

            if f"{name}.compressed_bias" in state_dict:
                layer.compressed_bias = state_dict[f"{name}.compressed_bias"]
                layer.bias_dtype = state_dict[f"{name}.bias_dtype"]
            elif f"{name}.bias" in state_dict and layer.bias is not None:
                layer.bias.data.copy_(state_dict[f"{name}.bias"])
