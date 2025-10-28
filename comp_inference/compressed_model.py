#!/usr/bin/env python3

import torch
import torch.nn as nn
from comp_inference.layers import compressed_layer_norm
from comp_inference.layers.compressed_linear import CompressedLinear
from comp_inference.layers.compressed_embedding import CompressedEmbedding
from comp_inference.layers.compressed_layer_norm import CompressedLayerNorm
from typing import Optional, Dict

from .utils.tied_embeddings import has_tied_embeddings


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
        self.has_tied_embeddings = has_tied_embeddings(self.model)
        self._replace_all_layers(self.model)

    def _replace_all_layers(
        self, module: nn.Module, shared_map: Optional[Dict[int, nn.Module]] = None
    ):
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
                    dtype=child.weight.dtype,
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
        return self

    def decompress(self):
        """Decompress all supported layers."""
        for layer in self.model.modules():
            if hasattr(layer, "decompress"):
                layer.decompress()
        return self

    def size_in_bytes(self) -> int:
        """Calculate total size of all unique (compressed or regular) weights in bytes."""
        total_size = 0

        for name, layer in self.model.named_modules():
            # Weight (compressed or not)
            if (
                hasattr(layer, "compressed_weight")
                and layer.compressed_weight is not None
            ):
                w = layer.compressed_weight
                total_size += w.element_size() * w.nelement()
            elif hasattr(layer, "weight"):
                w = layer.weight
                if w is not None:
                    total_size += w.element_size() * w.nelement()

            # Bias (compressed or not)
            if hasattr(layer, "compressed_bias") and layer.compressed_bias is not None:
                b = layer.compressed_bias
                total_size += b.element_size() * b.nelement()
            elif hasattr(layer, "bias") and layer.bias is not None:
                b = layer.bias
                total_size += b.element_size() * b.nelement()

        return total_size

    def forward(self, *args, **kwargs):
        """
        Forward pass: decompress layers on the fly if compressed.
        """
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

            print(f"Saving state for layer: {name}")
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

            print(f"Loading state for layer: {name}")

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

    def __getattr__(self, name):
        # Delegate attribute access to the model when not found in wrapper
        if name == "model":  # Prevent recursion
            return super().__getattr__(name)
        return getattr(self.model, name)
