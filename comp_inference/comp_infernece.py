import torch
import torch.nn as nn
from comp_inference.layers.compressed_linear import CompressedLinear
from comp_inference.layers.compressed_embedding import CompressedEmbedding


def replace_linear_with_compressed(module: nn.Module) -> nn.Module:
    """
    Recursively replace all nn.Linear layers with CompressedLinear.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            new_layer = CompressedLinear(
                child.in_features, child.out_features, bias=(child.bias is not None)
            )
            # Copy weights and bias
            new_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, new_layer)
        else:
            replace_linear_with_compressed(child)
    return module


def replace_embedding_with_compressed(module: nn.Module) -> nn.Module:
    """
    Recursively replace all nn.Embedding layers with CompressedEmbedding.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Embedding):
            new_layer = CompressedEmbedding(
                num_embeddings=child.num_embeddings,
                embedding_dim=child.embedding_dim,
                padding_idx=child.padding_idx,
                max_norm=child.max_norm,
                norm_type=child.norm_type,
                scale_grad_by_freq=child.scale_grad_by_freq,
                sparse=child.sparse,
            )

            # Copy weights
            new_layer.weight.data.copy_(child.weight.data)
            setattr(module, name, new_layer)
        else:
            replace_embedding_with_compressed(child)
    return module
