import torch
import torch.nn as nn
from comp_inference.layers.compressed_linear import CompressedLinear


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
