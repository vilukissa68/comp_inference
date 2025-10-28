#!/usr/bin/env python3

import torch.nn as nn


def has_tied_embeddings(model: nn.Module) -> bool:
    # Check if model has embeddings
    if not (
        hasattr(model, "get_input_embeddings")
        and hasattr(model, "get_output_embeddings")
    ):
        return False
    return (
        model.get_input_embeddings().weight.data_ptr()
        == model.get_output_embeddings().weight.data_ptr()
    )
