#!/usr/bin/env python3

from .comp_infernece import replace_linear_with_compressed
from .layers.compressed_linear import CompressedLinear

__all__ = ["replace_linear_with_compressed", "CompressedLinear"]
