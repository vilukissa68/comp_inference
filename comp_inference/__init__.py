#!/usr/bin/env python3

from .comp_infernece import (
    replace_linear_with_compressed,
    replace_embedding_with_compressed,
    replace_layer_norm_with_compressed,
    replace_all_with_compressed,
    fill_ans_table,
    rans_compress_module_weight
    rans_decompress_module_weight,

)
from .layers.compressed_linear import CompressedLinear
from .layers.compressed_embedding import CompressedEmbedding
from .layers.compressed_layer_norm import CompressedLayerNorm
from .compressed_model import CompressedModel
from .encoders.ans import ANSCompressor

__all__ = [
    "replace_linear_with_compressed",
    "CompressedLinear",
    "replace_embedding_with_compressed",
    "CompressedEmbedding",
    "replace_layer_norm_with_compressed",
    "CompressedLayerNorm",
    "replace_all_with_compressed",
]
