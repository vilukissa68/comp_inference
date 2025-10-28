#!/usr/bin/env python3

#!/usr/bin/env python3
# Implements nn.Embedding with weight compression and decompression
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from ..encoders.ans import ANSCompressor
from ..utils.dtype_enum import DTypeEnum, DTYPE_TO_ENUM, ENUM_TO_DTYPE


class CompressedEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, dtype=dtype)
        )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Store compressed form
        self.register_buffer("compressed_weight", None)

        # Buffer to track original dtype
        self.register_buffer(
            "weight_dtype",
            torch.tensor(DTYPE_TO_ENUM[self.weight.dtype], dtype=torch.int8),
            persistent=True,
        )

        self.encoder = ANSCompressor()

    def compress(self):
        """Compress embedding weight."""
        with torch.no_grad():
            self.compressed_weight = self.encoder.encode(self.weight)
            original_size = self.weight.element_size() * self.weight.nelement()
            compressed_size = (
                self.compressed_weight.element_size()
                * self.compressed_weight.nelement()
            )
            print(
                "[CompressedEmbedding] Compressed weight from {:.2f} MB to {:.2f} MB. Compression ratio: {:.2f}x".format(
                    original_size / (1024 * 1024),
                    compressed_size / (1024 * 1024),
                    compressed_size / original_size,
                )
            )
            # Remove full-precision weight to save memory
            del self._parameters["weight"]
            torch.cuda.empty_cache()

    def decompress(self):
        """Restore full-precision embedding weight."""
        with torch.no_grad():
            if self.compressed_weight is None:
                raise RuntimeError("Embedding not compressed!")

            w = self.encoder.decode(
                self.compressed_weight,
                dtype=ENUM_TO_DTYPE[self.weight_dtype.item()],
                shape=(self.num_embeddings, self.embedding_dim),
            )
            self.weight = nn.Parameter(w)
            self.compressed_weight = None

    def forward(self, x: torch.Tensor):
        if self.compressed_weight is not None:
            self.decompress()
        return F.embedding(
            x,
            self.weight,
            padding_idx=self.padding_idx,
        )
