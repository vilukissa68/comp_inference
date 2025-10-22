#!/usr/bin/env python3

#!/usr/bin/env python3
# Implements nn.Embedding with weight compression and decompression
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CompressedEmbedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # Store compressed form
        self.register_buffer("compressed_weight", None)

    def compress(self):
        """Compress embedding weight."""
        with torch.no_grad():
            w = self.weight
            scale = w.abs().max() / 127
            q_w = torch.clamp((w / scale).round(), -128, 127).to(torch.int8)

            self.compressed_weight = q_w
            self.scale.data = torch.tensor(scale, device=w.device)

            # Remove full-precision weight to save memory
            del self._parameters["weight"]
            torch.cuda.empty_cache()

    def decompress(self):
        """Restore full-precision embedding weight."""
        with torch.no_grad():
            if self.compressed_weight is None:
                raise RuntimeError("Embedding not compressed!")

            w = self.compressed_weight.float() * self.scale
            self.weight = nn.Parameter(w)

            # Free compressed version
            self.compressed_weight = None
            torch.cuda.empty_cache()

    def forward(self, input: torch.Tensor):
        if self.compressed_weight is not None:
            self.decompress()
        return F.embedding(
            input,
            self.weight,
            padding_idx=self.padding_idx,
        )
