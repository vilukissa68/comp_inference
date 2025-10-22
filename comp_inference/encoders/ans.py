#!/usr/bin/env python3

from nvidia import nvcomp
import cupy as cp
import torch


class ANSCompressor:
    """Class for encoding and decoding tensors using ANS compression."""

    def __init__(self):
        self.chunk_size = 65536
        # self.codec = nvcomp.Codec(algorithm="ANS", uncomp_chunk_size=self.chunk_size)
        self.codec = nvcomp.Codec(algorithm="ANS")
        pass

    def encode(self, tensor):
        tensor = tensor.contiguous().to("cuda")
        tensor = tensor.view(-1)
        nvarr = nvcomp.as_array(tensor)
        com_arr = self.codec.encode(nvarr)

        # Convert nvcomp Array → DLPack → Torch tensor
        torch_tensor = torch.utils.dlpack.from_dlpack(com_arr.to_dlpack())

        # Move compressed data to CPU for storage
        torch_tensor = torch_tensor.contiguous().to("cpu", non_blocking=False)

        print(f"Encoded tensor from {tensor.shape} → {torch_tensor.shape}")
        return torch_tensor

    def decode(self, bytestream, dtype, shape):
        print("Decoding bytestream of size:", bytestream.numel())

        bytestream = bytestream.contiguous().to("cuda", non_blocking=False)

        # Wrap as nvcomp array
        nvarr = nvcomp.as_array(bytestream)

        # Decode using nvcomp
        decoded_arr = self.codec.decode(nvarr)

        # Convert nvcomp Array → Torch tensor via DLPack
        decoded_tensor = torch.utils.dlpack.from_dlpack(decoded_arr.to_dlpack())

        # Reshape and cast dtype
        decoded_tensor = decoded_tensor.to(dtype).view(shape)

        print(f"Decoded tensor of shape: {decoded_tensor.shape}")
        return decoded_tensor
