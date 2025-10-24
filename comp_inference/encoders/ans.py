#!/usr/bin/env python3

from nvidia import nvcomp
import cupy as cp
import torch
import numpy as np


# class ANSCompressor:
#     """Class for encoding and decoding tensors using ANS compression."""

#     def __init__(self):
#         self.chunk_size = 16384  # 16 KB
#         self.codec = nvcomp.Codec(algorithm="ZSTD", uncomp_chunk_size=self.chunk_size)
#         # self.codec = nvcomp.Codec(algorithm="ANS")
#         pass

#     def encode(self, tensor):
#         print("Encoding tensor with shape:", tensor.shape, "and dtype:", tensor.dtype)
#         tensor = tensor.contiguous().to("cuda")
#         tensor = tensor.view(torch.uint8)  # Cast to uint8 for nvcomp compatibility
#         tensor = tensor.view(-1)
#         nvarr = nvcomp.as_array(tensor)
#         com_arr = self.codec.encode(nvarr)

#         # Convert nvcomp Array → DLPack → Torch tensor
#         torch_tensor = torch.utils.dlpack.from_dlpack(com_arr.to_dlpack())

#         # Move compressed data to CPU for storage
#         torch_tensor = torch_tensor.contiguous().to("cpu", non_blocking=False)

#         print(torch_tensor.shape)
#         print(torch_tensor[50:100])

#         return torch_tensor

#     def decode(self, bytestream, dtype, shape):
#         bytestream = bytestream.contiguous().to("cuda", non_blocking=False)

#         # Wrap as nvcomp array
#         nvarr = nvcomp.as_array(bytestream)

#         # Decode using nvcomp
#         decoded_arr = self.codec.decode(nvarr)

#         # Convert nvcomp Array → Torch tensor via DLPack
#         decoded_tensor = torch.utils.dlpack.from_dlpack(decoded_arr.to_dlpack())

#         # TODO: Figure out how to get rid of this .clone() for performance
#         decoded_tensor = decoded_tensor.view(dtype).view(shape).clone()
#         return decoded_tensor


class ANSCompressor:
    def __init__(self, algorithm="ANS", chunk_size=65536):
        self.chunk_size = chunk_size
        self.algorithm = algorithm
        self.codec = nvcomp.Codec(
            algorithm=self.algorithm, uncomp_chunk_size=self.chunk_size
        )

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        print(f"Encoding tensor with shape: {tensor.shape}, dtype: {tensor.dtype}")

        # Move to CPU and make contiguous
        tensor_cpu = tensor.contiguous().cpu()

        # Convert to bytes
        # raw_bytes = tensor_cpu.numpy().tobytes()

        raw_bytes_torch = tensor_cpu.view(torch.uint8).cpu().numpy().tobytes()

        # Wrap as nvcomp array (host)
        nvarr_h = nvcomp.as_array(raw_bytes_torch)

        # Move to device for encoding
        nvarr_d = nvarr_h.cuda()

        # Encode
        compressed = self.codec.encode(nvarr_d)

        # Convert nvcomp array to pytorch tensor/vector
        compressed_tensor = torch.frombuffer(
            compressed.cpu(), dtype=torch.uint8
        ).contiguous()

        return compressed_tensor

    def decode(self, compressed_tensor: torch.Tensor, dtype, shape) -> torch.Tensor:
        # Convert PyTorch tensor to nvcomp device array
        nvarr_d = nvcomp.as_array(compressed_tensor.cpu().numpy()).cuda()

        # Decode
        decoded_d = self.codec.decode(nvarr_d)

        # Reconstruct tensor
        decoded_tensor = torch.utils.dlpack.from_dlpack(decoded_d.to_dlpack())

        # Reshape and cast to original dtype
        decoded_tensor = decoded_tensor.view(dtype).view(shape).clone()

        return decoded_tensor
