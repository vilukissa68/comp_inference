#!/usr/bin/env python3

from nvidia import nvcomp
import torch


class ANSCompressor:
    def __init__(self, algorithm="ANS", chunk_size=64 * 1024):
        self.chunk_size = chunk_size
        self.algorithm = algorithm
        self.codec = nvcomp.Codec(
            algorithm=self.algorithm, uncomp_chunk_size=self.chunk_size
        )

    def encode(self, tensor: torch.Tensor) -> torch.Tensor:
        print(f"Encoding tensor with shape: {tensor.shape}, dtype: {tensor.dtype}")

        # Move to CPU and make contiguous
        tensor_cpu = tensor.contiguous().cpu()

        raw_bytes = tensor_cpu.view(torch.uint8).cpu().numpy().tobytes()

        # Wrap as nvcomp array
        nvarr_h = nvcomp.as_array(raw_bytes)

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
