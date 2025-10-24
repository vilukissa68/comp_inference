#!/usr/bin/env python3

from comp_inference import ANSCompressor

import unittest
import torch


class ANSCompressorTestCase(unittest.TestCase):
    def test_ans_compressor(self):
        DTYPE = torch.float16
        tensor_1 = torch.rand(1024, 1024, dtype=DTYPE).to("cuda")
        tensor_2 = torch.rand(16384, 16384, dtype=DTYPE).to("cuda")
        tensor_3 = torch.randint(0, 4, (1024, 1024), dtype=torch.uint8).cuda()

        # Add sparsity to tensor_2
        mask = torch.rand(1024, 1024, device="cuda") < 1.0  # 90% sparsity
        tensor_2 = tensor_2 * torch.zeros_like(tensor_2)

        print(tensor_2)

        compressor = ANSCompressor()
        compressed_1 = compressor.encode(tensor_1)
        compressed_2 = compressor.encode(tensor_2)
        compressed_3 = compressor.encode(tensor_3)

        size_1 = compressed_1.numel() * compressed_1.element_size()
        size_2 = compressed_2.numel() * compressed_2.element_size()
        size_3 = compressed_3.numel() * compressed_3.element_size()

        # assert size_2 < size_1

        print(f"Original size 1: {tensor_1.numel() * tensor_1.element_size()} bytes")
        print(
            f"Compressed size 1: {compressed_1.numel() * compressed_1.element_size()} bytes, compression ratio: {size_1 / (tensor_1.numel() * tensor_1.element_size()):.4f}"
        )
        print(f"Original size 2: {tensor_2.numel() * tensor_2.element_size()} bytes")
        print(
            f"Compressed size 2: {compressed_2.numel() * compressed_2.element_size()} bytes, compression ratio: {size_2 / (tensor_2.numel() * tensor_2.element_size()):.4f}"
        )
        print(f"Original size 3: {tensor_3.numel() * tensor_3.element_size()} bytes")
        print(
            f"Compressed size 3: {compressed_3.numel() * compressed_3.element_size()} bytes, compression ratio: {size_3 / (tensor_3.numel() * tensor_3.element_size()):.4f}"
        )

        decompressed_1 = compressor.decode(
            compressed_1, dtype=DTYPE, shape=tensor_1.shape
        )
        decompressed_2 = compressor.decode(
            compressed_2, dtype=DTYPE, shape=tensor_2.shape
        )
        decompressed_3 = compressor.decode(
            compressed_3, dtype=torch.uint8, shape=tensor_3.shape
        )

        print(tensor_1)
        print(decompressed_1)

        print(tensor_2)
        print(decompressed_2)

        print(tensor_3)
        print(decompressed_3)

        assert decompressed_1.dtype == tensor_1.dtype
        assert decompressed_2.dtype == tensor_2.dtype

        assert decompressed_1.shape == tensor_1.shape
        assert decompressed_2.shape == tensor_2.shape

        assert decompressed_3.dtype == tensor_3.dtype
        assert decompressed_3.shape == tensor_3.shape

        assert torch.allclose(tensor_1, decompressed_1, atol=1e-6)
        assert torch.allclose(tensor_2, decompressed_2, atol=1e-6)
        assert torch.equal(tensor_3, decompressed_3)


if __name__ == "__main__":
    unittest.main()
