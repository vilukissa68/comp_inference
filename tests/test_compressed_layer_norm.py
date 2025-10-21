#!/usr/bin/env python3

import torch
import unittest
from comp_inference import CompressedLayerNorm


class TestCompressedLayerNorm(unittest.TestCase):
    def setUp(self):
        self.normalized_shape = 8
        self.batch_size = 4
        self.seq_len = 10
        self.test_input = torch.randn(
            self.batch_size, self.seq_len, self.normalized_shape
        )

    def test_initialization(self):
        layer = CompressedLayerNorm(self.normalized_shape)
        self.assertEqual(layer.normalized_shape, (self.normalized_shape,))
        self.assertTrue(layer.elementwise_affine)
        self.assertIsNotNone(layer.weight)
        self.assertIsNotNone(layer.bias)

    def test_forward_pass(self):
        layer = CompressedLayerNorm(self.normalized_shape)
        output = layer(self.test_input)
        self.assertEqual(output.shape, self.test_input.shape)

    def test_compression_decompression(self):
        layer = CompressedLayerNorm(self.normalized_shape)
        original_weight = layer.weight.data.clone()
        original_bias = layer.bias.data.clone()

        layer.compress()
        self.assertFalse(hasattr(layer, "weight"))
        self.assertFalse(hasattr(layer, "bias"))
        self.assertIsNotNone(layer.compressed_weight)
        self.assertIsNotNone(layer.compressed_bias)

        layer.decompress()
        self.assertTrue(torch.allclose(layer.weight.data, original_weight, atol=1e-1))
        self.assertTrue(torch.allclose(layer.bias.data, original_bias, atol=1e-1))


if __name__ == "__main__":
    unittest.main()
