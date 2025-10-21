#!/usr/bin/env python3

import torch
import unittest
from comp_inference import CompressedEmbedding


class TestCompressedEmbedding(unittest.TestCase):
    def setUp(self):
        # Typical embedding parameters
        self.num_embeddings = 50
        self.embedding_dim = 8
        self.batch_size = 4
        self.seq_len = 6
        self.test_input = torch.randint(
            0, self.num_embeddings, (self.batch_size, self.seq_len)
        )

    def test_initialization(self):
        # Test if the embedding layer initializes correctly
        layer = CompressedEmbedding(self.num_embeddings, self.embedding_dim)
        self.assertEqual(layer.num_embeddings, self.num_embeddings)
        self.assertEqual(layer.embedding_dim, self.embedding_dim)
        self.assertTrue(hasattr(layer, "weight"))
        self.assertIsInstance(layer.weight, torch.nn.Parameter)

    def test_forward_pass(self):
        # Test the forward pass before compression
        layer = CompressedEmbedding(self.num_embeddings, self.embedding_dim)
        output = layer(self.test_input)
        self.assertEqual(
            output.shape, (self.batch_size, self.seq_len, self.embedding_dim)
        )

    def test_compression_decompression(self):
        # Test compression and decompression symmetry
        layer = CompressedEmbedding(self.num_embeddings, self.embedding_dim)
        original_weight = layer.weight.data.clone()

        layer.compress()
        # Confirm compression actually removed full-precision weights
        self.assertFalse(hasattr(layer, "weight"))
        self.assertIsNotNone(layer.compressed_weight)

        layer.decompress()
        # Confirm decompression restores weight and clears compressed version
        self.assertTrue(hasattr(layer, "weight"))
        self.assertIsNone(layer.compressed_weight)

        # Allow small error margin due to quantization rounding
        self.assertTrue(torch.allclose(layer.weight.data, original_weight, atol=1e-01))

    def test_forward_after_compression(self):
        # Ensure the layer can still produce valid output after compression
        layer = CompressedEmbedding(self.num_embeddings, self.embedding_dim)
        layer.compress()
        output = layer(self.test_input)  # should trigger decompression internally
        self.assertEqual(
            output.shape, (self.batch_size, self.seq_len, self.embedding_dim)
        )
        self.assertTrue(hasattr(layer, "weight"))  # should have been decompressed


if __name__ == "__main__":
    unittest.main()
