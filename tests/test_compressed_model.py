#!/usr/bin/env python3
import torch
import torch.nn as nn
import unittest
from comp_inference import replace_all_with_compressed
from comp_inference.layers.compressed_linear import CompressedLinear
from comp_inference.layers.compressed_embedding import CompressedEmbedding
from comp_inference.layers.compressed_layer_norm import CompressedLayerNorm


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 8)
        self.lin = nn.Linear(8, 4)
        self.norm = nn.LayerNorm(4)

    def forward(self, x):
        x = self.embed(x)
        x = x.mean(dim=1)
        x = self.lin(x)
        x = self.norm(x)
        return x


class TestCompressionPipeline(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.input_ids = torch.randint(0, 10, (2, 5))

    def test_replace_all_layers(self):
        compressed_model = replace_all_with_compressed(self.model)

        # Check layer types
        self.assertIsInstance(compressed_model.embed, CompressedEmbedding)
        self.assertIsInstance(compressed_model.lin, CompressedLinear)
        self.assertIsInstance(compressed_model.norm, CompressedLayerNorm)

        # Forward pass still works
        output = compressed_model(self.input_ids)
        self.assertEqual(output.shape, (2, 4))

    def test_compression_decompression_pipeline(self):
        # Original output
        original_output = self.model(self.input_ids)

        # Replace layers with compressed versions
        compressed_model = replace_all_with_compressed(self.model)

        # Compress all layers
        for module in compressed_model.modules():
            if hasattr(module, "compress"):
                module.compress()

        # Check that original parameters were removed
        for module in compressed_model.modules():
            if isinstance(
                module, (CompressedLinear, CompressedEmbedding, CompressedLayerNorm)
            ):
                self.assertFalse(hasattr(module, "weight"))

        # Decompress all layers and verify restoration
        for module in compressed_model.modules():
            if hasattr(module, "decompress"):
                module.decompress()
                self.assertTrue(hasattr(module, "weight"))

        # Forward pass after compression/decompression
        output = compressed_model(self.input_ids)

        # Check outputs are numerically close
        self.assertTrue(
            torch.allclose(output, original_output, atol=1e-2),
            "Output after compression/decompression differs from original",
        )

        # Also check shape
        self.assertEqual(output.shape, original_output.shape)


if __name__ == "__main__":
    unittest.main()
