#!/usr/bin/env python3


import unittest
import torch
import torch.nn as nn
from comp_inference import CompressedModel
from comp_inference.layers.compressed_linear import CompressedLinear
from comp_inference.layers.compressed_embedding import CompressedEmbedding
from comp_inference.layers.compressed_layer_norm import CompressedLayerNorm


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50, 16)
        self.fc1 = nn.Linear(16, 32)
        self.norm = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.embed(x)
        x = x.mean(dim=1)  # simulate pooling
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class TestCompressedModel(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel().to("cuda")
        self.input_data = torch.randint(0, 50, (4, 8)).to("cuda")
        self.compressed_model = CompressedModel(
            self.model,
            compress_layer_norm=False,
            compress_embedding=False,
            compress_linear=True,
        )  # separate instance

    def test_compress_decompress_lossless(self):
        self.compressed_model.compress()
        self.compressed_model.decompress()
        for orig_param, comp_param in zip(
            self.model.parameters(), self.compressed_model.model.parameters()
        ):
            self.assertTrue(torch.allclose(orig_param, comp_param, atol=1e-6))

    def test_compress_manual_decompress(self):
        # Test compression and decompression
        self.compressed_model.compress()
        # Check compressed weights exist and original params removed
        for layer in self.compressed_model.model.modules():
            if isinstance(
                layer, (CompressedLinear, CompressedEmbedding, CompressedLayerNorm)
            ):
                self.assertIsNotNone(layer.compressed_weight)
                if hasattr(layer, "bias") and layer.bias is not None:
                    self.assertIsNotNone(layer.compressed_bias)

        # Decompress and check weights restored
        self.compressed_model.decompress()
        for layer in self.compressed_model.model.modules():
            if isinstance(
                layer, (CompressedLinear, CompressedEmbedding, CompressedLayerNorm)
            ):
                self.assertIsNone(layer.compressed_weight)
                if hasattr(layer, "bias") and layer.bias is not None:
                    self.assertIsNone(layer.compressed_bias)

        # Forward pass after compression should match original closely
        orig_out = self.model(self.input_data)
        comp_out = self.compressed_model(self.input_data)
        self.assertTrue(orig_out.shape == comp_out.shape)
        self.assertTrue(torch.allclose(orig_out, comp_out, atol=1e-1))

    def test_forward_with_on_the_fly_decompression(self):
        # Compress the model
        self.compressed_model.compress()

        # Forward pass should decompress layers on the fly
        out = self.compressed_model(self.input_data)

        # Check output shape
        self.assertEqual(out.shape, (4, 10))

    def test_save_load_roundtrip(self):
        # Compress and save state
        self.compressed_model.compress()
        state = self.compressed_model.state_dict()

        # Load into a fresh compressed model
        new_model = CompressedModel(
            DummyModel(),
            compress_layer_norm=False,
            compress_embedding=False,
            compress_linear=False,
        )
        new_model.load_state_dict(state)

        # Check outputs are still close
        out1 = self.compressed_model(self.input_data)
        out2 = new_model(self.input_data)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-1))


if __name__ == "__main__":
    unittest.main()
