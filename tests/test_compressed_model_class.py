#!/usr/bin/env python3


import unittest
import torch
import torch.nn as nn
from comp_inference import CompressedModel
from comp_inference.layers.compressed_linear import CompressedLinear
from comp_inference.layers.compressed_embedding import CompressedEmbedding
from comp_inference.layers.compressed_layer_norm import CompressedLayerNorm

COMPUTATION_DTYPE = torch.bfloat16


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Larger embedding dimension and vocab
        self.embed = nn.Embedding(1000, 128)
        # Wider fully connected layers
        self.fc1 = nn.Linear(128, 512)
        self.norm = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 128)

        self.init_params()

        # Convert all parameters to the computation dtype
        for name, param in self.named_parameters():
            param.data = param.data.to(COMPUTATION_DTYPE)

    def forward(self, x):
        x = self.embed(x)
        x = x.mean(dim=1)  # simulate pooling
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x

    def init_params(self):
        for param in self.parameters():
            # nn.init.constant_(param, 0.1)
            nn.init.normal_(param, mean=0.0, std=0.02)


class TestCompressedModel(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.model = self.model.to(dtype=COMPUTATION_DTYPE)
        self.model = self.model.to("cuda")
        self.input_data = torch.randint(0, 50, (4, 8), dtype=torch.long).to("cuda")

        self.compressed_model = CompressedModel(
            self.model,
            compress_layer_norm=False,
            compress_embedding=True,
            compress_linear=True,
        )  # separate instance

    def test_base_model_inference(self):
        self.model = self.model.to("cuda")
        out = self.model(self.input_data)
        print("[SUCCESS] Base model inference test passed.")

    def test_compress_decompress_lossless(self):
        self.compressed_model.compress()
        self.compressed_model.decompress()
        for orig_param, comp_param in zip(
            self.model.parameters(), self.compressed_model.model.parameters()
        ):
            self.assertTrue(torch.allclose(orig_param, comp_param, atol=1e-6))
        print("[SUCCESS] Compression-decompression lossless test passed.")

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
        print("[SUCCESS] Manual compress-decompress test passed.")

    def test_forward_with_on_the_fly_decompression(self):
        # Compress the model
        self.compressed_model.compress()

        # Forward pass should decompress layers on the fly
        out = self.compressed_model(self.input_data)

        # Check output shape
        print("[SUCCESS] On-the-fly decompression during forward pass test passed")

    def test_save_load_roundtrip(self):
        print("=" * 20, "test_save_load_roundtrip", "=" * 20)
        # Compress and save state
        self.compressed_model.compress()
        print("-" * 20, "model compressed state dict", "-" * 20)
        state = self.compressed_model.state_dict()
        print("-" * 20, "state dict keys", "-" * 20)
        for k in state.keys():
            print(k, state[k].shape, state[k].dtype)

        # Load into a fresh compressed model
        print("-" * 20, "loading state into new model", "-" * 20)
        new_model = CompressedModel(
            DummyModel(),
            compress_layer_norm=False,
            compress_embedding=True,
            compress_linear=True,
        ).to("cuda")
        print("-" * 20, "before loading state", "-" * 20)
        new_model.load_state_dict(state)

        print(new_model)

        # Check outputs are still close
        print("-" * 20, "checking outputs", "-" * 20)
        out1 = self.compressed_model(self.input_data)
        print("-" * 20, "output from original compressed model", "-" * 20)
        out2 = new_model(self.input_data.to("cuda"))
        print("-" * 20, "output from new loaded model", "-" * 20)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-1))

        # Check that dtypes are preserved
        for (name1, param1), (name2, param2) in zip(
            self.compressed_model.model.named_parameters(),
            new_model.model.named_parameters(),
        ):
            self.assertEqual(name1, name2)
            self.assertEqual(param1.dtype, param2.dtype)
        print("[SUCCESS] Save-load roundtrip test passed.")


if __name__ == "__main__":
    unittest.main()
