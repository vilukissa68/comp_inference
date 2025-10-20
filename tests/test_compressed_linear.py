import torch
import unittest
from comp_inference import CompressedLinear


class TestCompressedLinear(unittest.TestCase):
    def setUp(self):
        # Set up any common test data or configurations
        self.input_dim = 10
        self.output_dim = 5
        self.batch_size = 4
        self.test_input = torch.randn(self.batch_size, self.input_dim)

    def test_initialization(self):
        # Test if the layer initializes correctly
        layer = CompressedLinear(self.input_dim, self.output_dim)
        self.assertEqual(layer.in_features, self.input_dim)
        self.assertEqual(layer.out_features, self.output_dim)

    def test_forward_pass(self):
        # Test the forward pass of the layer
        layer = CompressedLinear(self.input_dim, self.output_dim)
        output = layer(self.test_input)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_compression_decompression(self):
        # Test compression and decompression
        layer = CompressedLinear(self.input_dim, self.output_dim, bias=True)
        original_weight = layer.weight.data.clone()
        original_bias = layer.bias.data.clone() if layer.bias is not None else None

        layer.compress()
        self.assertFalse(hasattr(layer, "weight"))
        self.assertFalse(hasattr(layer, "bias"))

        layer.decompress()
        print(layer.weight.data)
        print(original_weight)
        self.assertTrue(torch.allclose(layer.weight.data, original_weight, atol=1e-01))
        if hasattr(layer, "bias") and layer.bias is not None:
            print(layer.bias.data)
            print(original_bias)
            self.assertTrue(torch.allclose(layer.bias.data, original_bias, atol=1e-01))


if __name__ == "__main__":
    unittest.main()
