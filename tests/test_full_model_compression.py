#!/usr/bin/env python3

import unittest
import torch
from comp_inference import CompressedModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use cuda device 1
torch.cuda.set_device(0)


class TestFullModelCompression(unittest.TestCase):
    def test_qwen3(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from comp_inference import CompressedModel
        import torch

        data_type = torch.bfloat16
        input_text = "Once upon a time"

        gt_output = None
        comp_output = None

        device_map = {"": "cuda:0"}

        # Load model
        model_name = "Qwen/Qwen3-0.6B"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=None,
            torch_dtype=data_type,
            device_map=device_map,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        # Get ground truth output
        inputs = tokenizer(input_text, return_tensors="pt").to(
            next(model.parameters()).device
        )

        with torch.no_grad():
            gt_output = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
                top_p=0.95,
            )

        compressed_model = CompressedModel(
            model,
            compress_layer_norm=False,
            compress_embedding=True,
            compress_linear=True,
        )
        size_uncompressed = compressed_model.size_in_bytes()
        compressed_model = compressed_model.compress()
        size_compressed = compressed_model.size_in_bytes()

        # Get compressed model output
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            comp_output = compressed_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
                top_p=0.95,
            )

        print(
            f"Model size: uncompressed = {size_uncompressed/1e6:.2f} MB, compressed = {size_compressed/1e6:.2f} MB, compression ratio = {size_compressed/size_uncompressed:.2f}x"
        )

        # Decode outputs and validate equality
        gt_text = tokenizer.decode(gt_output[0], skip_special_tokens=True)
        print("Generated Text (ground truth):", gt_text)

        comp_text = tokenizer.decode(comp_output[0], skip_special_tokens=True)
        print("Generated Text (compressed):", comp_text)

        self.assertEqual(gt_text, comp_text)
        self.assertEqual(gt_output.tolist(), comp_output.tolist())

        # Loop over parameters and check dtypes
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), compressed_model.model.named_parameters()
        ):
            self.assertEqual(name1, name2)
            self.assertEqual(param1.dtype, param2.dtype)
            self.assertTrue(torch.allclose(param1, param2, atol=1e-1))


if __name__ == "__main__":
    unittest.main()
