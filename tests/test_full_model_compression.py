#!/usr/bin/env python3

import unittest
from comp_inference import CompressedModel


class TestFullModelCompression(unittest.TestCase):
    def test_qwen3(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from comp_inference import CompressedModel
        import torch

        data_type = torch.bfloat16
        input_text = "Once upon a time"

        gt_output = None
        comp_output = None

        # Load model
        model_name = "Qwen/Qwen3-0.6B"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=data_type,
            device_map="auto",
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
            compress_embedding=False,
            compress_linear=True,
        )

        # Get compressed model output
        inputs = tokenizer(input_text, return_tensors="pt").to(
            next(compressed_model.parameters()).device
        )

        with torch.no_grad():
            comp_output = compressed_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                temperature=0.0,
                top_p=0.95,
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
