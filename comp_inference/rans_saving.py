#!/usr/bin/env python3

import torch
import torch.nn as nn
import os
import re

from safetensors.torch import save_file
from transformers import PreTrainedModel, PreTrainedTokenizer
from safetensors.torch import safe_open


def save_rans_model_package(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_dir: str
):
    """
    Saves the full model package: config.json, tokenizer files, and the compressed safetensors.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Setup rANS quantization config in model config
    if not hasattr(model.config, "quantization_config"):
        model.config.quantization_config = {}

    model.config.quantization_config["quant_method"] = "rans"
    model.config.quantization_config["compression"] = "rans_bfloat16"

    # 2. Save Config & Tokenizer
    print(f"Saving config and tokenizer to {output_dir}...")
    model.config.save_pretrained(output_dir)

    if tokenizer:
        tokenizer.save_pretrained(output_dir)

    # 3. Save Compressed Weights
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    pack_and_save_tensors(model, safetensors_path)


def pack_and_save_tensors(model, output_path: str):
    tensors = {}
    print(f"Packing model to {output_path}...")

    # Keep track of which parameters we have successfully compressed/saved
    handled_params = set()

    # Regex to find layer index (e.g. "layers.0.")
    layer_pattern = re.compile(r"layers\.(\d+)\.")

    for name, module in model.named_modules():
        if hasattr(module, "compressed") and module.compressed == "rans_bfloat16":
            # 1. Identify Layer Index
            match = layer_pattern.search(name)
            if not match:
                # This handles edge cases if you compressed lm_head or something outside layers
                print(
                    f"Note: Compressed module {name} outside standard layer structure."
                )
                vllm_module_name = name + "."
                prefix = ""
            else:
                layer_idx = match.group(1)

                vllm_module_name = None
                prefix = ""

                vllm_module_name = name + "."

            # 3. Save Compressed Tensors
            _save_rans_attributes(tensors, vllm_module_name, prefix, module)

            # 4. Mark original parameters as Handled
            # The standard parameter name is usually "{module_name}.weight"
            handled_params.add(f"{name}.weight")

            if hasattr(module, "bias") and module.bias is not None:
                handled_params.add(f"{name}.bias")

    # Iterate over ALL parameters in the model.
    # If we didn't compress it in Pass 1, save it raw here.
    for param_name, param in model.named_parameters():
        if param_name in handled_params:
            continue

        tensors[param_name] = param.data.cpu()

    # Metadata
    metadata = {"format": "pt", "compression_method": "rans_bfloat16"}

    save_file(tensors, output_path, metadata=metadata)
    print(f"Saved {len(tensors)} tensors.")


# (Ensure _save_rans_attributes is defined as in the previous answer)
def _save_rans_attributes(tensors, base_name, prefix, module):
    p = prefix

    # Exponent
    if hasattr(module, "exponent_compressed_weight"):
        tensors[
            f"{base_name}{p}rans_exp_stream"
        ] = module.exponent_compressed_weight.cpu()
        tensors[f"{base_name}{p}rans_exp_states"] = module.exponent_states.cpu().to(
            torch.int32
        )
        tensors[
            f"{base_name}{p}rans_exp_sizes"
        ] = module.exponent_output_sizes.cpu().to(torch.int32)
        tensors[f"{base_name}{p}rans_exp_freqs"] = module.exponent_freqs.cpu().to(
            torch.uint16
        )
        tensors[f"{base_name}{p}rans_exp_cdf"] = module.exponent_cdf.cpu().to(
            torch.uint16
        )
        exp_streams = module.exponent_num_streams
        is_exp_compressed = 1
    else:
        tensors[f"{base_name}{p}rans_exp_raw"] = module.exponent_raw.cpu()
        exp_streams = 0
        is_exp_compressed = 0

    # Mantissa
    if hasattr(module, "mantissa_compressed_weight"):
        tensors[
            f"{base_name}{p}rans_man_stream"
        ] = module.mantissa_compressed_weight.cpu()
        tensors[f"{base_name}{p}rans_man_states"] = module.mantissa_states.cpu().to(
            torch.int32
        )
        tensors[
            f"{base_name}{p}rans_man_sizes"
        ] = module.mantissa_output_sizes.cpu().to(torch.int32)
        tensors[f"{base_name}{p}rans_man_freqs"] = module.mantissa_freqs.cpu().to(
            torch.uint16
        )
        tensors[f"{base_name}{p}rans_man_cdf"] = module.mantissa_cdf.cpu().to(
            torch.uint16
        )
        man_streams = module.mantissa_num_streams
        is_man_compressed = 1
    else:
        tensors[f"{base_name}{p}rans_man_raw"] = module.mantissa_raw.cpu().flatten()
        man_streams = 0
        is_man_compressed = 0

    # Info
    shape = getattr(module, "input_shape", [])
    if not shape and hasattr(module, "expanded_size"):
        shape = [module.expanded_size]

    info_data = [
        1,
        module.expanded_size,
        is_exp_compressed,
        exp_streams,
        is_man_compressed,
        man_streams,
        len(shape),
    ] + list(shape)

    tensors[f"{base_name}{p}rans_info"] = torch.tensor(info_data, dtype=torch.int32)

    # Bias
    if hasattr(module, "bias") and module.bias is not None:
        # Bias names in vLLM split layers often depend on the specific kernel logic,
        # but for safety we save it attached to the mapped name.
        tensors[f"{base_name}{p}bias"] = module.bias.cpu()


def load_compressed_model_with_auto_model(
    model_name: str, safetensors_path: str, device="cpu"
):
    from transformers import AutoModelForCausalLM, AutoConfig

    config = AutoConfig.from_pretrained(model_name)
    with torch.device(device):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    load_compressed_model(model, safetensors_path, device=device)
    return model


def load_compressed_model(model: nn.Module, safetensors_path: str, device="cpu"):
    """
    Loads a rANS compressed model from a safetensors file into an initialized
    model skeleton.
    """
    print(f"Loading compressed model from {safetensors_path}...")

    with safe_open(safetensors_path, framework="pt", device=device) as f:
        # Get all keys to allow fast lookups
        file_keys = set(f.keys())

        loaded_count = 0

        for name, module in model.named_modules():
            # 1. Check if this module has compressed metadata in the file
            info_key = f"{name}.rans_info"

            if info_key in file_keys:
                # --- LOAD COMPRESSED LAYER ---
                _load_rans_layer(module, name, f)
                loaded_count += 1
            else:
                # --- LOAD STANDARD PARAMETERS ---
                # (Embeddings, LayerNorms, or uncompressed Linears)
                for param_name, param in module.named_parameters(recurse=False):
                    full_key = f"{name}.{param_name}" if name else param_name

                    if full_key in file_keys:
                        # Load data into existing parameter
                        tensor = f.get_tensor(full_key)

                        # Safety check for shape mismatch
                        if param.shape != tensor.shape:
                            print(
                                f"Warning: Shape mismatch for {full_key}. Expected {param.shape}, got {tensor.shape}"
                            )

                        param.data = tensor

                    # Note: We ignore missing keys here because they might belong
                    # to a parent/child module handled in a different iteration.

    print(f"Model loaded. {loaded_count} layers are compressed.")


def _load_rans_layer(module: nn.Module, prefix: str, f):
    """
    Helper to populate specific attributes of a compressed module.
    """
    # 1. Parse Info Tensor
    # Layout: [Version, ExpandedSize, ExpStreams, ManStreams, ShapeRank, Dim0, Dim1...]
    info = f.get_tensor(f"{prefix}.rans_info")

    # Extract scalars (using .item() to get Python ints)
    # version = info[0].item() # Unused for now
    expanded_size = info[1].item()
    is_exp_compressed = info[2].item()
    exp_num_streams = info[3].item()
    is_man_compressed = info[4].item()
    man_num_streams = info[5].item()
    shape_rank = info[6].item()

    # Reconstruct shape tuple
    input_shape = torch.Size(info[7 : 7 + shape_rank].tolist())

    # Set Module Attributes
    module.compressed = "rans_bfloat16"
    module.expanded_size = expanded_size
    module.input_shape = input_shape

    # 2. Load Exponent Data
    if is_exp_compressed > 0:
        module.exponent_compressed_weight = f.get_tensor(f"{prefix}.rans_exp_stream")
        module.exponent_states = f.get_tensor(f"{prefix}.rans_exp_states")
        module.exponent_output_sizes = f.get_tensor(f"{prefix}.rans_exp_sizes")
        module.exponent_freqs = f.get_tensor(
            f"{prefix}.rans_exp_freqs"
        )  # .to(torch.uint16)
        module.exponent_cdf = f.get_tensor(
            f"{prefix}.rans_exp_cdf"
        )  # .to(torch.uint16)

        module.exponent_num_streams = exp_num_streams
        module.exponent_stream_size = module.exponent_compressed_weight.numel()

        # Calculate total stream size for the buffer view logic
        # (This is usually just numel, but explicit check helps)
        module.exponent_total_stream_size = module.exponent_stream_size
    else:
        # Fallback raw
        if f"{prefix}.rans_exp_raw" in f.keys():
            module.exponent_raw = f.get_tensor(f"{prefix}.rans_exp_raw")
        else:
            raise ValueError(
                f"Corrupt file: {prefix} marked uncompressed exponent but raw data missing."
            )

    # 3. Load Mantissa Data
    if is_man_compressed > 0:
        module.mantissa_compressed_weight = f.get_tensor(f"{prefix}.rans_man_stream")
        module.mantissa_states = f.get_tensor(f"{prefix}.rans_man_states")
        module.mantissa_output_sizes = f.get_tensor(f"{prefix}.rans_man_sizes")
        module.mantissa_freqs = f.get_tensor(f"{prefix}.rans_man_freqs").to(
            torch.uint16
        )
        module.mantissa_cdf = f.get_tensor(f"{prefix}.rans_man_cdf").to(torch.uint16)

        module.mantissa_num_streams = man_num_streams
        module.mantissa_stream_size = module.mantissa_compressed_weight.numel()
        module.mantissa_total_stream_size = module.mantissa_stream_size
    else:
        # Fallback raw
        if f"{prefix}.rans_man_raw" in f.keys():
            module.mantissa_raw = f.get_tensor(f"{prefix}.rans_man_raw")
        else:
            raise ValueError(
                f"Corrupt file: {prefix} marked uncompressed mantissa but raw data missing."
            )

    # 4. Load Bias (if present)
    if f"{prefix}.bias" in f.keys():
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data = f.get_tensor(f"{prefix}.bias")

    # 5. CLEANUP: Delete the original weight
    # The model skeleton initialized a random weight. We must delete it
    # to save memory and ensure the forward pass crashes if we forget to decompress.
    if hasattr(module, "weight"):
        del module.weight
