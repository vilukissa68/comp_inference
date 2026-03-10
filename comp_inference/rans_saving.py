#!/usr/bin/env python3

import torch
import torch.nn as nn
import os
import re

from safetensors.torch import save_file
from transformers import PreTrainedModel, PreTrainedTokenizer
from safetensors.torch import safe_open

import os
import re
import torch
from safetensors.torch import save_file
from transformers import PreTrainedModel, PreTrainedTokenizer


def save_rans_model_package(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    fuse: bool = False,
    tied_lm_head: bool = False,
):
    """
    Saves the full model package: config.json, tokenizer files, and the compressed safetensors.
    :param fuse: If True, maps Q/K/V and Gate/Up projections into fused keys for vLLM.
                 If False, keeps them separate for standard PyTorch/HuggingFace loading.
    """
    if not fuse:
        output_dir = output_dir + "_unfused"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Setup rANS quantization config in model config
    if not hasattr(model.config, "quantization_config"):
        model.config.quantization_config = {}

    layer_configs = {}
    for name, module in model.named_modules():
        # Ensure that lm_head is not saved
        if tied_lm_head and "lm_head" in name:
            print(f"Skipping tied lm_head layer: {name}")
            continue

        if hasattr(module, "compressed") and module.compressed == "rans_bfloat16":
            th = getattr(
                module, "exponent_tile_height", getattr(module, "tile_height", 1024)
            )
            tw = getattr(
                module, "exponent_tile_width", getattr(module, "tile_width", 32)
            )

            # Conditionally map names for vLLM vs PyTorch
            if fuse:
                if "gate_proj" in name:
                    name = name.replace("gate_proj", "gate_up_proj")
                elif "q_proj" in name:
                    name = name.replace("q_proj", "qkv_proj")

                # If fuse is True, the compression script likely deleted k/v/up proj.
                # If they still exist in the module dict as ghosts, skip adding duplicate configs.
                if "k_proj" in name or "v_proj" in name or "up_proj" in name:
                    continue

            layer_configs[name] = {"tile_height": th, "tile_width": tw}

    # Inject the dictionary into the quantization config
    model.config.quantization_config["quant_method"] = "rans"
    model.config.quantization_config["compression"] = "rans_bfloat16"
    model.config.quantization_config["layer_configs"] = layer_configs
    model.config.quantization_config["default_tile_height"] = 1024
    model.config.quantization_config["default_tile_width"] = 32

    # Save the fuse status so the runtime loader knows what to expect!
    model.config.quantization_config["fused_projections"] = fuse

    # 2. Save Config & Tokenizer
    print(f"Saving config and tokenizer to {output_dir}...")
    model.config.save_pretrained(output_dir)

    if tokenizer:
        tokenizer.save_pretrained(output_dir)

    # 3. Save Compressed Weights
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    pack_and_save_tensors(model, safetensors_path, fuse=fuse, tied_lm_head=tied_lm_head)


# def pack_and_save_tensors(
#     model, output_path: str, fuse: bool = False, tied_lm_head: bool = False
# ):
#     tensors = {}
#     print(f"Packing model to {output_path} (Fused Mode: {fuse})...")

#     handled_params = set()
#     layer_pattern = re.compile(r"layers\.(\d+)\.")

#     for name, module in model.named_modules():
#         if hasattr(module, "compressed") and module.compressed == "rans_bfloat16":
#             # Conditionally map the safetensor key prefixes
#             mapped_name = name
#             if fuse:
#                 if "gate_proj" in mapped_name:
#                     mapped_name = mapped_name.replace("gate_proj", "gate_up_proj")
#                 elif "q_proj" in mapped_name:
#                     mapped_name = mapped_name.replace("q_proj", "qkv_proj")
#                 elif (
#                     "k_proj" in mapped_name
#                     or "v_proj" in mapped_name
#                     or "up_proj" in mapped_name
#                 ):
#                     # Skip redundant saves if the compression script already fused the tensor into Q/Gate
#                     handled_params.add(f"{name}.weight")
#                     if hasattr(module, "bias") and module.bias is not None:
#                         handled_params.add(f"{name}.bias")
#                     continue

#             match = layer_pattern.search(mapped_name)
#             if not match:
#                 print(
#                     f"Note: Compressed module {mapped_name} outside standard layer structure."
#                 )
#                 vllm_module_name = mapped_name + "."
#                 prefix = ""
#             else:
#                 vllm_module_name = mapped_name + "."
#                 prefix = ""

#             # 3. Save Compressed Tensors
#             _save_rans_attributes(tensors, vllm_module_name, prefix, module)

#             # 4. Mark original parameters as Handled
#             handled_params.add(f"{name}.weight")
#             if hasattr(module, "bias") and module.bias is not None:
#                 handled_params.add(f"{name}.bias")

#     # Iterate over ALL parameters in the model.
#     for param_name, param in model.named_parameters():
#         if param_name in handled_params:
#             continue
#         tensors[param_name] = param.data.cpu()

#     # Metadata
#     metadata = {
#         "format": "pt",
#         "compression_method": "rans_bfloat16",
#         "fused_projections": str(fuse),
#     }

#     save_file(tensors, output_path, metadata=metadata)
#     print(f"Saved {len(tensors)} tensors.")


def pack_and_save_tensors(
    model, output_path: str, fuse: bool = False, tied_lm_head: bool = False
):
    tensors = {}
    print(f"Packing model to {output_path} (Fused Mode: {fuse})...")

    handled_params = set()
    layer_pattern = re.compile(r"layers\.(\d+)\.")

    if tied_lm_head:
        print(
            "Tied LM Head detected: Explicitly dropping lm_head.weight from safetensors."
        )
        handled_params.add("lm_head.weight")
        handled_params.add("model.lm_head.weight")  # Add both common naming conventions

    for name, module in model.named_modules():
        if hasattr(module, "compressed") and module.compressed == "rans_bfloat16":
            # mapped_name = name
            # if fuse:
            #     if "gate_proj" in mapped_name:
            #         mapped_name = mapped_name.replace("gate_proj", "gate_up_proj")
            #     elif "q_proj" in mapped_name:
            #         mapped_name = mapped_name.replace("q_proj", "qkv_proj")
            #     elif (
            #         "k_proj" in mapped_name
            #         or "v_proj" in mapped_name
            #         or "up_proj" in mapped_name
            #     ):
            #         # Skip redundant saves if the compression script already fused the tensor into Q/Gate
            #         handled_params.add(f"{name}.weight")
            #         if hasattr(module, "bias") and module.bias is not None:
            #             handled_params.add(f"{name}.bias")
            #         continue

            match = layer_pattern.search(name)
            if not match:
                print(
                    f"Note: Compressed module {name} outside standard layer structure."
                )
                vllm_module_name = name + "."
                prefix = ""
            else:
                vllm_module_name = name + "."
                prefix = ""

            # 3. Save Compressed Tensors
            _save_rans_attributes(tensors, vllm_module_name, prefix, module)

            # 4. Mark original parameters as Handled
            handled_params.add(f"{name}.weight")
            if hasattr(module, "bias") and module.bias is not None:
                handled_params.add(f"{name}.bias")

    # Iterate over ALL parameters in the model.
    for param_name, param in model.named_parameters():
        if param_name in handled_params:
            continue
        tensors[param_name] = param.data.cpu()

    # Metadata
    metadata = {
        "format": "pt",
        "compression_method": "rans_bfloat16",
        "fused_projections": str(fuse),
    }

    save_file(tensors, output_path, metadata=metadata)
    print(f"Saved {len(tensors)} tensors.")


# def save_rans_model_package(
#     model: PreTrainedModel, tokenizer: PreTrainedTokenizer, output_dir: str
# ):
#     """
#     Saves the full model package: config.json, tokenizer files, and the compressed safetensors.
#     """
#     os.makedirs(output_dir, exist_ok=True)

#     # 1. Setup rANS quantization config in model config
#     if not hasattr(model.config, "quantization_config"):
#         model.config.quantization_config = {}

#     layer_configs = {}
#     for name, module in model.named_modules():
#         if hasattr(module, "compressed") and module.compressed == "rans_bfloat16":
#             # Extract the specific tile dimensions for this tensor
#             th = getattr(
#                 module, "exponent_tile_height", getattr(module, "tile_height", 1024)
#             )
#             tw = getattr(
#                 module, "exponent_tile_width", getattr(module, "tile_width", 32)
#             )

#             if "gate_proj" in name:
#                 name = name.replace("gate_proj", "gate_up_proj")
#             elif "q_proj" in name:
#                 name = name.replace("q_proj", "qkv_proj")

#             # Name will be exactly as it appears in the model structure
#             layer_configs[name] = {"tile_height": th, "tile_width": tw}

#     # Inject the dictionary into the quantization config
#     model.config.quantization_config["quant_method"] = "rans"
#     model.config.quantization_config["compression"] = "rans_bfloat16"
#     model.config.quantization_config["layer_configs"] = layer_configs
#     model.config.quantization_config["default_tile_height"] = 1024
#     model.config.quantization_config["default_tile_width"] = 32

#     # 2. Save Config & Tokenizer
#     print(f"Saving config and tokenizer to {output_dir}...")
#     model.config.save_pretrained(output_dir)

#     if tokenizer:
#         tokenizer.save_pretrained(output_dir)

#     # 3. Save Compressed Weights
#     safetensors_path = os.path.join(output_dir, "model.safetensors")
#     pack_and_save_tensors(model, safetensors_path)


# def pack_and_save_tensors(model, output_path: str):
#     tensors = {}
#     print(f"Packing model to {output_path}...")

#     # Keep track of which parameters we have successfully compressed/saved
#     handled_params = set()

#     # Regex to find layer index (e.g. "layers.0.")
#     layer_pattern = re.compile(r"layers\.(\d+)\.")

#     for name, module in model.named_modules():
#         if hasattr(module, "compressed") and module.compressed == "rans_bfloat16":
#             # 1. Identify Layer Index
#             match = layer_pattern.search(name)
#             if not match:
#                 # This handles edge cases if you compressed lm_head or something outside layers
#                 print(
#                     f"Note: Compressed module {name} outside standard layer structure."
#                 )
#                 vllm_module_name = name + "."
#                 prefix = ""
#             else:
#                 layer_idx = match.group(1)

#                 vllm_module_name = None
#                 prefix = ""

#                 vllm_module_name = name + "."

#             # 3. Save Compressed Tensors
#             _save_rans_attributes(tensors, vllm_module_name, prefix, module)

#             # 4. Mark original parameters as Handled
#             # The standard parameter name is usually "{module_name}.weight"
#             handled_params.add(f"{name}.weight")

#             if hasattr(module, "bias") and module.bias is not None:
#                 handled_params.add(f"{name}.bias")

#     # Iterate over ALL parameters in the model.
#     # If we didn't compress it in Pass 1, save it raw here.
#     for param_name, param in model.named_parameters():
#         if param_name in handled_params:
#             continue

#         tensors[param_name] = param.data.cpu()

#     # Metadata
#     metadata = {"format": "pt", "compression_method": "rans_bfloat16"}

#     save_file(tensors, output_path, metadata=metadata)
#     print(f"Saved {len(tensors)} tensors.")


def _save_rans_attributes(tensors, base_name, prefix, module):
    p = prefix

    num_tiles_n = 0
    num_tiles_k = 0
    tile_height = 0
    tile_width = 0

    # Helper to force dtypes and avoid signed-bit corruption
    def _to_u32(t):
        return t.cpu().to(torch.uint32)

    def _to_u8(t):
        return t.cpu().to(torch.uint8)

    def _validate_and_get(module, attr_name, expected_dtype):
        if not hasattr(module, attr_name):
            return None

        tensor = getattr(module, attr_name)

        # Check if the tensor matches the requirement
        if tensor.dtype != expected_dtype:
            # We log a warning and cast, but in a 'strict' mode you might raise RuntimeError
            print(
                f"WARNING: {attr_name} has dtype {tensor.dtype}, forcing to {expected_dtype}"
            )
            tensor = tensor.to(expected_dtype)

        return tensor.cpu()

    # 1. EXPONENT COMPRESSION DATA
    if hasattr(module, "exponent_compressed_weight"):
        num_tiles_n = getattr(module, "exponent_num_tiles_n", 0)
        num_tiles_k = getattr(module, "exponent_num_tiles_k", 0)
        tile_height = getattr(module, "exponent_tile_height", 0)
        tile_width = getattr(module, "exponent_tile_width", 0)
        exp_streams = getattr(module, "exponent_num_streams", 0)

        # The raw bitstream must be uint8
        tensors[f"{base_name}{p}rans_exp_stream"] = _validate_and_get(
            module, "exponent_compressed_weight", torch.uint8
        )

        # CRITICAL: States must be uint32 to prevent arithmetic shift corruption in Triton
        tensors[f"{base_name}{p}rans_exp_states"] = _validate_and_get(
            module, "exponent_states", torch.uint32
        )

        # Metadata / Tile Metrics
        tensors[f"{base_name}{p}rans_exp_tile_offsets"] = _validate_and_get(
            module, "exponent_tile_offsets", torch.uint32
        )

        tensors[f"{base_name}{p}rans_exp_tile_max_lens"] = _validate_and_get(
            module, "exponent_tile_max_lens", torch.uint32
        )

        # Decompression Tables
        tensors[f"{base_name}{p}rans_exp_tables"] = _validate_and_get(
            module, "exponent_tables", torch.uint32
        )

        tensors[f"{base_name}{p}rans_exp_slot_map"] = _validate_and_get(
            module, "exponent_slot_map", torch.uint8
        )
        is_exp_compressed = 1
    else:
        # Fallback for uncompressed raw exponents
        tensors[f"{base_name}{p}rans_exp_raw"] = module.exponent_raw.cpu()
        is_exp_compressed = 0
        exp_streams = 0

    # Mantissa
    if hasattr(module, "mantissa_compressed_weight"):
        tensors[
            f"{base_name}{p}rans_man_stream"
        ] = module.mantissa_compressed_weight.cpu()
        tensors[f"{base_name}{p}rans_man_states"] = module.mantissa_states.cpu().to(
            torch.uint32
        )
        # tensors[
        #     f"{base_name}{p}rans_man_sizes"
        # ] = module.mantissa_output_sizes.cpu().to(torch.uint32)
        tensors[f"{base_name}{p}rans_man_freqs"] = module.mantissa_freqs.cpu().to(
            torch.uint16
        )
        tensors[f"{base_name}{p}rans_man_cdf"] = module.mantissa_cdf.cpu().to(
            torch.uint16
        )
        man_streams = module.mantissa_num_streams
        is_man_compressed = 1
    else:
        tensors[f"{base_name}{p}rans_man_raw"] = _to_u8(module.mantissa_raw)
        is_man_compressed = 0
        man_streams = 0

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
        num_tiles_n,
        num_tiles_k,
        tile_height,
        tile_width,
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
    num_tiles_n = info[7].item() if len(info) > 7 else 0
    num_tiles_k = info[8].item() if len(info) > 8 else 0
    tile_height = info[9].item() if len(info) > 9 else 0
    tile_width = info[10].item() if len(info) > 10 else 0

    # Reconstruct shape tuple
    input_shape = torch.Size(info[11 : 11 + shape_rank].tolist())

    # Set Module Attributes
    module.compressed = "rans_bfloat16"
    module.expanded_size = expanded_size
    module.input_shape = input_shape
    module.tile_height = tile_height
    module.tile_width = tile_width

    # 2. Load Exponent Data
    if is_exp_compressed > 0:
        module.exponent_compressed_weight = f.get_tensor(f"{prefix}.rans_exp_stream")
        module.exponent_states = f.get_tensor(f"{prefix}.rans_exp_states")
        module.exponent_output_sizes = f.get_tensor(f"{prefix}.rans_exp_sizes")
        # module.exponent_freqs = f.get_tensor(
        #     f"{prefix}.rans_exp_freqs"
        # )
        # module.exponent_cdf = f.get_tensor(
        #     f"{prefix}.rans_exp_cdf"
        # )

        module.exponent_tables = f.get_tensor(f"{prefix}.rans_exp_tables")
        module.exponent_slot_map = f.get_tensor(f"{prefix}.rans_exp_slot_map")

        module.exponent_num_streams = exp_num_streams
        module.exponent_stream_size = module.exponent_compressed_weight.numel()

        # Calculate total stream size for the buffer view logic
        # (This is usually just numel, but explicit check helps)
        module.exponent_total_stream_size = module.exponent_stream_size

        if num_tiles_n > 0 and num_tiles_k > 0:
            module.exponent_num_tiles_n = num_tiles_n
            module.exponent_num_tiles_k = num_tiles_k

        if (
            hasattr(module, "exponent_tile_offsets")
            and f"{prefix}.rans_exp_tile_offsets" in f.keys()
        ):
            module.exponent_tile_offsets = f.get_tensor(
                f"{prefix}.rans_exp_tile_offsets"
            )

        if (
            hasattr(module, "exponent_tile_max_lens")
            and f"{prefix}.rans_exp_tile_max_lens" in f.keys()
        ):
            module.exponent_tile_max_lens = f.get_tensor(
                f"{prefix}.rans_exp_tile_max_lens"
            )

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
