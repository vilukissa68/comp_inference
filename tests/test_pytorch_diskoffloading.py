# import os
# import glob
# import time
# import threading
# import argparse
# import torch
# import pynvml
# from safetensors import safe_open
# from huggingface_hub import snapshot_download
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # IMPORT YOUR EXACT API
# from comp_inference.pytorch_runtime import load_compressed_model_with_auto_model


# class PowerProfiler:
#     """Background thread to poll NVIDIA GPU power consumption."""

#     def __init__(self, device_idx=0, poll_interval=0.01):
#         pynvml.nvmlInit()
#         self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
#         self.poll_interval = poll_interval
#         self.power_readings = []
#         self.running = False
#         self.thread = None

#     def _poll(self):
#         while self.running:
#             power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
#             self.power_readings.append(power_mw / 1000.0)  # Convert to Watts
#             time.sleep(self.poll_interval)

#     def start(self):
#         self.power_readings = []
#         self.running = True
#         self.thread = threading.Thread(target=self._poll)
#         self.thread.start()

#     def stop(self):
#         self.running = False
#         if self.thread:
#             self.thread.join()

#         if not self.power_readings:
#             return 0.0, 0.0
#         return sum(self.power_readings) / len(self.power_readings), max(
#             self.power_readings
#         )


# class MultiSafeOpen:
#     """Wraps multiple safetensors shards into a single continuous file interface."""

#     def __init__(self, file_paths, device="cuda"):
#         self.files = {}
#         self.key_to_file = {}
#         for path in file_paths:
#             f = safe_open(path, framework="pt", device=device)
#             self.files[path] = f
#             for k in f.keys():
#                 self.key_to_file[k] = path

#     def get_tensor(self, key):
#         path = self.key_to_file[key]
#         return self.files[path].get_tensor(key)

#     def keys(self):
#         return self.key_to_file.keys()


# # def apply_disk_offload(model, safetensors_paths, offload_ratio):
# #     """
# #     Evicts the last D% of transformer layers to disk.
# #     Attaches hooks to stream them natively into modules JIT.
# #     """
# #     if isinstance(safetensors_paths, str):
# #         safetensors_paths = [safetensors_paths]

# #     # Pre-map all keys to verify what exists on disk
# #     with torch.device("cpu"):
# #         multi_f = MultiSafeOpen(safetensors_paths, device="cpu")
# #         valid_keys = set(multi_f.keys())

# #     layers = model.model.layers
# #     num_offload = int(len(layers) * offload_ratio)
# #     if num_offload == 0:
# #         print("Offload ratio too low, running fully on GPU.")
# #         return

# #     start_idx = len(layers) - num_offload
# #     print(
# #         f"Offloading {num_offload}/{len(layers)} layers to disk (Layers {start_idx} to {len(layers)-1})..."
# #     )

# #     for i in range(start_idx, len(layers)):
# #         layer_prefix = f"model.layers.{i}"

# #         for name, submodule in layers[i].named_modules():
# #             if len(list(submodule.children())) > 0:
# #                 continue

# #             full_prefix = f"{layer_prefix}.{name}" if name else layer_prefix
# #             submodule_keys = [k for k in valid_keys if k.startswith(full_prefix + ".")]

# #             if not submodule_keys:
# #                 continue

# #             # 1. EVICT FROM GPU TO 0 BYTES
# #             for p_name, param in submodule.named_parameters(recurse=False):
# #                 if f"{full_prefix}.{p_name}" in valid_keys:
# #                     # THE FIX: Use param.device instead of "meta"
# #                     param.data = torch.empty(0, dtype=param.dtype, device=param.device)
# #             for b_name, buf in submodule.named_buffers(recurse=False):
# #                 if f"{full_prefix}.{b_name}" in valid_keys:
# #                     # THE FIX: Use buf.device instead of "meta"
# #                     buf.data = torch.empty(0, dtype=buf.dtype, device=buf.device)

# #             # 2. ATTACH PRE-HOOK (Disk -> CUDA)
# #             def pre_hook(
# #                 mod,
# #                 inputs,
# #                 prefix=full_prefix,
# #                 keys=submodule_keys,
# #                 sf_paths=safetensors_paths,
# #             ):
# #                 f = MultiSafeOpen(sf_paths, device="cuda")
# #                 for p_name, p in mod.named_parameters(recurse=False):
# #                     key = f"{prefix}.{p_name}"
# #                     if key in keys:
# #                         p.data = f.get_tensor(key)
# #                 for b_name, b in mod.named_buffers(recurse=False):
# #                     key = f"{prefix}.{b_name}"
# #                     if key in keys:
# #                         b.data = f.get_tensor(key)

# #             # 3. ATTACH POST-HOOK (CUDA -> 0 BYTES)
# #             def post_hook(
# #                 mod, inputs, outputs, prefix=full_prefix, keys=submodule_keys
# #             ):
# #                 for p_name, p in mod.named_parameters(recurse=False):
# #                     if f"{prefix}.{p_name}" in keys:
# #                         # THE FIX: Use p.device instead of "meta"
# #                         p.data = torch.empty(0, dtype=p.dtype, device=p.device)
# #                 for b_name, b in mod.named_buffers(recurse=False):
# #                     if f"{prefix}.{b_name}" in keys:
# #                         # THE FIX: Use b.device instead of "meta"
# #                         b.data = torch.empty(0, dtype=b.dtype, device=b.device)

# #             submodule.register_forward_pre_hook(pre_hook)
# #             submodule.register_forward_hook(post_hook)


# def apply_disk_offload(model, safetensors_paths, offload_ratio):
#     """
#     Evicts the last D% of transformer layers to disk.
#     Attaches hooks to stream them natively into modules JIT using setattr to protect Triton caching.
#     """
#     if isinstance(safetensors_paths, str):
#         safetensors_paths = [safetensors_paths]

#     multi_f = MultiSafeOpen(safetensors_paths, device="cpu")
#     valid_keys = set(multi_f.keys())

#     layers = model.model.layers
#     num_offload = int(len(layers) * offload_ratio)
#     if num_offload == 0:
#         print("Offload ratio too low, running fully on GPU.")
#         return

#     start_idx = len(layers) - num_offload
#     print(
#         f"Offloading {num_offload}/{len(layers)} layers to disk (Layers {start_idx} to {len(layers)-1})..."
#     )

#     for i in range(start_idx, len(layers)):
#         layer_prefix = f"model.layers.{i}"

#         for name, submodule in layers[i].named_modules():
#             if len(list(submodule.children())) > 0:
#                 continue

#             full_prefix = f"{layer_prefix}.{name}" if name else layer_prefix
#             submodule_keys = [k for k in valid_keys if k.startswith(full_prefix + ".")]

#             if not submodule_keys:
#                 continue

#             # 1. CAPTURE METADATA AND EVICT SAFELY
#             param_meta = {}
#             for p_name, param in submodule.named_parameters(recurse=False):
#                 if f"{full_prefix}.{p_name}" in valid_keys:
#                     param_meta[p_name] = (param.dtype, param.device)
#                     # Use setattr to completely replace the object
#                     setattr(
#                         submodule,
#                         p_name,
#                         torch.nn.Parameter(
#                             torch.empty(0, dtype=param.dtype, device=param.device),
#                             requires_grad=False,
#                         ),
#                     )

#             buf_meta = {}
#             for b_name, buf in submodule.named_buffers(recurse=False):
#                 if f"{full_prefix}.{b_name}" in valid_keys:
#                     buf_meta[b_name] = (buf.dtype, buf.device)
#                     # Use setattr to completely replace the object
#                     setattr(
#                         submodule,
#                         b_name,
#                         torch.empty(0, dtype=buf.dtype, device=buf.device),
#                     )

#             # 2. ATTACH PRE-HOOK (Disk -> CUDA via setattr)
#             def pre_hook(
#                 mod,
#                 inputs,
#                 prefix=full_prefix,
#                 f=multi_f,
#                 p_meta=param_meta,
#                 b_meta=buf_meta,
#             ):
#                 for p_name in p_meta.keys():
#                     key = f"{prefix}.{p_name}"
#                     new_t = f.get_tensor(key).to("cuda", non_blocking=True)
#                     setattr(mod, p_name, torch.nn.Parameter(new_t, requires_grad=False))
#                 for b_name in b_meta.keys():
#                     key = f"{prefix}.{b_name}"
#                     new_t = f.get_tensor(key).to("cuda", non_blocking=True)
#                     setattr(mod, b_name, new_t)

#             # 3. ATTACH POST-HOOK (CUDA -> 0 BYTES via setattr)
#             def post_hook(mod, inputs, outputs, p_meta=param_meta, b_meta=buf_meta):
#                 for p_name, (dtype, device) in p_meta.items():
#                     setattr(
#                         mod,
#                         p_name,
#                         torch.nn.Parameter(
#                             torch.empty(0, dtype=dtype, device=device),
#                             requires_grad=False,
#                         ),
#                     )
#                 for b_name, (dtype, device) in b_meta.items():
#                     setattr(mod, b_name, torch.empty(0, dtype=dtype, device=device))

#             submodule.register_forward_pre_hook(pre_hook)
#             submodule.register_forward_hook(post_hook)


# def run_benchmark(model, tokenizer, prompt, mode_name):
#     print(f"\n--- Running Benchmark: {mode_name} ---")
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

#     # ABSORB TRITON JIT PENALTY
#     print("Warming up JIT compiler and allocating KV caches...")
#     with torch.no_grad():
#         _ = model.generate(**inputs, max_new_tokens=2)
#     torch.cuda.synchronize()

#     profiler = PowerProfiler()
#     print("Benchmarking Generation...")

#     profiler.start()
#     start_time = time.perf_counter()

#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=16, use_cache=True)

#     torch.cuda.synchronize()
#     end_time = time.perf_counter()
#     avg_power, max_power = profiler.stop()

#     latency = end_time - start_time
#     generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]

#     print(f"[{mode_name}] Results:")
#     print(f"Total Time:      {latency:.3f} s")
#     print(f"Throughput:      {generated_tokens / latency:.2f} tokens/s")
#     print(f"Average Power:   {avg_power:.2f} W")
#     print(f"Peak Power:      {max_power:.2f} W")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         required=True,
#         help="Hugging Face Hub ID (e.g. Qwen/Qwen3-0.6B)",
#     )
#     parser.add_argument(
#         "--compressed_path", type=str, required=True, help="Path to rANS safetensors"
#     )
#     parser.add_argument(
#         "--offload_ratio",
#         type=float,
#         default=0.5,
#         help="Percentage of layers to keep on disk (0.0 to 1.0)",
#     )
#     parser.add_argument(
#         "--prompt",
#         type=str,
#         default="Explain the meaning of life in detail.",
#         help="Test prompt",
#     )

#     args = parser.parse_args()

#     # 1. FETCH UNCOMPRESSED SAFETENSORS PATHS
#     print(f"Ensuring {args.model_name} is downloaded from Hugging Face...")
#     local_dir = snapshot_download(
#         repo_id=args.model_name, allow_patterns=["*.safetensors"]
#     )
#     uncompressed_safetensors = glob.glob(os.path.join(local_dir, "*.safetensors"))

#     if not uncompressed_safetensors:
#         raise ValueError(f"No safetensors found in {local_dir}. Please check the repo.")

#     tokenizer = AutoTokenizer.from_pretrained(args.model_name)

#     # 2. TEST UNCOMPRESSED BASELINE WITH OFFLOAD
#     print("\n==================================================")
#     print(f"LOADING UNCOMPRESSED BASELINE: {args.model_name}")
#     print("==================================================")
#     uncompressed_model = AutoModelForCausalLM.from_pretrained(
#         args.model_name, torch_dtype=torch.bfloat16, device_map="cuda"
#     )
#     apply_disk_offload(uncompressed_model, uncompressed_safetensors, args.offload_ratio)

#     run_benchmark(
#         uncompressed_model,
#         tokenizer,
#         args.prompt,
#         f"Native FP16 (Offload {args.offload_ratio*100}%)",
#     )

#     del uncompressed_model
#     torch.cuda.empty_cache()

#     # 3. TEST RANS BACKEND WITH OFFLOAD
#     print("\n==================================================")
#     print("LOADING RANS BACKEND")
#     print("==================================================")
#     rans_model = load_compressed_model_with_auto_model(
#         model_id=args.model_name, safetensors_path=args.compressed_path
#     )
#     apply_disk_offload(rans_model, [args.compressed_path], args.offload_ratio)

#     run_benchmark(
#         rans_model,
#         tokenizer,
#         args.prompt,
#         f"rANS Backend (Offload {args.offload_ratio*100}%)",
#     )

import os
import time
import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Import your modules directly
from comp_inference.pytorch_runtime import RansLinear, RansEmbedding, RansLMHead


def build_rans_architecture(config, split_k=1):
    """Builds the meta-model and swaps in rANS modules BEFORE loading weights."""
    quant_config = getattr(config, "quantization_config", {})
    layer_configs = quant_config.get("layer_configs", {})
    default_th = quant_config.get("default_tile_height", 1024)
    default_tw = quant_config.get("default_tile_width", 32)

    # 1. Native Accelerate Meta Initialization (No garbage memory bugs!)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=torch.bfloat16, trust_remote_code=True
        )

    # 2. Swap standard modules for rANS modules while on the meta device
    replacements = []
    for name, module in model.named_modules():
        if "lm_head" in name or isinstance(
            module, (RansLinear, RansEmbedding, RansLMHead)
        ):
            continue

        th = layer_configs.get(name, {}).get("tile_height", default_th)
        tw = layer_configs.get(name, {}).get("tile_width", default_tw)

        if isinstance(module, torch.nn.Linear):
            # Trusting your call: Qwen3 doesn't use bias!
            replacements.append(
                (name, "linear", module.in_features, module.out_features, False, th, tw)
            )
        elif isinstance(module, torch.nn.Embedding):
            replacements.append(
                (
                    name,
                    "embedding",
                    module.num_embeddings,
                    module.embedding_dim,
                    False,
                    th,
                    tw,
                )
            )

    for name, layer_type, dim1, dim2, has_bias, th, tw in replacements:
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent_module = model.get_submodule(parent_name) if parent_name else model

        new_layer = (
            RansLinear(dim1, dim2, th, tw, split_k, has_bias)
            if layer_type == "linear"
            else RansEmbedding(dim1, dim2, th, tw)
        )
        setattr(parent_module, child_name, new_layer)

    if hasattr(model, "lm_head"):
        th = layer_configs.get("lm_head", {}).get("tile_height", default_th)
        tw = layer_configs.get("lm_head", {}).get("tile_width", default_tw)
        in_dim = model.lm_head.in_features
        out_dim = model.lm_head.out_features
        model.lm_head = RansLMHead(in_dim, out_dim, tile_k=th, tile_n=tw)

    return model


def create_offload_device_map(model, offload_ratio, offload_device="disk"):
    """Programmatically generates an accelerate device_map to split layers."""
    device_map = {}

    # Keep critical non-layer components on GPU
    device_map["model.embed_tokens"] = 0
    device_map["model.norm"] = 0
    device_map["lm_head"] = 0

    layers = model.model.layers
    num_offload = int(len(layers) * offload_ratio)
    start_offload_idx = len(layers) - num_offload

    for i in range(len(layers)):
        if i < start_offload_idx:
            device_map[f"model.layers.{i}"] = 0
        else:
            device_map[f"model.layers.{i}"] = offload_device

    return device_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, required=True, help="HF ID (e.g. Qwen/Qwen3-0.6B)"
    )
    parser.add_argument(
        "--compressed_path", type=str, required=True, help="Path to rANS safetensors"
    )
    parser.add_argument(
        "--offload_ratio",
        type=float,
        default=0.5,
        help="Percentage of layers to keep on disk",
    )
    parser.add_argument("--prompt", type=str, default="Who is Linus Torvalds?")
    args = parser.parse_args()

    print(f"\n[1] Preparing {args.model_name} Architecture via Accelerate...")
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Build the architecture full of our RansLinear modules on the meta device
    model = build_rans_architecture(config)

    # Calculate exact device mappings
    device_map = create_offload_device_map(
        model, args.offload_ratio, offload_device="disk"
    )
    print(f"\n[2] Device Map Generated (Offloading {args.offload_ratio*100}% to disk):")

    # --- THE CURE for Accelerate Shape Checking ---
    print("\n[2.5] Aligning Meta Tensor Shapes with Checkpoint Header...")
    from safetensors import safe_open

    with safe_open(args.compressed_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            shape = f.get_slice(key).get_shape()
            parts = key.split(".")
            if len(parts) > 1:
                module_name = ".".join(parts[:-1])
                buffer_name = parts[-1]
                try:
                    submod = model.get_submodule(module_name)
                    if hasattr(submod, buffer_name):
                        buf = getattr(submod, buffer_name)

                        # THE FIX: Respect PyTorch's strict Parameter vs Buffer typing
                        empty_tensor = torch.empty(
                            shape, dtype=buf.dtype, device="meta"
                        )
                        if isinstance(buf, torch.nn.Parameter):
                            setattr(
                                submod,
                                buffer_name,
                                torch.nn.Parameter(empty_tensor, requires_grad=False),
                            )
                        else:
                            setattr(submod, buffer_name, empty_tensor)
                except AttributeError:
                    pass
    # Fix the Accelerate tied weights warning
    model.tie_weights()

    offload_folder = "./offload_cache"
    os.makedirs(offload_folder, exist_ok=True)

    print(
        "\n[3] Dispatching Model via Accelerate (Disk -> GPU hooks natively applied)..."
    )
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=args.compressed_path,
        device_map=device_map,
        offload_folder=offload_folder,
        dtype=torch.bfloat16,
        offload_state_dict=True,
    )

    print("\n[4] Benchmarking Generation...")
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")

    # Warmup
    with torch.no_grad():
        _ = model.generate(
            **inputs,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    latency = end_time - start_time
    generated_tokens = outputs.shape[1] - inputs.input_ids.shape[1]

    print("\n" + "=" * 60)
    print(f"Output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    print("=" * 60)
    print(f"Throughput: {generated_tokens / latency:.2f} tokens/s")
    print(f"Latency:    {latency:.3f} s")
