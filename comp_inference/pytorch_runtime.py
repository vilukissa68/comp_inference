#!/usr/bin/env python3

# import torch
# import torch.nn as nn
# from safetensors.torch import load_file
# from transformers import AutoConfig, AutoModelForCausalLM

# from .rans_triton import (
#     rans_decomp_triton,
#     rans_decomp_triton_tiled,
# )

# from .linear_triton import (
#     fused_rans_linear_triton,
# )


# from .transposed_linear_triton import (
#     fused_rans_linear_transposed_triton,
# )

# from .embedding_triton import (
#     fused_rans_embedding_triton,
# )

# from .triton_gemm import (
#     triton_matmul,
# )


# class RansWorkspace:
#     _buffer = None

#     @classmethod
#     def get_workspace(cls, M, N, split_k, device):
#         elements = M * N * split_k
#         if cls._buffer is None or cls._buffer.numel() < elements:
#             cls._buffer = torch.zeros(elements * 2, device=device, dtype=torch.bfloat16)
#         return cls._buffer[:elements].view(split_k, M, N)


# class RansLinear(nn.Module):
#     def __init__(
#         self,
#         in_features: int,
#         out_features: int,
#         tile_k: int = 1024,
#         tile_n: int = 32,
#         split_k: int = 1,
#         has_bias: bool = False,
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.tile_k = tile_k
#         self.tile_n = tile_n
#         self.split_k = split_k

#         self.register_buffer("rans_exp_stream", torch.empty(0, dtype=torch.uint8))
#         self.register_buffer("rans_exp_states", torch.empty(0, dtype=torch.uint32))
#         self.register_buffer("rans_exp_tables", torch.empty(0, dtype=torch.uint32))
#         self.register_buffer("rans_exp_slot_map", torch.empty(0, dtype=torch.uint16))
#         self.register_buffer(
#             "rans_exp_tile_offsets", torch.empty(0, dtype=torch.uint32)
#         )
#         self.register_buffer(
#             "rans_exp_tile_max_lens", torch.empty(0, dtype=torch.uint32)
#         )
#         self.register_buffer("rans_man_raw", torch.empty(0, dtype=torch.uint8))
#         self.register_buffer("rans_info", torch.empty(0, dtype=torch.int32))
#         if has_bias:
#             self.register_buffer(
#                 "bias", torch.empty(out_features, dtype=torch.bfloat16)
#             )
#         else:
#             self.bias = None

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         M = x.view(-1, self.in_features).shape[0]

#         # Workspace uses out_features perfectly
#         workspace = RansWorkspace.get_workspace(
#             M, self.out_features, self.split_k, x.device
#         )

#         result = fused_rans_linear_triton(
#             x=x,
#             compressed_data=self.rans_exp_stream,
#             initial_states=self.rans_exp_states,
#             tables=self.rans_exp_tables,
#             slot_map=self.rans_exp_slot_map,
#             weight_shape=(self.in_features, self.out_features),
#             tile_offsets=self.rans_exp_tile_offsets,
#             tile_max_lens=self.rans_exp_tile_max_lens,
#             tile_k=self.tile_k,
#             tile_n=self.tile_n,
#             mantissas=self.rans_man_raw,
#             workspace=workspace,
#             SPLIT_K=self.split_k,
#             bias=self.bias,
#         )
#         return result


# class RansEmbedding(nn.Module):
#     def __init__(
#         self,
#         num_embeddings: int,
#         embedding_dim: int,
#         tile_k: int = 1024,
#         tile_n: int = 32,
#     ):
#         super().__init__()
#         self.num_embeddings = num_embeddings
#         self.embedding_dim = embedding_dim
#         self.tile_k = tile_k
#         self.tile_n = tile_n

#         self.register_buffer("rans_exp_stream", torch.empty(0, dtype=torch.uint8))
#         self.register_buffer("rans_exp_states", torch.empty(0, dtype=torch.uint32))
#         self.register_buffer("rans_exp_tables", torch.empty(0, dtype=torch.uint32))
#         self.register_buffer("rans_exp_slot_map", torch.empty(0, dtype=torch.uint16))
#         self.register_buffer(
#             "rans_exp_tile_offsets", torch.empty(0, dtype=torch.uint32)
#         )
#         self.register_buffer(
#             "rans_exp_tile_max_lens", torch.empty(0, dtype=torch.uint32)
#         )
#         self.register_buffer("rans_man_raw", torch.empty(0, dtype=torch.uint8))
#         self.register_buffer("rans_info", torch.empty(0, dtype=torch.int32))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         result = fused_rans_embedding_triton(
#             x=x,
#             compressed_data=self.rans_exp_stream,
#             initial_states=self.rans_exp_states,
#             tables=self.rans_exp_tables,
#             slot_map=self.rans_exp_slot_map,
#             weight_shape=(self.num_embeddings, self.embedding_dim),
#             tile_offsets=self.rans_exp_tile_offsets,
#             tile_max_lens=self.rans_exp_tile_max_lens,
#             tile_k=self.tile_k,
#             tile_n=self.tile_n,
#             mantissas=self.rans_man_raw,
#         )
#         return result


# class TiedRansLMHead(nn.Module):
#     def __init__(self, embed_tokens: RansEmbedding):
#         super().__init__()
#         self.embed_tokens = embed_tokens

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         M = x.view(-1, self.embed_tokens.embedding_dim).shape[0]
#         workspace = RansWorkspace.get_workspace(
#             M, self.embed_tokens.num_embeddings, 1, x.device
#         )
#         result = fused_rans_linear_transposed_triton(
#             x=x,
#             compressed_data=self.embed_tokens.rans_exp_stream,
#             initial_states=self.embed_tokens.rans_exp_states,
#             tables=self.embed_tokens.rans_exp_tables,
#             slot_map=self.embed_tokens.rans_exp_slot_map,
#             weight_shape=(
#                 self.embed_tokens.embedding_dim,
#                 self.embed_tokens.num_embeddings,
#             ),
#             tile_offsets=self.embed_tokens.rans_exp_tile_offsets,
#             tile_max_lens=self.embed_tokens.rans_exp_tile_max_lens,
#             tile_n=self.embed_tokens.tile_n,
#             tile_k=self.embed_tokens.tile_k,
#             mantissas=self.embed_tokens.rans_man_raw,
#             bias=None,
#             SPLIT_K=1,
#             workspace=workspace,
#         )
#         return result


# class RansLMHead(nn.Module):
#     def __init__(
#         self, in_features: int, out_features: int, tile_k: int = 1024, tile_n: int = 32
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features

#         self.tile_k = tile_k
#         self.tile_n = tile_n

#         self.register_buffer("rans_exp_stream", torch.empty(0, dtype=torch.uint8))
#         self.register_buffer("rans_exp_states", torch.empty(0, dtype=torch.uint32))
#         self.register_buffer("rans_exp_tables", torch.empty(0, dtype=torch.uint32))
#         self.register_buffer("rans_exp_slot_map", torch.empty(0, dtype=torch.uint16))
#         self.register_buffer(
#             "rans_exp_tile_offsets", torch.empty(0, dtype=torch.uint32)
#         )
#         self.register_buffer(
#             "rans_exp_tile_max_lens", torch.empty(0, dtype=torch.uint32)
#         )
#         self.register_buffer("rans_man_raw", torch.empty(0, dtype=torch.uint8))
#         self.register_buffer("rans_info", torch.empty(0, dtype=torch.int32))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         M = x.view(-1, self.in_features).shape[0]

#         workspace = RansWorkspace.get_workspace(M, self.out_features, 1, x.device)

#         result = fused_rans_linear_transposed_triton(
#             x=x,
#             compressed_data=self.rans_exp_stream,
#             initial_states=self.rans_exp_states,
#             tables=self.rans_exp_tables,
#             slot_map=self.rans_exp_slot_map,
#             weight_shape=(self.out_features, self.in_features),
#             tile_offsets=self.rans_exp_tile_offsets,
#             tile_max_lens=self.rans_exp_tile_max_lens,
#             tile_n=self.tile_n,
#             tile_k=self.tile_k,
#             mantissas=self.rans_man_raw,
#             workspace=workspace,
#             SPLIT_K=1,
#             bias=None,
#         )
#         return result


# class SharedFusedLayer:
#     """Caches the fused GEMM output so we don't re-compute it 3 times for Q, K, and V."""

#     def __init__(self, fused_layer):
#         self.fused_layer = fused_layer
#         self.last_x_ptr = None
#         self.last_out = None

#     def __call__(self, x):
#         if self.last_x_ptr != x.data_ptr():
#             self.last_out = self.fused_layer(x)
#             self.last_x_ptr = x.data_ptr()
#         return self.last_out


# class SliceProxy(nn.Module):
#     """Fools Hugging Face into thinking it's calling a separate Q/K/V linear layer."""

#     def __init__(self, shared_fused, start_idx, end_idx):
#         super().__init__()
#         self.shared_fused = shared_fused
#         self.start_idx = start_idx
#         self.end_idx = end_idx

#     def forward(self, x):
#         return self.shared_fused(x)[..., self.start_idx : self.end_idx]


# def load_compressed_model_with_auto_model(
#     model_id: str, safetensors_path: str, split_k: int = 1, compile_model: bool = False
# ):
#     print(f"Loading config for {model_id}...")
#     config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

#     quant_config = getattr(config, "quantization_config", {})
#     layer_configs = quant_config.get("layer_configs", {})
#     default_th = quant_config.get("default_tile_height", 1024)
#     default_tw = quant_config.get("default_tile_width", 32)

#     # 1. FAST INITIALIZATION
#     print("Initializing empty architecture on 'meta' device...")
#     with torch.device("meta"):
#         model = AutoModelForCausalLM.from_config(
#             config, torch_dtype=torch.bfloat16, trust_remote_code=True
#         )

#     print(f"Loading compressed safetensors from {safetensors_path}...")
#     state_dict = load_file(safetensors_path, device="cpu")

#     print("Swapping RANS modules...")
#     replacements = []
#     for name, module in model.named_modules():
#         if "lm_head" in name or isinstance(
#             module, (RansLinear, RansEmbedding, RansLMHead)
#         ):
#             continue

#         if f"{name}.rans_exp_stream" not in state_dict:
#             continue

#         th = layer_configs.get(name, {}).get("tile_height", default_th)
#         tw = layer_configs.get(name, {}).get("tile_width", default_tw)

#         if isinstance(module, nn.Linear):
#             replacements.append(
#                 (
#                     name,
#                     "linear",
#                     module.in_features,
#                     module.out_features,
#                     module.bias is not None,
#                     th,
#                     tw,
#                 )
#             )
#         elif isinstance(module, nn.Embedding):
#             replacements.append(
#                 (
#                     name,
#                     "embedding",
#                     module.num_embeddings,
#                     module.embedding_dim,
#                     False,
#                     th,
#                     tw,
#                 )
#             )

#     for name, layer_type, dim1, dim2, has_bias, th, tw in replacements:
#         parent_name = ".".join(name.split(".")[:-1])
#         child_name = name.split(".")[-1]
#         parent_module = model.get_submodule(parent_name) if parent_name else model

#         new_layer = (
#             RansLinear(dim1, dim2, th, tw, split_k, has_bias)
#             if layer_type == "linear"
#             else RansEmbedding(dim1, dim2, th, tw)
#         )
#         setattr(parent_module, child_name, new_layer)

#     if hasattr(model, "lm_head") and "lm_head.rans_exp_stream" in state_dict:
#         th = layer_configs.get("lm_head", {}).get("tile_height", default_th)
#         tw = layer_configs.get("lm_head", {}).get("tile_width", default_tw)
#         in_dim = model.lm_head.in_features
#         out_dim = model.lm_head.out_features
#         model.lm_head = RansLMHead(in_dim, out_dim, tile_k=th, tile_n=tw)

#     # 2. ALLOCATE REAL MEMORY
#     print("Allocating memory structures...")
#     # NOTE: You can change this to "cuda" directly if you want maximum speed and don't need CPU offloading
#     model.to_empty(device="cpu")

#     # 3. THE ROPE CURE: Fix the uninitialized memory!
#     print("Curing RoPE corruption...")
#     base = getattr(config, "rotary_emb_base", 1000000.0)  # Qwen defaults to high base
#     for name, module in model.named_modules():
#         if hasattr(module, "inv_freq") and module.inv_freq is not None:
#             dim = module.inv_freq.shape[0] * 2
#             inv_freq = 1.0 / (
#                 base
#                 ** (
#                     torch.arange(
#                         0, dim, 2, dtype=torch.float32, device=module.inv_freq.device
#                     )
#                     / dim
#                 )
#             )
#             with torch.no_grad():
#                 module.inv_freq.copy_(inv_freq)

#     # 4. ASSIGN POINTERS
#     print("Assigning data pointers...")
#     model_state = dict(model.named_parameters())
#     model_state.update(dict(model.named_buffers()))

#     for name, target_tensor in model_state.items():
#         if name in state_dict:
#             target_tensor.data = state_dict[name]

#     print("Moving model to CUDA...")
#     model = model.cuda()
#     print("Model loaded and ready!")
#     return model


#!/usr/bin/env python3

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from .linear_triton import fused_rans_linear_triton
from .transposed_linear_triton import fused_rans_linear_transposed_triton
from .embedding_triton import fused_rans_embedding_triton


class RansWorkspace:
    _buffer = None

    @classmethod
    def get_workspace(cls, M, N, split_k, device):
        elements = M * N * split_k
        if cls._buffer is None or cls._buffer.numel() < elements:
            cls._buffer = torch.zeros(elements * 2, device=device, dtype=torch.bfloat16)
        return cls._buffer[:elements].view(split_k, M, N)


class RansLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tile_k: int = 1024,
        tile_n: int = 32,
        split_k: int = 1,
        has_bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tile_k = tile_k
        self.tile_n = tile_n
        self.split_k = split_k

        self.register_buffer("rans_exp_stream", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("rans_exp_states", torch.empty(0, dtype=torch.uint32))
        self.register_buffer("rans_exp_tables", torch.empty(0, dtype=torch.uint32))
        self.register_buffer("rans_exp_slot_map", torch.empty(0, dtype=torch.uint16))
        self.register_buffer(
            "rans_exp_tile_offsets", torch.empty(0, dtype=torch.uint32)
        )
        self.register_buffer(
            "rans_exp_tile_max_lens", torch.empty(0, dtype=torch.uint32)
        )
        self.register_buffer("rans_man_raw", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("rans_info", torch.empty(0, dtype=torch.int32))
        if has_bias:
            self.register_buffer(
                "bias", torch.empty(out_features, dtype=torch.bfloat16)
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        M = x.view(-1, self.in_features).shape[0]
        workspace = RansWorkspace.get_workspace(
            M, self.out_features, self.split_k, device
        )

        # UNINTRUSIVE ON-DEMAND OFFLOADING
        # If already on CUDA, .to() does nothing. If on CPU, it moves it right before compute.
        return fused_rans_linear_triton(
            x=x,
            compressed_data=self.rans_exp_stream.to(device, non_blocking=True),
            initial_states=self.rans_exp_states.to(device, non_blocking=True),
            tables=self.rans_exp_tables.to(device, non_blocking=True),
            slot_map=self.rans_exp_slot_map.to(device, non_blocking=True),
            weight_shape=(self.in_features, self.out_features),
            tile_offsets=self.rans_exp_tile_offsets.to(device, non_blocking=True),
            tile_max_lens=self.rans_exp_tile_max_lens.to(device, non_blocking=True),
            tile_k=self.tile_k,
            tile_n=self.tile_n,
            mantissas=self.rans_man_raw.to(device, non_blocking=True),
            workspace=workspace,
            SPLIT_K=self.split_k,
            bias=self.bias.to(device, non_blocking=True)
            if self.bias is not None
            else None,
        )


class RansLMHead(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, tile_k: int = 1024, tile_n: int = 32
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tile_k = tile_k
        self.tile_n = tile_n

        self.register_buffer("rans_exp_stream", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("rans_exp_states", torch.empty(0, dtype=torch.uint32))
        self.register_buffer("rans_exp_tables", torch.empty(0, dtype=torch.uint32))
        self.register_buffer("rans_exp_slot_map", torch.empty(0, dtype=torch.uint16))
        self.register_buffer(
            "rans_exp_tile_offsets", torch.empty(0, dtype=torch.uint32)
        )
        self.register_buffer(
            "rans_exp_tile_max_lens", torch.empty(0, dtype=torch.uint32)
        )
        self.register_buffer("rans_man_raw", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("rans_info", torch.empty(0, dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        M = x.view(-1, self.in_features).shape[0]
        workspace = RansWorkspace.get_workspace(M, self.out_features, 1, device)

        return fused_rans_linear_transposed_triton(
            x=x,
            compressed_data=self.rans_exp_stream.to(device, non_blocking=True),
            initial_states=self.rans_exp_states.to(device, non_blocking=True),
            tables=self.rans_exp_tables.to(device, non_blocking=True),
            slot_map=self.rans_exp_slot_map.to(device, non_blocking=True),
            weight_shape=(self.out_features, self.in_features),
            tile_offsets=self.rans_exp_tile_offsets.to(device, non_blocking=True),
            tile_max_lens=self.rans_exp_tile_max_lens.to(device, non_blocking=True),
            tile_n=self.tile_n,
            tile_k=self.tile_k,
            mantissas=self.rans_man_raw.to(device, non_blocking=True),
            workspace=workspace,
            SPLIT_K=1,
            bias=None,
        )


class RansEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        tile_k: int = 1024,
        tile_n: int = 32,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tile_k = tile_k
        self.tile_n = tile_n

        self.register_buffer("rans_exp_stream", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("rans_exp_states", torch.empty(0, dtype=torch.uint32))
        self.register_buffer("rans_exp_tables", torch.empty(0, dtype=torch.uint32))
        self.register_buffer("rans_exp_slot_map", torch.empty(0, dtype=torch.uint16))
        self.register_buffer(
            "rans_exp_tile_offsets", torch.empty(0, dtype=torch.uint32)
        )
        self.register_buffer(
            "rans_exp_tile_max_lens", torch.empty(0, dtype=torch.uint32)
        )
        self.register_buffer("rans_man_raw", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("rans_info", torch.empty(0, dtype=torch.int32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        return fused_rans_embedding_triton(
            x=x,
            compressed_data=self.rans_exp_stream.to(device, non_blocking=True),
            initial_states=self.rans_exp_states.to(device, non_blocking=True),
            tables=self.rans_exp_tables.to(device, non_blocking=True),
            slot_map=self.rans_exp_slot_map.to(device, non_blocking=True),
            weight_shape=(self.num_embeddings, self.embedding_dim),
            tile_offsets=self.rans_exp_tile_offsets.to(device, non_blocking=True),
            tile_max_lens=self.rans_exp_tile_max_lens.to(device, non_blocking=True),
            tile_k=self.tile_k,
            tile_n=self.tile_n,
            mantissas=self.rans_man_raw.to(device, non_blocking=True),
        )


def load_compressed_model_with_auto_model(
    model_id: str, safetensors_path: str, split_k: int = 1, compile_model: bool = False
):
    print(
        f"Loading FULL baseline {model_id} to CPU to ensure mathematically valid LayerNorms..."
    )
    # 1. LOAD THE REAL MODEL (Fixes LayerNorms and Masks entirely)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True
    )

    config = model.config
    quant_config = getattr(config, "quantization_config", {})
    layer_configs = quant_config.get("layer_configs", {})
    default_th = quant_config.get("default_tile_height", 1024)
    default_tw = quant_config.get("default_tile_width", 32)

    print(f"Loading compressed safetensors from {safetensors_path}...")
    state_dict = load_file(safetensors_path, device="cpu")

    print("Swapping RANS modules...")
    replacements = []
    for name, module in model.named_modules():
        if "lm_head" in name or isinstance(
            module, (RansLinear, RansEmbedding, RansLMHead)
        ):
            continue

        if f"{name}.rans_exp_stream" not in state_dict:
            continue

        th = layer_configs.get(name, {}).get("tile_height", default_th)
        tw = layer_configs.get(name, {}).get("tile_width", default_tw)

        if isinstance(module, nn.Linear):
            replacements.append(
                (
                    name,
                    "linear",
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    th,
                    tw,
                )
            )
        elif isinstance(module, nn.Embedding):
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

    if hasattr(model, "lm_head") and "lm_head.rans_exp_stream" in state_dict:
        th = layer_configs.get("lm_head", {}).get("tile_height", default_th)
        tw = layer_configs.get("lm_head", {}).get("tile_width", default_tw)
        in_dim = model.lm_head.in_features
        out_dim = model.lm_head.out_features
        model.lm_head = RansLMHead(in_dim, out_dim, tile_k=th, tile_n=tw)

    print("Assigning compressed data pointers...")
    model_state = dict(model.named_buffers())
    model_state.update(dict(model.named_parameters()))

    # 2. INJECT RANS BYTES (Ignores LayerNorms, they stay correctly loaded from HF)
    for name, target_tensor in model_state.items():
        if name in state_dict:
            target_tensor.data = state_dict[name].contiguous()

    print("Moving model to CUDA...")
    model = model.cuda()
    print("Model loaded and ready!")
    return model
