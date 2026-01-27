import torch
import numpy as np
import gguf 
import re
import json
import os
from transformers.utils.hub import cached_file


def hf_to_gguf_name(name: str) -> str:
    new_name = name
    new_name = re.sub(r'^model\.layers\.(\d+)', r'blk.\1', new_name)
    new_name = new_name.replace("model.embed_tokens", "token_embd")
    new_name = new_name.replace("model.norm", "output_norm")
    new_name = new_name.replace("lm_head", "output")
    new_name = new_name.replace("self_attn.q_proj", "attn_q")
    new_name = new_name.replace("self_attn.k_proj", "attn_k")
    new_name = new_name.replace("self_attn.v_proj", "attn_v")
    new_name = new_name.replace("self_attn.o_proj", "attn_output")
    new_name = new_name.replace("mlp.gate_proj", "ffn_gate")
    new_name = new_name.replace("mlp.up_proj", "ffn_up")
    new_name = new_name.replace("mlp.down_proj", "ffn_down")
    new_name = new_name.replace("input_layernorm", "attn_norm")
    new_name = new_name.replace("post_attention_layernorm", "ffn_norm")
    if new_name.endswith(".weight"):
        new_name = new_name[:-7]
    return new_name

def sanitize_merges(raw_merges):
    if not raw_merges:
        return []
    
    sanitized = []
    for m in raw_merges:
        if isinstance(m, str):
            sanitized.append(m)
        elif hasattr(m, "__getitem__") and len(m) >= 2:
            sanitized.append(f"{m[0]} {m[1]}")
        else:
            sanitized.append(str(m))
    return sanitized

# TODO: Generalize tokenizer support
def write_tokenizer(gguf_writer, tokenizer, model_path_or_name):
    merges = []

    # Try extracting from the tokenizer instance backend
    if hasattr(tokenizer, "backend_tokenizer"):
        try:
            raw_merges = tokenizer.backend_tokenizer.model.get_merges()
            merges = sanitize_merges(raw_merges)
        except Exception:
            pass

    # Look for merges.txt file
    if not merges:
        json_path = os.path.join(model_path_or_name, "tokenizer.json")
        if not os.path.exists(json_path):
            try:
                json_path = cached_file(model_path_or_name, "tokenizer.json")
            except Exception:
                json_path = None

        if json_path and os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    raw_merges = data.get("model", {}).get("merges", [])
                    merges = sanitize_merges(raw_merges)
            except Exception:
                pass

    if not merges:
        raise ValueError("Could not extract BPE merges.")

    gguf_writer.add_tokenizer_model("gpt2")
    gguf_writer.add_tokenizer_pre("qwen2")
    gguf_writer.add_token_merges(merges)

    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    sorted_tokens = [b""] * vocab_size

    for token, id in vocab.items():
        if isinstance(token, str):
            sorted_tokens[id] = token.encode("utf-8", errors="ignore")
        else:
            sorted_tokens[id] = token

    for i in range(vocab_size):
        if sorted_tokens[i] == b"":
            sorted_tokens[i] = b"<unk>"

    gguf_writer.add_token_list(sorted_tokens)

    tok_types = [1] * vocab_size
    for sid in tokenizer.all_special_ids:
        if sid < vocab_size:
            tok_types[sid] = 3
    
    gguf_writer.add_token_types(tok_types)
    gguf_writer.add_token_scores([0.0] * vocab_size)

    if tokenizer.bos_token_id is not None:
        gguf_writer.add_bos_token_id(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        gguf_writer.add_eos_token_id(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        gguf_writer.add_pad_token_id(tokenizer.pad_token_id)

def save_rans_model_gguf(model, tokenizer, output_path: str, model_name: str = None):
    print(f"Packing model to GGUF: {output_path}...")

    
    arch = "qwen3" 
    gguf_writer = gguf.GGUFWriter(output_path, arch)

    # Metadata
    gguf_writer.add_architecture()
    gguf_writer.add_block_count(model.config.num_hidden_layers)
    gguf_writer.add_context_length(getattr(model.config, "max_position_embeddings", 32768))
    gguf_writer.add_embedding_length(model.config.hidden_size)
    gguf_writer.add_feed_forward_length(model.config.intermediate_size)
    gguf_writer.add_head_count(model.config.num_attention_heads)
    
    num_kv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    gguf_writer.add_head_count_kv(num_kv)

    head_dim = model.config.hidden_size // model.config.num_attention_heads
    gguf_writer.add_rope_dimension_count(head_dim)
    gguf_writer.add_rope_freq_base(getattr(model.config, "rope_theta", 1000000.0))

    eps = getattr(model.config, "rms_norm_eps", getattr(model.config, "layer_norm_epsilon", 1e-6))
    gguf_writer.add_layer_norm_rms_eps(eps)
    
    gguf_writer.add_string("quantization.method", "rans_bfloat16")

    # Tokenizer
    write_tokenizer(gguf_writer, tokenizer, model_name)

    # Tensors
    for name, module in model.named_modules():
        
        # Internal helper to handle names and types automatically
        def add(suffix, tensor):

            clean_name = hf_to_gguf_name(name)
            key = f"{clean_name}.{suffix}"

            # Type casts for GGUF
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.view(torch.int16)
            elif tensor.dtype == torch.uint8:
                tensor = tensor.view(torch.int8) 
            elif tensor.dtype == torch.uint16:
                tensor = tensor.view(torch.int16)
            elif tensor.dtype == torch.uint32:
                tensor = tensor.view(torch.int32)

            data = tensor.detach().cpu().numpy()
            gguf_writer.add_tensor(key, data)

        # Handle rANS compressed modules
        if hasattr(module, "compressed") and module.compressed == "rans_bfloat16":
            
            # Exponent
            if hasattr(module, "exponent_compressed_weight"):
                add("rans_exp_stream", module.exponent_compressed_weight)
                add("rans_exp_states", module.exponent_states)
                add("rans_exp_sizes",  module.exponent_output_sizes)
                add("rans_exp_freqs",  module.exponent_freqs)
                add("rans_exp_cdf",    module.exponent_cdf)
                exp_streams = module.exponent_num_streams
                is_exp_compressed = 1
            else:
                # Use raw fallback
                add("rans_exp_raw", module.exponent_raw)
                exp_streams = 0
                is_exp_compressed = 0

            # Mantissa
            if hasattr(module, "mantissa_compressed_weight"):
                add("rans_man_stream", module.mantissa_compressed_weight)
                add("rans_man_states", module.mantissa_states)
                add("rans_man_sizes",  module.mantissa_output_sizes)
                add("rans_man_freqs",  module.mantissa_freqs)
                add("rans_man_cdf",    module.mantissa_cdf)
                man_streams = module.mantissa_num_streams
                is_man_compressed = 1
            else:
                # Use raw fallback (flattened)
                raw_u8 = module.mantissa_raw.flatten()
                add("rans_man_raw", raw_u8)
                man_streams = 0
                is_man_compressed = 0

            # Info Tensor
            shape = getattr(module, "input_shape", [])
            if not shape and hasattr(module, "expanded_size"):
                shape = [module.expanded_size]

            info_data = [
                1, # Version
                module.expanded_size,
                is_exp_compressed,
                exp_streams,
                is_man_compressed,
                man_streams,
                len(shape)
            ] + list(shape)

            add("rans_info", torch.tensor(info_data, dtype=torch.int32))

            # Bias
            if hasattr(module, "bias") and module.bias is not None:
                add("bias", module.bias.float())

        else:
            for param_name, param in module.named_parameters(recurse=False):
                if param_name == "weight" and hasattr(module, "compressed"):
                    continue
                
                # Shorten name
                full_name = f"{name}.{param_name}" if name else param_name
                clean_key = hf_to_gguf_name(full_name)
                
                # Convert data
                data = param.detach().cpu()
                if data.dtype == torch.bfloat16:
                    data = data.view(torch.int16)
                
                gguf_writer.add_tensor(clean_key, data.numpy())

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"GGUF file saved: {output_path}")
