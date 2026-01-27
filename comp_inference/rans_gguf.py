import torch
import numpy as np
import gguf

def save_rans_model_gguf(model, output_path: str):
    print(f"Packing model to GGUF: {output_path}...")
    
    # Initialize GGUF Writer
    arch = model.config.model_type
    gguf_writer = gguf.GGUFWriter(output_path, arch)

    # Basic Architecture Metadata
    gguf_writer.add_architecture() # Adds 'general.architecture'
    gguf_writer.add_block_count(model.config.num_hidden_layers)
    gguf_writer.add_context_length(getattr(model.config, "max_position_embeddings", 2048))
    gguf_writer.add_embedding_length(model.config.hidden_size)
    gguf_writer.add_feed_forward_length(model.config.intermediate_size)
    gguf_writer.add_head_count(model.config.num_attention_heads)
    
    # Add custom flag so C++ knows to look for rANS tensors
    gguf_writer.add_string("quantization.method", "rans_bfloat16")

    # Pack tensors
    for name, module in model.named_modules():
        
        # Helper to add tensor to GGUF
        # GGUF expects numpy arrays.
        def add(suffix, tensor):
            # GGUF keys usually look like "blk.0.attn_q.weight"
            # For now, we keep your HF names "model.layers.0..." for consistency
            # but standard GGUF tools might expect remapping.
            key = f"{name}.{suffix}"

            # Check tensor type
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.view(torch.int16)
            elif tensor.dtype == torch.uint8:
                tensor = tensor.view(torch.int8)
            elif tensor.dtype == torch.uint16:
                tensor = tensor.view(torch.int16)
            elif tensor.dtype == torch.uint32:
                tensor = tensor.view(torch.int32)

            data = tensor.detach().cpu().numpy()
            print(f"Adding tensor: {key}, shape: {data.shape}, dtype: {data.dtype}")
            gguf_writer.add_tensor(key, data)

        if hasattr(module, "compressed") and module.compressed == "rans_bfloat16":
            
            # Pack Exponent
            if hasattr(module, "exponent_compressed_weight"):
                add("rans_exp_stream", module.exponent_compressed_weight)
                add("rans_exp_states", module.exponent_states)
                add("rans_exp_sizes",  module.exponent_output_sizes)
                add("rans_exp_freqs",  module.exponent_freqs)
                add("rans_exp_cdf",    module.exponent_cdf)
                exp_streams = module.exponent_num_streams
                is_exp_compressed = 1
            else:
                add("rans_exp_raw", module.exponent_raw)
                exp_streams = 0
                is_exp_compressed = 0

            # B. Pack Mantissa
            if hasattr(module, "mantissa_compressed_weight"):
                add("rans_man_stream", module.mantissa_compressed_weight)
                add("rans_man_states", module.mantissa_states)
                add("rans_man_sizes",  module.mantissa_output_sizes)
                add("rans_man_freqs",  module.mantissa_freqs)
                add("rans_man_cdf",    module.mantissa_cdf)
                man_streams = module.mantissa_num_streams
                is_man_compressed = 1
            else:
                add("rans_man_raw", module.mantissa_raw)
                man_streams = 0
                is_man_compressed = 0

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

            # Manually create tensor for info
            info_tensor = torch.tensor(info_data, dtype=torch.int32)
            add("rans_info", info_tensor)

            if hasattr(module, "bias") and module.bias is not None:
                add("bias", module.bias.float())

        else:
            # Handle uncompressed parameters
            for param_name, param in module.named_parameters(recurse=False):
                if param_name == "weight" and hasattr(module, "compressed"):
                    continue
                
                full_name = f"{name}.{param_name}" if name else param_name
                
                add(full_name, param)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"GGUF file saved: {output_path}")
