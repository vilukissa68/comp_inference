import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from comp_inference import (
    fused_rans_linear_triton,
    rans_compress_module_weight,
)


def replicate_qwen_vllm_logic():
    # 1. SETUP EXACT ARCHITECTURE
    # Qwen-0.5B/0.6B typically has Hidden Size K=896 or 1024.
    # Your log shows K=1024 and N=4096 for the fused QKV.
    model_id = "Qwen/Qwen3-0.6B"  # Adjust if you want to test on 0.5B or another model
    device = "cuda"
    prompt = "Who is Linus Torvalds?"

    print(f"--- Loading {model_id} and Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device)

    # 2. CAPTURE REAL HIDDEN STATES (Interception)
    # We run the prompt and catch the data right before the first QKV projection.
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    captured_x = []

    def hook_fn(module, input, output):
        captured_x.append(input[0].detach())

    # The first attention block's QKV projection
    target_layer = model.model.layers[0].self_attn.q_proj
    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        model(inputs.input_ids)
    handle.remove()

    # Physical reality from your log: M=8, K=1024
    x_real = captured_x[0].view(-1, captured_x[0].shape[-1])
    M, K_actual = x_real.shape

    # 3. CONSTRUCT THE FUSED QKV WEIGHT
    # Qwen's QKV is often 3 separate layers; we fuse them to match your N=4096 log.
    q_w = model.model.layers[0].self_attn.q_proj.weight
    k_w = model.model.layers[0].self_attn.k_proj.weight
    v_w = model.model.layers[0].self_attn.v_proj.weight

    # Fuse them: Resulting shape is [N, K] -> [4096, 1024]
    fused_weight = torch.cat([q_w, k_w, v_w], dim=0)
    N_actual = fused_weight.shape[0]

    print(f"Captured Activations: {x_real.shape}")
    print(f"Fused Weight Shape:   {fused_weight.shape}")

    # 4. COMPRESSION (Using your Module API)
    class FusedModule(nn.Module):
        def __init__(self, w):
            super().__init__()
            # Your API expects [K, N] so it can use weight_height=K
            self.weight = nn.Parameter(w.t().contiguous())
            self.bias = None

    module = FusedModule(fused_weight)
    rans_compress_module_weight(module)

    # 5. REFERENCE CALCULATION (Golden)
    # Standard: [M, K] @ [K, N]
    result_golden = torch.matmul(x_real, fused_weight.t())

    # 6. FUSED TRITON EXECUTION (The 'Broken' Path)
    # We use the EXACT same call you have in vLLM.
    result_fused = fused_rans_linear_triton(
        x=x_real,
        compressed_data=module.exponent_compressed_weight.cuda(),
        tile_offsets=module.exponent_tile_offsets.cuda(),
        tile_max_lens=module.exponent_tile_max_lens.cuda(),
        initial_states=module.exponent_states.cuda(),
        mantissas=module.mantissa_raw.cuda(),
        slot_map=module.exponent_slot_map.cuda(),
        tables=module.exponent_tables.cuda(),
        weight_height=K_actual,  # 1024
        weight_width=N_actual,  # 4096
        num_tiles_n_global=N_actual // 32,  # Assuming tile size of 32 for N dimension
        bias=None,
    )

    # 7. COMPARISON
    max_diff = torch.max(torch.abs(result_golden - result_fused)).item()
    print(f"\nVerification for prompt: '{prompt}'")
    print(f"Max Absolute Difference: {max_diff:.6f}")

    if max_diff < 0.1:
        print("✅ Standalone matches real model activations.")
    else:
        print("❌ Divergence found in real model replication!")


if __name__ == "__main__":
    replicate_qwen_vllm_logic()
