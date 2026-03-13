#!/usr/bin/env python3
try:
    import torch
    from . import ccore
except ImportError as e:
    # This helps debug if the compilation failed or the .so file is missing
    print(
        f"CRITICAL: Could not import ccore extension. Did you run 'pip install -e .'?: {e}"
    )
    ccore = None


from .comp_inference import (
    get_rans_lut,
    rans_compress_module_weight,
    rans_compress_qkv_fused,
    rans_compress_gate_up_fused,
    rans_decompress_module_weight,
    extract_exp_and_mantissa,
    reconstruct_from_exp_and_mantissa,
    interleave_mantissas,
    uninterleave_mantissas,
)

from .rans_saving import (
    save_rans_model_package,
    pack_and_save_tensors,
    load_compressed_model,
    load_compressed_model_with_auto_model,
)

from .rans_gguf import save_rans_model_gguf

from .rans_triton import (
    rans_decomp_triton,
    rans_decomp_triton_tiled,
)

from .linear_triton import (
    fused_rans_linear_triton,
    fused_rans_linear_triton_uncoalesced,
)


from .transposed_linear_triton import (
    fused_rans_linear_transposed_triton,
    fused_rans_linear_transposed_triton_uncoalesced,
)

from .embedding_triton import (
    fused_rans_embedding_triton,
    fused_rans_embedding_triton_uncoalesced,
)

from .triton_gemm import (
    triton_matmul,
)

from .pytorch_runtime import load_compressed_model_with_auto_model
