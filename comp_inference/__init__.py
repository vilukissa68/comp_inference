#!/usr/bin/env python3

try:
    from . import ccore
except ImportError as e:
    # This helps debug if the compilation failed or the .so file is missing
    print(f"CRITICAL: Could not import ccore extension. Did you run 'pip install -e .'?: {e}")
    ccore = None


from .comp_inference import (
    get_rans_lut,
    rans_compress_module_weight,
    rans_decompress_module_weight,
)
