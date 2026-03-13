#!/usr/bin/env python3

import torch
import triton
import triton.language as tl


def fused_rans_embedding_triton_2(
    x,
    compressed_data,
    initial_states,
    tables,
    slot_map,
    weight_shape,
    tile_offsets,
    tile_max_lens,
    tile_k,
    tile_n,
    mantissas,
    out=None,
):
    # K is Vocab Size, N is Hidden Dimension
    K, N = weight_shape

    # Flatten token IDs in case input is [batch, seq_len]
    x_flat = x.view(-1)
    M_input = x_flat.numel()

    # Calculate grid constraints
    TILES_N = (N + tile_n - 1) // tile_n
    TILES_K = (K + tile_k - 1) // tile_k

    if out is None:
        output = torch.empty((M_input, N), device=x.device, dtype=torch.bfloat16)
    else:
        output = out.view(M_input, N)

    # 2D Grid: [Tokens, Hidden Dimension Tiles]
    grid = (M_input, TILES_N)

    fused_rans_embedding_kernel[grid](
        x_flat,
        compressed_data,
        tile_offsets,
        tile_max_lens,
        initial_states,
        mantissas,
        output,
        slot_map,
        tables,
        M=M_input,
        N=N,
        K=K,
        num_tiles_n=TILES_N,
        stride_out_m=output.stride(0),
        stride_out_n=output.stride(1),
        TILE_N=tile_n,
        TILE_K=tile_k,
        PROB_BITS=12,
        PROB_MASK=4095,
        RANS_L=1 << 16,
        num_warps=4,
        num_stages=4,  # Sequential loop, keep stages at 1
    )

    # Return exactly formatted to [batch, seq_len, hidden_dim]
    return output.view(*x.shape, N)


@triton.jit
def fused_rans_embedding_kernel(
    x_ptr,
    compressed_data,
    tile_offsets,
    tile_max_lens,
    initial_states,
    mantissas_ptr,
    output_ptr,
    slot_map,
    tables,
    M,  # Total number of tokens in the batch
    N,  # Hidden dimension size
    K,  # Vocab size
    num_tiles_n,
    stride_out_m,
    stride_out_n,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    PROB_BITS: tl.constexpr,
    PROB_MASK: tl.constexpr,
    RANS_L: tl.constexpr,
):
    pid_m = tl.program_id(0)  # Which token we are processing
    pid_n = tl.program_id(1)  # Which hidden-dim tile we are decoding

    # 1. Boundary Check for Batch/Sequence
    if pid_m >= M:
        return

    lane_id = tl.arange(0, TILE_N)
    global_n = pid_n * TILE_N + lane_id
    n_valid = global_n < N

    # 2. Get the Token ID
    tok = tl.load(x_ptr + pid_m).to(tl.int32)

    # If token is out of vocabulary (e.g., padding), write zeros and exit
    if tok >= K or tok < 0:
        tl.store(
            output_ptr + pid_m * stride_out_m + global_n * stride_out_n,
            0.0,
            mask=n_valid,
        )
        return

    # 3. Calculate the Exact Tile and Row Destination
    k_tile_idx = tok // TILE_K
    target_row = tok % TILE_K
    tile_id = k_tile_idx * num_tiles_n + pid_n

    # 4. Load Metadata
    tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)
    tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)

    state = tl.load(initial_states + (tile_id * TILE_N + lane_id)).to(tl.uint32)
    current_byte_row = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)
    tile_data_base = compressed_data + tile_start + lane_id

    curr_row = 0
    while curr_row < target_row:
        slot = state & PROB_MASK
        exp_sym = tl.load(slot_map + slot, mask=n_valid, other=0).to(tl.uint32)

        packed = tl.load(tables + exp_sym, mask=n_valid, other=0)
        state = (packed & 0xFFFF) * (state >> PROB_BITS) + (
            slot - ((packed >> 16) & 0xFFFF)
        )

        for _ in range(2):
            needs_renorm = (state < RANS_L) & n_valid & (current_byte_row >= 0)
            ptr = tile_data_base + (current_byte_row * TILE_N)
            state = tl.where(
                needs_renorm,
                (state << 8) | tl.load(ptr, mask=needs_renorm, other=0).to(tl.uint32),
                state,
            )
            current_byte_row -= tl.where(needs_renorm, 1, 0)

        curr_row += 1

    # 6. We are now exactly at the target row. Extract the true Exponent!
    slot = state & PROB_MASK
    exp_sym = tl.load(slot_map + slot, mask=n_valid, other=0).to(tl.uint16)

    # 7. Fetch the corresponding Mantissa directly from memory
    tile_man_base = tile_id * TILE_K * TILE_N
    bk_man_base = tile_man_base + target_row * TILE_N + lane_id
    raw_man = tl.load(mantissas_ptr + bk_man_base, mask=n_valid, other=0).to(tl.uint16)

    # 8. Reconstruct exact BFloat16
    w_int = ((raw_man & 0x80) << 8) | ((exp_sym & 0xFF) << 7) | (raw_man & 0x7F)
    w_val = w_int.to(tl.uint16).to(tl.bfloat16, bitcast=True)

    # 9. Store directly to the activation output buffer
    tl.store(
        output_ptr + pid_m * stride_out_m + global_n * stride_out_n, w_val, mask=n_valid
    )


@triton.jit
def fused_rans_embedding_kernel_ilp2(
    x_ptr,
    compressed_data,
    tile_offsets,
    tile_max_lens,
    initial_states,
    mantissas_ptr,
    output_ptr,
    slot_map,
    tables,
    M,
    N,
    K,
    num_tiles_n,
    stride_out_m,
    stride_out_n,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    PROB_BITS: tl.constexpr,
    PROB_MASK: tl.constexpr,
    RANS_L: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_m >= M:
        return

    lane_id = tl.arange(0, TILE_N)
    global_n = pid_n * TILE_N + lane_id
    n_valid = global_n < N

    tok = tl.load(x_ptr + pid_m).to(tl.int32)
    if tok >= K or tok < 0:
        tl.store(
            output_ptr + pid_m * stride_out_m + global_n * stride_out_n,
            0.0,
            mask=n_valid,
        )
        return

    # --- ILP2 GEOMETRY ---
    k_tile_idx = tok // TILE_K
    rel_row = tok % TILE_K
    HALF_K = TILE_K // 2
    is_bottom = rel_row >= HALF_K

    # Logic: if in bottom half, we fast-forward relative to the START of the bottom half
    target_inner_row = rel_row % HALF_K
    tile_id = k_tile_idx * num_tiles_n + pid_n

    # Load Metadata
    tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)
    tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)

    # State Selection: Load A if top, B if bottom
    # Stream B starts TILE_N after Stream A
    stream_offset = tl.where(is_bottom, TILE_N, 0)
    state = tl.load(
        initial_states + (tile_id * TILE_N * 2 + stream_offset + lane_id)
    ).to(tl.uint32)

    current_byte_row = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)
    # Bitstream interleaved: A (lane), B (lane) -> ROW_STRIDE = 2 * TILE_N
    tile_data_base = compressed_data + tile_start + stream_offset + lane_id
    ROW_STRIDE = TILE_N * 2

    # Fast-forward to the target row inside the half-tile
    curr_row = 0
    while curr_row < target_inner_row:
        slot = state & PROB_MASK
        exp_sym = tl.load(slot_map + slot, mask=n_valid, other=0).to(tl.uint32)
        packed = tl.load(tables + exp_sym, mask=n_valid, other=0)
        state = (packed & 0xFFFF) * (state >> PROB_BITS) + (
            slot - ((packed >> 16) & 0xFFFF)
        )

        for _ in range(2):
            needs_renorm = (state < RANS_L) & n_valid & (current_byte_row >= 0)
            ptr = tile_data_base + (current_byte_row * ROW_STRIDE)
            state = tl.where(
                needs_renorm,
                (state << 8) | tl.load(ptr, mask=needs_renorm, other=0).to(tl.uint32),
                state,
            )
            current_byte_row -= tl.where(needs_renorm, 1, 0)
        curr_row += 1

    # Extract Exponent and Mantissa
    slot = state & PROB_MASK
    exp_sym = tl.load(slot_map + slot, mask=n_valid, other=0).to(tl.uint16)

    # Mantissa Indexing: Respect the stacked ILP2 blocks
    tile_man_base = tile_id * TILE_K * TILE_N
    half_tile_offset = tl.where(is_bottom, HALF_K * TILE_N, 0)
    bk_man_base = tile_man_base + half_tile_offset + target_inner_row * TILE_N + lane_id
    raw_man = tl.load(mantissas_ptr + bk_man_base, mask=n_valid, other=0).to(tl.uint16)

    # Reconstruct
    w_int = ((raw_man & 0x80) << 8) | ((exp_sym & 0xFF) << 7) | (raw_man & 0x7F)
    tl.store(
        output_ptr + pid_m * stride_out_m + global_n * stride_out_n,
        w_int.to(tl.uint16).to(tl.bfloat16, bitcast=True),
        mask=n_valid,
    )


def fused_rans_embedding_triton(
    x,
    compressed_data,
    initial_states,
    tables,
    slot_map,
    weight_shape,
    tile_offsets,
    tile_max_lens,
    tile_k,
    tile_n,
    mantissas,
    out=None,
):
    K, N = weight_shape  # K = Vocab, N = Hidden
    x_flat = x.view(-1)
    M_input = x_flat.numel()

    TILES_N = (N + tile_n - 1) // tile_n

    if out is None:
        output = torch.empty((M_input, N), device=x.device, dtype=torch.bfloat16)
    else:
        output = out.view(M_input, N)

    grid = (M_input, TILES_N)

    fused_rans_embedding_kernel_ilp2[grid](
        x_flat,
        compressed_data,
        tile_offsets,
        tile_max_lens,
        initial_states,
        mantissas,
        output,
        slot_map,
        tables,
        M=M_input,
        N=N,
        K=K,
        num_tiles_n=TILES_N,
        stride_out_m=output.stride(0),
        stride_out_n=output.stride(1),
        TILE_N=tile_n,
        TILE_K=tile_k,
        PROB_BITS=12,
        PROB_MASK=4095,
        RANS_L=1 << 16,
        num_warps=4,
        num_stages=1,  # Stages=1 is best for sequential rANS loops
    )

    return output.view(*x.shape, N)


@triton.jit
def fused_rans_embedding_kernel_uncoalesced(
    x_ptr,
    compressed_data,
    stream_offsets,  # NEW: Absolute byte offset for each stream
    stream_sizes,  # NEW: Exact byte length of each stream
    initial_states,
    mantissas_ptr,
    output_ptr,
    slot_map,
    tables,
    M,
    N,
    K,
    num_tiles_n,
    stride_out_m,
    stride_out_n,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    PROB_BITS: tl.constexpr,
    PROB_MASK: tl.constexpr,
    RANS_L: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    if pid_m >= M:
        return

    lane_id = tl.arange(0, TILE_N)
    global_n = pid_n * TILE_N + lane_id
    n_valid = global_n < N

    # Look up the token ID
    tok = tl.load(x_ptr + pid_m).to(tl.int32)
    if tok >= K or tok < 0:
        tl.store(
            output_ptr + pid_m * stride_out_m + global_n * stride_out_n,
            0.0,
            mask=n_valid,
        )
        return

    # --- GEOMETRY ---
    k_tile_idx = tok // TILE_K
    rel_row = tok % TILE_K
    tile_id = k_tile_idx * num_tiles_n + pid_n

    # Dense Packing: 1 offset/size per lane
    stream_id = tile_id * TILE_N + lane_id

    stream_offset = tl.load(stream_offsets + stream_id).to(tl.int64)
    stream_size = tl.load(stream_sizes + stream_id).to(tl.int64)
    state = tl.load(initial_states + stream_id).to(tl.uint32)

    # Initialize the backward read pointer at the end of the stream
    current_byte_idx = stream_size - 1
    stream_base_ptr = compressed_data + stream_offset

    # --- THE FAST-FORWARD ---
    # Because compression encoded from TILE_K-1 down to 0, decoding forwards yields row 0, 1, 2...
    # We decode and discard elements until we reach our target relative row.
    for _ in range(rel_row):
        slot = state & PROB_MASK
        exp_sym = tl.load(slot_map + slot, mask=n_valid, other=0).to(tl.uint32)
        packed = tl.load(tables + exp_sym, mask=n_valid, other=0)
        state = (packed & 0xFFFF) * (state >> PROB_BITS) + (
            slot - ((packed >> 16) & 0xFFFF)
        )

        for _ in range(2):
            needs_renorm = (state < RANS_L) & n_valid & (current_byte_idx >= 0)
            ptr = stream_base_ptr + tl.maximum(current_byte_idx, 0)
            state = tl.where(
                needs_renorm,
                (state << 8) | tl.load(ptr, mask=needs_renorm, other=0).to(tl.uint32),
                state,
            )
            current_byte_idx -= tl.where(needs_renorm, 1, 0)

    # --- EXTRACT TARGET WEIGHT ---
    slot = state & PROB_MASK
    exp_sym = tl.load(slot_map + slot, mask=n_valid, other=0).to(tl.uint16)

    # Mantissa Indexing: Clean 3D array access [tile_id, rel_row, lane_id]
    tile_man_base = tile_id * TILE_K * TILE_N
    bk_man_base = tile_man_base + rel_row * TILE_N + lane_id
    raw_man = tl.load(mantissas_ptr + bk_man_base, mask=n_valid, other=0).to(tl.uint16)

    # Reconstruct bfloat16 and Store
    w_int = ((raw_man & 0x80) << 8) | ((exp_sym & 0xFF) << 7) | (raw_man & 0x7F)
    tl.store(
        output_ptr + pid_m * stride_out_m + global_n * stride_out_n,
        w_int.to(tl.uint16).to(tl.bfloat16, bitcast=True),
        mask=n_valid,
    )


def fused_rans_embedding_triton_uncoalesced(
    x,
    compressed_data,
    initial_states,
    tables,
    slot_map,
    weight_shape,
    stream_offsets,  # UPDATED
    stream_sizes,  # UPDATED
    tile_k,  # rANS tile height
    tile_n,  # rANS tile width
    mantissas,
    out=None,
):
    K, N = weight_shape  # K = Vocab, N = Hidden
    x_flat = x.view(-1)
    M_input = x_flat.numel()

    TILES_N = (N + tile_n - 1) // tile_n

    if out is None:
        output = torch.empty((M_input, N), device=x.device, dtype=torch.bfloat16)
    else:
        output = out.view(M_input, N)

    grid = (M_input, TILES_N)

    fused_rans_embedding_kernel_uncoalesced[grid](
        x_flat,
        compressed_data,
        stream_offsets,
        stream_sizes,
        initial_states,
        mantissas,
        output,
        slot_map,
        tables,
        M=M_input,
        N=N,
        K=K,
        num_tiles_n=TILES_N,
        stride_out_m=output.stride(0),
        stride_out_n=output.stride(1),
        TILE_N=tile_n,
        TILE_K=tile_k,
        PROB_BITS=12,
        PROB_MASK=4095,
        RANS_L=1 << 16,
        num_warps=4,
        num_stages=1,  # Stages=1 remains best for sequential rANS loops
    )

    return output.view(*x.shape, N)
