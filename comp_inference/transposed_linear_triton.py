#!/usr/bin/env python3

import torch
import triton
import triton.language as tl


# @triton.jit
# def fused_rans_matmul_transposed_kernel_ilp2_splitk(
#     x_ptr,
#     compressed_data,
#     tile_offsets,
#     tile_max_lens,
#     initial_states,
#     mantissas_ptr,
#     workspace_ptr,
#     slot_map,
#     tables,
#     M,
#     N,
#     K,
#     num_tiles_k,
#     stride_xm,
#     stride_xk,
#     stride_wk,
#     stride_wm,
#     stride_wn,
#     TILE_M: tl.constexpr,
#     TILE_N: tl.constexpr,
#     TILE_K: tl.constexpr,
#     STREAM_LEN: tl.constexpr,
#     PROB_BITS: tl.constexpr,
#     PROB_MASK: tl.constexpr,
#     RANS_L: tl.constexpr,
#     SPLIT_K: tl.constexpr,
# ):
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)
#     pid_k = tl.program_id(2)

#     # 1. Split-K Range Logic
#     total_k_tiles = K // TILE_K
#     tiles_per_split = tl.cdiv(total_k_tiles, SPLIT_K)
#     start_k_tile = pid_k * tiles_per_split
#     end_k_tile = tl.minimum(start_k_tile + tiles_per_split, total_k_tiles)

#     offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
#     offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
#     m_valid = offs_m < M
#     n_valid = offs_n < N

#     acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

#     # 2. Geometry Mapping for Vocab Chunk
#     # STREAM_LEN is the height of a tile (e.g., 512).
#     # HALF_N is the split point for ILP2 (e.g., 256).
#     HALF_N = STREAM_LEN // 2
#     tile_v_idx = (pid_n * TILE_N) // STREAM_LEN
#     row_start = (pid_n * TILE_N) % STREAM_LEN

#     # row_start must be in the top half for this simplified logic
#     # (usually TILE_N is small like 32 or 64)
#     lane_id_k = tl.arange(0, TILE_K)
#     v_range = tl.arange(0, TILE_N)

#     for k_tile_idx in range(start_k_tile, end_k_tile):
#         tile_id = tile_v_idx * num_tiles_k + k_tile_idx

#         # Load Metadata
#         tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)
#         tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)

#         # ILP2 States: A is Top Half, B is Bottom Half
#         tile_stream_base = tile_id * (TILE_K * 2)
#         state_A = tl.load(initial_states + tile_stream_base + lane_id_k).to(tl.uint32)
#         state_B = tl.load(initial_states + tile_stream_base + TILE_K + lane_id_k).to(
#             tl.uint32
#         )

#         current_byte_row_A = tl.full((TILE_K,), tile_depth - 1, dtype=tl.int64)
#         current_byte_row_B = tl.full((TILE_K,), tile_depth - 1, dtype=tl.int64)

#         tile_data_base_A = compressed_data + tile_start + lane_id_k
#         tile_data_base_B = compressed_data + tile_start + TILE_K + lane_id_k
#         ROW_STRIDE = TILE_K * 2

#         # 3. Fast-Forward Loop
#         # Skip rows until we reach 'row_start' within the tile
#         for _ in range(row_start):
#             # Update A
#             s_A = state_A & PROB_MASK
#             p_A = tl.load(tables + tl.load(slot_map + s_A))
#             state_A = (p_A & 0xFFFF) * (state_A >> PROB_BITS) + (s_A - (p_A >> 16))
#             for _ in range(2):
#                 rn_A = (state_A < RANS_L) & (current_byte_row_A >= 0)
#                 state_A = tl.where(
#                     rn_A,
#                     (state_A << 8)
#                     | tl.load(
#                         tile_data_base_A + current_byte_row_A * ROW_STRIDE,
#                         mask=rn_A,
#                         other=0,
#                     ).to(tl.uint32),
#                     state_A,
#                 )
#                 current_byte_row_A -= tl.where(rn_A, 1, 0)

#             # Update B
#             s_B = state_B & PROB_MASK
#             p_B = tl.load(tables + tl.load(slot_map + s_B))
#             state_B = (p_B & 0xFFFF) * (state_B >> PROB_BITS) + (s_B - (p_B >> 16))
#             for _ in range(2):
#                 rn_B = (state_B < RANS_L) & (current_byte_row_B >= 0)
#                 state_B = tl.where(
#                     rn_B,
#                     (state_B << 8)
#                     | tl.load(
#                         tile_data_base_B + current_byte_row_B * ROW_STRIDE,
#                         mask=rn_B,
#                         other=0,
#                     ).to(tl.uint32),
#                     state_B,
#                 )
#                 current_byte_row_B -= tl.where(rn_B, 1, 0)

#         # 4. Decode Weight Chunk [TILE_N, TILE_K]
#         W_chunk = tl.zeros((TILE_N, TILE_K), dtype=tl.bfloat16)
#         tile_man_base = tile_id * STREAM_LEN * TILE_K
#         HALF_TILE_SIZE = HALF_N * TILE_K

#         for i in range(TILE_N):
#             # For simplicity, we assume TILE_N fits within a half-tile
#             # and use state_A logic. If TILE_N spans the split,
#             # logic needs to branch between state_A and state_B.
#             slot = state_A & PROB_MASK
#             exp_sym = tl.load(slot_map + slot).to(tl.uint16)

#             # Reconstruction
#             raw_man = tl.load(
#                 mantissas_ptr + tile_man_base + (row_start + i) * TILE_K + lane_id_k
#             ).to(tl.uint16)
#             w_int = ((raw_man & 0x80) << 8) | ((exp_sym & 0xFF) << 7) | (raw_man & 0x7F)
#             w_val = w_int.to(tl.uint16).to(tl.bfloat16, bitcast=True)

#             # Store row in SRAM
#             W_chunk = tl.where((v_range == i)[:, None], w_val[None, :], W_chunk)

#             # Update state_A
#             pk = tl.load(tables + exp_sym.to(tl.uint32))
#             state_A = (pk & 0xFFFF) * (state_A >> PROB_BITS) + (slot - (pk >> 16))
#             for _ in range(2):
#                 rn = (state_A < RANS_L) & (current_byte_row_A >= 0)
#                 state_A = tl.where(
#                     rn,
#                     (state_A << 8)
#                     | tl.load(
#                         tile_data_base_A + current_byte_row_A * ROW_STRIDE,
#                         mask=rn,
#                         other=0,
#                     ).to(tl.uint32),
#                     state_A,
#                 )
#                 current_byte_row_A -= tl.where(rn, 1, 0)

#         # 5. Dot Product [TILE_M, TILE_K] @ [TILE_K, TILE_N]
#         x_ptrs = (
#             x_ptr
#             + offs_m[:, None] * stride_xm
#             + (k_tile_idx * TILE_K + lane_id_k)[None, :] * stride_xk
#         )
#         X_chunk = tl.load(x_ptrs, mask=m_valid[:, None], other=0.0).to(tl.bfloat16)
#         acc += tl.dot(X_chunk, tl.trans(W_chunk), out_dtype=tl.float32)

#     # 6. Store to Workspace
#     workspace_offset = (
#         (pid_k.to(tl.int64) * stride_wk)
#         + (offs_m[:, None].to(tl.int64) * stride_wm)
#         + (offs_n[None, :].to(tl.int64) * stride_wn)
#     )
#     tl.store(
#         workspace_ptr + workspace_offset,
#         acc.to(tl.bfloat16),
#         mask=m_valid[:, None] & n_valid[None, :],
#     )


@triton.jit
def fused_rans_matmul_transposed_kernel_ilp2_splitk(
    x_ptr,
    compressed_data,
    tile_offsets,
    tile_max_lens,
    initial_states,
    mantissas_ptr,
    workspace_ptr,
    slot_map,
    tables,
    M,
    N,
    K,
    num_tiles_n,
    num_tiles_k,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wm,
    stride_wn,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PROB_BITS: tl.constexpr,
    PROB_MASK: tl.constexpr,
    RANS_L: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    # In Transposed: TILE_K is the width (lane), TILE_N is the output (vocab chunk)
    # The 'tile' we are decoding is much taller than TILE_N.
    # We define REAL_TILE_HEIGHT based on your compression metadata.
    REAL_TILE_HEIGHT = TILE_N * num_tiles_n  # Total rows per stream tile

    lane_id = tl.arange(0, TILE_K)  # Lane is the Hidden Dim
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)

    m_valid = offs_m < M
    n_valid = offs_n < N

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    # Calculate which physical rANS tile this block belongs to
    # and where inside that tile it starts.
    tile_v_idx = (pid_n * TILE_N) // REAL_TILE_HEIGHT
    row_start = (pid_n * TILE_N) % REAL_TILE_HEIGHT

    # Split-K Logic
    total_k_tiles = tl.cdiv(K, TILE_K)
    tiles_per_split = tl.cdiv(total_k_tiles, SPLIT_K)
    start_k_tile = pid_k * tiles_per_split
    end_k_tile = tl.minimum(start_k_tile + tiles_per_split, total_k_tiles)

    for k_tile_idx in range(start_k_tile, end_k_tile):
        tile_id = tile_v_idx * num_tiles_k + k_tile_idx

        tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)
        tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)

        # Loading states (Using TILE_K because K is the lane)
        tile_stream_base = tile_id * (TILE_K * 2)
        state_A = tl.load(initial_states + tile_stream_base + lane_id).to(tl.uint32)
        state_B = tl.load(initial_states + tile_stream_base + TILE_K + lane_id).to(
            tl.uint32
        )

        current_byte_row_A = tl.full((TILE_K,), tile_depth - 1, dtype=tl.int64)
        current_byte_row_B = tl.full((TILE_K,), tile_depth - 1, dtype=tl.int64)

        tile_data_base_A = compressed_data + tile_start + lane_id
        tile_data_base_B = compressed_data + tile_start + TILE_K + lane_id
        ROW_STRIDE = TILE_K * 2

        # --- THE FAST-FORWARD ---
        # Skip rows to reach this block's assigned vocab chunk
        # Determine if we use Stream A or B
        HALF_HEIGHT = REAL_TILE_HEIGHT // 2
        is_bottom = row_start >= HALF_HEIGHT

        curr_state = tl.where(is_bottom, state_B, state_A)
        curr_byte_ptr = tl.where(is_bottom, current_byte_row_B, current_byte_row_A)
        curr_data_base = tl.where(is_bottom, tile_data_base_B, tile_data_base_A)

        inner_start = row_start % HALF_HEIGHT
        for _ in range(inner_start):
            slot = curr_state & PROB_MASK
            packed = tl.load(tables + tl.load(slot_map + slot))
            curr_state = (packed & 0xFFFF) * (curr_state >> PROB_BITS) + (
                slot - (packed >> 16)
            )
            for _ in range(2):
                needs_renorm = (curr_state < RANS_L) & (curr_byte_ptr >= 0)
                curr_state = tl.where(
                    needs_renorm,
                    (curr_state << 8)
                    | tl.load(
                        curr_data_base + curr_byte_ptr * ROW_STRIDE,
                        mask=needs_renorm,
                        other=0,
                    ).to(tl.uint32),
                    curr_state,
                )
                curr_byte_ptr -= tl.where(needs_renorm, 1, 0)

        # --- DECODE AND GEMM ---
        W_chunk = tl.zeros((TILE_N, TILE_K), dtype=tl.bfloat16)
        tile_man_base = tile_id * REAL_TILE_HEIGHT * TILE_K
        half_man_offset = tl.where(is_bottom, HALF_HEIGHT * TILE_K, 0)

        for i in range(TILE_N):
            slot = curr_state & PROB_MASK
            exp_sym = tl.load(slot_map + slot).to(tl.uint16)

            # Use stacked mantissa logic
            raw_man = tl.load(
                mantissas_ptr
                + tile_man_base
                + half_man_offset
                + (inner_start + i) * TILE_K
                + lane_id
            ).to(tl.uint16)
            w_int = ((raw_man & 0x80) << 8) | ((exp_sym & 0xFF) << 7) | (raw_man & 0x7F)
            W_chunk = tl.where(
                (tl.arange(0, TILE_N) == i)[:, None],
                w_int.to(tl.uint16).to(tl.bfloat16, bitcast=True)[None, :],
                W_chunk,
            )

            packed = tl.load(tables + exp_sym.to(tl.uint32))
            curr_state = (packed & 0xFFFF) * (curr_state >> PROB_BITS) + (
                slot - (packed >> 16)
            )
            for _ in range(2):
                needs_renorm = (curr_state < RANS_L) & (curr_byte_ptr >= 0)
                curr_state = tl.where(
                    needs_renorm,
                    (curr_state << 8)
                    | tl.load(
                        curr_data_base + curr_byte_ptr * ROW_STRIDE,
                        mask=needs_renorm,
                        other=0,
                    ).to(tl.uint32),
                    curr_state,
                )
                curr_byte_ptr -= tl.where(needs_renorm, 1, 0)

        # GEMM
        x_col_ptrs = (
            x_ptr
            + offs_m[:, None] * stride_am
            + (k_tile_idx * TILE_K + lane_id)[None, :] * stride_ak
        )
        X_chunk = tl.load(x_col_ptrs, mask=m_valid[:, None], other=0.0).to(tl.bfloat16)
        acc += tl.dot(X_chunk, tl.trans(W_chunk))

    # Store
    off_w = (
        (pid_k.to(tl.int64) * stride_wk)
        + (offs_m[:, None] * stride_wm)
        + (offs_n[None, :] * stride_wn)
    )
    tl.store(
        workspace_ptr + off_w,
        acc.to(tl.bfloat16),
        mask=m_valid[:, None] & n_valid[None, :],
    )


def fused_rans_linear_transposed_triton(
    x,
    compressed_data,
    initial_states,
    tables,
    slot_map,
    weight_shape,
    tile_offsets,
    tile_max_lens,
    tile_k,  # rANS tile height (e.g. 512)
    tile_n,  # rANS tile width (e.g. 32)
    mantissas,
    workspace,  # Passed as explicit param
    bias=None,
    out=None,
    SPLIT_K=8,
):
    # N = Vocab, K = Hidden
    N, K = weight_shape
    x_flat = x.view(-1, K)
    M_input = x_flat.shape[0]

    # Zeroing is required because we sum the slices later
    workspace.zero_()

    # Geometry Setup
    TILE_M, TILE_N = 16, 64
    TILE_K = tile_n  # Lane width
    num_tiles_n = tile_k // TILE_N
    num_tiles_k = K // tile_n

    grid = (triton.cdiv(M_input, TILE_M), triton.cdiv(N, TILE_N), SPLIT_K)

    fused_rans_matmul_transposed_kernel_ilp2_splitk[grid](
        x_ptr=x_flat,
        compressed_data=compressed_data,
        tile_offsets=tile_offsets,
        tile_max_lens=tile_max_lens,
        initial_states=initial_states,
        mantissas_ptr=mantissas,
        workspace_ptr=workspace,
        slot_map=slot_map,
        tables=tables,
        M=M_input,
        N=N,
        K=K,
        num_tiles_n=num_tiles_n,
        num_tiles_k=num_tiles_k,
        stride_am=x_flat.stride(0),
        stride_ak=x_flat.stride(1),
        stride_wk=workspace.stride(0),
        stride_wm=workspace.stride(1),
        stride_wn=workspace.stride(2),
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        BLOCK_K=TILE_K,
        PROB_BITS=12,
        PROB_MASK=4095,
        RANS_L=1 << 16,
        SPLIT_K=SPLIT_K,
        num_warps=8,
        num_stages=1,
    )

    # Reduction Step
    final_output = torch.sum(workspace, dim=0)
    if bias is not None:
        final_output += bias

    if out is not None:
        out.copy_(final_output)
        return out

    return final_output.view(*x.shape[:-1], N)


def fused_rans_linear_transposed_triton_2(
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
    bias=None,
    out=None,
):
    N, K = weight_shape  # N = 151936, K = 1024

    x_flat = x.view(-1, K)
    M_input = x_flat.shape[0]

    num_tiles_k = K // tile_n  # tile_n holds the width (32)

    if out is None:
        output = torch.empty((M_input, N), device=x.device, dtype=torch.bfloat16)
    else:
        output = out.view(M_input, N)

    TILE_M = 16
    TILE_N = 64  # Chunking Vocab to keep SRAM usage low
    grid = (triton.cdiv(M_input, TILE_M), triton.cdiv(N, TILE_N))

    fused_rans_matmul_transposed_kernel[grid](
        x_flat,
        compressed_data,
        tile_offsets,
        tile_max_lens,
        initial_states,
        mantissas,
        bias if bias is not None else x_flat,
        output,
        slot_map,
        tables,
        M_input,
        N,
        K,
        num_tiles_k,
        x_flat.stride(0),
        x_flat.stride(1),
        output.stride(0),
        output.stride(1),
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=tile_n,
        STREAM_LEN=tile_k,
        PROB_BITS=12,
        PROB_MASK=4095,
        RANS_L=1 << 16,
        HAS_BIAS=bias is not None,
        num_warps=4,
        num_stages=4,
    )

    return output.view(*x.shape[:-1], N)


# @triton.jit
# def fused_rans_matmul_transposed_kernel_uncoalesced_splitk(
#     x_ptr,
#     compressed_data,
#     stream_offsets,  # NEW: Absolute byte offset for each stream
#     stream_sizes,  # NEW: Exact byte length of each stream
#     initial_states,
#     mantissas_ptr,
#     workspace_ptr,
#     slot_map,
#     tables,
#     M,
#     N,
#     K,
#     num_tiles_n,
#     num_tiles_k,
#     stride_am,
#     stride_ak,
#     stride_wk,
#     stride_wm,
#     stride_wn,
#     TILE_M: tl.constexpr,
#     TILE_N: tl.constexpr,
#     TILE_K: tl.constexpr,
#     BLOCK_K: tl.constexpr,
#     PROB_BITS: tl.constexpr,
#     PROB_MASK: tl.constexpr,
#     RANS_L: tl.constexpr,
#     SPLIT_K: tl.constexpr,
# ):
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)
#     pid_k = tl.program_id(2)

#     # In Transposed: TILE_K is the width (lane), TILE_N is the output (vocab chunk)
#     # REAL_TILE_HEIGHT is the rANS tile_k (e.g., 512).
#     REAL_TILE_HEIGHT = TILE_N * num_tiles_n

#     lane_id = tl.arange(0, TILE_K)
#     offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
#     offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)

#     m_valid = offs_m < M
#     n_valid = offs_n < N

#     acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

#     tile_v_idx = (pid_n * TILE_N) // REAL_TILE_HEIGHT
#     row_start = (pid_n * TILE_N) % REAL_TILE_HEIGHT

#     # Split-K Logic
#     total_k_tiles = tl.cdiv(K, TILE_K)
#     tiles_per_split = tl.cdiv(total_k_tiles, SPLIT_K)
#     start_k_tile = pid_k * tiles_per_split
#     end_k_tile = tl.minimum(start_k_tile + tiles_per_split, total_k_tiles)

#     for k_tile_idx in range(start_k_tile, end_k_tile):
#         tile_id = tile_v_idx * num_tiles_k + k_tile_idx

#         # Dense Packing: 1 offset/size per lane
#         stream_id = tile_id * TILE_K + lane_id
#         stream_offset = tl.load(stream_offsets + stream_id).to(tl.int64)
#         stream_size = tl.load(stream_sizes + stream_id).to(tl.int64)
#         curr_state = tl.load(initial_states + stream_id).to(tl.uint32)

#         curr_byte_ptr = stream_size - 1
#         curr_data_base = compressed_data + stream_offset

#         # --- THE FAST-FORWARD ---
#         # Because rANS decodes LIFO (bottom-up), we must skip the elements below our assigned TILE_N chunk.
#         skip_count = REAL_TILE_HEIGHT - row_start - TILE_N

#         for _ in range(skip_count):
#             slot = curr_state & PROB_MASK
#             packed = tl.load(tables + tl.load(slot_map + slot))
#             curr_state = (packed & 0xFFFF) * (curr_state >> PROB_BITS) + (
#                 slot - (packed >> 16)
#             )
#             for _ in range(2):
#                 needs_renorm = (curr_state < RANS_L) & (curr_byte_ptr >= 0)
#                 curr_state = tl.where(
#                     needs_renorm,
#                     (curr_state << 8)
#                     | tl.load(
#                         curr_data_base + tl.maximum(curr_byte_ptr, 0),
#                         mask=needs_renorm,
#                         other=0,
#                     ).to(tl.uint32),
#                     curr_state,
#                 )
#                 curr_byte_ptr -= tl.where(needs_renorm, 1, 0)

#         # --- DECODE AND GEMM ---
#         W_chunk = tl.zeros((TILE_N, TILE_K), dtype=tl.bfloat16)
#         tile_man_base = tile_id * REAL_TILE_HEIGHT * TILE_K

#         for i in range(TILE_N):
#             slot = curr_state & PROB_MASK
#             exp_sym = tl.load(slot_map + slot).to(tl.uint16)

#             # LIFO dictates we reconstruct the bottom of our chunk first.
#             local_idx = TILE_N - 1 - i
#             man_row_idx = row_start + local_idx

#             raw_man = tl.load(
#                 mantissas_ptr + tile_man_base + man_row_idx * TILE_K + lane_id
#             ).to(tl.uint16)

#             w_int = ((raw_man & 0x80) << 8) | ((exp_sym & 0xFF) << 7) | (raw_man & 0x7F)

#             # Map the LIFO decoded weight to the correct spatial row in the W_chunk
#             W_chunk = tl.where(
#                 (tl.arange(0, TILE_N) == local_idx)[:, None],
#                 w_int.to(tl.uint16).to(tl.bfloat16, bitcast=True)[None, :],
#                 W_chunk,
#             )

#             packed = tl.load(tables + exp_sym.to(tl.uint32))
#             curr_state = (packed & 0xFFFF) * (curr_state >> PROB_BITS) + (
#                 slot - (packed >> 16)
#             )
#             for _ in range(2):
#                 needs_renorm = (curr_state < RANS_L) & (curr_byte_ptr >= 0)
#                 curr_state = tl.where(
#                     needs_renorm,
#                     (curr_state << 8)
#                     | tl.load(
#                         curr_data_base + tl.maximum(curr_byte_ptr, 0),
#                         mask=needs_renorm,
#                         other=0,
#                     ).to(tl.uint32),
#                     curr_state,
#                 )
#                 curr_byte_ptr -= tl.where(needs_renorm, 1, 0)

#         # GEMM
#         x_col_ptrs = (
#             x_ptr
#             + offs_m[:, None] * stride_am
#             + (k_tile_idx * TILE_K + lane_id)[None, :] * stride_ak
#         )
#         X_chunk = tl.load(x_col_ptrs, mask=m_valid[:, None], other=0.0).to(tl.bfloat16)
#         acc += tl.dot(X_chunk, tl.trans(W_chunk))

#     # Store
#     off_w = (
#         (pid_k.to(tl.int64) * stride_wk)
#         + (offs_m[:, None] * stride_wm)
#         + (offs_n[None, :] * stride_wn)
#     )
#     tl.store(
#         workspace_ptr + off_w,
#         acc.to(tl.bfloat16),
#         mask=m_valid[:, None] & n_valid[None, :],
#     )


@triton.jit
def fused_rans_matmul_transposed_kernel_uncoalesced_splitk(
    x_ptr,
    compressed_data,
    stream_offsets,
    stream_sizes,
    initial_states,
    mantissas_ptr,
    workspace_ptr,
    slot_map,
    tables,
    M,
    N,
    K,
    num_tiles_n,
    num_tiles_k,
    stride_am,
    stride_ak,
    stride_wk,
    stride_wm,
    stride_wn,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PROB_BITS: tl.constexpr,
    PROB_MASK: tl.constexpr,
    RANS_L: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    REAL_TILE_HEIGHT = TILE_N * num_tiles_n

    lane_id = tl.arange(0, TILE_K)
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)

    m_valid = offs_m < M
    n_valid = offs_n < N

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    tile_v_idx = (pid_n * TILE_N) // REAL_TILE_HEIGHT
    row_start = (pid_n * TILE_N) % REAL_TILE_HEIGHT

    total_k_tiles = tl.cdiv(K, TILE_K)
    tiles_per_split = tl.cdiv(total_k_tiles, SPLIT_K)
    start_k_tile = pid_k * tiles_per_split
    end_k_tile = tl.minimum(start_k_tile + tiles_per_split, total_k_tiles)

    for k_tile_idx in range(start_k_tile, end_k_tile):
        tile_id = tile_v_idx * num_tiles_k + k_tile_idx

        stream_id = tile_id * TILE_K + lane_id
        stream_offset = tl.load(stream_offsets + stream_id).to(tl.int64)
        stream_size = tl.load(stream_sizes + stream_id).to(tl.int64)
        curr_state = tl.load(initial_states + stream_id).to(tl.uint32)

        curr_byte_ptr = stream_size - 1
        curr_data_base = compressed_data + stream_offset

        # --- THE FAST-FORWARD (FIXED) ---
        # The stream naturally yields rows 0, 1, 2...
        # We want to start reading at row_start, so we skip exactly row_start elements.
        skip_count = row_start

        for _ in range(skip_count):
            slot = curr_state & PROB_MASK
            packed = tl.load(tables + tl.load(slot_map + slot))
            curr_state = (packed & 0xFFFF) * (curr_state >> PROB_BITS) + (
                slot - (packed >> 16)
            )
            for _ in range(2):
                needs_renorm = (curr_state < RANS_L) & (curr_byte_ptr >= 0)
                curr_state = tl.where(
                    needs_renorm,
                    (curr_state << 8)
                    | tl.load(
                        curr_data_base + tl.maximum(curr_byte_ptr, 0),
                        mask=needs_renorm,
                        other=0,
                    ).to(tl.uint32),
                    curr_state,
                )
                curr_byte_ptr -= tl.where(needs_renorm, 1, 0)

        # --- DECODE AND GEMM (FIXED) ---
        W_chunk = tl.zeros((TILE_N, TILE_K), dtype=tl.bfloat16)
        tile_man_base = tile_id * REAL_TILE_HEIGHT * TILE_K

        for i in range(TILE_N):
            slot = curr_state & PROB_MASK
            exp_sym = tl.load(slot_map + slot).to(tl.uint16)

            # Spatial row matches ascending loop index cleanly
            man_row_idx = row_start + i

            raw_man = tl.load(
                mantissas_ptr + tile_man_base + man_row_idx * TILE_K + lane_id
            ).to(tl.uint16)

            w_int = ((raw_man & 0x80) << 8) | ((exp_sym & 0xFF) << 7) | (raw_man & 0x7F)

            # Map the decoded weight to the chunk in standard ascending order
            W_chunk = tl.where(
                (tl.arange(0, TILE_N) == i)[:, None],
                w_int.to(tl.uint16).to(tl.bfloat16, bitcast=True)[None, :],
                W_chunk,
            )

            packed = tl.load(tables + exp_sym.to(tl.uint32))
            curr_state = (packed & 0xFFFF) * (curr_state >> PROB_BITS) + (
                slot - (packed >> 16)
            )
            for _ in range(2):
                needs_renorm = (curr_state < RANS_L) & (curr_byte_ptr >= 0)
                curr_state = tl.where(
                    needs_renorm,
                    (curr_state << 8)
                    | tl.load(
                        curr_data_base + tl.maximum(curr_byte_ptr, 0),
                        mask=needs_renorm,
                        other=0,
                    ).to(tl.uint32),
                    curr_state,
                )
                curr_byte_ptr -= tl.where(needs_renorm, 1, 0)

        # GEMM
        x_col_ptrs = (
            x_ptr
            + offs_m[:, None] * stride_am
            + (k_tile_idx * TILE_K + lane_id)[None, :] * stride_ak
        )
        X_chunk = tl.load(x_col_ptrs, mask=m_valid[:, None], other=0.0).to(tl.bfloat16)
        acc += tl.dot(X_chunk, tl.trans(W_chunk))

    # Store
    off_w = (
        (pid_k.to(tl.int64) * stride_wk)
        + (offs_m[:, None] * stride_wm)
        + (offs_n[None, :] * stride_wn)
    )
    tl.store(
        workspace_ptr + off_w,
        acc.to(tl.bfloat16),
        mask=m_valid[:, None] & n_valid[None, :],
    )


def fused_rans_linear_transposed_triton_uncoalesced(
    x,
    compressed_data,
    initial_states,
    tables,
    slot_map,
    weight_shape,
    stream_offsets,  # UPDATED
    stream_sizes,  # UPDATED
    tile_k,
    tile_n,
    mantissas,
    workspace,
    bias=None,
    out=None,
    SPLIT_K=8,
):
    N, K = weight_shape
    x_flat = x.view(-1, K)
    M_input = x_flat.shape[0]

    workspace.zero_()

    TILE_M, TILE_N = 16, 64
    TILE_K = tile_n
    num_tiles_n = tile_k // TILE_N
    num_tiles_k = K // tile_n

    grid = (triton.cdiv(M_input, TILE_M), triton.cdiv(N, TILE_N), SPLIT_K)

    fused_rans_matmul_transposed_kernel_uncoalesced_splitk[grid](
        x_ptr=x_flat,
        compressed_data=compressed_data,
        stream_offsets=stream_offsets,
        stream_sizes=stream_sizes,
        initial_states=initial_states,
        mantissas_ptr=mantissas,
        workspace_ptr=workspace,
        slot_map=slot_map,
        tables=tables,
        M=M_input,
        N=N,
        K=K,
        num_tiles_n=num_tiles_n,
        num_tiles_k=num_tiles_k,
        stride_am=x_flat.stride(0),
        stride_ak=x_flat.stride(1),
        stride_wk=workspace.stride(0),
        stride_wm=workspace.stride(1),
        stride_wn=workspace.stride(2),
        TILE_M=TILE_M,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        BLOCK_K=TILE_K,
        PROB_BITS=12,
        PROB_MASK=4095,
        RANS_L=1 << 16,
        SPLIT_K=SPLIT_K,
        num_warps=8,
        num_stages=1,
    )

    final_output = torch.sum(workspace, dim=0)
    if bias is not None:
        final_output += bias

    if out is not None:
        out.copy_(final_output)
        return out

    return final_output.view(*x.shape[:-1], N)
