#!/usr/bin/env python3

import torch
import triton
import triton.language as tl


@triton.jit
def fused_rans_matmul_kernel_with_bias_ilp2(
    x_ptr,
    compressed_data,
    tile_offsets,
    tile_max_lens,
    initial_states,
    mantissas_ptr,
    bias_ptr,
    output_ptr,
    slot_map,
    tables,
    M,
    N,
    K,
    num_tiles_n,
    num_tiles_k,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PROB_BITS: tl.constexpr,
    PROB_MASK: tl.constexpr,
    RANS_L: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    lane_id = tl.arange(0, TILE_N)

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    global_n = pid_n * TILE_N + lane_id

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_valid = global_n < N
    m_valid = offs_m < M

    x_row_ptrs = x_ptr + offs_m[:, None] * stride_am

    # We split the tile height in half for the two ILP streams
    HALF_TILE_K = TILE_K // 2

    for k_tile_idx in range(0, tl.cdiv(K, TILE_K)):
        tile_id = k_tile_idx * num_tiles_n + pid_n

        tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)
        tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)

        # --- ILP STATE LOADING ---
        # The tile has TILE_N * 2 streams. A is the first half, B is the second half.
        tile_stream_base = tile_id * (TILE_N * 2)
        state_A = tl.load(initial_states + tile_stream_base + lane_id).to(tl.uint32)
        state_B = tl.load(initial_states + tile_stream_base + TILE_N + lane_id).to(
            tl.uint32
        )

        current_byte_row_A = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)
        current_byte_row_B = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)

        local_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

        tile_man_base = tile_id * TILE_K * TILE_N

        # The compressed data layout is wide: (TILE_N * 2) per row.
        tile_data_base_A = compressed_data + tile_start + lane_id
        tile_data_base_B = compressed_data + tile_start + TILE_N + lane_id
        ROW_STRIDE = TILE_N * 2

        # --- ILP LOOP: Runs for HALF the tile height ---
        for bk_start in range(0, HALF_TILE_K, BLOCK_K):
            # K-dimension bases for Top Half (A) and Bottom Half (B)
            k_base_A = k_tile_idx * TILE_K + bk_start
            k_base_B = k_tile_idx * TILE_K + HALF_TILE_K + bk_start

            # Advance pointers for both halves
            x_col_ptrs_A = x_row_ptrs + k_base_A * stride_ak
            x_col_ptrs_B = x_row_ptrs + k_base_B * stride_ak

            man_ptrs_A = mantissas_ptr + tile_man_base + bk_start * TILE_N + lane_id
            man_ptrs_B = (
                mantissas_ptr
                + tile_man_base
                + (HALF_TILE_K + bk_start) * TILE_N
                + lane_id
            )

            for i in range(BLOCK_K):
                k_in_bounds_A = (k_base_A + i) < K
                k_in_bounds_B = (k_base_B + i) < K
                mask_n_A = k_in_bounds_A & n_valid
                mask_n_B = k_in_bounds_B & n_valid
                mask_m_A = k_in_bounds_A & m_valid[:, None]
                mask_m_B = k_in_bounds_B & m_valid[:, None]

                # 1. INDEPENDENT MEMORY FETCHES
                x_col_A = tl.load(x_col_ptrs_A, mask=mask_m_A, other=0.0).to(
                    tl.bfloat16
                )
                x_col_B = tl.load(x_col_ptrs_B, mask=mask_m_B, other=0.0).to(
                    tl.bfloat16
                )
                x_col_ptrs_A += stride_ak
                x_col_ptrs_B += stride_ak

                raw_man_A = tl.load(man_ptrs_A, mask=mask_n_A, other=0).to(tl.uint16)
                raw_man_B = tl.load(man_ptrs_B, mask=mask_n_B, other=0).to(tl.uint16)
                man_ptrs_A += TILE_N
                man_ptrs_B += TILE_N

                # 2. STATE LOOKUPS
                slot_A = state_A & PROB_MASK
                slot_B = state_B & PROB_MASK
                exp_sym_A = tl.load(slot_map + slot_A, mask=mask_n_A, other=0).to(
                    tl.uint16
                )
                exp_sym_B = tl.load(slot_map + slot_B, mask=mask_n_B, other=0).to(
                    tl.uint16
                )

                # 3. WEIGHT RECONSTRUCTION & ACCUMULATION
                w_int_A = (
                    ((raw_man_A & 0x80) << 8) | (exp_sym_A << 7) | (raw_man_A & 0x7F)
                )
                w_int_B = (
                    ((raw_man_B & 0x80) << 8) | (exp_sym_B << 7) | (raw_man_B & 0x7F)
                )
                w_val_A = w_int_A.to(tl.uint16).to(tl.bfloat16, bitcast=True)
                w_val_B = w_int_B.to(tl.uint16).to(tl.bfloat16, bitcast=True)

                local_acc += (x_col_A * w_val_A[None, :]).to(tl.float32)
                local_acc += (x_col_B * w_val_B[None, :]).to(tl.float32)

                # 4. STATE UPDATES
                packed_A = tl.load(tables + exp_sym_A, mask=mask_n_A, other=0)
                packed_B = tl.load(tables + exp_sym_B, mask=mask_n_B, other=0)
                state_A = (packed_A & 0xFFFF) * (state_A >> PROB_BITS) + (
                    slot_A - (packed_A >> 16)
                )
                state_B = (packed_B & 0xFFFF) * (state_B >> PROB_BITS) + (
                    slot_B - (packed_B >> 16)
                )

                # 5. RENORMALIZATION STREAM A
                needs_renorm_1_A = (
                    (state_A < RANS_L) & mask_n_A & (current_byte_row_A >= 0)
                )
                ptr_1_A = tile_data_base_A + (
                    tl.maximum(current_byte_row_A, 0) * ROW_STRIDE
                )
                state_A = tl.where(
                    needs_renorm_1_A,
                    (state_A << 8)
                    | tl.load(ptr_1_A, mask=needs_renorm_1_A, other=0).to(tl.uint32),
                    state_A,
                )
                current_byte_row_A -= tl.where(needs_renorm_1_A, 1, 0)

                needs_renorm_2_A = (
                    (state_A < RANS_L) & mask_n_A & (current_byte_row_A >= 0)
                )
                ptr_2_A = tile_data_base_A + (
                    tl.maximum(current_byte_row_A, 0) * ROW_STRIDE
                )
                state_A = tl.where(
                    needs_renorm_2_A,
                    (state_A << 8)
                    | tl.load(ptr_2_A, mask=needs_renorm_2_A, other=0).to(tl.uint32),
                    state_A,
                )
                current_byte_row_A -= tl.where(needs_renorm_2_A, 1, 0)

                # 6. RENORMALIZATION STREAM B
                needs_renorm_1_B = (
                    (state_B < RANS_L) & mask_n_B & (current_byte_row_B >= 0)
                )
                ptr_1_B = tile_data_base_B + (
                    tl.maximum(current_byte_row_B, 0) * ROW_STRIDE
                )
                state_B = tl.where(
                    needs_renorm_1_B,
                    (state_B << 8)
                    | tl.load(ptr_1_B, mask=needs_renorm_1_B, other=0).to(tl.uint32),
                    state_B,
                )
                current_byte_row_B -= tl.where(needs_renorm_1_B, 1, 0)

                needs_renorm_2_B = (
                    (state_B < RANS_L) & mask_n_B & (current_byte_row_B >= 0)
                )
                ptr_2_B = tile_data_base_B + (
                    tl.maximum(current_byte_row_B, 0) * ROW_STRIDE
                )
                state_B = tl.where(
                    needs_renorm_2_B,
                    (state_B << 8)
                    | tl.load(ptr_2_B, mask=needs_renorm_2_B, other=0).to(tl.uint32),
                    state_B,
                )
                current_byte_row_B -= tl.where(needs_renorm_2_B, 1, 0)

        acc += local_acc

    acc_bf16 = acc.to(tl.bfloat16)
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + global_n, mask=n_valid, other=0.0).to(
            tl.bfloat16
        )
        acc_bf16 = tl.where(m_valid[:, None], acc_bf16 + bias_vals[None, :], acc_bf16)

    tl.store(
        output_ptr + (offs_m[:, None] * stride_cm + global_n[None, :] * stride_cn),
        acc_bf16,
        mask=m_valid[:, None] & n_valid[None, :],
    )


# Batch optimized ilp2+split-k
def fused_rans_linear_triton(
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
    accum_block_size=32,
    bias=None,
    out=None,
    SPLIT_K=8,
    workspace=None,
):
    K, N = weight_shape

    # Safely flatten X for 2D matrix math without forcing a memory copy
    x_flat = x.view(-1, K)
    M_input = x_flat.shape[0]

    TILES_N = (N + tile_n - 1) // tile_n
    TILES_K = (K + tile_k - 1) // tile_k

    # --- FIX 1 & 3: Correct 3D Shape and torch.empty ---
    # We require a buffer of (SPLIT_K, M_input, N) to avoid race conditions.
    if workspace is None:
        workspace = torch.empty(
            (SPLIT_K, M_input, N), dtype=torch.bfloat16, device=x.device
        )
    else:
        # If using your RansWorkspace, ensure we view it as the correct 3D shape
        workspace = workspace.view(SPLIT_K, M_input, N)

    TILE_M = 16

    # 2. LAUNCH 3D GRID
    grid = (triton.cdiv(M_input, TILE_M), triton.cdiv(N, tile_n), SPLIT_K)

    fused_rans_matmul_kernel_ilp2_splitk[grid](
        x_flat,
        compressed_data,
        tile_offsets,
        tile_max_lens,
        initial_states,
        mantissas,
        workspace,
        slot_map,
        tables,
        M_input,
        N,
        K,
        TILES_N,
        TILES_K,
        # --- FIX 2: Pass actual strides to Triton so we don't need .contiguous() ---
        x_flat.stride(0),
        x_flat.stride(1),
        workspace.stride(0),  # SPLIT_K stride
        workspace.stride(1),  # M stride
        workspace.stride(2),  # N stride
        TILE_M=TILE_M,
        TILE_N=tile_n,
        TILE_K=tile_k,
        BLOCK_K=accum_block_size,
        PROB_BITS=12,
        PROB_MASK=4095,
        RANS_L=1 << 16,
        SPLIT_K=SPLIT_K,
    )

    # --- FIX 4: Zero-Copy In-Place Reduction ---
    if out is not None:
        # Reduce directly into vLLM's pre-allocated output buffer
        torch.sum(workspace, dim=0, out=out)
        if bias is not None:
            out += bias
        return out
    else:
        # If no out buffer provided, create one natively formatted to x's original shape
        final_output = torch.sum(workspace, dim=0)
        if bias is not None:
            final_output += bias
        return final_output.view(*x.shape[:-1], N)


@triton.jit
def fused_rans_matmul_kernel_ilp2_splitk(
    x_ptr,
    compressed_data,
    tile_offsets,
    tile_max_lens,
    initial_states,
    mantissas_ptr,
    workspace_ptr,  # NEW: We write partial sums here instead of output_ptr
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
    stride_wm,  # NEW: Workspace strides
    stride_wn,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PROB_BITS: tl.constexpr,
    PROB_MASK: tl.constexpr,
    RANS_L: tl.constexpr,
    SPLIT_K: tl.constexpr,  # NEW: How many blocks share a column
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)  # NEW: The Split-K ID

    lane_id = tl.arange(0, TILE_N)
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    global_n = pid_n * TILE_N + lane_id
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)

    n_valid = global_n < N
    m_valid = offs_m < M
    x_row_ptrs = x_ptr + offs_m[:, None] * stride_am

    HALF_TILE_K = TILE_K // 2

    # --- SPLIT-K LOGIC: Calculate which K-tiles this specific block owns ---
    total_k_tiles = tl.cdiv(K, TILE_K)
    tiles_per_split = tl.cdiv(total_k_tiles, SPLIT_K)

    start_k_tile = pid_k * tiles_per_split
    end_k_tile = tl.minimum(start_k_tile + tiles_per_split, total_k_tiles)

    # ONLY loop through this block's assigned tiles!
    for k_tile_idx in range(start_k_tile, end_k_tile):
        tile_id = k_tile_idx * num_tiles_n + pid_n

        tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)
        tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)

        tile_stream_base = tile_id * (TILE_N * 2)
        state_A = tl.load(initial_states + tile_stream_base + lane_id).to(tl.uint32)
        state_B = tl.load(initial_states + tile_stream_base + TILE_N + lane_id).to(
            tl.uint32
        )

        current_byte_row_A = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)
        current_byte_row_B = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)

        local_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

        tile_man_base = tile_id * TILE_K * TILE_N
        HALF_TILE_SIZE = HALF_TILE_K * TILE_N

        # The compressed data layout is wide: (TILE_N * 2) per row.
        tile_data_base_A = compressed_data + tile_start + lane_id
        tile_data_base_B = compressed_data + tile_start + TILE_N + lane_id
        ROW_STRIDE = TILE_N * 2

        # --- ILP LOOP: Runs for HALF the tile height ---
        for bk_start in range(0, HALF_TILE_K, BLOCK_K):
            # K-dimension bases for Top Half (A) and Bottom Half (B)
            k_base_A = k_tile_idx * TILE_K + bk_start
            k_base_B = k_tile_idx * TILE_K + HALF_TILE_K + bk_start

            # Advance pointers for both halves
            x_col_ptrs_A = x_row_ptrs + k_base_A * stride_ak
            x_col_ptrs_B = x_row_ptrs + k_base_B * stride_ak

            man_ptrs_A = mantissas_ptr + tile_man_base + bk_start * TILE_N + lane_id
            man_ptrs_B = (
                mantissas_ptr
                + tile_man_base
                + (HALF_TILE_K + bk_start) * TILE_N
                + lane_id
            )
            # man_ptrs_A = mantissas_ptr + tile_man_base + (bk_start * TILE_N) + lane_id
            # man_ptrs_B = (
            #     mantissas_ptr
            #     + tile_man_base
            #     + HALF_TILE_SIZE
            #     + (bk_start * TILE_N)
            #     + lane_id
            # )

            for i in range(BLOCK_K):
                k_in_bounds_A = (k_base_A + i) < K
                k_in_bounds_B = (k_base_B + i) < K
                mask_n_A = k_in_bounds_A & n_valid
                mask_n_B = k_in_bounds_B & n_valid
                mask_m_A = k_in_bounds_A & m_valid[:, None]
                mask_m_B = k_in_bounds_B & m_valid[:, None]

                # 1. INDEPENDENT MEMORY FETCHES
                x_col_A = tl.load(x_col_ptrs_A, mask=mask_m_A, other=0.0).to(
                    tl.bfloat16
                )
                x_col_B = tl.load(x_col_ptrs_B, mask=mask_m_B, other=0.0).to(
                    tl.bfloat16
                )
                x_col_ptrs_A += stride_ak
                x_col_ptrs_B += stride_ak

                raw_man_A = tl.load(man_ptrs_A, mask=mask_n_A, other=0).to(tl.uint16)
                raw_man_B = tl.load(man_ptrs_B, mask=mask_n_B, other=0).to(tl.uint16)
                man_ptrs_A += TILE_N
                man_ptrs_B += TILE_N

                # 2. STATE LOOKUPS
                slot_A = state_A & PROB_MASK
                slot_B = state_B & PROB_MASK
                exp_sym_A = tl.load(slot_map + slot_A, mask=mask_n_A, other=0).to(
                    tl.uint16
                )
                exp_sym_B = tl.load(slot_map + slot_B, mask=mask_n_B, other=0).to(
                    tl.uint16
                )

                # 3. WEIGHT RECONSTRUCTION & ACCUMULATION
                w_int_A = (
                    ((raw_man_A & 0x80) << 8) | (exp_sym_A << 7) | (raw_man_A & 0x7F)
                )
                w_int_B = (
                    ((raw_man_B & 0x80) << 8) | (exp_sym_B << 7) | (raw_man_B & 0x7F)
                )
                w_val_A = w_int_A.to(tl.uint16).to(tl.bfloat16, bitcast=True)
                w_val_B = w_int_B.to(tl.uint16).to(tl.bfloat16, bitcast=True)

                local_acc += (x_col_A * w_val_A[None, :]).to(tl.float32)
                local_acc += (x_col_B * w_val_B[None, :]).to(tl.float32)

                # 4. STATE UPDATES
                packed_A = tl.load(tables + exp_sym_A, mask=mask_n_A, other=0)
                packed_B = tl.load(tables + exp_sym_B, mask=mask_n_B, other=0)
                state_A = (packed_A & 0xFFFF) * (state_A >> PROB_BITS) + (
                    slot_A - (packed_A >> 16)
                )
                state_B = (packed_B & 0xFFFF) * (state_B >> PROB_BITS) + (
                    slot_B - (packed_B >> 16)
                )

                # 5. RENORMALIZATION STREAM A
                needs_renorm_1_A = (
                    (state_A < RANS_L) & mask_n_A & (current_byte_row_A >= 0)
                )
                ptr_1_A = tile_data_base_A + (
                    tl.maximum(current_byte_row_A, 0) * ROW_STRIDE
                )
                state_A = tl.where(
                    needs_renorm_1_A,
                    (state_A << 8)
                    | tl.load(ptr_1_A, mask=needs_renorm_1_A, other=0).to(tl.uint32),
                    state_A,
                )
                current_byte_row_A -= tl.where(needs_renorm_1_A, 1, 0)

                needs_renorm_2_A = (
                    (state_A < RANS_L) & mask_n_A & (current_byte_row_A >= 0)
                )
                ptr_2_A = tile_data_base_A + (
                    tl.maximum(current_byte_row_A, 0) * ROW_STRIDE
                )
                state_A = tl.where(
                    needs_renorm_2_A,
                    (state_A << 8)
                    | tl.load(ptr_2_A, mask=needs_renorm_2_A, other=0).to(tl.uint32),
                    state_A,
                )
                current_byte_row_A -= tl.where(needs_renorm_2_A, 1, 0)

                # 6. RENORMALIZATION STREAM B
                needs_renorm_1_B = (
                    (state_B < RANS_L) & mask_n_B & (current_byte_row_B >= 0)
                )
                ptr_1_B = tile_data_base_B + (
                    tl.maximum(current_byte_row_B, 0) * ROW_STRIDE
                )
                state_B = tl.where(
                    needs_renorm_1_B,
                    (state_B << 8)
                    | tl.load(ptr_1_B, mask=needs_renorm_1_B, other=0).to(tl.uint32),
                    state_B,
                )
                current_byte_row_B -= tl.where(needs_renorm_1_B, 1, 0)

                needs_renorm_2_B = (
                    (state_B < RANS_L) & mask_n_B & (current_byte_row_B >= 0)
                )
                ptr_2_B = tile_data_base_B + (
                    tl.maximum(current_byte_row_B, 0) * ROW_STRIDE
                )
                state_B = tl.where(
                    needs_renorm_2_B,
                    (state_B << 8)
                    | tl.load(ptr_2_B, mask=needs_renorm_2_B, other=0).to(tl.uint32),
                    state_B,
                )
                current_byte_row_B -= tl.where(needs_renorm_2_B, 1, 0)

        acc += local_acc
    # --- SPLIT-K WRITEBACK ---
    # We DO NOT add bias here! If we did, we would add the bias SPLIT_K times.
    acc_bf16 = acc.to(tl.bfloat16)

    # Calculate offset using the absolute 3D strides passed from Python
    workspace_offset = (
        (pid_k.to(tl.int64) * stride_wk)
        + (offs_m[:, None].to(tl.int64) * stride_wm)
        + (global_n[None, :].to(tl.int64) * stride_wn)
    )

    tl.store(
        workspace_ptr + workspace_offset,
        acc_bf16,
        mask=m_valid[:, None] & n_valid[None, :],
    )


# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_K": 32}, num_warps=4, num_stages=3),
#         triton.Config({"BLOCK_K": 64}, num_warps=4, num_stages=3),
#         triton.Config({"BLOCK_K": 128}, num_warps=4, num_stages=4),
#         triton.Config({"BLOCK_K": 64}, num_warps=8, num_stages=3),
#     ],
#     key=["M", "N"],
# )
@triton.jit
def fused_rans_matmul_kernel_with_bias(
    x_ptr,
    compressed_data,
    tile_offsets,
    tile_max_lens,
    initial_states,
    mantissas_ptr,
    bias_ptr,
    output_ptr,
    slot_map,
    tables,
    M,
    N,
    K,
    num_tiles_n,
    num_tiles_k,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PROB_BITS: tl.constexpr,
    PROB_MASK: tl.constexpr,
    RANS_L: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    lane_id = tl.arange(0, TILE_N)

    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    global_n = pid_n * TILE_N + lane_id

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_valid = global_n < N
    m_valid = offs_m < M

    x_row_ptrs = x_ptr + offs_m * stride_am

    for k_tile_idx in range(0, tl.cdiv(K, TILE_K)):
        tile_id = k_tile_idx * num_tiles_n + pid_n

        tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)
        tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)
        state = tl.load(initial_states + (tile_id * TILE_N + lane_id)).to(tl.uint32)
        current_byte_row = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)

        local_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

        tile_man_base = tile_id * TILE_K * TILE_N
        tile_data_base = compressed_data + tile_start + lane_id

        for bk_start in range(0, TILE_K, BLOCK_K):
            k_base = k_tile_idx * TILE_K + bk_start

            # --- OPTIMIZATION 1: Pointer Advancing ---
            x_col_ptrs = x_row_ptrs + k_base * stride_ak
            man_ptrs = mantissas_ptr + tile_man_base + bk_start * TILE_N + lane_id

            # --- OPTIMIZATION 2: Standard Range Loop ---
            # Standard range allows LLVM to pipeline memory without crashing the compiler
            for i in range(BLOCK_K):
                # ---------------------------------------------------------
                # BUG FIX: Separate masks for the M and N dimensions!
                # ---------------------------------------------------------
                k_in_bounds = (k_base + i) < K
                mask_n = k_in_bounds & n_valid
                mask_m = k_in_bounds & m_valid

                # INDEPENDENT FETCH
                x_col = tl.load(x_col_ptrs, mask=mask_m, other=0.0).to(tl.bfloat16)
                x_col_ptrs += stride_ak

                raw_man = tl.load(man_ptrs, mask=mask_n, other=0).to(tl.uint16)
                man_ptrs += TILE_N

                # DEPENDENT FETCH
                slot = state & PROB_MASK
                exp_sym = tl.load(slot_map + slot, mask=mask_n, other=0).to(tl.uint16)

                # Weight Reconstruction & Direct Accumulation
                w_int = ((raw_man & 0x80) << 8) | (exp_sym << 7) | (raw_man & 0x7F)
                w_val = w_int.to(tl.uint16).to(tl.bfloat16, bitcast=True)
                local_acc += (x_col[:, None] * w_val[None, :]).to(tl.float32)

                # Update rANS State
                packed = tl.load(tables + exp_sym, mask=mask_n, other=0)
                state = (packed & 0xFFFF) * (state >> PROB_BITS) + (
                    slot - (packed >> 16)
                )

                # --- OPTIMIZATION 3: Flat Renorm Loop (Memory Safe) ---
                # Renorm 1
                needs_renorm_1 = (state < RANS_L) & mask_n & (current_byte_row >= 0)

                # PREVENTS ILLEGAL MEMORY CRASHES: Force the row index to 0 even if masked out
                safe_row_1 = tl.maximum(current_byte_row, 0)
                ptr_1 = tile_data_base + (safe_row_1 * TILE_N)

                state = tl.where(
                    needs_renorm_1,
                    (state << 8)
                    | tl.load(ptr_1, mask=needs_renorm_1, other=0).to(tl.uint32),
                    state,
                )
                current_byte_row -= tl.where(needs_renorm_1, 1, 0)

                # Renorm 2
                needs_renorm_2 = (state < RANS_L) & mask_n & (current_byte_row >= 0)

                # PREVENTS ILLEGAL MEMORY CRASHES
                safe_row_2 = tl.maximum(current_byte_row, 0)
                ptr_2 = tile_data_base + (safe_row_2 * TILE_N)

                state = tl.where(
                    needs_renorm_2,
                    (state << 8)
                    | tl.load(ptr_2, mask=needs_renorm_2, other=0).to(tl.uint32),
                    state,
                )
                current_byte_row -= tl.where(needs_renorm_2, 1, 0)
        acc += local_acc

    acc_bf16 = acc.to(tl.bfloat16)
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + global_n, mask=n_valid, other=0.0).to(
            tl.bfloat16
        )
        acc_bf16 = tl.where(m_valid[:, None], acc_bf16 + bias_vals[None, :], acc_bf16)

    tl.store(
        output_ptr + (offs_m[:, None] * stride_cm + global_n[None, :] * stride_cn),
        acc_bf16,
        mask=m_valid[:, None] & n_valid[None, :],
    )


def fused_rans_linear_triton_2(
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
    accum_block_size=16,
    bias=None,
    out=None,
):
    K, N = weight_shape

    # Single view; reused for shape check, strides, and kernel argument
    x_flat = x.view(-1, K)
    if not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()
    M_input, K_input = x_flat.shape

    # assert (
    #     K_input == K
    # ), f"Input K dimension ({K_input}) does not match expected K ({K})"

    TILES_N = (N + tile_n - 1) // tile_n
    TILES_K = (K + tile_k - 1) // tile_k

    expected_tiles = TILES_K * TILES_N
    expected_streams = expected_tiles * tile_n

    # if initial_states.numel() != expected_streams:
    #     raise ValueError(
    #         f"Initial states count ({initial_states.numel()}) does not match expected ({expected_streams}) based on tiling config."
    #     )

    # if tile_offsets.numel() != expected_tiles:
    #     raise ValueError(
    #         f"tile_offsets numel ({tile_offsets.numel()}) != expected ({expected_tiles}). "
    #         f"K-sharding may have failed in the loader."
    #     )

    expected_mantissa_size = K * N
    # if mantissas.numel() != expected_mantissa_size:
    #     raise ValueError(
    #         f"mantissa numel ({mantissas.numel()}) != expected ({expected_mantissa_size})"
    #     )

    stride_am = x_flat.stride(0)
    stride_ak = x_flat.stride(1)

    # Allocate output buffer
    if out is None:
        output = torch.empty((M_input, N), device=x.device, dtype=torch.bfloat16)
    else:
        output = out.view(M_input, N)

    # Kernel grid
    TILE_M = 16
    grid = (triton.cdiv(M_input, TILE_M), triton.cdiv(N, tile_n))

    # Launch kernel
    # Since rANS is sequential num_stages should be 1
    fused_rans_matmul_kernel_with_bias_ilp2[grid](
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
        TILES_N,
        TILES_K,
        stride_am,
        stride_ak,
        output.stride(0),
        output.stride(1),
        TILE_M=TILE_M,
        TILE_N=tile_n,
        TILE_K=tile_k,
        BLOCK_K=32,
        PROB_BITS=12,
        PROB_MASK=4095,
        RANS_L=1 << 16,
        HAS_BIAS=bias is not None,
        num_stages=4,
        num_warps=4,
    )

    return output.view(*x.shape[:-1], N)


def fused_rans_linear_triton_splitk(
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
    accum_block_size=32,
    bias=None,
    out=None,
    SPLIT_K=8,
    workspace=None,
):
    K, N = weight_shape

    # Safely flatten X for 2D matrix math without forcing a memory copy
    x_flat = x.view(-1, K)
    M_input = x_flat.shape[0]

    TILES_N = (N + tile_n - 1) // tile_n
    TILES_K = (K + tile_k - 1) // tile_k

    # We require a buffer of (SPLIT_K, M_input, N) to avoid race conditions.
    if workspace is None:
        workspace = torch.empty(
            (SPLIT_K, M_input, N), dtype=torch.bfloat16, device=x.device
        )
    else:
        # If using your RansWorkspace, ensure we view it as the correct 3D shape
        workspace = workspace.view(SPLIT_K, M_input, N)

    TILE_M = 16

    # 2. LAUNCH 3D GRID
    grid = (triton.cdiv(M_input, TILE_M), triton.cdiv(N, tile_n), SPLIT_K)

    fused_rans_matmul_kernel_splitk[grid](
        x_flat,
        compressed_data,
        tile_offsets,
        tile_max_lens,
        initial_states,
        mantissas,
        workspace,
        slot_map,
        tables,
        M_input,
        N,
        K,
        TILES_N,
        TILES_K,
        x_flat.stride(0),
        x_flat.stride(1),
        workspace.stride(0),  # SPLIT_K stride
        workspace.stride(1),  # M stride
        workspace.stride(2),  # N stride
        TILE_M=TILE_M,
        TILE_N=tile_n,
        TILE_K=tile_k,
        BLOCK_K=accum_block_size,
        PROB_BITS=12,
        PROB_MASK=4095,
        RANS_L=1 << 16,
        SPLIT_K=SPLIT_K,
    )

    # Zero-Copy In-Place Reduction
    if out is not None:
        # Reduce directly into vLLM's pre-allocated output buffer
        torch.sum(workspace, dim=0, out=out)
        if bias is not None:
            out += bias
        return out
    else:
        # If no out buffer provided, create one natively formatted to x's original shape
        final_output = torch.sum(workspace, dim=0)
        if bias is not None:
            final_output += bias
        return final_output.view(*x.shape[:-1], N)


@triton.jit
def fused_rans_matmul_kernel_splitk(
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
    pid_k = tl.program_id(2)  # The Split-K ID

    lane_id = tl.arange(0, TILE_N)
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

    global_n = pid_n * TILE_N + lane_id
    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)

    n_valid = global_n < N
    m_valid = offs_m < M
    x_row_ptrs = x_ptr + offs_m[:, None] * stride_am

    # --- SPLIT-K LOGIC: Calculate which K-tiles this specific block owns ---
    total_k_tiles = tl.cdiv(K, TILE_K)
    tiles_per_split = tl.cdiv(total_k_tiles, SPLIT_K)

    start_k_tile = pid_k * tiles_per_split
    end_k_tile = tl.minimum(start_k_tile + tiles_per_split, total_k_tiles)

    # ONLY loop through this block's assigned tiles!
    for k_tile_idx in range(start_k_tile, end_k_tile):
        tile_id = k_tile_idx * num_tiles_n + pid_n

        tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)
        tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)

        # Non-ILP stream base (1 state per lane)
        tile_stream_base = tile_id * TILE_N
        state = tl.load(initial_states + tile_stream_base + lane_id).to(tl.uint32)

        current_byte_row = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)

        local_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

        tile_man_base = tile_id * TILE_K * TILE_N

        # The compressed data layout is standard: TILE_N per row.
        tile_data_base = compressed_data + tile_start + lane_id
        ROW_STRIDE = TILE_N

        # --- LOOP: Runs for full tile height ---
        for bk_start in range(0, TILE_K, BLOCK_K):
            k_base = k_tile_idx * TILE_K + bk_start

            x_col_ptrs = x_row_ptrs + k_base * stride_ak
            man_ptrs = mantissas_ptr + tile_man_base + bk_start * TILE_N + lane_id

            for i in range(BLOCK_K):
                k_in_bounds = (k_base + i) < K
                mask_n = k_in_bounds & n_valid
                mask_m = k_in_bounds & m_valid[:, None]

                # 1. MEMORY FETCHES
                x_col = tl.load(x_col_ptrs, mask=mask_m, other=0.0).to(tl.bfloat16)
                x_col_ptrs += stride_ak

                raw_man = tl.load(man_ptrs, mask=mask_n, other=0).to(tl.uint16)
                man_ptrs += TILE_N

                # 2. STATE LOOKUPS
                slot = state & PROB_MASK
                exp_sym = tl.load(slot_map + slot, mask=mask_n, other=0).to(tl.uint16)

                # 3. WEIGHT RECONSTRUCTION & ACCUMULATION
                w_int = ((raw_man & 0x80) << 8) | (exp_sym << 7) | (raw_man & 0x7F)
                w_val = w_int.to(tl.uint16).to(tl.bfloat16, bitcast=True)

                local_acc += (x_col * w_val[None, :]).to(tl.float32)

                # 4. STATE UPDATES
                packed = tl.load(tables + exp_sym, mask=mask_n, other=0)
                state = (packed & 0xFFFF) * (state >> PROB_BITS) + (
                    slot - (packed >> 16)
                )

                # 5. RENORMALIZATION
                needs_renorm_1 = (state < RANS_L) & mask_n & (current_byte_row >= 0)
                ptr_1 = tile_data_base + (tl.maximum(current_byte_row, 0) * ROW_STRIDE)
                state = tl.where(
                    needs_renorm_1,
                    (state << 8)
                    | tl.load(ptr_1, mask=needs_renorm_1, other=0).to(tl.uint32),
                    state,
                )
                current_byte_row -= tl.where(needs_renorm_1, 1, 0)

                needs_renorm_2 = (state < RANS_L) & mask_n & (current_byte_row >= 0)
                ptr_2 = tile_data_base + (tl.maximum(current_byte_row, 0) * ROW_STRIDE)
                state = tl.where(
                    needs_renorm_2,
                    (state << 8)
                    | tl.load(ptr_2, mask=needs_renorm_2, other=0).to(tl.uint32),
                    state,
                )
                current_byte_row -= tl.where(needs_renorm_2, 1, 0)

        acc += local_acc

    # --- SPLIT-K WRITEBACK ---
    acc_bf16 = acc.to(tl.bfloat16)

    # Calculate offset using the absolute 3D strides passed from Python
    workspace_offset = (
        (pid_k.to(tl.int64) * stride_wk)
        + (offs_m[:, None].to(tl.int64) * stride_wm)
        + (global_n[None, :].to(tl.int64) * stride_wn)
    )

    tl.store(
        workspace_ptr + workspace_offset,
        acc_bf16,
        mask=m_valid[:, None] & n_valid[None, :],
    )
