#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

RANS_L = tl.constexpr(1 << 15)
PROB_BITS = tl.constexpr(8)
PROB_MASK = tl.constexpr((1 << 8) - 1)


@triton.jit
def _assemble_bf16(exp_symbol, raw_man_byte):
    """
    Reconstructs bfloat16 bits: [S|EEEEEEEE|MMMMMMM]
    exp_symbol: 8 bits
    raw_man_byte: [S|MMMMMMM]
    """
    # Extract sign: top bit of mantissa byte shifted to 15th bit
    sign = (raw_man_byte.to(tl.uint16) & 0x80) << 8
    # Shift exponent: 8 bits shifted to 7th bit
    exponent = (exp_symbol.to(tl.uint16) & 0xFF) << 7
    # Mask 7 bits of mantissa
    mantissa = raw_man_byte.to(tl.uint16) & 0x7F

    # Combine and bitcast to bfloat16
    bits = sign | exponent | mantissa
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _fused_rans_linear_kernel(
    x_ptr,
    exp_stream_ptr,
    man_ptr,
    bias_ptr,
    out_ptr,
    states_ptr,
    slot_map_ptr,
    sym_info_ptr,
    sizes_ptr,
    M,
    N,
    K,
    total_lanes,
    B,
    HAS_BIAS: tl.constexpr,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    PROB_BITS: tl.constexpr,
    PROB_MASK: tl.constexpr,
    RANS_L: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    width = GROUP_SIZE_M * num_pid_n
    group_id = pid // width
    group_size = tl.minimum(num_pid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    n_mask = offs_n < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    k_range = tl.arange(0, BLOCK_SIZE_K)

    for k in range(0, K, BLOCK_SIZE_K):
        seg_id = k // B
        # The stride for gids MUST be the derived N from the metadata
        gids = (seg_id * N) + offs_n
        gid_mask = n_mask & (gids < total_lanes)

        state = tl.load(states_ptr + gids, mask=gid_mask).to(tl.uint32)
        byte_offset = tl.load(sizes_ptr + gids, mask=gid_mask).to(tl.int32) - 1
        exp_buffer = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.int32)

        # --- PHASE 1: DECOMPRESS ---
        syms_in_seg = tl.minimum(BLOCK_SIZE_K, K - k)
        for i in range(BLOCK_SIZE_K):
            row_mask = gid_mask & (i < syms_in_seg)
            slot = state & PROB_MASK
            symbol = tl.load(slot_map_ptr + slot, mask=row_mask)
            exp_buffer = tl.where(k_range[:, None] == i, symbol[None, :], exp_buffer)

            packed = tl.load(sym_info_ptr + symbol.to(tl.int32), mask=row_mask)
            state = tl.where(
                row_mask,
                (packed & 0xFFFF) * (state >> PROB_BITS) + (slot - (packed >> 16)),
                state,
            )

            for _ in range(2):
                renorm_mask = (state < RANS_L) & row_mask & (byte_offset >= 0)
                # Stride is N (derived from metadata)
                ptr = exp_stream_ptr + gids + (byte_offset.to(tl.int64) * N)
                state = tl.where(
                    renorm_mask,
                    (state << 8)
                    | tl.load(ptr, mask=renorm_mask, other=0).to(tl.uint32),
                    state,
                )
                byte_offset -= tl.where(renorm_mask, 1, 0)

        # --- PHASE 2: COMPUTE ---
        offs_k = k + k_range
        tile_x = tl.load(
            x_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        tile_man = tl.load(
            man_ptr + offs_k[:, None] * N + offs_n[None, :],
            mask=(offs_k[:, None] < K) & n_mask[None, :],
            other=0,
        )

        # bf16 assembly: [S(1) | E(8) | M(7)]
        w_bits = (
            ((tile_man.to(tl.uint32) & 0x80) << 8)
            | ((exp_buffer.to(tl.uint32) & 0xFF) << 7)
            | (tile_man.to(tl.uint32) & 0x7F)
        )
        w_bf16 = w_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)

        if BLOCK_SIZE_M == 1:
            # w_bf16: (512, 32)

            # 1. Expand tile_x to match the tile_n dimension
            # Shape: (512, 32)
            x_broadcast = tl.broadcast_to(
                tile_x.to(tl.float32)[0, :, None], (BLOCK_SIZE_K, BLOCK_SIZE_N)
            )

            # 2. Element-wise multiply and sum across K
            # sum((512, 32) * (512, 32), axis=0) -> (32,)
            partial_sum = tl.sum(x_broadcast * w_bf16.to(tl.float32), axis=0)

            # 3. Add to accumulator: (1, 32)
            accumulator += partial_sum[None, :]
        else:
            # GEMM path for Prefill
            accumulator = tl.dot(tile_x.to(tl.bfloat16), w_bf16, acc=accumulator)

    # --- PHASE 3: STORE ---
    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + offs_n, mask=n_mask).to(tl.float32)[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(out_ptrs, accumulator.to(tl.bfloat16), mask=(offs_m[:, None] < M) & n_mask)


def fused_rans_linear_triton(
    x,
    exp_stream,
    man_stream,
    exp_states,
    exp_tables,
    exp_slot_map,
    exp_sizes,
    bias,
    output_shape,
    weight_shape,
):
    # 1. Geometry Setup
    x_flat = x.reshape(-1, x.shape[-1])
    M, K = x_flat.shape
    _, K_weight = weight_shape  # We use the model's K for the reduction limit

    B = 512
    segments_per_col = (K_weight + B - 1) // B
    total_lanes = exp_states.numel()

    # 2. DERIVE N FROM METADATA (The Source of Truth)
    # This ensures the kernel's (seg_id * N) + offs_n indexing matches the compressor
    meta_N = total_lanes // segments_per_col

    # 3. Allocation
    # We MUST allocate for meta_N to maintain lane alignment
    output = torch.empty((M, meta_N), device=x.device, dtype=torch.bfloat16)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"])
        * triton.cdiv(meta_N, META["BLOCK_SIZE_N"]),
    )

    # 4. Kernel Launch
    _fused_rans_linear_kernel[grid](
        x_ptr=x_flat,
        exp_stream_ptr=exp_stream,
        man_ptr=man_stream,
        bias_ptr=bias,
        out_ptr=output,
        states_ptr=exp_states,
        slot_map_ptr=exp_slot_map,
        sym_info_ptr=exp_tables,
        sizes_ptr=exp_sizes,
        M=M,
        N=meta_N,
        K=K_weight,
        total_lanes=total_lanes,
        B=B,
        HAS_BIAS=(bias is not None),
        stride_am=x_flat.stride(0),
        stride_ak=x_flat.stride(1),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        BLOCK_SIZE_M=1 if M == 1 else 32,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=B,
        GROUP_SIZE_M=8,
        PROB_BITS=12,
        PROB_MASK=4095,
        RANS_L=1 << 16,
        num_warps=4,
        num_stages=1,
    )

    # 5. Return and Slicing
    # If meta_N (4096) is larger than requested N (1024), we slice.
    # This handles fused QKV where only one part is being requested.
    requested_N = output_shape[-1]
    return output[:, :requested_N].view(*x.shape[:-1], requested_N)


@triton.jit
def rans_decompress_kernel_triton(
    compressed_streams,
    initial_states,
    stream_sizes,
    output,
    slot_map,
    tables,
    num_streams,
    output_height,
    output_width,
    PROB_BITS: tl.constexpr,
    PROB_MASK: tl.constexpr,
    RANS_L: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    B = 512
    pid = tl.program_id(0)
    gid = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_mask = gid < num_streams

    col = gid % output_width
    seg_id = gid // output_width
    seg_start_row = seg_id * B

    # Load initial metadata
    state = tl.load(initial_states + gid, mask=n_mask).to(tl.uint32)
    byte_offset = tl.load(stream_sizes + gid, mask=n_mask).to(tl.int32) - 1

    # Calculate exactly how many symbols this specific stream handles
    # For the last segment, this might be 384 instead of 512
    syms_in_seg = tl.minimum(B, output_height - seg_start_row)

    for i in range(B):
        # The row_mask must now strictly check if the loop index 'i'
        # is still within the segment's actual symbol count.
        row_mask = n_mask & (i < syms_in_seg)

        current_row = seg_start_row + i

        # 1. Decode Symbol
        slot = state & PROB_MASK
        symbol = tl.load(slot_map + slot, mask=row_mask)

        # 2. Store (Masked)
        out_ptr = output + current_row.to(tl.int64) * output_width + col
        tl.store(out_ptr, symbol, mask=row_mask)

        # 3. Update State ONLY if row_mask is active
        packed_val = tl.load(tables + symbol.to(tl.int32), mask=row_mask)
        freq = packed_val & 0xFFFF
        cdf = (packed_val >> 16) & 0xFFFF

        # We use tl.where to ensure that if row_mask is False, the state
        # is NOT updated. It remains frozen for the remaining iterations of B.
        new_state = freq * (state >> PROB_BITS) + (slot - cdf)
        state = tl.where(row_mask, new_state, state)

        # 4. Renormalization (Must also be gated by row_mask)
        for _ in range(2):
            # Needs_renorm now includes row_mask to prevent "ghost" reads
            needs_renorm = (state < RANS_L) & row_mask
            read_mask = needs_renorm & (byte_offset >= 0)

            ptr_offset = tl.maximum(byte_offset, 0).to(tl.int64) * num_streams + gid
            val = tl.load(compressed_streams + ptr_offset, mask=read_mask, other=0).to(
                tl.uint32
            )

            state = tl.where(read_mask, (state << 8) | val, state)

            # byte_offset ONLY decrements if a real read happened
            byte_offset -= tl.where(read_mask, 1, 0)


def rans_decomp_triton(
    compressed_streams, initial_states, tables, slot_map, stream_sizes, output_shape
):
    # If out_shape is (N, D), usually num_streams = N * D
    # but based on your comment, num_streams is likely your parallel state count.
    num_streams = len(initial_states)

    output = torch.empty(
        output_shape, device=compressed_streams.device, dtype=torch.uint8
    )

    grid = lambda meta: (triton.cdiv(num_streams, meta["BLOCK_SIZE"]),)

    print("Output Shape:", output_shape)
    print("Num Streams:", num_streams)
    print("Num initial states:", len(initial_states))
    print("Num stream sizes:", len(stream_sizes))
    print(f"Tables Dtype: {tables.dtype}")
    print(f"DEBUG: Max initial state: {initial_states.max()}")
    print(f"DEBUG: Max stream size: {stream_sizes.max()}")
    print(f"DEBUG: Slot map sum: {slot_map.sum()}")
    rans_decompress_kernel_triton[grid](
        compressed_streams,
        initial_states=initial_states,
        stream_sizes=stream_sizes,
        output=output,
        slot_map=slot_map,
        tables=tables,
        num_streams=num_streams,
        output_height=output_shape[0],
        output_width=output_shape[1],
        PROB_BITS=12,
        PROB_MASK=4095,
        RANS_L=1 << 16,
        BLOCK_SIZE=128,
        num_warps=4,
    )

    print("Decomp kernel done.")

    return output
