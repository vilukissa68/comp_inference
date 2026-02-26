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


triton.jit_constants = {"TRITON_FORCE_FULL_UNROLL": 1, "TRITON_DEBUG": 1}


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
        row_mask = n_mask & (i < syms_in_seg)

        current_row = seg_start_row + i

        # Decode next symbol
        slot = state & PROB_MASK
        symbol = tl.load(slot_map + slot, mask=row_mask)

        # Store symbol to output
        out_ptr = output + current_row.to(tl.int64) * output_width + col
        tl.store(out_ptr, symbol, mask=row_mask)

        # Update state
        packed_val = tl.load(tables + symbol.to(tl.int32), mask=row_mask)
        freq = packed_val & 0xFFFF
        cdf = (packed_val >> 16) & 0xFFFF

        # Mask for new state
        new_state = freq * (state >> PROB_BITS) + (slot - cdf)
        state = tl.where(row_mask, new_state, state)

        # Renormalization loop
        # NOTE: For uint8 this loop needs at most 2 iterations
        for _ in range(2):
            needs_renorm = (state < RANS_L) & row_mask
            read_mask = needs_renorm & (byte_offset >= 0)

            ptr_offset = tl.maximum(byte_offset, 0).to(tl.int64) * num_streams + gid
            val = tl.load(compressed_streams + ptr_offset, mask=read_mask, other=0).to(
                tl.uint32
            )

            state = tl.where(read_mask, (state << 8) | val, state)

            byte_offset -= tl.where(read_mask, 1, 0)


def rans_decomp_triton(
    compressed_streams, initial_states, tables, slot_map, stream_sizes, output_shape
):
    num_streams = len(initial_states)

    output = torch.empty(
        output_shape, device=compressed_streams.device, dtype=torch.uint8
    )

    grid = lambda meta: (triton.cdiv(num_streams, meta["BLOCK_SIZE"]),)

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
        num_warps=8,
    )

    return output


@triton.jit
def rans_decompress_tiled_kernel_triton(
    compressed_data,
    tile_offsets,
    tile_max_lens,
    initial_states,
    output,
    slot_map,
    tables,
    num_tiles_n,
    total_height,
    total_width,
    TILE_K: tl.constexpr,
    TILE_N: tl.constexpr,
    PROB_BITS: tl.constexpr,
    PROB_MASK: tl.constexpr,
    RANS_L: tl.constexpr,
):
    # Map coordinates
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    tile_id = pid_k * num_tiles_n + pid_n

    lane_id = tl.arange(0, TILE_N)
    global_col = pid_n * TILE_N + lane_id
    global_stream_id = tile_id * TILE_N + lane_id

    # Load metadata
    tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)
    tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)

    current_byte_row = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)

    n_mask = global_col < total_width
    state = tl.load(initial_states + global_stream_id, mask=n_mask, other=0).to(
        tl.uint32
    )

    start_row = pid_k * TILE_K
    syms_in_tile = tl.minimum(TILE_K, total_height - start_row)

    actual_tile_width = tl.minimum(TILE_N, total_width - pid_n * TILE_N)
    out_col_base = output + global_col.to(tl.int64)

    # Decoding loop
    for i in range(TILE_K):
        row_mask = n_mask & (i < syms_in_tile)

        # Decode Symbol
        slot = state & PROB_MASK
        symbol = tl.load(slot_map + slot, mask=row_mask, other=0)

        # Store
        out_ptr = out_col_base + (start_row + i) * total_width
        tl.store(out_ptr, symbol, mask=row_mask)

        # Update State
        packed_val = tl.load(tables + symbol.to(tl.int32), mask=row_mask, other=0)
        freq = packed_val & 0xFFFF
        cdf = (packed_val >> 16) & 0xFFFF

        new_state = freq * (state >> PROB_BITS) + (slot - cdf)
        state = tl.where(row_mask, new_state, state)

        # Renormalization loop
        for _ in range(2):
            needs_renorm = (state < RANS_L) & row_mask & (current_byte_row >= 0)

            # Tile strided memory access
            ptr = (
                compressed_data
                + tile_start
                + (current_byte_row * actual_tile_width)
                + lane_id
            )
            val = tl.load(ptr, mask=needs_renorm, other=0).to(tl.uint32)

            state = tl.where(needs_renorm, (state << 8) | val, state)
            current_byte_row -= tl.where(needs_renorm, 1, 0)


def rans_decomp_triton_tiled(
    compressed_streams,
    initial_states,
    tables,
    slot_map,
    output_shape,
    tile_offsets,
    tile_max_lens,
    tile_k=1024,
    tile_n=32,
):
    output = torch.empty(
        output_shape, device=compressed_streams.device, dtype=torch.uint8
    )

    K, N = output_shape
    num_tiles_n = (N + tile_n - 1) // tile_n
    num_tiles_k = (K + tile_k - 1) // tile_k
    grid = (num_tiles_n, num_tiles_k)

    rans_decompress_tiled_kernel_triton[grid](
        compressed_data=compressed_streams,
        tile_offsets=tile_offsets,
        tile_max_lens=tile_max_lens,
        initial_states=initial_states,
        output=output,
        slot_map=slot_map,
        tables=tables,
        num_tiles_n=num_tiles_n,
        total_height=output_shape[0],
        total_width=output_shape[1],
        TILE_K=tile_k,
        TILE_N=tile_n,
        PROB_BITS=12,
        PROB_MASK=4095,
        RANS_L=1 << 16,
        num_stages=1,
        num_warps=8,
    )

    return output


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

    # Accumulator
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    global_n = pid_n * TILE_N + lane_id

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_valid = global_n < N
    m_valid = offs_m < M

    # Precompute row base pointers for x to avoid recomputing offs_m * stride_am each block
    x_row_ptrs = x_ptr + offs_m * stride_am

    bk_range = tl.arange(0, BLOCK_K)

    for k_tile_idx in range(0, tl.cdiv(K, TILE_K)):
        tile_id = k_tile_idx * num_tiles_n + pid_n

        # Load Metadata
        tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)
        tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)

        state = tl.load(initial_states + (tile_id * TILE_N + lane_id)).to(tl.uint32)
        current_byte_row = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)

        # Local accumulator for this tile
        local_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

        # Pre-compute tile addressing
        tile_man_base = tile_id * TILE_K * TILE_N
        tile_data_base = compressed_data + tile_start + lane_id

        for bk_start in range(0, TILE_K, BLOCK_K):
            w_tile = tl.zeros((BLOCK_K, TILE_N), dtype=tl.bfloat16)

            k_base = k_tile_idx * TILE_K + bk_start
            bk_man_base = tile_man_base + bk_start * TILE_N + lane_id

            for i in range(BLOCK_K):
                # fmt: off
                mask_k = ((k_base + i) < K) & n_valid
                # fmt: on

                # Decode
                slot = state & PROB_MASK
                exp_sym = tl.load(slot_map + slot, mask=mask_k, other=0).to(tl.uint16)

                # Interleaved mantissa indexing
                raw_man = tl.load(
                    mantissas_ptr + bk_man_base + i * TILE_N, mask=mask_k, other=0
                ).to(tl.uint16)

                # BF16 reconstruction
                w_int = ((raw_man & 0x80) << 8) | (exp_sym << 7) | (raw_man & 0x7F)

                # Narrow to 16 bits, then reinterpret as bf16
                w_val = w_int.to(tl.uint16).to(tl.bfloat16, bitcast=True)

                row_mask = bk_range == i
                w_tile = tl.where(row_mask[:, None], w_val[None, :], w_tile)

                # State Update
                packed = tl.load(tables + exp_sym, mask=mask_k, other=0)
                state = (packed & 0xFFFF) * (state >> PROB_BITS) + (
                    slot - (packed >> 16)
                )

                # Renormalization looop
                for _ in range(2):
                    needs_renorm = (state < RANS_L) & mask_k & (current_byte_row >= 0)
                    ptr = tile_data_base + (current_byte_row * TILE_N)
                    state = tl.where(
                        needs_renorm,
                        (state << 8)
                        | tl.load(ptr, mask=needs_renorm, other=0).to(tl.uint32),
                        state,
                    )
                    current_byte_row -= tl.where(needs_renorm, 1, 0)

            # Local matmul
            offs_bk = k_base + bk_range
            x_tile = tl.load(
                x_row_ptrs[:, None] + offs_bk[None, :] * stride_ak,
                mask=m_valid[:, None] & (offs_bk[None, :] < K),
                other=0.0,
            ).to(tl.bfloat16)

            local_acc = tl.dot(
                x_tile, w_tile, local_acc, out_dtype=tl.float32, allow_tf32=False
            )

        # Accumulate to global accumulator after processing each chunk
        acc += local_acc.to(tl.bfloat16)

    # Cast to bf16 for biasing
    acc_bf16 = acc.to(tl.bfloat16)

    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + global_n, mask=n_valid, other=0.0).to(
            tl.bfloat16
        )

        # Add bias
        acc_bf16 = tl.where(m_valid[:, None], acc_bf16 + bias_vals[None, :], acc_bf16)

    # Store final
    tl.store(
        output_ptr + (offs_m[:, None] * stride_cm + global_n[None, :] * stride_cn),
        acc_bf16,
        mask=m_valid[:, None] & n_valid[None, :],
    )


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
    accum_block_size=128,
    bias=None,
    out=None,
):
    K, N = weight_shape

    # Single view; reused for shape check, strides, and kernel argument
    x_flat = x.view(-1, K)
    if not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()
    M_input, K_input = x_flat.shape

    assert (
        K_input == K
    ), f"Input K dimension ({K_input}) does not match expected K ({K})"

    TILES_N = (N + tile_n - 1) // tile_n
    TILES_K = (K + tile_k - 1) // tile_k

    expected_tiles = TILES_K * TILES_N
    expected_streams = expected_tiles * tile_n

    if initial_states.numel() != expected_streams:
        raise ValueError(
            f"Initial states count ({initial_states.numel()}) does not match expected ({expected_streams}) based on tiling config."
        )

    if tile_offsets.numel() != expected_tiles:
        raise ValueError(
            f"tile_offsets numel ({tile_offsets.numel()}) != expected ({expected_tiles}). "
            f"K-sharding may have failed in the loader."
        )

    expected_mantissa_size = K * N
    if mantissas.numel() != expected_mantissa_size:
        raise ValueError(
            f"mantissa numel ({mantissas.numel()}) != expected ({expected_mantissa_size})"
        )

    stride_am = x_flat.stride(0)
    stride_ak = x_flat.stride(1)

    # Allocate output buffer
    if out is None:
        output = torch.empty((M_input, N), device=x.device, dtype=torch.bfloat16)
    else:
        output = out.view(M_input, N)

    # Kernel grid
    TILE_M = 32
    grid = (triton.cdiv(M_input, TILE_M), triton.cdiv(N, tile_n))

    # Launch kernel
    # Since rANS is sequential num_stages should be 1
    fused_rans_matmul_kernel_with_bias[grid](
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
        BLOCK_K=accum_block_size,
        PROB_BITS=12,
        PROB_MASK=4095,
        RANS_L=1 << 16,
        HAS_BIAS=bias is not None,
        num_stages=1,
        num_warps=8,
    )

    return output.view(*x.shape[:-1], N)
