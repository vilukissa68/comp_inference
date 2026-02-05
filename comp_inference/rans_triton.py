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


# @triton.jit
# def _fused_rans_linear_kernel(
#     x_ptr,
#     exp_stream_ptr,
#     man_ptr,
#     bias_ptr,
#     out_ptr,
#     states_ptr,
#     slot_map_ptr,
#     sym_info_ptr,
#     sizes_ptr,
#     M,
#     N,
#     K,
#     total_lanes,
#     B,
#     HAS_BIAS: tl.constexpr,
#     stride_am,
#     stride_ak,
#     stride_cm,
#     stride_cn,
#     BLOCK_SIZE_M: tl.constexpr,
#     BLOCK_SIZE_N: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
#     PROB_BITS: tl.constexpr,
#     PROB_MASK: tl.constexpr,
#     RANS_L: tl.constexpr,
# ):
#     pid = tl.program_id(0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

#     width = GROUP_SIZE_M * num_pid_n
#     group_id = pid // width
#     group_size = tl.minimum(num_pid_m - group_id * GROUP_SIZE_M, GROUP_SIZE_M)
#     pid_m = group_id * GROUP_SIZE_M + (pid % group_size)
#     pid_n = (pid % width) // group_size

#     offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#     n_mask = offs_n < N

#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#     k_range = tl.arange(0, BLOCK_SIZE_K)

#     for k in range(0, K, BLOCK_SIZE_K):
#         seg_id = k // B
#         # The stride for gids MUST be the derived N from the metadata
#         gids = (seg_id * N) + offs_n
#         gid_mask = n_mask & (gids < total_lanes)

#         state = tl.load(states_ptr + gids, mask=gid_mask).to(tl.uint32)
#         byte_offset = tl.load(sizes_ptr + gids, mask=gid_mask).to(tl.int32) - 1
#         exp_buffer = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.int32)

#         # --- PHASE 1: DECOMPRESS ---
#         syms_in_seg = tl.minimum(BLOCK_SIZE_K, K - k)
#         for i in range(BLOCK_SIZE_K):
#             row_mask = gid_mask & (i < syms_in_seg)
#             slot = state & PROB_MASK
#             symbol = tl.load(slot_map_ptr + slot, mask=row_mask)
#             exp_buffer = tl.where(k_range[:, None] == i, symbol[None, :], exp_buffer)

#             packed = tl.load(sym_info_ptr + symbol.to(tl.int32), mask=row_mask)
#             state = tl.where(
#                 row_mask,
#                 (packed & 0xFFFF) * (state >> PROB_BITS) + (slot - (packed >> 16)),
#                 state,
#             )

#             for _ in range(2):
#                 renorm_mask = (state < RANS_L) & row_mask & (byte_offset >= 0)
#                 # Stride is N (derived from metadata)
#                 ptr = exp_stream_ptr + gids + (byte_offset.to(tl.int64) * N)
#                 state = tl.where(
#                     renorm_mask,
#                     (state << 8)
#                     | tl.load(ptr, mask=renorm_mask, other=0).to(tl.uint32),
#                     state,
#                 )
#                 byte_offset -= tl.where(renorm_mask, 1, 0)

#         # --- PHASE 2: COMPUTE ---
#         offs_k = k + k_range
#         tile_x = tl.load(
#             x_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
#             mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
#             other=0.0,
#         )
#         tile_man = tl.load(
#             man_ptr + offs_k[:, None] * N + offs_n[None, :],
#             mask=(offs_k[:, None] < K) & n_mask[None, :],
#             other=0,
#         )

#         # bf16 assembly: [S(1) | E(8) | M(7)]
#         w_bits = (
#             ((tile_man.to(tl.uint32) & 0x80) << 8)
#             | ((exp_buffer.to(tl.uint32) & 0xFF) << 7)
#             | (tile_man.to(tl.uint32) & 0x7F)
#         )
#         w_bf16 = w_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)

#         if BLOCK_SIZE_M == 1:
#             # w_bf16: (512, 32)

#             # 1. Expand tile_x to match the tile_n dimension
#             # Shape: (512, 32)
#             x_broadcast = tl.broadcast_to(
#                 tile_x.to(tl.float32)[0, :, None], (BLOCK_SIZE_K, BLOCK_SIZE_N)
#             )

#             # 2. Element-wise multiply and sum across K
#             # sum((512, 32) * (512, 32), axis=0) -> (32,)
#             partial_sum = tl.sum(x_broadcast * w_bf16.to(tl.float32), axis=0)

#             # 3. Add to accumulator: (1, 32)
#             accumulator += partial_sum[None, :]
#         else:
#             # GEMM path for Prefill
#             accumulator = tl.dot(tile_x.to(tl.bfloat16), w_bf16, acc=accumulator)

#     # --- PHASE 3: STORE ---
#     if HAS_BIAS:
#         accumulator += tl.load(bias_ptr + offs_n, mask=n_mask).to(tl.float32)[None, :]

#     out_ptrs = out_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
#     tl.store(out_ptrs, accumulator.to(tl.bfloat16), mask=(offs_m[:, None] < M) & n_mask)


# def fused_rans_linear_triton(
#     x,
#     exp_stream,
#     man_stream,
#     exp_states,
#     exp_tables,
#     exp_slot_map,
#     exp_sizes,
#     bias,
#     output_shape,
#     weight_shape,
# ):
#     # 1. Geometry Setup
#     x_flat = x.reshape(-1, x.shape[-1])
#     M, K = x_flat.shape
#     _, K_weight = weight_shape  # We use the model's K for the reduction limit

#     B = 512
#     segments_per_col = (K_weight + B - 1) // B
#     total_lanes = exp_states.numel()

#     # 2. DERIVE N FROM METADATA (The Source of Truth)
#     # This ensures the kernel's (seg_id * N) + offs_n indexing matches the compressor
#     meta_N = total_lanes // segments_per_col

#     # 3. Allocation
#     # We MUST allocate for meta_N to maintain lane alignment
#     output = torch.empty((M, meta_N), device=x.device, dtype=torch.bfloat16)

#     grid = lambda META: (
#         triton.cdiv(M, META["BLOCK_SIZE_M"])
#         * triton.cdiv(meta_N, META["BLOCK_SIZE_N"]),
#     )

#     # 4. Kernel Launch
#     _fused_rans_linear_kernel[grid](
#         x_ptr=x_flat,
#         exp_stream_ptr=exp_stream,
#         man_ptr=man_stream,
#         bias_ptr=bias,
#         out_ptr=output,
#         states_ptr=exp_states,
#         slot_map_ptr=exp_slot_map,
#         sym_info_ptr=exp_tables,
#         sizes_ptr=exp_sizes,
#         M=M,
#         N=meta_N,
#         K=K_weight,
#         total_lanes=total_lanes,
#         B=B,
#         HAS_BIAS=(bias is not None),
#         stride_am=x_flat.stride(0),
#         stride_ak=x_flat.stride(1),
#         stride_cm=output.stride(0),
#         stride_cn=output.stride(1),
#         BLOCK_SIZE_M=1 if M == 1 else 32,
#         BLOCK_SIZE_N=32,
#         BLOCK_SIZE_K=B,
#         GROUP_SIZE_M=8,
#         PROB_BITS=12,
#         PROB_MASK=4095,
#         RANS_L=1 << 16,
#         num_warps=4,
#         num_stages=1,
#     )

#     # 5. Return and Slicing
#     # If meta_N (4096) is larger than requested N (1024), we slice.
#     # This handles fused QKV where only one part is being requested.
#     requested_N = output_shape[-1]
#     return output[:, :requested_N].view(*x.shape[:-1], requested_N)


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


# Working impl
# @triton.jit
# def rans_decompress_tiled_kernel_triton(
#     compressed_data,
#     tile_offsets,
#     tile_max_lens,  # Explicitly pass the un-padded depth
#     initial_states,
#     output,
#     slot_map,
#     tables,
#     num_tiles_n,
#     total_height,
#     total_width,
#     TILE_K: tl.constexpr,
#     TILE_N: tl.constexpr,
#     PROB_BITS: tl.constexpr,
#     PROB_MASK: tl.constexpr,
#     RANS_L: tl.constexpr,
# ):
#     # 1. Coordinate Mapping
#     pid_n = tl.program_id(0)
#     pid_k = tl.program_id(1)
#     tile_id = pid_k * num_tiles_n + pid_n

#     lane_id = tl.arange(0, TILE_N)
#     global_col = pid_n * TILE_N + lane_id
#     global_stream_id = tile_id * TILE_N + lane_id

#     # 2. Metadata Loading
#     tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)

#     # --- FIX STARTS HERE ---
#     # Load the real row depth of this tile, ignoring HW alignment padding
#     tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)

#     # Initialize pointer to the LAST real row (where Row 0's bytes are)
#     # We use Left-Padding in C++, so Row 0 is at (tile_depth - 1)
#     current_byte_row = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)
#     # --- FIX ENDS HERE ---

#     # 3. Stream & Row Guards
#     n_mask = global_col < total_width
#     state = tl.load(initial_states + global_stream_id, mask=n_mask, other=0).to(
#         tl.uint32
#     )

#     start_row = pid_k * TILE_K
#     syms_in_tile = tl.minimum(TILE_K, total_height - start_row)

#     # 4. Decoding Loop
#     for i in range(TILE_K):
#         row_mask = n_mask & (i < syms_in_tile)

#         # Decode Symbol
#         slot = state & PROB_MASK
#         symbol = tl.load(slot_map + slot, mask=row_mask, other=0)

#         # Store Result
#         out_ptr = output + (start_row + i).to(tl.int64) * total_width + global_col
#         tl.store(out_ptr, symbol, mask=row_mask)

#         # Update State
#         packed_val = tl.load(tables + symbol.to(tl.int32), mask=row_mask, other=0)
#         freq = packed_val & 0xFFFF
#         cdf = (packed_val >> 16) & 0xFFFF

#         new_state = freq * (state >> PROB_BITS) + (slot - cdf)
#         state = tl.where(row_mask, new_state, state)

#         # Renormalize (Pull bytes from the interleaved stack)
#         for _ in range(2):
#             # Guard against reading past the start of the tile's data
#             # (current_byte_row >= 0) stops us from hitting the padding at index 0
#             needs_renorm = (state < RANS_L) & row_mask & (current_byte_row >= 0)

#             ptr = compressed_data + tile_start + (current_byte_row * TILE_N) + lane_id
#             val = tl.load(ptr, mask=needs_renorm, other=0).to(tl.uint32)

#             state = tl.where(needs_renorm, (state << 8) | val, state)
#             current_byte_row -= tl.where(needs_renorm, 1, 0)


@triton.jit
def rans_decompress_tiled_kernel_triton(
    compressed_data,
    tile_offsets,
    tile_max_lens,  # Explicitly pass the un-padded depth
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
    # 1. Coordinate Mapping
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    tile_id = pid_k * num_tiles_n + pid_n

    lane_id = tl.arange(0, TILE_N)
    global_col = pid_n * TILE_N + lane_id
    global_stream_id = tile_id * TILE_N + lane_id

    # 2. Metadata Loading
    tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)

    # --- FIX STARTS HERE ---
    # Load the real row depth of this tile, ignoring HW alignment padding
    tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)

    # Initialize pointer to the LAST real row (where Row 0's bytes are)
    # We use Left-Padding in C++, so Row 0 is at (tile_depth - 1)
    current_byte_row = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)
    # --- FIX ENDS HERE ---

    # 3. Stream & Row Guards
    n_mask = global_col < total_width
    state = tl.load(initial_states + global_stream_id, mask=n_mask, other=0).to(
        tl.uint32
    )

    start_row = pid_k * TILE_K
    syms_in_tile = tl.minimum(TILE_K, total_height - start_row)

    # 4. Decoding Loop
    for i in range(TILE_K):
        row_mask = n_mask & (i < syms_in_tile)

        # Decode Symbol
        slot = state & PROB_MASK
        symbol = tl.load(slot_map + slot, mask=row_mask, other=0)

        # Store Result
        out_ptr = (
            output
            + (start_row + i).to(tl.int64) * total_width
            + global_col.to(tl.int64)
        )
        tl.store(out_ptr, symbol, mask=row_mask)

        # Update State
        packed_val = tl.load(tables + symbol.to(tl.int32), mask=row_mask, other=0)
        freq = packed_val & 0xFFFF
        cdf = (packed_val >> 16) & 0xFFFF

        new_state = freq * (state >> PROB_BITS) + (slot - cdf)
        state = tl.where(row_mask, new_state, state)

        # Renormalize (Pull bytes from the interleaved stack)
        for _ in range(2):
            # Guard against reading past the start of the tile's data
            # (current_byte_row >= 0) stops us from hitting the padding at index 0
            needs_renorm = (state < RANS_L) & row_mask & (current_byte_row >= 0)

            # Dynamically calculate how many active streams exist in this specific tile
            actual_tile_width = tl.minimum(TILE_N, total_width - pid_n * TILE_N)

            # Use the actual width as the memory stride
            ptr = (
                compressed_data
                + tile_start
                + (current_byte_row * actual_tile_width)
                + lane_id
            )
            # ptr = compressed_data + tile_start + (current_byte_row * TILE_N) + lane_id
            val = tl.load(ptr, mask=needs_renorm, other=0).to(tl.uint32)

            state = tl.where(needs_renorm, (state << 8) | val, state)
            current_byte_row -= tl.where(needs_renorm, 1, 0)


def rans_decomp_triton_tiled(
    compressed_streams,
    initial_states,
    tables,
    slot_map,
    output_shape,
    tile_offsets,  # Added to handle the jump-table
    tile_max_lens,
    tile_k=1024,
    tile_n=32,
):
    num_streams = len(initial_states)
    output = torch.empty(
        output_shape, device=compressed_streams.device, dtype=torch.uint8
    )

    # K, N = output_shape
    # TILES_N = (N + tile_n - 1) // tile_n
    # TILES_K = (K + tile_k - 1) // tile_k

    # # Grid: (Tiles_N, Tiles_K)
    # # This matches the 2D layout produced by the C++ encoder
    # grid = (TILES_N, TILES_K)
    K, N = output_shape
    num_tiles_n = (N + tile_n - 1) // tile_n
    num_tiles_k = (K + tile_k - 1) // tile_k
    # num_tiles_n = triton.cdiv(N, tile_n)
    # num_tiles_k = triton.cdiv(K, tile_k)

    print("K:", K)
    print("N:", N)

    print("TILE_K:", tile_k)
    print("TILE_N:", tile_n)

    print("Num Tiles K:", num_tiles_k)
    print("Num Tiles N:", num_tiles_n)

    print("Output shape:", output.shape)

    print("Tile offsets:", tile_offsets.shape)
    print("Tile offset[0]", tile_offsets[0])
    print("Tile max lens:", tile_max_lens.shape)

    # 3. Grid: (Number of Tiles in N, Number of Tiles in K)
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
        num_warps=4,
    )

    return output


# Almost works
# @triton.jit
# def fused_rans_matmul_kernel_with_bias(
#     x_ptr,
#     compressed_data,
#     tile_offsets,
#     tile_max_lens,
#     initial_states,
#     mantissas_ptr,
#     bias_ptr,
#     output_ptr,
#     slot_map,
#     tables,
#     M,
#     N,
#     K,
#     num_tiles_n,
#     num_tiles_k,
#     stride_am,
#     stride_ak,
#     stride_cm,
#     stride_cn,
#     TILE_M: tl.constexpr,
#     TILE_N: tl.constexpr,
#     TILE_K: tl.constexpr,
#     BLOCK_K: tl.constexpr,
#     PROB_BITS: tl.constexpr,
#     PROB_MASK: tl.constexpr,
#     RANS_L: tl.constexpr,
#     HAS_BIAS: tl.constexpr,
# ):
#     pid_m = tl.program_id(0)
#     pid_n = tl.program_id(1)
#     lane_id = tl.arange(0, TILE_N)

#     acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
#     acc_bf16 = tl.zeros((TILE_M, TILE_N), dtype=tl.bfloat16)
#     global_n = pid_n * TILE_N + lane_id

#     for k_tile_idx in range(0, tl.cdiv(K, TILE_K)):
#         # tile_id identifies the [TILE_K, TILE_N] block
#         tile_id = k_tile_idx * num_tiles_n + pid_n

#         # Load Metadata
#         tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)
#         tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)

#         # Fresh state for THIS tile
#         state = tl.load(initial_states + (tile_id * TILE_N + lane_id)).to(tl.uint32)
#         current_byte_row = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)

#         for bk_start in range(0, TILE_K, BLOCK_K):
#             w_tile = tl.zeros((BLOCK_K, TILE_N), dtype=tl.bfloat16)

#             for i in range(BLOCK_K):
#                 k_idx = k_tile_idx * TILE_K + bk_start + i
#                 # fmt: off
#                 mask_k = (k_idx < K) & (global_n < N)
#                 # fmt: on

#                 # Decode
#                 slot = state & PROB_MASK
#                 exp_sym = tl.load(slot_map + slot, mask=mask_k, other=0).to(tl.uint16)

#                 # Mantissa Indexing
#                 m_offset = (
#                     (tile_id * TILE_K * TILE_N) + ((bk_start + i) * TILE_N) + lane_id
#                 )
#                 raw_man = tl.load(mantissas_ptr + m_offset, mask=mask_k, other=0).to(
#                     tl.uint16
#                 )

#                 # --- BF16 RECONSTRUCT FIX ---
#                 # Combine components into a 16-bit word (stored in a 32-bit register initially)
#                 w_int = (
#                     ((raw_man & 0x80) << 8) | ((exp_sym & 0xFF) << 7) | (raw_man & 0x7F)
#                 )

#                 # Narrow to 16 bits, then reinterpret as bf16
#                 w_val = w_int.to(tl.uint16).to(tl.bfloat16, bitcast=True)

#                 # Stack row into w_tile
#                 row_mask = tl.arange(0, BLOCK_K) == i
#                 w_tile = tl.where(row_mask[:, None], w_val[None, :], w_tile)

#                 # State Update
#                 packed = tl.load(tables + exp_sym, mask=mask_k, other=0)
#                 state = (packed & 0xFFFF) * (state >> PROB_BITS) + (
#                     slot - ((packed >> 16) & 0xFFFF)
#                 )

#                 for _ in range(2):
#                     needs_renorm = (state < RANS_L) & mask_k & (current_byte_row >= 0)
#                     ptr = (
#                         compressed_data
#                         + tile_start
#                         + (current_byte_row * TILE_N)
#                         + lane_id
#                     )
#                     state = tl.where(
#                         needs_renorm,
#                         (state << 8)
#                         | tl.load(ptr, mask=needs_renorm, other=0).to(tl.uint32),
#                         state,
#                     )
#                     current_byte_row -= tl.where(needs_renorm, 1, 0)

#             # Matmul
#             offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
#             offs_bk = (k_tile_idx * TILE_K + bk_start) + tl.arange(0, BLOCK_K)
#             x_tile = tl.load(
#                 x_ptr + (offs_m[:, None] * stride_am + offs_bk[None, :] * stride_ak),
#                 mask=(offs_m[:, None] < M) & (offs_bk[None, :] < K),
#                 other=0.0,
#             ).to(tl.bfloat16)
#             acc = tl.dot(x_tile, w_tile, acc, out_dtype=tl.float32, allow_tf32=False)
#             acc_bf16 = acc.to(tl.bfloat16)

#     # if HAS_BIAS:
#     #     offs_n = pid_n * TILE_N + lane_id
#     #     # Mask N here to avoid out-of-bounds bias reads
#     #     bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(
#     #         tl.float32
#     #     )
#     #     # Optional: Only add bias to valid M rows to save a few cycles
#     #     mask_m = (pid_m * TILE_M + tl.arange(0, TILE_M)) < M
#     #     acc = tl.where(mask_m[:, None], acc + bias_vals[None, :], acc)

#     if HAS_BIAS:
#         offs_n = pid_n * TILE_N + lane_id
#         # 2. Load bias directly as bfloat16
#         bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(
#             tl.bfloat16
#         )

#         # Optional: Only add bias to valid M rows to save a few cycles
#         mask_m = (pid_m * TILE_M + tl.arange(0, TILE_M)) < M

#         # 3. Add them together in bfloat16 space
#         acc_bf16 = tl.where(mask_m[:, None], acc_bf16 + bias_vals[None, :], acc_bf16)

#     # Store
#     out_m = pid_m * TILE_M + tl.arange(0, TILE_M)
#     out_n = pid_n * TILE_N + lane_id
#     tl.store(
#         output_ptr + (out_m[:, None] * stride_cm + out_n[None, :] * stride_cn),
#         acc_bf16.to(tl.bfloat16),
#         mask=(out_m[:, None] < M) & (out_n[None, :] < N),
#     )


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

    # 1. Main Accumulator for the entire K dimension
    acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    global_n = pid_n * TILE_N + lane_id

    for k_tile_idx in range(0, tl.cdiv(K, TILE_K)):
        # tile_id identifies the [TILE_K, TILE_N] block
        tile_id = k_tile_idx * num_tiles_n + pid_n

        # Load Metadata
        tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)
        tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)

        # Fresh state for THIS tile
        state = tl.load(initial_states + (tile_id * TILE_N + lane_id)).to(tl.uint32)
        current_byte_row = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)

        # 2. Local accumulator strictly for this 1024-element chunk
        local_acc = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)

        for bk_start in range(0, TILE_K, BLOCK_K):
            w_tile = tl.zeros((BLOCK_K, TILE_N), dtype=tl.bfloat16)

            for i in range(BLOCK_K):
                k_idx = k_tile_idx * TILE_K + bk_start + i
                # fmt: off
                mask_k = (k_idx < K) & (global_n < N)
                # fmt: on

                # Decode
                slot = state & PROB_MASK
                exp_sym = tl.load(slot_map + slot, mask=mask_k, other=0).to(tl.uint16)

                # Mantissa Indexing
                m_offset = (
                    (tile_id * TILE_K * TILE_N) + ((bk_start + i) * TILE_N) + lane_id
                )
                raw_man = tl.load(mantissas_ptr + m_offset, mask=mask_k, other=0).to(
                    tl.uint16
                )

                # --- BF16 RECONSTRUCT FIX ---
                # Combine components into a 16-bit word (stored in a 32-bit register initially)
                w_int = (
                    ((raw_man & 0x80) << 8) | ((exp_sym & 0xFF) << 7) | (raw_man & 0x7F)
                )

                # Narrow to 16 bits, then reinterpret as bf16
                w_val = w_int.to(tl.uint16).to(tl.bfloat16, bitcast=True)

                # Stack row into w_tile
                row_mask = tl.arange(0, BLOCK_K) == i
                w_tile = tl.where(row_mask[:, None], w_val[None, :], w_tile)

                # State Update
                packed = tl.load(tables + exp_sym, mask=mask_k, other=0)
                state = (packed & 0xFFFF) * (state >> PROB_BITS) + (
                    slot - ((packed >> 16) & 0xFFFF)
                )

                for _ in range(2):
                    needs_renorm = (state < RANS_L) & mask_k & (current_byte_row >= 0)
                    ptr = (
                        compressed_data
                        + tile_start
                        + (current_byte_row * TILE_N)
                        + lane_id
                    )
                    state = tl.where(
                        needs_renorm,
                        (state << 8)
                        | tl.load(ptr, mask=needs_renorm, other=0).to(tl.uint32),
                        state,
                    )
                    current_byte_row -= tl.where(needs_renorm, 1, 0)

            # Matmul (accumulating strictly into the LOCAL accumulator)
            offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
            offs_bk = (k_tile_idx * TILE_K + bk_start) + tl.arange(0, BLOCK_K)
            x_tile = tl.load(
                x_ptr + (offs_m[:, None] * stride_am + offs_bk[None, :] * stride_ak),
                mask=(offs_m[:, None] < M) & (offs_bk[None, :] < K),
                other=0.0,
            ).to(tl.bfloat16)

            # Note: Feed 'local_acc' in, write 'local_acc' out.
            local_acc = tl.dot(
                x_tile, w_tile, local_acc, out_dtype=tl.float32, allow_tf32=False
            )

        # 3. Add the completed 1024-chunk to the main accumulator.
        # This double-cast perfectly emulates cuBLAS atomic summing of Split-K chunks!
        acc += local_acc.to(tl.bfloat16)

    # 4. Final bfloat16 casting to match PyTorch F.linear boundary
    acc_bf16 = acc.to(tl.bfloat16)

    if HAS_BIAS:
        offs_n = pid_n * TILE_N + lane_id
        # Load bias directly as bfloat16
        bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(
            tl.bfloat16
        )

        # Optional: Only add bias to valid M rows to save a few cycles
        mask_m = (pid_m * TILE_M + tl.arange(0, TILE_M)) < M

        # Add them together in bfloat16 space
        acc_bf16 = tl.where(mask_m[:, None], acc_bf16 + bias_vals[None, :], acc_bf16)

    # 5. Store
    out_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    out_n = pid_n * TILE_N + lane_id
    tl.store(
        output_ptr + (out_m[:, None] * stride_cm + out_n[None, :] * stride_cn),
        acc_bf16,
        mask=(out_m[:, None] < M) & (out_n[None, :] < N),
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
    x_2d = x.view(-1, K)
    M_input, K_input = x_2d.shape

    assert (
        K_input == K
    ), f"Input K dimension ({K_input}) does not match expected K ({K})"

    # TILE_K, TILE_N = 1024, 32
    TILES_N = (
        N + tile_n - 1
    ) // tile_n  # Total number of tiles in the N dimension (global constant)
    TILES_K = (
        K + tile_k - 1
    ) // tile_k  # Total number of tiles in the K dimension (depends on weight K)

    print(f"Input M: {M_input}, K: {K}, N: {N}")
    print(f"Tile size K: {tile_k}, Tile size N: {tile_n}")
    print(f"Tiles in K dimension: {TILES_K}")
    print(f"Tiles in N dimension: {TILES_N}")
    print(f"Total tiles (K x N): {TILES_K * TILES_N}")
    print(f"Expected streams (tiles x tile_n): {TILES_K * TILES_N * tile_n}")

    expected_tiles = TILES_K * TILES_N
    expected_streams = (
        expected_tiles * tile_n
    )  # Each tile corresponds to tile_n streams, each stream decodes tile_k rows

    if initial_states.numel() != expected_streams:
        raise ValueError(
            f"Initial states count ({initial_states.numel()}) does not match expected ({expected_streams}) based on tiling config."
        )

    # --- AGGRESSIVE LOGGING ---
    print(f"\n[RANS DEBUG] Starting fused_rans_linear_triton")
    print(f"  > Logical Shapes: M={M_input}, K={K}, N={N}")
    print(f"  > Tiling Config: TILE_K={tile_k}, TILE_N={tile_n}, BLOCK_K=32")
    print(f"  > Calculated Grid: num_tiles_k={TILES_K}, num_tiles_n={TILES_N}")
    print(f"  > Expected Metadata Elements: {expected_tiles}")

    # Tensor Inspection
    print(
        f"  > Tensor 'tile_offsets': shape={tile_offsets.shape}, dtype={tile_offsets.dtype}, device={tile_offsets.device}"
    )
    print(
        f"  > Tensor 'tile_max_lens': shape={tile_max_lens.shape}, dtype={tile_max_lens.dtype}"
    )
    print(
        f"  > Tensor 'initial_states': shape={initial_states.shape}, numel={initial_states.numel()}"
    )
    print(f"  > Tensor 'mantissas': shape={mantissas.shape}, numel={mantissas.numel()}")
    print(f"  > Global N Tiles (Constant): {TILES_N}")

    # Critical Sanity Checks
    if tile_offsets.numel() != expected_tiles:
        print(
            f"  !! ERROR: tile_offsets numel ({tile_offsets.numel()}) != expected ({expected_tiles})"
        )
        print(
            f"     This means K-sharding failed in the loader. Kernel will OOB for K > 1024."
        )

    expected_mantissa_size = K * N
    if mantissas.numel() != expected_mantissa_size:
        print(
            f"  !! ERROR: mantissa numel ({mantissas.numel()}) != expected ({expected_mantissa_size})"
        )

    # 2. Preparation
    x_flat = x.view(-1, K)
    if not x_flat.is_contiguous():
        print("  > Warning: x is non-contiguous. Forcing contiguous.")
        x_flat = x_flat.contiguous()

    stride_am = x_flat.stride(0)
    stride_ak = x_flat.stride(1)
    print(f"  > Strides: stride_am={stride_am}, stride_ak={stride_ak}")

    # 3. Output Buffer
    if out is None:
        output = torch.empty((M_input, N), device=x.device, dtype=torch.bfloat16)
    else:
        output = out.view(M_input, N)

    # 4. Kernel Grid
    TILE_M = 64
    grid = (triton.cdiv(M_input, TILE_M), triton.cdiv(N, tile_n))
    print(f"  > Launching Grid: {grid}")

    # 5. Launch
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
        num_stages=2,
        num_warps=4,
    )

    print(f"[RANS DEBUG] Kernel Finished\n")
    return output.view(*x.shape[:-1], N)


# def fused_rans_linear_triton(
#     x,
#     compressed_data,
#     initial_states,
#     tables,
#     slot_map,
#     weight_shape,
#     tile_offsets,
#     tile_max_lens,
#     tile_k,
#     tile_n,
#     mantissas,
#     bias=None,
# ):
#     # 1. DERIVE DIMENSIONS FROM REALITY, NOT METADATA
#     # K is the feature dimension of the input tensor
#     # K = x.shape[-1]
#     # M is the total number of tokens (works for 2D or 3D)
#     # K, N = weight_shape
#     K = weight_shape[0]
#     N = weight_shape[1]
#     M = x.numel() // K
#     TILE_M = 16

#     num_tiles_n = triton.cdiv(N, tile_n)
#     num_tiles_k = triton.cdiv(K, tile_k)
#     num_tiles_m = triton.cdiv(M, TILE_M)

#     grid = (num_tiles_m, num_tiles_n)

#     output = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

#     print("K:", K)
#     print("N:", N)
#     print("M:", M)

#     print("TILE_K:", tile_k)
#     print("TILE_N:", tile_n)
#     print("TILE_M:", TILE_M)

#     print("Num Tiles K:", num_tiles_k)
#     print("Num Tiles N:", num_tiles_n)
#     print("Num Tiles M:", num_tiles_m)

#     print("Input shape:", x.shape)
#     print("Mantissas shape:", mantissas.shape)
#     print("Weight shape (K,N):", weight_shape)
#     print("Output shape:", output.shape)

#     print("Tile offsets:", tile_offsets.shape)
#     print("Tile offset[0]", tile_offsets[0])
#     print("Tile max lens:", tile_max_lens.shape)

#     # 5. Launch
#     fused_rans_matmul_kernel_with_bias[grid](
#         x_ptr=x,
#         compressed_data=compressed_data,
#         tile_offsets=tile_offsets,
#         tile_max_lens=tile_max_lens,
#         initial_states=initial_states,
#         mantissas_ptr=mantissas,
#         bias_ptr=bias if bias is not None else x,
#         output_ptr=output,
#         slot_map=slot_map,
#         tables=tables,
#         M=M,
#         N=N,
#         K=K,
#         NUM_TILES_N=num_tiles_n,
#         stride_am=x.stride(0),
#         stride_ak=x.stride(1),
#         stride_cm=output.stride(0),
#         stride_cn=output.stride(1),
#         TILE_M=TILE_M,
#         TILE_N=tile_n,
#         TILE_K=tile_k,
#         BLOCK_K=32,
#         PROB_BITS=12,
#         PROB_MASK=4095,
#         RANS_L=1 << 16,
#         HAS_BIAS=bias is not None,
#         num_stages=1,
#         num_warps=1,
#     )

#     return output.view(M, N)
