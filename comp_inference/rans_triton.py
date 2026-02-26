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
def rans_decompress_tiled_kernel_triton_ilp2(
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
    # 1. Map Coordinates
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    tile_id = pid_k * num_tiles_n + pid_n

    lane_id = tl.arange(0, TILE_N)
    global_col = pid_n * TILE_N + lane_id
    n_mask = global_col < total_width

    # 2. ILP2 State Loading (Doubled Streams)
    # Layout: [Stream A0...AN, Stream B0...BN]
    tile_stream_base = tile_id * (TILE_N * 2)
    state_A = tl.load(
        initial_states + tile_stream_base + lane_id, mask=n_mask, other=0
    ).to(tl.uint32)
    state_B = tl.load(
        initial_states + tile_stream_base + TILE_N + lane_id, mask=n_mask, other=0
    ).to(tl.uint32)

    # 3. Metadata and Bitstream Pointers
    tile_start = tl.load(tile_offsets + tile_id).to(tl.int64)
    tile_depth = tl.load(tile_max_lens + tile_id).to(tl.int64)

    current_byte_row_A = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)
    current_byte_row_B = tl.full((TILE_N,), tile_depth - 1, dtype=tl.int64)

    # Bitstream Stride is 2x Tile Width
    ROW_STRIDE = TILE_N * 2
    tile_data_base_A = compressed_data + tile_start + lane_id
    tile_data_base_B = compressed_data + tile_start + TILE_N + lane_id

    start_row = pid_k * TILE_K
    HALF_TILE_K = TILE_K // 2

    # 4. Decoding Loop (ILP2)
    for i in range(HALF_TILE_K):
        # Calculate Row Indices for Top and Bottom halves
        row_idx_A = start_row + i
        row_idx_B = start_row + i + HALF_TILE_K

        mask_A = n_mask & (row_idx_A < total_height)
        mask_B = n_mask & (row_idx_B < total_height)

        # --- DECODE PHASE ---
        slot_A = state_A & PROB_MASK
        slot_B = state_B & PROB_MASK

        sym_A = tl.load(slot_map + slot_A, mask=mask_A, other=0)
        sym_B = tl.load(slot_map + slot_B, mask=mask_B, other=0)

        # --- STORE PHASE ---
        tl.store(output + row_idx_A * total_width + global_col, sym_A, mask=mask_A)
        tl.store(output + row_idx_B * total_width + global_col, sym_B, mask=mask_B)

        # --- STATE UPDATE PHASE ---
        pk_A = tl.load(tables + sym_A.to(tl.int32), mask=mask_A, other=0)
        pk_B = tl.load(tables + sym_B.to(tl.int32), mask=mask_B, other=0)

        state_A = (pk_A & 0xFFFF) * (state_A >> PROB_BITS) + (slot_A - (pk_A >> 16))
        state_B = (pk_B & 0xFFFF) * (state_B >> PROB_BITS) + (slot_B - (pk_B >> 16))

        # --- RENORMALIZATION PHASE (A) ---
        for _ in range(2):
            renorm_A = (state_A < RANS_L) & mask_A & (current_byte_row_A >= 0)
            ptr_A = tile_data_base_A + (current_byte_row_A * ROW_STRIDE)
            val_A = tl.load(ptr_A, mask=renorm_A, other=0).to(tl.uint32)
            state_A = tl.where(renorm_A, (state_A << 8) | val_A, state_A)
            current_byte_row_A -= tl.where(renorm_A, 1, 0)

        # --- RENORMALIZATION PHASE (B) ---
        for _ in range(2):
            renorm_B = (state_B < RANS_L) & mask_B & (current_byte_row_B >= 0)
            ptr_B = tile_data_base_B + (current_byte_row_B * ROW_STRIDE)
            val_B = tl.load(ptr_B, mask=renorm_B, other=0).to(tl.uint32)
            state_B = tl.where(renorm_B, (state_B << 8) | val_B, state_B)
            current_byte_row_B -= tl.where(renorm_B, 1, 0)


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

    # Validate ILP2 state count
    expected_streams = (num_tiles_n * num_tiles_k) * tile_n * 2
    if initial_states.numel() != expected_streams:
        raise ValueError(f"Initial states size mismatch. Expected {expected_streams}")

    grid = (num_tiles_n, num_tiles_k)

    rans_decompress_tiled_kernel_triton_ilp2[grid](
        compressed_data=compressed_streams,
        tile_offsets=tile_offsets,
        tile_max_lens=tile_max_lens,
        initial_states=initial_states,
        output=output,
        slot_map=slot_map,
        tables=tables,
        num_tiles_n=num_tiles_n,
        total_height=K,
        total_width=N,
        TILE_K=tile_k,
        TILE_N=tile_n,
        PROB_BITS=12,
        PROB_MASK=4095,
        RANS_L=1 << 16,
        num_stages=1,
        num_warps=4,
    )

    return output


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
        num_warps=4,
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


def rans_decomp_triton_tiled2(
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
        num_warps=4,
    )

    return output
