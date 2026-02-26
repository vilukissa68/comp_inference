#pragma once
#include "rans.cuh"
#include <cuda.h>

#define CHECKPOINT_INTERVAL 128

template <typename RansConfig>
__global__ void
rans_compress_kernel_tiled_ilp2(RansTiledEncoderCtx<RansConfig> ctx) {
    using symbol_t = typename RansConfig::symbol_t;
    using state_t = typename RansConfig::state_t;
    using io_t = typename RansConfig::io_t;
    using sym_info_t = typename RansConfig::sym_info_t;

    // 1. Load symbol info table into shared memory
    __shared__ sym_info_t s_sym_info[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        s_sym_info[i] = ctx.tables.sym_info[i];
    }
    __syncthreads();

    uint32_t tile_idx_n = blockIdx.x;
    uint32_t tile_idx_k = blockIdx.y;

    // Thread handles ONE column, but outputs TWO streams (Top and Bottom
    // halves)
    uint32_t local_col = threadIdx.x;
    uint32_t global_col = tile_idx_n * ctx.tile_width + local_col;

    if (global_col >= ctx.total_width)
        return;

    // --- ILP MAPPING ---
    // Every tile now produces (tile_width * 2) streams.
    // We group all Top Half streams together, then all Bottom Half streams
    // together to ensure coalesced memory reads in the Triton decompressor.
    uint32_t base_stream_idx =
        (tile_idx_k * ctx.num_tiles_n + tile_idx_n) * (ctx.tile_width * 2);

    uint32_t global_stream_id_A = base_stream_idx + local_col; // Top Half
    uint32_t global_stream_id_B =
        base_stream_idx + ctx.tile_width + local_col; // Bottom Half

    state_t state_A = RansConfig::rans_l;
    state_t state_B = RansConfig::rans_l;

    uint32_t out_idx_A = 0;
    uint32_t out_idx_B = 0;
    uint32_t values_encoded_A = 0;
    uint32_t values_encoded_B = 0;

    uint32_t start_row = tile_idx_k * ctx.tile_height;
    uint32_t half_tile = ctx.tile_height / 2;
    const state_t x_max_base =
        ((RansConfig::rans_l >> RansConfig::prob_bits) << RansConfig::io_bits);

    // --- COMPUTE PHASE B: Bottom Half (Rows: tile_height-1 down to half_tile)
    // ---
    for (int i = (int)ctx.tile_height - 1; i >= (int)half_tile; --i) {
        uint32_t row = start_row + i;
        if (row >= ctx.total_height)
            continue;

        symbol_t sym = ctx.symbols[row * ctx.total_width + global_col];
        auto info = s_sym_info[sym];
        state_t x_max = x_max_base * info.freq;

        values_encoded_B++;
        while (state_B >= x_max) {
            if (out_idx_B < ctx.stream_capacity) {
                // ctx.num_streams here MUST be the doubled total!
                ctx.output[out_idx_B * ctx.num_streams + global_stream_id_B] =
                    (io_t)(state_B & RansConfig::io_mask);
                out_idx_B++;
            } else {
                ctx.success = false;
                break;
            }
            state_B >>= RansConfig::io_bits;
        }
        state_B = ((state_B / info.freq) << RansConfig::prob_bits) +
                  (state_B % info.freq) + info.cdf;
    }

    // --- COMPUTE PHASE A: Top Half (Rows: half_tile-1 down to 0) ---
    for (int i = (int)half_tile - 1; i >= 0; --i) {
        uint32_t row = start_row + i;
        if (row >= ctx.total_height)
            continue;

        symbol_t sym = ctx.symbols[row * ctx.total_width + global_col];
        auto info = s_sym_info[sym];
        state_t x_max = x_max_base * info.freq;

        values_encoded_A++;
        while (state_A >= x_max) {
            if (out_idx_A < ctx.stream_capacity) {
                ctx.output[out_idx_A * ctx.num_streams + global_stream_id_A] =
                    (io_t)(state_A & RansConfig::io_mask);
                out_idx_A++;
            } else {
                ctx.success = false;
                break;
            }
            state_A >>= RansConfig::io_bits;
        }
        state_A = ((state_A / info.freq) << RansConfig::prob_bits) +
                  (state_A % info.freq) + info.cdf;
    }

    // --- WRITEBACK PHASE ---
    ctx.final_states[global_stream_id_A] = state_A;
    ctx.stream_sizes[global_stream_id_A] = out_idx_A;
    ctx.values_encoded[global_stream_id_A] = values_encoded_A;

    ctx.final_states[global_stream_id_B] = state_B;
    ctx.stream_sizes[global_stream_id_B] = out_idx_B;
    ctx.values_encoded[global_stream_id_B] = values_encoded_B;
}

template <typename RansConfig>
__global__ void
rans_compress_kernel_tiled_ilp(RansTiledEncoderCtx<RansConfig> ctx) {
    using symbol_t = typename RansConfig::symbol_t;
    using state_t = typename RansConfig::state_t;
    using io_t = typename RansConfig::io_t;
    using sym_info_t = typename RansConfig::sym_info_t;

    // Load symbol info table into shared memory
    __shared__ sym_info_t s_sym_info[256];

    // Cooperative loading using blockDim.x (32 threads)
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        s_sym_info[i] = ctx.tables.sym_info[i];
    }
    __syncthreads();

    uint32_t tile_idx_n = blockIdx.x;
    uint32_t tile_idx_k = blockIdx.y;

    // --- ILP MAPPING ---
    // Thread handles two columns offset by blockDim.x (e.g., 0 and 32)
    uint32_t local_col_A = threadIdx.x;
    uint32_t local_col_B = threadIdx.x + blockDim.x;

    uint32_t N = ctx.total_width;
    uint32_t K = ctx.total_height;

    uint32_t global_stream_id_A =
        (tile_idx_k * ctx.num_tiles_n + tile_idx_n) * ctx.tile_width +
        local_col_A;
    uint32_t global_stream_id_B =
        (tile_idx_k * ctx.num_tiles_n + tile_idx_n) * ctx.tile_width +
        local_col_B;

    uint32_t global_col_A = tile_idx_n * ctx.tile_width + local_col_A;
    uint32_t global_col_B = tile_idx_n * ctx.tile_width + local_col_B;

    bool valid_A = (global_stream_id_A < ctx.num_streams) && (global_col_A < N);
    bool valid_B = (global_stream_id_B < ctx.num_streams) && (global_col_B < N);

    // If both are out of bounds, bail early
    if (!valid_A && !valid_B)
        return;

    state_t state_A = RansConfig::rans_l;
    state_t state_B = RansConfig::rans_l;

    uint32_t out_idx_A = 0;
    uint32_t out_idx_B = 0;
    uint32_t values_encoded_A = 0;
    uint32_t values_encoded_B = 0;

    uint32_t start_row = tile_idx_k * ctx.tile_height;
    const state_t x_max_base =
        ((RansConfig::rans_l >> RansConfig::prob_bits) << RansConfig::io_bits);

    for (int i = (int)ctx.tile_height - 1; i >= 0; --i) {
        uint32_t row = start_row + i;
        if (row >= K)
            continue;

        // --- MEMORY FETCH PHASE ---
        // Both threads fetch simultaneously (Coalesced reads!)
        symbol_t sym_A = valid_A ? ctx.symbols[row * N + global_col_A] : 0;
        symbol_t sym_B = valid_B ? ctx.symbols[row * N + global_col_B] : 0;

        auto info_A = s_sym_info[sym_A];
        auto info_B = s_sym_info[sym_B];

        state_t x_max_A = x_max_base * info_A.freq;
        state_t x_max_B = x_max_base * info_B.freq;

        // --- COMPUTE PHASE A ---
        if (valid_A) {
            values_encoded_A++;
            while (state_A >= x_max_A) {
                if (out_idx_A < ctx.stream_capacity) {
                    ctx.output[out_idx_A * ctx.num_streams +
                               global_stream_id_A] =
                        (io_t)(state_A & RansConfig::io_mask);
                    out_idx_A++;
                } else {
                    ctx.success = false;
                    break;
                }
                state_A >>= RansConfig::io_bits;
            }
            state_A = ((state_A / info_A.freq) << RansConfig::prob_bits) +
                      (state_A % info_A.freq) + info_A.cdf;
        }

        // --- COMPUTE PHASE B ---
        // (Compiler interleaves this with Phase A)
        if (valid_B) {
            values_encoded_B++;
            while (state_B >= x_max_B) {
                if (out_idx_B < ctx.stream_capacity) {
                    ctx.output[out_idx_B * ctx.num_streams +
                               global_stream_id_B] =
                        (io_t)(state_B & RansConfig::io_mask);
                    out_idx_B++;
                } else {
                    ctx.success = false;
                    break;
                }
                state_B >>= RansConfig::io_bits;
            }
            state_B = ((state_B / info_B.freq) << RansConfig::prob_bits) +
                      (state_B % info_B.freq) + info_B.cdf;
        }
    }

    // --- WRITEBACK PHASE ---
    if (valid_A) {
        ctx.final_states[global_stream_id_A] = state_A;
        ctx.stream_sizes[global_stream_id_A] = out_idx_A;
        ctx.values_encoded[global_stream_id_A] = values_encoded_A;
    }
    if (valid_B) {
        ctx.final_states[global_stream_id_B] = state_B;
        ctx.stream_sizes[global_stream_id_B] = out_idx_B;
        ctx.values_encoded[global_stream_id_B] = values_encoded_B;
    }
}

template <typename RansConfig>
__global__ void
rans_compress_kernel_tiled(RansTiledEncoderCtx<RansConfig> ctx) {
    using symbol_t = typename RansConfig::symbol_t;
    using state_t = typename RansConfig::state_t;
    using io_t = typename RansConfig::io_t;
    using sym_info_t = typename RansConfig::sym_info_t;

    // Load symbol info table into shared memory for fast access
    __shared__ sym_info_t s_sym_info[256];

    // Cooperative loading (all threads help load the table)
    for (int i = threadIdx.x; i < 256; i += ctx.tile_width) {
        s_sym_info[i] = ctx.tables.sym_info[i];
    }

    __syncthreads();

    // Map GID to 2D tile grid
    uint32_t tile_idx_n = blockIdx.x; // Tile x coordinate
    uint32_t tile_idx_k = blockIdx.y; // Tile y coordinate
    uint32_t local_col = threadIdx.x; // Width of tile
    uint32_t values_encoded = 0;
    uint32_t N = ctx.total_width;
    uint32_t K = ctx.total_height;
    uint32_t global_stream_id =
        (tile_idx_k * ctx.num_tiles_n + tile_idx_n) * ctx.tile_width +
        local_col;

    if (global_stream_id >= ctx.num_streams)
        return;

    uint32_t global_col = tile_idx_n * ctx.tile_width + local_col;
    if (global_col >= ctx.total_width)
        return;

    state_t state = RansConfig::rans_l;
    uint32_t out_idx = 0;
    uint32_t start_row = tile_idx_k * ctx.tile_height; // Start from the
    const state_t x_max_base =
        ((RansConfig::rans_l >> RansConfig::prob_bits) << RansConfig::io_bits);

    for (int i = (int)ctx.tile_height - 1; i >= 0; --i) {
        uint32_t row = start_row + i;
        if (row >= K) {
            continue;
        }

        symbol_t symbol = ctx.symbols[row * N + global_col];
        values_encoded++;
        auto info = s_sym_info[symbol];

        // Renormalization
        state_t x_max = x_max_base * info.freq;
        while (state >= x_max) {
            if (out_idx < ctx.stream_capacity) { // Guard against overflow
                ctx.output[out_idx * ctx.num_streams + global_stream_id] =
                    (io_t)(state & RansConfig::io_mask);
                out_idx++;
            } else {
                ctx.success = false; // Mark failure
                break;
            }
            state >>= RansConfig::io_bits;
        }

        // Update state
        state = ((state / info.freq) << RansConfig::prob_bits) +
                (state % info.freq) + info.cdf;
    }

    // Record the actual work done for C++ post-processing
    ctx.final_states[global_stream_id] = state;
    ctx.stream_sizes[global_stream_id] = out_idx;
    ctx.values_encoded[global_stream_id] = values_encoded;
}

template <typename RansConfig>
__global__ void rans_compress_kernel(RansEncoderCtx<RansConfig> ctx) {

    using symbol_t = typename RansConfig::symbol_t;
    using state_t = typename RansConfig::state_t;
    using io_t = typename RansConfig::io_t;
    using sym_info_t = typename RansConfig::sym_info_t;

    // Load symbol info table into shared memory for fast access
    __shared__ sym_info_t s_sym_info[256];

    // Cooperative loading (all threads help load the table)
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        s_sym_info[i] = ctx.tables.sym_info[i];
    }

    __syncthreads();

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= ctx.num_streams) {
        return;
    }

    // 1. Map GID to the 2D grid
    uint32_t col = gid % ctx.input_width; // ctx.stride is the tensor width
    uint32_t segment_id = gid / ctx.input_width; // Which "chunk" vertically

    // 2. Each stream processes exactly B symbols (or fewer if at the very
    uint32_t B = 512; // This should be passed in ctx uint32_t
    uint32_t syms_left = B;

    // Start with initial state
    state_t state = RansConfig::rans_l;

    if (syms_left == 0) {
        // NOTE: This should be avoided with careful thread launching
        ctx.final_states[gid] = state;
        ctx.output_sizes[gid] = 0;
        return;
    }

    // 3. Start at the bottom of the segment and move up
    // The absolute index of the LAST symbol in this stream's segment:
    uint32_t segment_start_row = segment_id * B;
    uint32_t syms_in_this_segment =
        (segment_start_row < ctx.input_height)
            ? min(B, ctx.input_height - segment_start_row)
            : 0;

    // 2. Map idx to the LAST valid row of this segment
    uint32_t last_row_in_seg = segment_start_row + syms_in_this_segment - 1;
    uint32_t idx = (last_row_in_seg * ctx.input_width) + col;

    // Number of encoded symbols
    uint32_t out_idx = 0;

    // Pre compute x_max base
    const state_t x_max_base =
        ((RansConfig::rans_l >> RansConfig::prob_bits) << RansConfig::io_bits);

    // Compression loop
    for (uint32_t i = 0; i < syms_in_this_segment; ++i) {

        // Checkpointing
        if (i % CHECKPOINT_INTERVAL == 0) {
            uint32_t tile_k_idx = i / CHECKPOINT_INTERVAL;
            // [32-bit State | 32-bit Out_Idx (Offset)]
            RansCheckpoint rc;
            rc.state = state;
            rc.offset = out_idx;
            // Store in a strided way so each stream's checkpoints are
            // contiguous
            ctx.checkpoints[tile_k_idx * ctx.num_streams + gid] = rc;
        }

        // Read next symbol
        symbol_t s = ctx.symbols[idx];

        // Move idx pointer to next symbol (interleaven streams)
        // idx -= ctx.num_streams;
        idx -= ctx.input_width;

        // Load symbol probability info
        auto info = s_sym_info[s];
        state_t freq = info.freq;
        state_t start = info.cdf;

        // Renormalization
        state_t x_max = x_max_base * freq;
        while (state >= x_max) {
            if (out_idx < ctx.stream_capacity) {
                // Strided write to global memory
                ctx.output[out_idx * ctx.num_streams + gid] =
                    static_cast<io_t>(state & RansConfig::io_mask);
                out_idx++;
            } else {
                ctx.success = false;
                break;
            }

            state >>= RansConfig::io_bits;
        }

        // Update state / encode symbol
        // C(x,s) = (floor(x / f[s]) << n) + (x % f[s]) + CDF[s]
        // NOTE: The div here is an annoying cost we can't really avoid
        state =
            ((state / freq) << RansConfig::prob_bits) + (state % freq) + start;
    }

    // Encoding done
    // Save final state needed for decoding
    ctx.final_states[gid] = state;

    // Save number of values written
    ctx.output_sizes[gid] = out_idx;
}

// template <typename RansConfig>
// __global__ void rans_compress_kernel(RansEncoderCtx<RansConfig> ctx) {

//     using symbol_t = typename RansConfig::symbol_t;
//     using state_t = typename RansConfig::state_t;
//     using io_t = typename RansConfig::io_t;
//     using sym_info_t = typename RansConfig::sym_info_t;

//     // Load symbol info table into shared memory for fast access
//     __shared__ sym_info_t s_sym_info[256];

//     // Cooperative loading (all threads help load the table)
//     for (int i = threadIdx.x; i < 256; i += blockDim.x) {
//         s_sym_info[i] = ctx.tables.sym_info[i];
//     }

//     __syncthreads();

//     uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (gid >= ctx.num_streams) {
//         return;
//     }

//     state_t state = RansConfig::rans_l;

//     // Find start of data for thread
//     // io_t *stream = &ctx.output[gid * ctx.stream_capacity];
//     // This can be removed now

//     // Number of encoded symbols
//     uint32_t out_idx = 0;

//     // How many symbols to process
//     uint32_t syms_left =
//         (ctx.input_size + ctx.num_streams - 1 - gid) / ctx.num_streams;

//     if (syms_left == 0) {
//         // Too many threads launched
//         // NOTE: This should be avoided with careful thread launching
//         ctx.final_states[gid] = state;
//         ctx.output_sizes[gid] = 0;
//         return;
//     }

//     // Start iteration from the end to have data in correct order
//     uint32_t idx = gid + (syms_left - 1) * ctx.num_streams;

//     // Pre compute x_max base
//     const state_t x_max_base =
//         ((RansConfig::rans_l >> RansConfig::prob_bits) <<
//         RansConfig::io_bits);

//     // Compression loop
//     for (uint32_t i = 0; i < syms_left; ++i) {

//         // Checkpointing
//         if (i % CHECKPOINT_INTERVAL == 0) {
//             uint32_t tile_k_idx = i / CHECKPOINT_INTERVAL;
//             // [32-bit State | 32-bit Out_Idx (Offset)]
//             RansCheckpoint rc;
//             rc.state = state;
//             rc.offset = out_idx;
//             // Store in a strided way so each stream's checkpoints are
//             // contiguous
//             ctx.checkpoints[tile_k_idx * ctx.num_streams + gid] = rc;
//         }

//         // Read next symbol
//         symbol_t s = ctx.symbols[idx];

//         // Move idx pointer to next symbol (interleaven streams)
//         idx -= ctx.num_streams;

//         // Load symbol probability info
//         auto info = ctx.tables.sym_info[s];
//         state_t freq = info.freq;
//         state_t start = info.cdf;

//         // Renormalization
//         state_t x_max = x_max_base * freq;

//         while (state >= x_max) {
//             if (out_idx < ctx.stream_capacity) {
//                 // Write state to stream
//                 // stream[out_idx] =
//                 //     static_cast<io_t>(state & RansConfig::io_mask);

//                 // Strided write to global memory
//                 ctx.output[out_idx * ctx.num_streams + gid] =
//                     static_cast<io_t>(state & RansConfig::io_mask);
//                 out_idx++;
//             } else {
//                 ctx.success = false;
//                 break;
//             }

//             state >>= RansConfig::io_bits;
//         }

//         // Update state / encode symbol
//         // C(x,s) = (floor(x / f[s]) << n) + (x % f[s]) + CDF[s]
//         // NOTE: The div here is an annoying cost we can't really avoid
//         state =
//             ((state / freq) << RansConfig::prob_bits) + (state % freq) +
//             start;
//     }

//     // Encoding done
//     // Save final state needed for decoding
//     ctx.final_states[gid] = state;

//     // Save number of values written
//     ctx.output_sizes[gid] = out_idx;
// }

template <typename RansConfig>
__global__ void rans_decompress_kernel(RansDecoderCtx<RansConfig> ctx) {
    using symbol_t = typename RansConfig::symbol_t;
    using state_t = typename RansConfig::state_t;
    using io_t = typename RansConfig::io_t;
    using sym_info_t = typename RansConfig::sym_info_t;

    // Total shared mem usage ~5KB (Safe for 100% occupancy)
    __shared__ sym_info_t s_sym_info[256];
    __shared__ symbol_t s_slot_map[RansConfig::prob_scale];

    // Cooperative loading: All threads in the block help load the tables
    int tid = threadIdx.x;

    // Load Symbol Info Table
    for (int i = tid; i < 256; i += blockDim.x) {
        s_sym_info[i] = ctx.tables.sym_info[i];
    }

    // Load Slot Mapping Table
    for (int i = tid; i < RansConfig::prob_scale; i += blockDim.x) {
        s_slot_map[i] = ctx.tables.slot_to_sym[i];
    }

    // Ensure all data is loaded before proceeding
    __syncthreads();

    // Setup
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= ctx.num_streams)
        return;

    uint32_t stream_size = ctx.input_sizes[gid];
    int32_t stream_offset = (int32_t)stream_size - 1;

    uint32_t out_idx = gid;
    const uint32_t out_stride = ctx.num_streams;

    const uint32_t stride = ctx.num_streams;
    // Points to the *current* byte to be read
    const io_t *input_ptr = ctx.input + (stream_offset * stride) + gid;
    // We also keep the base pointer to check for underflow (safety)
    const io_t *input_base = ctx.input + gid;

    state_t state = ctx.initial_states[gid];

    // Decompression loop
    for (uint32_t i = 0; i < ctx.output_size; i++) {
        uint32_t slot = state & RansConfig::prob_mask;

        symbol_t s = s_slot_map[slot];

        // Write Output
        ctx.output[out_idx] = s;
        out_idx += out_stride;

        // NOTE: Make sure your host code sends RansSymInfoPacked or
        // equivalent
        sym_info_t info = s_sym_info[s];

        // Update State
        state =
            info.freq * (state >> RansConfig::prob_bits) + (slot - info.cdf);

        // NOTE: For int8 IO we know that unrolling twice is enough
        // Unrolled renormalization for up to 2 reads
        if (state < RansConfig::rans_l) {
            // Safety check: ensure we don't read before start of stream
            if (input_ptr >= input_base) {
                state = (state << RansConfig::io_bits) | *input_ptr;
                input_ptr -= stride; // Move pointer back by 1 row
            }

            // Second byte check
            if (state < RansConfig::rans_l) {
                if (input_ptr >= input_base) {
                    state = (state << RansConfig::io_bits) | *input_ptr;
                    input_ptr -= stride;
                }
            }
        }
    }
}
