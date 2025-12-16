#include "rans.cuh"
#include <cuda.h>

template <typename RansConfig>
__global__ void rans_compress_kernel(RansEncoderCtx<RansConfig> ctx) {

    using symbol_t = typename RansConfig::symbol_t;
    using state_t = typename RansConfig::state_t;
    using io_t = typename RansConfig::io_t;

	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= ctx.num_streams) {
        return;
    }

    state_t state = RansConfig::rans_l;

    // Find start of data for thread
    io_t *stream = &ctx.output[gid * ctx.stream_capacity];

    // Number of encoded symbols
    uint32_t out_idx = 0;

    // How many symbols to process
    uint32_t syms_left =
        (ctx.input_size + ctx.num_streams - 1 - gid) / ctx.num_streams;

    if (syms_left == 0) {
        // Too many threads launched
        // NOTE: This should be avoided with careful thread launching
        ctx.final_states[gid] = state;
        ctx.output_sizes[gid] = 0;
        return;
    }

    // Start iteration from the end to have data in correct order
    uint32_t idx = gid + (syms_left - 1) * ctx.num_streams;

    // Compression loop
    for (uint32_t i = 0; i < syms_left; ++i) {
        // Read next symbol
        symbol_t s = ctx.symbols[idx];

        // Move idx pointer to next symbol (interleaven streams)
        idx -= ctx.num_streams;

        // Load symbol probability info
        auto info = ctx.tables.sym_info[s];
        state_t freq = info.freq;
        state_t start = info.cdf;

        // Renormalization
        state_t x_max = ((RansConfig::rans_l >> RansConfig::prob_bits)
                         << RansConfig::io_bits) *
                        freq;

        while (state >= x_max) {
            if (out_idx < ctx.stream_capacity) {
                // Write state to stream
                stream[out_idx] =
                    static_cast<io_t>(state & RansConfig::io_mask);
                out_idx++;
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

template <typename RansConfig>
__global__ void rans_decompress_kernel(RansDecoderCtx<RansConfig> ctx) {
    using symbol_t = typename RansConfig::symbol_t;
    using state_t = typename RansConfig::state_t;
    using io_t = typename RansConfig::io_t;

    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= ctx.num_streams) return;

    // 1. INPUT SETUP
    const io_t* stream_base = &ctx.input[gid * ctx.stream_capacity];
    uint32_t stream_size = ctx.input_sizes[gid];
    
    // IMPORTANT: Read backwards from the end of the stream
    int32_t stream_offset = (int32_t)stream_size - 1;

    // 2. OUTPUT SETUP
    uint32_t out_idx = gid;
    uint32_t out_stride = ctx.num_streams;

    state_t state = ctx.initial_states[gid];

    for(uint32_t i = 0; i < ctx.output_size; i++) {
        uint32_t slot = state & RansConfig::prob_mask;
        symbol_t s = ctx.tables.slot_to_sym[slot];

        ctx.output[out_idx] = s;
        out_idx += out_stride;

        auto info = ctx.tables.sym_info[s];
        state = info.freq * (state >> RansConfig::prob_bits) + (slot - info.cdf);

        // Renormalize
        while (state < RansConfig::rans_l && stream_offset >= 0) {
            io_t value = stream_base[stream_offset];
            stream_offset--; // Move backwards
            state = (state << RansConfig::io_bits) | value;
        }
    }
}
