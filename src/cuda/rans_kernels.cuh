#include "rans.cuh"
#include <cuda.h>

template <typename RansConfig>
__global__ void rand_compress_kernel(RansEncoderCtx<RansConfig> ctx) {

    using symbol_t = typename RansConfig::symbol_t;
    using state_t = typename RansConfig::state_t;
    using io_t = typename RansConfig::io_t;

    uint32_t gid = blockGid.x * blockDim.x + threadGid.x;
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
        ctx.steam_sizes[gid] = 0;
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
