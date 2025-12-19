#pragma once
#include "rans.cuh"
#include <cuda.h>

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

    state_t state = RansConfig::rans_l;

    // Find start of data for thread
    //io_t *stream = &ctx.output[gid * ctx.stream_capacity];
    // This can be removed now

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

	// Pre compute x_max base
	const state_t x_max_base = ((RansConfig::rans_l >> RansConfig::prob_bits)
					<< RansConfig::io_bits);

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
		state_t x_max = x_max_base * freq;

        while (state >= x_max) {
            if (out_idx < ctx.stream_capacity) {
                // Write state to stream
                // stream[out_idx] =
                //     static_cast<io_t>(state & RansConfig::io_mask);
                
                // Strided write to global memory 
                ctx.output[out_idx * ctx.num_streams + gid] =
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
    if (gid >= ctx.num_streams) return;


	uint32_t stream_size = ctx.input_sizes[gid];
	int32_t stream_offset = (int32_t)stream_size - 1;

    uint32_t out_idx = gid;
    uint32_t out_stride = ctx.num_streams;

    uint32_t stride = ctx.num_streams;
    // Points to the *current* byte to be read
    const io_t* input_ptr = ctx.input + (stream_offset * stride) + gid;
    // We also keep the base pointer to check for underflow (safety)
    const io_t* input_base = ctx.input + gid;

    state_t state = ctx.initial_states[gid];

	// Decompression loop
    for(uint32_t i = 0; i < ctx.output_size; i++) {
        uint32_t slot = state & RansConfig::prob_mask;

        symbol_t s = s_slot_map[slot];

        // Write Output
        ctx.output[out_idx] = s;
        out_idx += out_stride;

        // NOTE: Make sure your host code sends RansSymInfoPacked or equivalent
        sym_info_t info = s_sym_info[s];

        // Update State
        state = info.freq * (state >> RansConfig::prob_bits) + (slot - info.cdf);

        // Original renormalization loop
        // while (state < RansConfig::rans_l && stream_offset >= 0) {
        //     // Read from input stream (Strided Access)
		// 	io_t value = ctx.input[stream_offset * ctx.num_streams + gid];
        //     stream_offset--;
        //     state = (state << RansConfig::io_bits) | value;
        // }

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
