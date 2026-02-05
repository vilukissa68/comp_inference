#pragma once
#include "rans.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/default_gemm.h>

template <typename RansConfig, int TILE_K, int TILE_N> struct RansWeightLoader {
    using FragmentB = cutlass::Array<cutlass::half_t, TILE_K * TILE_N / 32>;
    using symbol_t = typename RansConfig::symbol_t;
    using state_t = typename RansConfig::state_t;
    using io_t = typename RansConfig::io_t;
    using sym_info_t = typename RansConfig::sym_info_t;

    struct Params {
        const io_t *stream_ptr;
        const uint8_t *mantissa_ptr;
        const io_t *input_base;
        uint32_t stride;
        uint32_t *initial_states;
    };

    struct SharedStorage {
        sym_info_t s_sym_info[256];
        uint8_t s_slot_map[RansConfig::prob_size];
    };
    __device__ static __forceinline__ void
    renormalize_warp_sync(uint32_t &state, const uint8_t *&input_ptr,
                          uint32_t stride, const uint8_t *input_base) {

// Standard rANS 8-bit renormalization requires max 2 reads
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            // Warp sync to determine which lanes need renormalization
            uint32_t mask =
                __ballot_sync(0xFFFFFFFF, state < RansConfig::rans_l);
            if (mask == 0) {
                // exit early if the whole warp is satisfied
                break;
            }
            // Active threads fetch from their interleaved pointer
            if (state < RansConfig::rans_l && input_ptr >= input_base) {
                state = (state << RansConfig::io_bits) | (*input_ptr);
                input_ptr -= stride;
            }
        }
    }

    __device__ static __forceinline__ uint16_t
    assemble_bf16(uint8_t exp_symbol, uint8_t raw_man_byte) {
        /* Reconstruct bf6 from exponent and mantissa = SEEEEEEEEMMMMMMM */

        // Extract sign bit
        uint16_t sign = (uint16_t)(raw_man_byte & 0x80) << 8;

        // Shift exponent
        uint16_t exponent = (uint16_t)(exp_symbol & 0xFF) << 7;

        // Mask first bit of mantissa
        uint16_t mantissa = (uint16_t)(raw_man_byte & 0x7F);

        // Combine
        return sign | exponent | mantissa;
    }

    __device__ void
    load_and_decompress(FragmentB &frag, Params &p,
                        typename RansConfig::SharedStorage &storage) {

        // Use threadIdx.x to get the local thread ID within the block
        int tid = threadIdx.x;

        // Initalize state
        state_t state = p.initial_states[tid];
        const io_t *ptr = p.stream_ptr + (tid * p.stride);
        const io_t *ptr_man = p.mantissa_ptr + (tid * p.stride);

#pragma unroll
        for (int i = 0; i < frag.size(); ++i) {
            // rANS for exponent

            uint32_t slot = state & RansConfig::prob_mask;

            // Get symbol from slot
            symbol_t exp_s = storage.s_slot_map[slot];

            // Get symbol info
            sym_info_t info = storage.s_sym_info[exp_s];

            // Update state
            state = info.freq * (state >> RansConfig::prob_bits) +
                    (slot - info.cdf);

            // Warp coalesced renormalization
            renormalize_warp_sync(state, ptr, p.stride, p.input_base);

            // Read mantissa byte
            uint8_t man_byte = *ptr_man; // TODO: Fix this
            ptr_man -= p.stride;

            // Assemble bf16 from exponent symbol and mantissa byte
            uint16_t bf16_bits = assemble_bf16(exp_s, man_byte);
            frag[i] = reinterpret_cast<cutlass::half_t &>(bf16_bits);
        }
    }
};
