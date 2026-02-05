#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

#define CHUNK_SIZE 4096

// Pack symbol info for faster access during decoding
// NOTE: This is safe for PROB_SCALE <= 2**15
struct __align__(4) RansSymInfoPacked {
    uint16_t freq;
    uint16_t cdf;
};

// Wide symbol info for decoding
// NOTE: This is safe for PROB_SCALE > 2**15
struct __align__(8) RansSymInfoWide {
    uint32_t freq;
    uint32_t cdf;
};

// RANS type configuration struct
template <typename T_symbol, typename T_state, typename T_io, int T_prob_bits>
struct RansTraits {
    using symbol_t = T_symbol;
    using state_t = T_state;
    using io_t = T_io;

    // Probability model parameters
    static constexpr int prob_bits = T_prob_bits;
    static constexpr int prob_scale = 1 << T_prob_bits;
    static constexpr T_state prob_mask = static_cast<T_state>(prob_scale - 1);

    // Vocabulary size
    static constexpr int vocab_size = 1 << (sizeof(T_symbol) * 8);

    // I/O parameters
    static constexpr int io_bits = sizeof(T_io) * 8;
    static constexpr int io_mask = (1 << io_bits) - 1;

    // Lower bound for renormalization
    static constexpr int state_l_exp =
        (io_bits == 8) ? 16 : 24; // 2**16 for byte, 2**24 for word
    static constexpr state_t rans_l = static_cast<state_t>(1) << state_l_exp;

    // Use packed CDF and frequency info if possible
    using sym_info_t =
        typename std::conditional<(prob_bits > 16),
                                  RansSymInfoWide,  // True: Use 64-bit struct
                                  RansSymInfoPacked // False: Use 32-bit struct
                                  >::type;
};

// CDF and Frequency tables for encoding
template <typename RansConfig> struct RansTablesCore {
    using sym_info_t = typename RansConfig::sym_info_t;

    const sym_info_t __restrict__ *sym_info;
};

// CDF, Frequency, and Symbol tables for decoding
template <typename RansConfig> struct RansTablesFull {
    using symbol_t = typename RansConfig::symbol_t;
    using sym_info_t = typename RansConfig::sym_info_t;

    const sym_info_t __restrict__ *sym_info;

    // Map from slot to symbol table[PROB_SCALE]
    const symbol_t __restrict__ *slot_to_sym;
};

// Checkpoint structure for intermediate states
struct RansCheckpoint {
    uint32_t state;
    uint32_t offset;
};

template <typename RansConfig> struct RansEncoderCtx {
    using symbol_t = typename RansConfig::symbol_t;
    using state_t = typename RansConfig::state_t;
    using io_t = typename RansConfig::io_t;
    using sym_info_t = typename RansConfig::sym_info_t;

    // Was compression successful
    bool success;

    // Input data
    const symbol_t *__restrict__ symbols;
    uint32_t input_size; // This is same for each stream

    // Output buffer
    io_t *__restrict__ output;
    state_t *final_states;
    uint32_t *output_sizes;

    // Capacity of each interleaved stream
    uint32_t stream_capacity;

    // Stride between values in a stream, should match width of the tensor for
    // coalesced access in decoder
    uint32_t input_height;
    uint32_t input_width;

    // Total number of interleaved streams
    uint32_t num_streams;

    RansTablesCore<RansConfig> tables;
    RansCheckpoint *checkpoints;
};

template <typename RansConfig> struct RansTiledEncoderCtx {
    using symbol_t = typename RansConfig::symbol_t;
    using state_t = typename RansConfig::state_t;
    using io_t = typename RansConfig::io_t;
    using sym_info_t = typename RansConfig::sym_info_t;

    // --- Control & Status ---
    bool success;
    uint32_t tile_height;
    uint32_t tile_width; // Expected 32 (Warp Width)
    uint32_t num_tiles_k;
    uint32_t num_tiles_n;

    // --- Input (Weights) ---
    const symbol_t *__restrict__ symbols;
    uint32_t total_height; // K
    uint32_t total_width;  // N

    // --- Output (The Compressed Texture) ---
    // 1. The bitstream buffer where ALL tiles are packed
    io_t *__restrict__ output;

    // 2. Tile-specific starting info
    // Size: [NumTilesN, NumTilesK, TILE_N]
    // Indexed by: (tile_id * TILE_N) + local_col
    state_t *__restrict__ final_states;
    uint32_t *__restrict__ stream_sizes;
    uint32_t *__restrict__ values_encoded;

    // 3. The Map: Where each tile starts in output_bytes
    // Size: [NumTilesN, NumTilesK]
    // Note: We pass this to the Write Pass, calculated from the Analyze Pass
    // uint32_t *__restrict__ tile_offsets;
    uint32_t num_streams;
    uint32_t stream_capacity;

    // --- Table Info ---
    RansTablesCore<RansConfig> tables;
};

template <typename RansConfig> struct RansDecoderCtx {
    using symbol_t = typename RansConfig::symbol_t;
    using state_t = typename RansConfig::state_t;
    using io_t = typename RansConfig::io_t;
    using sym_info_t = typename RansConfig::sym_info_t;

    const io_t *__restrict__ input;
    const state_t *__restrict__ initial_states;

    const uint32_t *__restrict__ input_sizes;

    symbol_t *__restrict__ output;
    const uint32_t output_size;

    // Capacity of each interleaved stream
    const uint32_t stream_capacity;

    // Total number of interleaved streams
    const uint32_t num_streams;

    const RansTablesFull<RansConfig> tables;
};
