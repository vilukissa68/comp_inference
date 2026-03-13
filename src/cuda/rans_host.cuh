#pragma once
#include <algorithm>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include "rans.cuh"
#include "rans_kernels.cuh"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            throw std::runtime_error("CUDA Error");                            \
        }                                                                      \
    } while (0)

struct RansConfig8 : public RansTraits<uint8_t, uint32_t, uint8_t, 12> {
    using sym_info_t = RansSymInfoPacked;
};

template <typename Config> struct RansWorkspace {
    using symbol_t = typename Config::symbol_t;
    using io_t = typename Config::io_t;
    using sym_info_t = typename Config::sym_info_t;
    using state_t = typename Config::state_t;

    // Pointers
    sym_info_t *d_sym_info = nullptr;
    symbol_t *d_symbols = nullptr;
    io_t *d_output = nullptr;
    state_t *d_states = nullptr;
    uint32_t *d_sizes = nullptr;

    symbol_t *d_slot_map = nullptr;
    io_t *d_input = nullptr;
    state_t *d_init_states = nullptr;
    uint32_t *d_input_sizes = nullptr;
    symbol_t *d_decoded_output = nullptr;
    RansCheckpoint *d_checkpoints = nullptr;
    uint32_t *d_values_encoded = nullptr;

    // Individual Capacities (in bytes) to prevent overflow
    size_t cap_sym_info = 0;
    size_t cap_symbols = 0;
    size_t cap_output = 0;
    size_t cap_states = 0;
    size_t cap_sizes = 0;
    size_t cap_slot_map = 0;
    size_t cap_input = 0;
    size_t cap_init_states = 0;
    size_t cap_input_sizes = 0;
    size_t cap_decoded_output = 0;
    size_t cap_checkpoints = 0;
    size_t cap_values_encoded = 0;

    RansWorkspace() = default;

    void realloc_if_needed(void **ptr, size_t &current_cap,
                           size_t needed_bytes) {
        if (needed_bytes > current_cap) {
            // Safety margin: 1.1x growth to prevent frequent reallocs
            size_t alloc_size = (size_t)(needed_bytes * 1.1);

            // Align to 256 bytes for performance
            if (alloc_size % 256 != 0)
                alloc_size += (256 - (alloc_size % 256));

            if (*ptr) {
                cudaDeviceSynchronize();
                CUDA_CHECK(cudaFree(*ptr));
                *ptr = nullptr;
            }

            cudaError_t err = cudaMalloc(ptr, alloc_size);
            if (err == cudaErrorMemoryAllocation) {
                std::cerr << "!! CRITICAL OOM ERROR !!\n";
                std::cerr << "Requested: " << (double)alloc_size / (1024 * 1024)
                          << " MB\n";
                std::cerr
                    << "This is likely caused by PyTorch reserving all VRAM.\n";
                std::cerr << "Try running 'torch.cuda.empty_cache()' in Python "
                             "before this call.\n";
                throw std::runtime_error("CUDA Out of Memory in RansWorkspace");
            }
            CUDA_CHECK(err);

            current_cap = alloc_size;
        }
    }

    ~RansWorkspace() {
        if (d_sym_info)
            cudaFree(d_sym_info);
        if (d_symbols)
            cudaFree(d_symbols);
        if (d_output)
            cudaFree(d_output);
        if (d_states)
            cudaFree(d_states);
        if (d_sizes)
            cudaFree(d_sizes);
        if (d_slot_map)
            cudaFree(d_slot_map);
        if (d_input)
            cudaFree(d_input);
        if (d_init_states)
            cudaFree(d_init_states);
        if (d_input_sizes)
            cudaFree(d_input_sizes);
        if (d_decoded_output)
            cudaFree(d_decoded_output);
        if (d_checkpoints)
            cudaFree(d_checkpoints);
        if (d_values_encoded)
            cudaFree(d_values_encoded);
    }

    void resize(size_t input_sz_bytes, uint32_t streams,
                uint32_t cap_per_stream) {
        size_t total_out = (size_t)streams * cap_per_stream * sizeof(io_t);

        realloc_if_needed((void **)&d_sym_info, cap_sym_info,
                          Config::vocab_size * sizeof(sym_info_t));
        realloc_if_needed((void **)&d_symbols, cap_symbols, input_sz_bytes);
        realloc_if_needed((void **)&d_output, cap_output, total_out);
        realloc_if_needed((void **)&d_states, cap_states,
                          streams * sizeof(state_t));
        realloc_if_needed((void **)&d_sizes, cap_sizes,
                          streams * sizeof(uint32_t));
        realloc_if_needed((void **)&d_checkpoints, cap_checkpoints,
                          streams * CHECKPOINT_INTERVAL *
                              sizeof(RansCheckpoint));
        realloc_if_needed((void **)&d_values_encoded, cap_values_encoded,
                          streams * cap_per_stream * sizeof(uint32_t));
    }

    void resize_dec(size_t in_bytes, size_t out_bytes, uint32_t streams) {
        realloc_if_needed((void **)&d_slot_map, cap_slot_map,
                          Config::prob_scale * sizeof(symbol_t));
        realloc_if_needed((void **)&d_input, cap_input, in_bytes);
        realloc_if_needed((void **)&d_init_states, cap_init_states,
                          streams * sizeof(state_t));
        realloc_if_needed((void **)&d_input_sizes, cap_input_sizes,
                          streams * sizeof(uint32_t));
        realloc_if_needed((void **)&d_decoded_output, cap_decoded_output,
                          out_bytes);
        realloc_if_needed((void **)&d_checkpoints, cap_checkpoints,
                          streams * CHECKPOINT_INTERVAL *
                              sizeof(RansCheckpoint));
        realloc_if_needed((void **)&d_values_encoded, cap_values_encoded,
                          streams * out_bytes / sizeof(symbol_t) *
                              sizeof(uint32_t));
    }
};

struct KernelConfig {
    int block_size;
    uint32_t num_streams;
};

template <typename Config> struct StreamConfigurator {
    static KernelConfig suggest(size_t height, size_t width, size_t B) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        int min_grid_size, best_block_size;
        cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &best_block_size,
                                           rans_compress_kernel<Config>, 0, 0);

        // --- LOGICAL MAPPING ---
        // How many segments of height B do we need to cover the total height H?
        uint32_t segments_per_col = (height + B - 1) / B;

        // Total streams = (Segments per Column) * (Number of Columns)
        uint32_t final_streams = segments_per_col * (uint32_t)width;

        return {best_block_size, final_streams};
    }
};

template <typename Config> struct RansResultPointers {
    bool success;
    const typename Config::io_t *stream;
    const typename Config::state_t *final_states;
    const uint32_t *output_sizes;
    uint32_t num_streams;
    size_t stream_len;
    typename Config::sym_info_t *tables;                // 256 values
    std::vector<typename Config::symbol_t> slot_to_sym; // PROB_SCALE values
    RansCheckpoint *checkpoints; // num_streams * CHECKPOINT_INTERVAL
};

template <typename Config>
RansResultPointers<Config> rans_compress_cuda(
    RansWorkspace<Config> &ws, const typename Config::symbol_t *host_data,
    size_t input_size, const uint16_t *host_freqs, const uint16_t *host_cdf,
    const std::pair<size_t, size_t> shape, KernelConfig k_conf) {
    using symbol_t = typename Config::symbol_t;
    using io_t = typename Config::io_t;
    using sym_info_t = typename Config::sym_info_t;

    uint32_t num_streams = k_conf.num_streams;
    size_t syms_per_stream = (input_size + num_streams - 1) / num_streams;
    uint32_t capacity = (uint32_t)(syms_per_stream * 1.25) + 64;

    // Pack tables
    std::vector<sym_info_t> host_sym_info(Config::vocab_size);
    for (int i = 0; i < Config::vocab_size; ++i) {
        host_sym_info[i].freq = host_freqs[i];
        host_sym_info[i].cdf = host_cdf[i];
    }

    // Generate slot to symbol map
    std::vector<symbol_t> host_slot_map(Config::prob_scale);
    for (int i = 0; i < Config::vocab_size; ++i) {
        for (int j = 0; j < host_freqs[i]; ++j) {
            host_slot_map[host_cdf[i] + j] = (symbol_t)i;
        }
    }

    ws.resize(input_size * sizeof(symbol_t), num_streams, capacity);

    cudaStream_t stream = 0;
    CUDA_CHECK(cudaMemcpyAsync(ws.d_sym_info, host_sym_info.data(),
                               Config::vocab_size * sizeof(sym_info_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_symbols, host_data,
                               input_size * sizeof(symbol_t),
                               cudaMemcpyHostToDevice, stream));

    // Print shape
    std::cout << "Input shape: (" << shape.first << ", " << shape.second
              << ")\n";
    RansEncoderCtx<Config> ctx;
    ctx.success = true;
    ctx.num_streams = num_streams;
    ctx.stream_capacity = capacity;
    ctx.symbols = ws.d_symbols;
    ctx.input_height = shape.first; // K
    ctx.input_width = shape.second; // N
    const_cast<uint32_t &>(ctx.input_size) = (uint32_t)input_size;
    ctx.output = ws.d_output;
    ctx.final_states = ws.d_states;
    ctx.output_sizes = ws.d_sizes;
    ctx.tables.sym_info = ws.d_sym_info;
    ctx.checkpoints = ws.d_checkpoints;

    int block = k_conf.block_size;
    int grid = (num_streams + block - 1) / block;

    // Launch kernel
    rans_compress_kernel<Config><<<grid, block, 0, stream>>>(ctx);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return {
        ctx.success,
        ws.d_output, // Compressed stream
        ws.d_states, // Final states
        ws.d_sizes,  // Length of each stream
        num_streams, // Number of streams
        (size_t)num_streams *
            capacity,    // Total allocated stream length in bytes
        ws.d_sym_info,   // Symbol info table
        host_slot_map,   // Slot to symbol map
        ws.d_checkpoints // Checkpoints
    };
}

template <typename Config> struct RansTiledResultPointers {
    bool success;
    const typename Config::io_t *stream;
    const typename Config::state_t *final_states;
    const uint32_t *output_sizes;
    uint32_t num_streams;
    size_t stream_len;
    size_t capacity_per_stream;
    uint32_t tile_height;
    uint32_t tile_width;
    uint32_t num_tiles_n; // Number of tiles across the width (N / 32)
    uint32_t num_tiles_k; // Number of tiles across the height (K / 128)
    typename Config::sym_info_t *tables;                // 256 values
    std::vector<typename Config::symbol_t> slot_to_sym; // PROB_SCALE values
    uint32_t *values_encoded;
};

template <typename Config>
RansTiledResultPointers<Config> rans_compress_tiled_uncoalesced_cuda(
    RansWorkspace<Config> &ws, const typename Config::symbol_t *host_data,
    size_t input_size, const uint16_t *host_freqs, const uint16_t *host_cdf,
    const std::pair<size_t, size_t> shape, const uint32_t tile_height,
    const uint32_t tile_width) {

    std::cout << "Starting tiled compression...\n";

    using io_t = typename Config::io_t;
    using symbol_t = typename Config::symbol_t;
    using sym_info_t = typename Config::sym_info_t;

    uint32_t K = shape.first;
    uint32_t N = shape.second;

    // Ensure that tile height is multiple of 16
    if (tile_height % 16 != 0) {
        std::cerr << "Tile height must be a multiple of 16 for proper state "
                     "management.\n";
        throw std::runtime_error("Invalid tile height");
    }

    uint32_t num_tiles_k = (K + tile_height - 1) / tile_height;
    uint32_t num_tiles_n = (N + tile_width - 1) / tile_width;
    uint32_t total_tiles = num_tiles_k * num_tiles_n;

    // --- THE FIX: Remove the * 2 since we dropped ILP2 ---
    uint32_t num_streams = total_tiles * tile_width;

    std::cout << "Calculated num tiles K: " << num_tiles_k
              << ", num tiles N: " << num_tiles_n
              << ", total tiles: " << total_tiles
              << ", num streams: " << num_streams
              << ", tile height: " << tile_height
              << ", tile width: " << tile_width << "\n";

    // Check that input_size matches total shape
    if (input_size != K * N) {
        std::cerr << "Input size (" << input_size
                  << ") does not match expected size from shape (" << K * N
                  << ")\n";
        throw std::runtime_error("Input size mismatch in tiled compression");
    }

    // We use a fixed capacity per stream for the one-pass workspace
    size_t syms_per_stream = (input_size + num_streams - 1) / num_streams;
    uint32_t capacity = (uint32_t)(syms_per_stream * 1.25) + 64;

    ws.resize(K * N * sizeof(symbol_t), num_streams, capacity);

    // Pack tables
    std::vector<sym_info_t> host_sym_info(Config::vocab_size);
    for (int i = 0; i < Config::vocab_size; ++i) {
        host_sym_info[i].freq = host_freqs[i];
        host_sym_info[i].cdf = host_cdf[i];
    }

    // Generate slot to symbol map
    std::vector<symbol_t> host_slot_map(Config::prob_scale);
    for (int i = 0; i < Config::vocab_size; ++i) {
        for (int j = 0; j < host_freqs[i]; ++j) {
            host_slot_map[host_cdf[i] + j] = (symbol_t)i;
        }
    }

    cudaStream_t stream = 0;

    // Move tables to GPU
    CUDA_CHECK(cudaMemcpyAsync(ws.d_sym_info, host_sym_info.data(),
                               Config::vocab_size * sizeof(sym_info_t),
                               cudaMemcpyHostToDevice, stream));

    // Move input symbols to GPU
    CUDA_CHECK(cudaMemcpyAsync(ws.d_symbols, host_data,
                               input_size * sizeof(symbol_t),
                               cudaMemcpyHostToDevice, stream));

    // 2. Configure Context
    RansTiledEncoderCtx<Config> ctx;
    ctx.success = true;
    ctx.tile_height = tile_height;
    ctx.tile_width = tile_width;
    ctx.num_tiles_k = num_tiles_k;
    ctx.num_tiles_n = num_tiles_n;
    ctx.symbols = ws.d_symbols;
    ctx.total_height = K;
    ctx.total_width = N;
    ctx.output = ws.d_output;
    ctx.final_states = ws.d_states;
    ctx.stream_sizes = ws.d_sizes;
    ctx.num_streams = num_streams;
    ctx.stream_capacity = capacity;
    ctx.tables.sym_info = ws.d_sym_info;
    ctx.values_encoded = ws.d_values_encoded;

    std::cout << "Got tile height: " << tile_height
              << ", tile width: " << tile_width
              << ", num tiles K: " << num_tiles_k
              << ", num tiles N: " << num_tiles_n
              << ", total streams: " << num_streams
              << ", capacity per stream: " << capacity << "\n";

    // --- THE FIX: Launch the single-pass kernel ---
    dim3 block(tile_width, 1);           // width, height
    dim3 grid(num_tiles_n, num_tiles_k); // width, height

    rans_compress_kernel_tiled_uncoalesced<Config>
        <<<grid, block, 0, stream>>>(ctx);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // --- 5. RETURN TILED RESULT ---
    return {
        ctx.success,
        ws.d_output, // Compressed stream
        ws.d_states, // Final states
        ws.d_sizes,  // Length of each stream
        num_streams, // Number of streams
        (size_t)num_streams *
            capacity, // Total allocated stream length in bytes
        capacity,
        tile_height,        // Height
        tile_width,         // Width
        num_tiles_n,        // Number of tiles across width
        num_tiles_k,        // Number of tiles across height
        ws.d_sym_info,      // Symbol info table
        host_slot_map,      // Slot to symbol map
        ws.d_values_encoded // Buffer with the actual encoded values
    };
}
template <typename Config>
RansTiledResultPointers<Config> rans_compress_tiled_cuda_ilp2(
    RansWorkspace<Config> &ws, const typename Config::symbol_t *host_data,
    size_t input_size, const uint16_t *host_freqs, const uint16_t *host_cdf,
    const std::pair<size_t, size_t> shape, const uint32_t tile_height,
    const uint32_t tile_width) {
    std::cout << "Starting tiled compression...\n";

    using io_t = typename Config::io_t;
    using symbol_t = typename Config::symbol_t;
    using sym_info_t = typename Config::sym_info_t;

    uint32_t K = shape.first;
    uint32_t N = shape.second;
    // uint32_t tile_height = 1024;
    // uint32_t tile_width = 32;

    // Ensure that tile height is multiple of 16
    if (tile_height % 16 != 0) {
        std::cerr << "Tile height must be a multiple of 16 for proper state "
                     "management.\n";
        throw std::runtime_error("Invalid tile height");
    }

    uint32_t num_tiles_k = (K + tile_height - 1) / tile_height;
    uint32_t num_tiles_n = (N + tile_width - 1) / tile_width;
    uint32_t total_tiles = num_tiles_k * num_tiles_n;
    // uint32_t num_streams = total_tiles * tile_width;
    //  ILP 2
    uint32_t num_streams = total_tiles * tile_width * 2;

    std::cout << "Calculated num tiles K: " << num_tiles_k
              << ", num tiles N: " << num_tiles_n
              << ", total tiles: " << total_tiles
              << ", num streams: " << num_streams
              << ", tile height: " << tile_height
              << ", tile width: " << tile_width << "\n";

    // Check that input_size matches total shape
    if (input_size != K * N) {
        std::cerr << "Input size (" << input_size
                  << ") does not match expected size from shape (" << K * N
                  << ")\n";
        throw std::runtime_error("Input size mismatch in tiled compression");
    }

    // We use a fixed capacity per stream for the one-pass workspace
    size_t syms_per_stream = (input_size + num_streams - 1) / num_streams;
    uint32_t capacity = (uint32_t)(syms_per_stream * 1.25) + 64;

    ws.resize(K * N * sizeof(symbol_t), num_streams, capacity);

    // Pack tables
    std::vector<sym_info_t> host_sym_info(Config::vocab_size);
    for (int i = 0; i < Config::vocab_size; ++i) {
        host_sym_info[i].freq = host_freqs[i];
        host_sym_info[i].cdf = host_cdf[i];
    }

    // Generate slot to symbol map
    std::vector<symbol_t> host_slot_map(Config::prob_scale);
    for (int i = 0; i < Config::vocab_size; ++i) {
        for (int j = 0; j < host_freqs[i]; ++j) {
            host_slot_map[host_cdf[i] + j] = (symbol_t)i;
        }
    }

    cudaStream_t stream = 0;

    // Move tables to GPU
    CUDA_CHECK(cudaMemcpyAsync(ws.d_sym_info, host_sym_info.data(),
                               Config::vocab_size * sizeof(sym_info_t),
                               cudaMemcpyHostToDevice, stream));

    // Move input symbols to GPU
    CUDA_CHECK(cudaMemcpyAsync(ws.d_symbols, host_data,
                               input_size * sizeof(symbol_t),
                               cudaMemcpyHostToDevice, stream));

    // Values encoded buffer

    // 2. Configure Context
    RansTiledEncoderCtx<Config> ctx;
    ctx.success = true;
    ctx.tile_height = tile_height;
    ctx.tile_width = tile_width;
    ctx.num_tiles_k = num_tiles_k;
    ctx.num_tiles_n = num_tiles_n;
    ctx.symbols = ws.d_symbols;
    ctx.total_height = K;
    ctx.total_width = N;
    ctx.output = ws.d_output;
    ctx.final_states = ws.d_states;
    ctx.stream_sizes = ws.d_sizes;
    ctx.num_streams = num_streams;
    ctx.stream_capacity = capacity;
    ctx.tables.sym_info = ws.d_sym_info;
    ctx.values_encoded = ws.d_values_encoded;

    std::cout << "Got tile height: " << tile_height
              << ", tile width: " << tile_width
              << ", num tiles K: " << num_tiles_k
              << ", num tiles N: " << num_tiles_n
              << ", total streams: " << num_streams
              << ", capacity per stream: " << capacity << "\n";

    size_t threads_ilp = tile_width / 2;

    dim3 block(tile_width, 1); // width, height
    // dim3 block(threads_ilp, 1);          // width, height
    dim3 grid(num_tiles_n, num_tiles_k); // width, height
    // rans_compress_kernel_tiled<Config><<<grid, block, 0, stream>>>(ctx);
    rans_compress_kernel_tiled_ilp2<Config><<<grid, block, 0, stream>>>(ctx);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // --- 5. RETURN TILED RESULT ---
    // Note: Use cloned vectors/tensors as this result goes to Python/Triton
    return {
        ctx.success,
        ws.d_output, // Compressed stream
        ws.d_states, // Final states
        ws.d_sizes,  // Length of each stream
        num_streams, // Number of streams
        (size_t)num_streams *
            capacity, // Total allocated stream length in bytes
        capacity,
        tile_height,        // Height
        tile_width,         // Width
        num_tiles_n,        // Number of tiles across width
        num_tiles_k,        // Number of tiles across height
        ws.d_sym_info,      // Symbol info table
        host_slot_map,      // Slot to symbol map
        ws.d_values_encoded // Buffer with the actual encoded values (before
                            // bit packing)
    };
}

template <typename Config>
RansTiledResultPointers<Config> rans_compress_tiled_cuda(
    RansWorkspace<Config> &ws, const typename Config::symbol_t *host_data,
    size_t input_size, const uint16_t *host_freqs, const uint16_t *host_cdf,
    const std::pair<size_t, size_t> shape, const uint32_t tile_height,
    const uint32_t tile_width) {
    std::cout << "Starting tiled compression...\n";

    using io_t = typename Config::io_t;
    using symbol_t = typename Config::symbol_t;
    using sym_info_t = typename Config::sym_info_t;

    uint32_t K = shape.first;
    uint32_t N = shape.second;

    // Ensure that tile height is multiple of 16
    if (tile_height % 16 != 0) {
        std::cerr << "Tile height must be a multiple of 16 for proper state "
                     "management.\n";
        throw std::runtime_error("Invalid tile height");
    }

    uint32_t num_tiles_k = (K + tile_height - 1) / tile_height;
    uint32_t num_tiles_n = (N + tile_width - 1) / tile_width;
    uint32_t total_tiles = num_tiles_k * num_tiles_n;

    // Standard baseline: 1 stream per tile column
    uint32_t num_streams = total_tiles * tile_width;

    std::cout << "Calculated num tiles K: " << num_tiles_k
              << ", num tiles N: " << num_tiles_n
              << ", total tiles: " << total_tiles
              << ", num streams: " << num_streams
              << ", tile height: " << tile_height
              << ", tile width: " << tile_width << "\n";

    // Check that input_size matches total shape
    if (input_size != K * N) {
        std::cerr << "Input size (" << input_size
                  << ") does not match expected size from shape (" << K * N
                  << ")\n";
        throw std::runtime_error("Input size mismatch in tiled compression");
    }

    // We use a fixed capacity per stream for the one-pass workspace
    size_t syms_per_stream = (input_size + num_streams - 1) / num_streams;
    uint32_t capacity = (uint32_t)(syms_per_stream * 1.25) + 64;

    ws.resize(K * N * sizeof(symbol_t), num_streams, capacity);

    // Pack tables
    std::vector<sym_info_t> host_sym_info(Config::vocab_size);
    for (int i = 0; i < Config::vocab_size; ++i) {
        host_sym_info[i].freq = host_freqs[i];
        host_sym_info[i].cdf = host_cdf[i];
    }

    // Generate slot to symbol map
    std::vector<symbol_t> host_slot_map(Config::prob_scale);
    for (int i = 0; i < Config::vocab_size; ++i) {
        for (int j = 0; j < host_freqs[i]; ++j) {
            host_slot_map[host_cdf[i] + j] = (symbol_t)i;
        }
    }

    cudaStream_t stream = 0;

    // Move tables to GPU
    CUDA_CHECK(cudaMemcpyAsync(ws.d_sym_info, host_sym_info.data(),
                               Config::vocab_size * sizeof(sym_info_t),
                               cudaMemcpyHostToDevice, stream));

    // Move input symbols to GPU
    CUDA_CHECK(cudaMemcpyAsync(ws.d_symbols, host_data,
                               input_size * sizeof(symbol_t),
                               cudaMemcpyHostToDevice, stream));

    // 2. Configure Context
    RansTiledEncoderCtx<Config> ctx;
    ctx.success = true;
    ctx.tile_height = tile_height;
    ctx.tile_width = tile_width;
    ctx.num_tiles_k = num_tiles_k;
    ctx.num_tiles_n = num_tiles_n;
    ctx.symbols = ws.d_symbols;
    ctx.total_height = K;
    ctx.total_width = N;
    ctx.output = ws.d_output;
    ctx.final_states = ws.d_states;
    ctx.stream_sizes = ws.d_sizes;
    ctx.num_streams = num_streams;
    ctx.stream_capacity = capacity;
    ctx.tables.sym_info = ws.d_sym_info;
    ctx.values_encoded = ws.d_values_encoded;

    std::cout << "Got tile height: " << tile_height
              << ", tile width: " << tile_width
              << ", num tiles K: " << num_tiles_k
              << ", num tiles N: " << num_tiles_n
              << ", total streams: " << num_streams
              << ", capacity per stream: " << capacity << "\n";

    dim3 block(tile_width, 1);           // width, height
    dim3 grid(num_tiles_n, num_tiles_k); // width, height

    rans_compress_kernel_tiled<Config><<<grid, block, 0, stream>>>(ctx);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // --- 5. RETURN TILED RESULT ---
    // Note: Use cloned vectors/tensors as this result goes to Python/Triton
    return {
        ctx.success,
        ws.d_output, // Compressed stream
        ws.d_states, // Final states
        ws.d_sizes,  // Length of each stream
        num_streams, // Number of streams
        (size_t)num_streams *
            capacity, // Total allocated stream length in bytes
        capacity,
        tile_height,        // Height
        tile_width,         // Width
        num_tiles_n,        // Number of tiles across width
        num_tiles_k,        // Number of tiles across height
        ws.d_sym_info,      // Symbol info table
        host_slot_map,      // Slot to symbol map
        ws.d_values_encoded // Buffer with the actual encoded values (before bit
                            // packing)
    };
}

template <typename Config>
std::pair<const typename Config::symbol_t *, float> rans_decompress_cuda_ws(
    RansWorkspace<Config> &ws, const typename Config::io_t *stream_ptr,
    size_t stream_size_bytes, const typename Config::state_t *states_ptr,
    const uint32_t *sizes_ptr, uint32_t num_streams,
    uint32_t symbols_per_stream, const uint16_t *host_freqs,
    const uint16_t *host_cdf) {
    using symbol_t = typename Config::symbol_t;
    using sym_info_t = typename Config::sym_info_t;

    // NOTE: This allocation should be handeld in python
    if (ws.d_sym_info == nullptr) {
        std::cout << "Ws.d_sym_info is null, allocating...\n";
        CUDA_CHECK(cudaMalloc(&ws.d_sym_info,
                              Config::vocab_size * sizeof(sym_info_t)));
    }
    if (ws.d_slot_map == nullptr) {
        std::cout << "Ws.d_slot_map is null, allocating...\n";
        CUDA_CHECK(
            cudaMalloc(&ws.d_slot_map, Config::prob_scale * sizeof(symbol_t)));
    }
    uint32_t capacity_per_stream = stream_size_bytes / num_streams;

    std::vector<sym_info_t> host_sym_info(Config::vocab_size);
    std::vector<symbol_t> host_slot_map(Config::prob_scale);

    for (int i = 0; i < Config::vocab_size; ++i) {
        host_sym_info[i].freq = host_freqs[i];
        host_sym_info[i].cdf = host_cdf[i];
        for (int j = 0; j < host_freqs[i]; ++j)
            host_slot_map[host_cdf[i] + j] = (symbol_t)i;
    }

    size_t input_bytes = stream_size_bytes;
    size_t output_bytes =
        (size_t)symbols_per_stream * num_streams * sizeof(symbol_t);

    ws.resize_dec(input_bytes, output_bytes, num_streams);

    cudaStream_t stream = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    CUDA_CHECK(cudaMemcpyAsync(ws.d_sym_info, host_sym_info.data(),
                               Config::vocab_size * sizeof(sym_info_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_slot_map, host_slot_map.data(),
                               Config::prob_scale * sizeof(symbol_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_input, stream_ptr, input_bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_init_states, states_ptr,
                               num_streams * sizeof(typename Config::state_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_input_sizes, sizes_ptr,
                               num_streams * sizeof(uint32_t),
                               cudaMemcpyHostToDevice, stream));

    RansDecoderCtx<Config> ctx{
        ws.d_input,         ws.d_init_states,
        ws.d_input_sizes,   ws.d_decoded_output,
        symbols_per_stream, capacity_per_stream,
        num_streams,        {ws.d_sym_info, ws.d_slot_map}};
    // ctx.input = ws.d_input;
    // ctx.initial_states = ws.d_init_states;
    // ctx.input_sizes = ws.d_input_sizes;
    // ctx.output = ws.d_decoded_output;
    // ctx.output_size = symbols_per_stream;
    // ctx.stream_capacity = capacity_per_stream;
    // ctx.num_streams = num_streams;
    // ctx.tables.sym_info = ws.d_sym_info;
    // ctx.tables.slot_to_sym = ws.d_slot_map;

    // Log ctx
    std::cout << "RANS Decompression Context:\n";
    std::cout << "  Num streams: " << num_streams << "\n";
    std::cout << "  Symbols per stream: " << symbols_per_stream << "\n";
    std::cout << "  Stream capacity (io_t): " << capacity_per_stream << "\n";
    std::cout << "  Input size (bytes): " << input_bytes << "\n";
    std::cout << "  Output size (symbols): "
              << (output_bytes / sizeof(symbol_t)) << "\n";

    int min_grid, best_block;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &best_block,
                                       rans_decompress_kernel<Config>, 0, 0);
    int grid = (num_streams + best_block - 1) / best_block;

    cudaEventRecord(start, stream);
    rans_decompress_kernel<Config><<<grid, best_block, 0, stream>>>(ctx);
    cudaEventRecord(stop, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return {ws.d_decoded_output, ms};
}

template <typename Config>
void rans_decompress_cuda(
    const typename Config::io_t *__restrict__ d_stream,
    const size_t stream_size_bytes,
    const typename Config::state_t *__restrict__ d_states,
    const uint32_t *__restrict__ d_sizes, const uint32_t num_streams,
    const uint32_t symbols_per_stream, const uint32_t *__restrict__ d_tables,
    const typename Config::symbol_t *__restrict__ slot_to_sym,
    typename Config::symbol_t *__restrict__ d_output) {

    using symbol_t = typename Config::symbol_t;
    using sym_info_t = typename Config::sym_info_t;

    uint32_t capacity_per_stream = stream_size_bytes / num_streams;

    RansTablesFull<Config> tables{
        reinterpret_cast<const sym_info_t *>(d_tables), slot_to_sym};

    RansDecoderCtx<Config> ctx{d_stream,
                               d_states,
                               d_sizes,
                               d_output,
                               symbols_per_stream,
                               capacity_per_stream,
                               num_streams,
                               tables};

    size_t input_bytes = stream_size_bytes;
    size_t output_bytes =
        (size_t)symbols_per_stream * num_streams * sizeof(symbol_t);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // cudaStream_t stream = 0;

    // Compute optimal grid and launch kernel
    int min_grid, best_block;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &best_block,
                                       rans_decompress_kernel<Config>, 0, 0);
    int grid = (num_streams + best_block - 1) / best_block;
    rans_decompress_kernel<Config><<<grid, best_block, 0, stream>>>(ctx);

    // NOTE: This sync can propably be removed
    // CUDA_CHECK(cudaStreamSynchronize(stream));
}
