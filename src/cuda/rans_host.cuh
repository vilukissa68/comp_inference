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
    ctx.num_streams = num_streams;
    ctx.stream_capacity = capacity;
    ctx.symbols = ws.d_symbols;
    ctx.input_height = shape.first;
    ctx.input_width = shape.second;
    const_cast<uint32_t &>(ctx.input_size) = (uint32_t)input_size;
    ctx.output = ws.d_output;
    ctx.final_states = ws.d_states;
    ctx.output_sizes = ws.d_sizes;
    ctx.tables.sym_info = ws.d_sym_info;
    ctx.checkpoints = ws.d_checkpoints;

    int block = k_conf.block_size;
    int grid = (num_streams + block - 1) / block;
    ctx.success = true;

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
