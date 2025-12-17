#include <iostream>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <iomanip>
#include <algorithm> // For std::max, std::min

#include "rans.cuh"
#include "rans_kernels.cuh"

#define CUDA_CHECK(call) \
    do { \
         cudaError_t err = call; \
         if (err != cudaSuccess) { \
             fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                     cudaGetErrorString(err), __FILE__, __LINE__); \
            throw std::runtime_error("CUDA Error"); \
         } \
     } while (0)

template <typename T>
struct PinnedAllocator {
    using value_type = T;
    T* allocate(std::size_t n) {
        T* ptr = nullptr;
        if (n > 0) CUDA_CHECK(cudaMallocHost(&ptr, n * sizeof(T)));
        return ptr;
    }
    void deallocate(T* ptr, std::size_t) {
        if (ptr) cudaFreeHost(ptr);
    }
};
template <typename T>
using PinnedVector = std::vector<T, PinnedAllocator<T>>;

template <typename Config>
struct RansWorkspace {
    using symbol_t = typename Config::symbol_t;
    using io_t = typename Config::io_t;
    using sym_info_t = typename Config::sym_info_t;
    using state_t = typename Config::state_t;

    sym_info_t* d_sym_info = nullptr;
    symbol_t* d_symbols = nullptr;
    io_t* d_output = nullptr;
    state_t* d_states = nullptr;
    uint32_t* d_sizes = nullptr;
    symbol_t* d_slot_map = nullptr;
    io_t* d_input = nullptr;
    state_t* d_init_states = nullptr;
    uint32_t* d_input_sizes = nullptr;
    symbol_t* d_decoded_output = nullptr;

    size_t cap_bytes = 0;
    
    RansWorkspace() = default;
    ~RansWorkspace() {
        if(d_sym_info) cudaFree(d_sym_info);
        if(d_symbols) cudaFree(d_symbols);
        if(d_output) cudaFree(d_output);
        if(d_states) cudaFree(d_states);
        if(d_sizes) cudaFree(d_sizes);
        if(d_slot_map) cudaFree(d_slot_map);
        if(d_input) cudaFree(d_input);
        if(d_init_states) cudaFree(d_init_states);
        if(d_input_sizes) cudaFree(d_input_sizes);
        if(d_decoded_output) cudaFree(d_decoded_output);
    }

    void resize(size_t input_sz, uint32_t streams, uint32_t cap_per_stream) {
        size_t total_out = (size_t)streams * cap_per_stream * sizeof(io_t);
        if (total_out > cap_bytes) {
            if(d_sym_info) cudaFree(d_sym_info);
            if(d_symbols) cudaFree(d_symbols);
            if(d_output) cudaFree(d_output);
            if(d_states) cudaFree(d_states);
            if(d_sizes) cudaFree(d_sizes);
            
            CUDA_CHECK(cudaMalloc(&d_sym_info, Config::vocab_size * sizeof(sym_info_t)));
            CUDA_CHECK(cudaMalloc(&d_symbols, input_sz * sizeof(symbol_t)));
            CUDA_CHECK(cudaMalloc(&d_output, total_out));
            CUDA_CHECK(cudaMalloc(&d_states, streams * sizeof(state_t)));
            CUDA_CHECK(cudaMalloc(&d_sizes, streams * sizeof(uint32_t)));
            cap_bytes = total_out;
        }
    }
    
    void resize_dec(size_t in_bytes, size_t out_bytes, uint32_t streams) {
        if(!d_slot_map) CUDA_CHECK(cudaMalloc(&d_slot_map, Config::prob_scale * sizeof(symbol_t)));
        if(!d_input) CUDA_CHECK(cudaMalloc(&d_input, in_bytes));
        if(!d_init_states) CUDA_CHECK(cudaMalloc(&d_init_states, streams * sizeof(state_t)));
        if(!d_input_sizes) CUDA_CHECK(cudaMalloc(&d_input_sizes, streams * sizeof(uint32_t)));
        if(!d_decoded_output) CUDA_CHECK(cudaMalloc(&d_decoded_output, out_bytes));
    }
};

struct RansConfig8 : public RansTraits<uint8_t, uint32_t, uint8_t, 12> {
    using sym_info_t = RansSymInfoPacked;
};

template <typename Config>
struct RansResult {
    PinnedVector<typename Config::io_t> stream;
    PinnedVector<typename Config::state_t> final_states;
    PinnedVector<uint32_t> output_sizes;
    uint32_t num_streams; // Store this for the decoder
};


struct KernelConfig {
    int block_size;
    uint32_t num_streams;
};

template <typename Config>
struct StreamConfigurator {
    static KernelConfig suggest(size_t total_data_size) {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        // Determine optimal Block Size for this GPU
        int min_grid_size, best_block_size;
        // This calculates best block size based on registers/shared mem of the kernel
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, 
            &best_block_size, 
            rans_compress_kernel<Config>, 
            0, 0);

        // Determine Optimal Number of Streams
        // Heuristic: We want a chunk size around 4KB - 16KB for rANS.
        // Too small (<512B) = Thread overhead dominates.
        // Too large (>64KB) = Divergence issues & tail effects.
        
        // Target 4KB per stream as a baseline sweet spot
        uint32_t target_chunk_size = 4096; 
        
        // Calculate streams based on target chunk
        uint32_t suggested_streams = (total_data_size + target_chunk_size - 1) / target_chunk_size;

        // Constraint A: Saturation
        // We need enough blocks to hide latency. 
        // 1080 Ti (28 SMs) vs 4090 (128 SMs).
        // Aim for at least 4 blocks per SM (latency hiding).
        uint32_t min_streams_for_saturation = prop.multiProcessorCount * 4 * best_block_size;
        
        // If 4KB chunks aren't enough to fill the GPU, chop them smaller (down to 512B)
        if (suggested_streams < min_streams_for_saturation) {
            suggested_streams = min_streams_for_saturation;
        }

        // Constraint B: Hardware Limits
        // Don't exceed maximum grid dimensions (usually 2^31-1, unlikely to hit)
        // Ensure streams isn't larger than total bytes (1 byte per stream minimum)
        suggested_streams = std::min((uint32_t)total_data_size, suggested_streams);

        // Ensure at least 1 stream
        if (suggested_streams == 0) suggested_streams = 1;

        return { best_block_size, suggested_streams };
    }
};


template <typename Config>
RansResult<Config> rans_compress_cuda(
    RansWorkspace<Config>& ws,
    const PinnedVector<typename Config::symbol_t> &host_data,
    const uint16_t *host_freqs, const uint16_t *host_cdf, 
    KernelConfig k_conf)
{
    using symbol_t = typename Config::symbol_t;
    using io_t = typename Config::io_t;
    using sym_info_t = typename Config::sym_info_t;

    uint32_t num_streams = k_conf.num_streams;
    size_t input_size = host_data.size();
    size_t syms_per_stream = (input_size + num_streams - 1) / num_streams;
    // Padding + Safety for compression expansion
    uint32_t capacity = (uint32_t)(syms_per_stream * 1.25) + 64; 

    std::vector<sym_info_t> host_sym_info(Config::vocab_size);
    for (int i = 0; i < Config::vocab_size; ++i) {
        host_sym_info[i].freq = host_freqs[i];
        host_sym_info[i].cdf = host_cdf[i];
    }

    ws.resize(input_size, num_streams, capacity);

    cudaStream_t stream = 0;
    CUDA_CHECK(cudaMemcpyAsync(ws.d_sym_info, host_sym_info.data(), Config::vocab_size * sizeof(sym_info_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_symbols, host_data.data(), input_size * sizeof(symbol_t), cudaMemcpyHostToDevice, stream));

    RansEncoderCtx<Config> ctx;
    ctx.num_streams = num_streams;
    ctx.stream_capacity = capacity;
    ctx.symbols = ws.d_symbols;
    const_cast<uint32_t &>(ctx.input_size) = (uint32_t)input_size;
    ctx.output = ws.d_output;
    ctx.final_states = ws.d_states;
    ctx.output_sizes = ws.d_sizes;
    ctx.tables.sym_info = ws.d_sym_info;

    // Use the auto-tuned block size
    int block = k_conf.block_size;
    int grid = (num_streams + block - 1) / block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    rans_compress_kernel<Config><<<grid, block, 0, stream>>>(ctx);
    cudaEventRecord(stop, stream);

    RansResult<Config> result;
    result.num_streams = num_streams; // Save for decoder
    result.stream.resize(num_streams * capacity);
    result.final_states.resize(num_streams);
    result.output_sizes.resize(num_streams);

    CUDA_CHECK(cudaMemcpyAsync(result.stream.data(), ws.d_output, num_streams * capacity * sizeof(io_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.final_states.data(), ws.d_states, num_streams * sizeof(typename Config::state_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(result.output_sizes.data(), ws.d_sizes, num_streams * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double gib_per_sec = (double)input_size / (ms * 1e-3) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "[Encoder] Time: " << std::fixed << std::setprecision(3) << ms << " ms | Throughput: " << gib_per_sec << " GiB/s"
              << " | Grid: " << grid << " Block: " << block << " Streams: " << num_streams << "\n";

    cudaEventDestroy(start); cudaEventDestroy(stop);
    return result;
}

template <typename Config>
PinnedVector<typename Config::symbol_t> rans_decompress_cuda(
    RansWorkspace<Config>& ws,
    PinnedVector<typename Config::io_t> &host_stream_data,
    const PinnedVector<typename Config::state_t> &host_states, 
    const PinnedVector<uint32_t> &host_sizes,
    uint32_t num_streams, // Must match what was used in compression
    uint32_t symbols_per_stream,
    const uint16_t *host_freqs, const uint16_t *host_cdf) 
{
    using symbol_t = typename Config::symbol_t;
    using sym_info_t = typename Config::sym_info_t;

    // Recalculate capacity based on actual data provided
    // NOTE: In a real application, you might store this in a header, but here we derive it
    uint32_t capacity_per_stream = host_stream_data.size() / num_streams;

    std::vector<sym_info_t> host_sym_info(Config::vocab_size);
    std::vector<symbol_t> host_slot_map(Config::prob_scale);

    for (int i = 0; i < Config::vocab_size; ++i) {
        host_sym_info[i].freq = host_freqs[i];
        host_sym_info[i].cdf = host_cdf[i];
        for(int j=0; j<host_freqs[i]; ++j) host_slot_map[host_cdf[i] + j] = (symbol_t)i;
    }

    size_t input_bytes = host_stream_data.size() * sizeof(typename Config::io_t);
    size_t output_bytes = (size_t)symbols_per_stream * num_streams * sizeof(symbol_t);

    ws.resize_dec(input_bytes, output_bytes, num_streams);
    cudaStream_t stream = 0;

    CUDA_CHECK(cudaMemcpyAsync(ws.d_sym_info, host_sym_info.data(), Config::vocab_size * sizeof(sym_info_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_slot_map, host_slot_map.data(), Config::prob_scale * sizeof(symbol_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_input, host_stream_data.data(), input_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_init_states, host_states.data(), num_streams * sizeof(typename Config::state_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(ws.d_input_sizes, host_sizes.data(), num_streams * sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

    RansDecoderCtx<Config> ctx;
    ctx.input = ws.d_input;
    ctx.initial_states = ws.d_init_states;
    ctx.input_sizes = ws.d_input_sizes;
    ctx.output = ws.d_decoded_output;
    ctx.output_size = symbols_per_stream;
    ctx.stream_capacity = capacity_per_stream;
    ctx.num_streams = num_streams;
    ctx.tables.sym_info = ws.d_sym_info;
    ctx.tables.slot_to_sym = ws.d_slot_map;

    // Use Occupancy API for Decompression Block Size too
    int min_grid, best_block;
    cudaOccupancyMaxPotentialBlockSize(&min_grid, &best_block, rans_decompress_kernel<Config>, 0, 0);
    
    int grid = (num_streams + best_block - 1) / best_block;

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    rans_decompress_kernel<Config><<<grid, best_block, 0, stream>>>(ctx);
    cudaEventRecord(stop, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    double gib_per_sec = (double)output_bytes / (ms * 1e-3) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "[Decoder] Time: " << std::fixed << std::setprecision(3) << ms << " ms | Throughput: " << gib_per_sec << " GiB/s"
              << " | Block: " << best_block << "\n";

    cudaEventDestroy(start); cudaEventDestroy(stop);

    PinnedVector<symbol_t> result(num_streams * symbols_per_stream);
    CUDA_CHECK(cudaMemcpyAsync(result.data(), ws.d_decoded_output, output_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

int main() {
    cudaFree(0); 

    const size_t TOTAL_SIZE = 256 * 1024 * 1024; // 256 MB

    std::cout << "Allocating " << (double)TOTAL_SIZE / (1024*1024) << " MiB Pinned Memory...\n";
    PinnedVector<uint8_t> original_data(TOTAL_SIZE);
    
    // Generate data
    for (size_t i = 0; i < TOTAL_SIZE; ++i) original_data[i] = (i * 13) % 32;

    std::vector<uint16_t> freqs(256), cdf(256);
    int used_symbols = 32, high_freq = 120;
    for(int i = 0; i < 256; ++i) freqs[i] = (i < used_symbols) ? high_freq : 1;
    int sum = 0;
    for(int f : freqs) sum += f;
    if (sum < 4096) freqs[0] += (4096 - sum);
    sum = 0;
    for (int i = 0; i < 256; ++i) { cdf[i] = sum; sum += freqs[i]; }

    RansWorkspace<RansConfig8> workspace;

    try {
        auto config = StreamConfigurator<RansConfig8>::suggest(original_data.size());
        
        std::cout << "Auto-Tuner Selected:\n"
                  << "  Block Size: " << config.block_size << "\n"
                  << "  Streams:    " << config.num_streams << "\n"
                  << "  Chunk Size: " << original_data.size() / config.num_streams << " bytes\n";

        std::cout << "Starting Compression...\n";
        auto encoded = rans_compress_cuda<RansConfig8>(
            workspace, original_data, freqs.data(), cdf.data(), config);
        
        size_t stream_payload_bytes = 0;
        // Note: d_sizes is on GPU, result.output_sizes is on CPU (Pinned). We use the CPU vector.
        for (uint32_t size : encoded.output_sizes) {
            stream_payload_bytes += size;
        }

        size_t state_overhead = encoded.final_states.size() * sizeof(uint32_t);
        size_t size_header_overhead = encoded.output_sizes.size() * sizeof(uint32_t);
        size_t table_overhead = 256 * sizeof(uint16_t) * 2; // Freq + CDF

        size_t total_compressed_size = stream_payload_bytes + state_overhead + size_header_overhead + table_overhead;

        double total_mb = (double)total_compressed_size / (1024.0 * 1024.0);
        double orig_mb = (double)TOTAL_SIZE / (1024.0 * 1024.0);
        double bits_per_symbol = ((double)total_compressed_size * 8.0) / TOTAL_SIZE;
        double ratio = (double)TOTAL_SIZE / total_compressed_size;

        std::cout << "\n--- Statistics ---\n";
        std::cout << "Original Size     : " << std::fixed << std::setprecision(2) << orig_mb << " MiB\n";
        std::cout << "Compressed Payload: " << (double)stream_payload_bytes / (1024*1024) << " MiB\n";
        std::cout << "Total Size (w/ Ovh):" << total_mb << " MiB\n";
        std::cout << "Compression Ratio : " << std::setprecision(2) << ratio << ":1 (" 
                  << (float)total_compressed_size / TOTAL_SIZE * 100.0f << "%)\n";
        std::cout << "Bits per symbol   : " << std::setprecision(4) << bits_per_symbol << " bits\n";
        std::cout << "Overhead          : " << (double)(state_overhead + size_header_overhead) / 1024.0 << " KiB\n";

        
        uint32_t syms_per_stream = (TOTAL_SIZE + encoded.num_streams - 1) / encoded.num_streams;
        
        std::cout << "\nStarting Decompression...\n";
        auto decoded = rans_decompress_cuda<RansConfig8>(
            workspace, encoded.stream,
            encoded.final_states, encoded.output_sizes,
            encoded.num_streams, syms_per_stream, freqs.data(), cdf.data());

        // Verification
        size_t check_len = std::min(original_data.size(), decoded.size());
        int errors = 0;
        for(size_t i=0; i<check_len; ++i) {
            if(original_data[i] != decoded[i]) {
                if(errors++ < 5) std::cout << "Mismatch " << i << "\n";
            }
        }
        if(errors == 0) std::cout << "SUCCESS: Verification Passed.\n";

    } catch (const std::exception &e) {
        std::cerr << "Ex: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
