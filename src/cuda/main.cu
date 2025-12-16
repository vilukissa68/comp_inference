#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

// Include your header definitions
#include "rans.cuh"
#include "rans_kernels.cuh"

// Forward declare the kernel (assuming it's in rans_kernels.cu/cuh)
// We correct the typo 'rand_' to 'rans_' here, assuming standard naming.
template <typename RansConfig>
__global__ void rans_compress_kernel(RansEncoderCtx<RansConfig> ctx);

// ============================================================================
// CONCRETE CONFIGURATION
// ============================================================================
// We extend RansTraits to define 'sym_info_t', which RansTablesCore requires.
struct RansConfig8 : public RansTraits<uint8_t, uint32_t, uint8_t, 12> {
    // Select packed struct (4 bytes) because prob_bits (12) <= 16
    using sym_info_t = RansSymInfoPacked;
};

// ============================================================================
// HELPER MACROS & STRUCTS
// ============================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            throw std::runtime_error("CUDA Error");                            \
        }                                                                      \
    } while (0)

template <typename Config> struct RansResult {
    std::vector<typename Config::io_t> stream;
    std::vector<typename Config::state_t> final_states;
    std::vector<uint32_t> output_sizes;
};

// ============================================================================
// HOST WRAPPER IMPLEMENTATION
// ============================================================================

template <typename Config>
RansResult<Config> rans_compress_cuda(
    const typename Config::symbol_t *host_data, size_t input_size,
    const uint16_t *host_freqs, // Source freqs (usually from python/cpu)
    const uint16_t *host_cdf,   // Source cdf
    uint32_t num_streams) {
    using SymbolT = typename Config::symbol_t;
    using StateT = typename Config::state_t;
    using IOT = typename Config::io_t;
    using SymInfoT = typename Config::sym_info_t;

    // 1. Calculate Capacity
    // Estimate size per stream + padding for safety
    size_t syms_per_stream = (input_size + num_streams - 1) / num_streams;
    uint32_t capacity = (uint32_t)(syms_per_stream * 1.2) + 32;

    // ------------------------------------------------------------------------
    // 2. PREPARE TABLES (Pack separate arrays into RansSymInfo)
    // ------------------------------------------------------------------------
    std::vector<SymInfoT> host_sym_info(Config::vocab_size);
    for (int i = 0; i < Config::vocab_size; ++i) {
        // We cast to the types defined in the struct (uint16_t or uint32_t)
        host_sym_info[i].freq =
            static_cast<decltype(host_sym_info[i].freq)>(host_freqs[i]);
        host_sym_info[i].cdf =
            static_cast<decltype(host_sym_info[i].cdf)>(host_cdf[i]);
    }

    SymInfoT *d_sym_info;
    CUDA_CHECK(cudaMalloc(&d_sym_info, Config::vocab_size * sizeof(SymInfoT)));
    CUDA_CHECK(cudaMemcpy(d_sym_info, host_sym_info.data(),
                          Config::vocab_size * sizeof(SymInfoT),
                          cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------------
    // 3. ALLOCATE BUFFERS
    // ------------------------------------------------------------------------
    SymbolT *d_symbols;
    IOT *d_output;
    StateT *d_states;
    uint32_t *d_sizes;

    size_t input_bytes = input_size * sizeof(SymbolT);
    size_t stream_bytes = num_streams * capacity * sizeof(IOT);
    size_t states_bytes = num_streams * sizeof(StateT);
    size_t sizes_bytes = num_streams * sizeof(uint32_t);

    CUDA_CHECK(cudaMalloc(&d_symbols, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, stream_bytes));
    CUDA_CHECK(cudaMalloc(&d_states, states_bytes));
    CUDA_CHECK(cudaMalloc(&d_sizes, sizes_bytes));

    CUDA_CHECK(
        cudaMemcpy(d_symbols, host_data, input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sizes, 0, sizes_bytes)); // Safety clear

    // ------------------------------------------------------------------------
    // 4. SETUP CONTEXT
    // ------------------------------------------------------------------------
    RansEncoderCtx<Config> ctx;

    // Config
    ctx.num_streams = num_streams;
    ctx.stream_capacity = capacity;

    // Input
    ctx.symbols = d_symbols;
    // We cannot construct 'const uint32_t' member directly in list init for a
    // device struct passed by value easily without a constructor, but for POD
    // structs in CUDA, assignment works if we cast away const or init properly.
    // Standard CUDA practice: const_cast or set before copy.
    const_cast<uint32_t &>(ctx.input_size) = (uint32_t)input_size;

    // Output
    ctx.output = d_output;
    ctx.final_states = d_states;
    ctx.output_sizes = d_sizes;

    // Tables
    ctx.tables.sym_info = d_sym_info;

    // ------------------------------------------------------------------------
    // 5. LAUNCH
    // ------------------------------------------------------------------------
    int block_size = 256;
    int grid_size = (num_streams + block_size - 1) / block_size;

    rans_compress_kernel<Config><<<grid_size, block_size>>>(ctx);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------------------
    // 6. RETRIEVE
    // ------------------------------------------------------------------------
    RansResult<Config> result;
    result.stream.resize(num_streams * capacity);
    result.final_states.resize(num_streams);
    result.output_sizes.resize(num_streams);

    CUDA_CHECK(cudaMemcpy(result.stream.data(), d_output, stream_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.final_states.data(), d_states, states_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.output_sizes.data(), d_sizes, sizes_bytes,
                          cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------------
    // 7. CLEANUP
    // ------------------------------------------------------------------------
    cudaFree(d_symbols);
    cudaFree(d_output);
    cudaFree(d_states);
    cudaFree(d_sizes);
    cudaFree(d_sym_info);

    return result;
}

// ============================================================================
// EXAMPLE USAGE
// ============================================================================
int main() {
    // 1. Setup Dummy Data
    const int N = 100000;
    const int STREAMS = 256;
    std::vector<uint8_t> data(N);
    for (int i = 0; i < N; ++i)
        data[i] = i % 256;

    // 2. Setup Dummy Tables (Uniform)
    std::vector<uint16_t> freqs(256, 16); // 4096 / 256 = 16
    std::vector<uint16_t> cdf(256);
    int sum = 0;
    for (int i = 0; i < 256; ++i) {
        cdf[i] = sum;
        sum += freqs[i];
    }

    try {
        // 3. Run
        auto res = rans_compress_cuda<RansConfig8>(data.data(), N, freqs.data(),
                                                   cdf.data(), STREAMS);

        std::cout << "Compressed " << N << " bytes into " << STREAMS
                  << " streams.\n";
        std::cout << "Stream 0 size: " << res.output_sizes[0] << " bytes.\n";
        std::cout << "Stream 0 state: " << res.final_states[0] << "\n";

    } catch (const std::exception &e) {
        std::cerr << "FAILED: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
