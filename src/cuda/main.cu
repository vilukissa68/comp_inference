#include <iostream>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

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

template <typename Config>
struct RansResult {
    std::vector<typename Config::io_t> stream;
    std::vector<typename Config::state_t> final_states;
    std::vector<uint32_t> output_sizes;
};

struct RansConfig8 : public RansTraits<uint8_t, uint32_t, uint8_t, 12> {
    using sym_info_t = RansSymInfoPacked;
};

template <typename Config>
RansResult<Config> rans_compress_cuda(
    const typename Config::symbol_t *host_data, size_t input_size,
    const uint16_t *host_freqs, const uint16_t *host_cdf, uint32_t num_streams) 
{
    using symbol_t = typename Config::symbol_t;
    using io_t = typename Config::io_t;
    using sym_info_t = typename Config::sym_info_t;

    size_t syms_per_stream = (input_size + num_streams - 1) / num_streams;
    uint32_t capacity = (uint32_t)(syms_per_stream * 1.2) + 32;

    std::vector<sym_info_t> host_sym_info(Config::vocab_size);
    for (int i = 0; i < Config::vocab_size; ++i) {
        host_sym_info[i].freq = static_cast<decltype(host_sym_info[i].freq)>(host_freqs[i]);
        host_sym_info[i].cdf = static_cast<decltype(host_sym_info[i].cdf)>(host_cdf[i]);
    }

    sym_info_t *d_sym_info;
    symbol_t *d_symbols;
    io_t *d_output;
    typename Config::state_t *d_states;
    uint32_t *d_sizes;

    CUDA_CHECK(cudaMalloc(&d_sym_info, Config::vocab_size * sizeof(sym_info_t)));
    CUDA_CHECK(cudaMalloc(&d_symbols, input_size * sizeof(symbol_t)));
    CUDA_CHECK(cudaMalloc(&d_output, num_streams * capacity * sizeof(io_t)));
    CUDA_CHECK(cudaMalloc(&d_states, num_streams * sizeof(typename Config::state_t)));
    CUDA_CHECK(cudaMalloc(&d_sizes, num_streams * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_sym_info, host_sym_info.data(), Config::vocab_size * sizeof(sym_info_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_symbols, host_data, input_size * sizeof(symbol_t), cudaMemcpyHostToDevice));

    RansEncoderCtx<Config> ctx;
    ctx.num_streams = num_streams;
    ctx.stream_capacity = capacity;
    ctx.symbols = d_symbols;
    const_cast<uint32_t &>(ctx.input_size) = (uint32_t)input_size;
    ctx.output = d_output;
    ctx.final_states = d_states;
    ctx.output_sizes = d_sizes;
    ctx.tables.sym_info = d_sym_info;

    int block_size = 256;
    int grid_size = (num_streams + block_size - 1) / block_size;
    rans_compress_kernel<Config><<<grid_size, block_size>>>(ctx);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    RansResult<Config> result;
    result.stream.resize(num_streams * capacity);
    result.final_states.resize(num_streams);
    result.output_sizes.resize(num_streams);

    CUDA_CHECK(cudaMemcpy(result.stream.data(), d_output, num_streams * capacity * sizeof(io_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.final_states.data(), d_states, num_streams * sizeof(typename Config::state_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.output_sizes.data(), d_sizes, num_streams * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cudaFree(d_sym_info); cudaFree(d_symbols); cudaFree(d_output); cudaFree(d_states); cudaFree(d_sizes);
    return result;
}

template <typename Config>
std::vector<typename Config::symbol_t> rans_decompress_cuda(
    const typename Config::io_t *host_stream_data, size_t stream_data_size,
    const typename Config::state_t *host_states, const uint32_t *host_stream_sizes,
    uint32_t num_streams, uint32_t capacity_per_stream, uint32_t symbols_per_stream,
    const uint16_t *host_freqs, const uint16_t *host_cdf) 
{
    using symbol_t = typename Config::symbol_t;
    using sym_info_t = typename Config::sym_info_t;

    std::vector<sym_info_t> host_sym_info(Config::vocab_size);
    std::vector<symbol_t> host_slot_map(Config::prob_scale);

    for (int i = 0; i < Config::vocab_size; ++i) {
        host_sym_info[i].freq = (decltype(host_sym_info[i].freq))host_freqs[i];
        host_sym_info[i].cdf = (decltype(host_sym_info[i].cdf))host_cdf[i];
        for(int j=0; j<host_freqs[i]; ++j) host_slot_map[host_cdf[i] + j] = (symbol_t)i;
    }

    sym_info_t *d_sym_info;
    symbol_t *d_slot_map;
    typename Config::io_t *d_input;
    typename Config::state_t *d_init_states;
    uint32_t *d_input_sizes;
    symbol_t *d_output;

    size_t input_bytes = stream_data_size * sizeof(typename Config::io_t);
    size_t output_bytes = symbols_per_stream * num_streams * sizeof(symbol_t);

    CUDA_CHECK(cudaMalloc(&d_sym_info, Config::vocab_size * sizeof(sym_info_t)));
    CUDA_CHECK(cudaMalloc(&d_slot_map, Config::prob_scale * sizeof(symbol_t)));
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_init_states, num_streams * sizeof(typename Config::state_t)));
    CUDA_CHECK(cudaMalloc(&d_input_sizes, num_streams * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    CUDA_CHECK(cudaMemcpy(d_sym_info, host_sym_info.data(), Config::vocab_size * sizeof(sym_info_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_slot_map, host_slot_map.data(), Config::prob_scale * sizeof(symbol_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input, host_stream_data, input_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_init_states, host_states, num_streams * sizeof(typename Config::state_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_sizes, host_stream_sizes, num_streams * sizeof(uint32_t), cudaMemcpyHostToDevice));

    RansDecoderCtx<Config> ctx;
    ctx.input = d_input;
    ctx.initial_states = d_init_states;
    ctx.input_sizes = d_input_sizes;
    ctx.output = d_output;
    ctx.output_size = symbols_per_stream;
    ctx.stream_capacity = capacity_per_stream;
    ctx.num_streams = num_streams;
    ctx.tables.sym_info = d_sym_info;
    ctx.tables.slot_to_sym = d_slot_map;

    int block_size = 256;
    int grid_size = (num_streams + block_size - 1) / block_size;
    rans_decompress_kernel<Config><<<grid_size, block_size>>>(ctx);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<symbol_t> result(num_streams * symbols_per_stream);
    CUDA_CHECK(cudaMemcpy(result.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_sym_info); cudaFree(d_slot_map); cudaFree(d_input);
    cudaFree(d_init_states); cudaFree(d_input_sizes); cudaFree(d_output);
    return result;
}

int main() {
    const int STREAMS = 4096;
    const int SYMBOLS_PER_STREAM = 39000;
    const int N = STREAMS * SYMBOLS_PER_STREAM;

    std::vector<uint8_t> original_data(N);
    for (int i = 0; i < N; ++i) original_data[i] = (i * 7) % 32;

    std::vector<uint16_t> freqs(256), cdf(256);
    int used_symbols = 32, high_freq = 120;
    for(int i = 0; i < 256; ++i) freqs[i] = (i < used_symbols) ? high_freq : 1;
    
    int sum = 0;
    for(int f : freqs) sum += f;
    if (sum < 4096) freqs[0] += (4096 - sum);
    
    sum = 0;
    for (int i = 0; i < 256; ++i) { cdf[i] = sum; sum += freqs[i]; }

    try {
        auto encoded = rans_compress_cuda<RansConfig8>(
            original_data.data(), N, freqs.data(), cdf.data(), STREAMS);

        uint32_t capacity = encoded.stream.size() / STREAMS;
        
        auto decoded = rans_decompress_cuda<RansConfig8>(
            encoded.stream.data(), encoded.stream.size(),
            encoded.final_states.data(), encoded.output_sizes.data(),
            STREAMS, capacity, SYMBOLS_PER_STREAM, freqs.data(), cdf.data());

        if (original_data.size() != decoded.size()) throw std::runtime_error("Size mismatch");
        
        int errors = 0;
        for(size_t i=0; i<original_data.size(); ++i) {
            if(original_data[i] != decoded[i]) {
                if(errors++ < 5) std::cout << "Mismatch " << i << ": " << (int)original_data[i] << "!=" << (int)decoded[i] << "\n";
            }
        }

        if(errors == 0) std::cout << "SUCCESS. Size: " << N << "\n";
        else std::cout << "FAILURE: " << errors << " errors.\n";

		size_t stream_payload_bytes = 0;
		for (uint32_t size : encoded.output_sizes) {
			stream_payload_bytes += size;
		}

		size_t state_bytes = encoded.final_states.size() * sizeof(uint32_t);

		size_t table_bytes = 256 * sizeof(uint16_t);

		size_t header_bytes = encoded.output_sizes.size() * sizeof(uint32_t);

		size_t total_file_size = stream_payload_bytes + state_bytes + table_bytes + header_bytes;

		std::cout << "--- True File Size Breakdown ---\n";
		std::cout << "1. Model (Freq Table) : " << table_bytes << " bytes\n";
		std::cout << "2. Header (Stream Sz) : " << header_bytes << " bytes\n";
		std::cout << "3. States (Init Vals) : " << state_bytes << " bytes\n";
		std::cout << "4. Payload (Data)     : " << stream_payload_bytes << " bytes\n";
		std::cout << "--------------------------------\n";
		std::cout << "TOTAL SIZE            : " << total_file_size << " bytes\n";
		std::cout << "Original Size         : " << N << " bytes\n";
		std::cout << "Compression Ratio     : " << (float)total_file_size / N * 100.0f << "%\n";

    } catch (const std::exception &e) {
        std::cerr << "Ex: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
