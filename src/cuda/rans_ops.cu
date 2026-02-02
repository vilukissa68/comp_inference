#include "rans.hpp"
#include "rans_host.cuh"
#include <cstring>
#include <vector>

struct RansWorkspaceWrapper {
    RansWorkspace<RansConfig8> internal;
};

RansManager::RansManager(size_t max_data_size_hint) {
    ws = new RansWorkspaceWrapper();
}

RansManager::~RansManager() { delete ws; }

RansManager::CompressResult RansManager::compress(const uint8_t *data,
                                                  size_t size,
                                                  const uint16_t *freqs,
                                                  const uint16_t *cdf) {
    auto config = StreamConfigurator<RansConfig8>::suggest(size);

    auto gpu_result = rans_compress_cuda<RansConfig8>(ws->internal, data, size,
                                                      freqs, cdf, config);

    // std::vector<uint8_t> stream_vec(gpu_result.stream_len);
    std::vector<uint32_t> states_vec(gpu_result.num_streams);
    std::vector<uint32_t> sizes_vec(gpu_result.num_streams);
    std::vector<uint32_t> tables_vec(256);
    // std::vector<uint16_t> slots_vec(4096); // PROB_SCALE = 4096

    // CUDA_CHECK(cudaMemcpy(stream_vec.data(), gpu_result.stream,
    // gpu_result.stream_len, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(states_vec.data(), gpu_result.final_states,
                          gpu_result.num_streams * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sizes_vec.data(), gpu_result.output_sizes,
                          gpu_result.num_streams * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tables_vec.data(), gpu_result.tables,
                          256 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    // CUDA_CHECK(cudaMemcpy(slots_vec.data(), gpu_result.slot_to_sym,
    //                       4096 * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    // Find longest stream length
    uint32_t max_len = 0;
    for (uint32_t sz : sizes_vec) {
        if (sz > max_len) {
            max_len = sz;
        }
    }

    // Allocate trimmed stream vector
    size_t trimmed_size = (size_t)max_len * gpu_result.num_streams;
    std::vector<uint8_t> stream_vec(trimmed_size);

    // Copy trimmed data from GPU
    CUDA_CHECK(cudaMemcpy(stream_vec.data(), gpu_result.stream, trimmed_size,
                          cudaMemcpyDeviceToHost));

    return {
        gpu_result.success,
        stream_vec,
        states_vec,
        sizes_vec,
        tables_vec,
        gpu_result.slot_to_sym,
        gpu_result.num_streams,
        stream_vec.size() // Return the TRIMMED size
    };
}

float RansManager::decompress(const uint8_t *stream, size_t stream_size,
                              const uint32_t *states, const uint32_t *sizes,
                              uint32_t num_streams, size_t output_size,
                              const uint16_t *freqs, const uint16_t *cdf,
                              uint8_t *output_buffer) {
    uint32_t syms_per_stream = (output_size + num_streams - 1) / num_streams;

    auto result = rans_decompress_cuda_ws<RansConfig8>(
        ws->internal, stream, stream_size, states, sizes, num_streams,
        syms_per_stream, freqs, cdf);

    CUDA_CHECK(cudaMemcpy(output_buffer, result.first, output_size,
                          cudaMemcpyDeviceToHost));

    return result.second;
}

void decompress(const uint8_t *stream, size_t stream_size,
                const uint32_t *states, const uint32_t *sizes,
                uint32_t num_streams, size_t output_size,
                const uint32_t *tables, const uint8_t *slot_map,
                uint8_t *output_buffer) {
    uint32_t syms_per_stream = (output_size + num_streams - 1) / num_streams;

    rans_decompress_cuda<RansConfig8>(stream, stream_size, states, sizes,
                                      num_streams, syms_per_stream, tables,
                                      slot_map, output_buffer);
}
