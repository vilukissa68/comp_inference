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

RansManager::CompressResult
RansManager::compress(const uint8_t *data, size_t size, const uint16_t *freqs,
                      const uint16_t *cdf,
                      const std::pair<size_t, size_t> shape,
                      size_t min_block_size) {

    auto height = shape.first;
    auto width = shape.second;
    auto config =
        StreamConfigurator<RansConfig8>::suggest(height, width, min_block_size);

    auto gpu_result = rans_compress_cuda<RansConfig8>(
        ws->internal, data, size, freqs, cdf, shape, config);

    // std::vector<uint8_t> stream_vec(gpu_result.stream_len);
    std::vector<uint32_t> states_vec(gpu_result.num_streams);
    std::vector<uint32_t> sizes_vec(gpu_result.num_streams);
    std::vector<uint32_t> tables_vec(256);
    std::vector<uint64_t> checkpoints_vec(
        gpu_result.num_streams *
        CHECKPOINT_INTERVAL); // CHECKPOINT_INTERVAL = 128

    CUDA_CHECK(cudaMemcpy(states_vec.data(), gpu_result.final_states,
                          gpu_result.num_streams * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sizes_vec.data(), gpu_result.output_sizes,
                          gpu_result.num_streams * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tables_vec.data(), gpu_result.tables,
                          256 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(checkpoints_vec.data(), gpu_result.checkpoints,
                          gpu_result.num_streams * CHECKPOINT_INTERVAL *
                              sizeof(uint64_t),
                          cudaMemcpyDeviceToHost));

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

    return {gpu_result.success, stream_vec,
            states_vec,         sizes_vec,
            tables_vec,         gpu_result.slot_to_sym,
            checkpoints_vec,    gpu_result.num_streams,
            stream_vec.size()};
}

RansManager::TiledCompressResult RansManager::compress_tiled(
    const uint8_t *data, size_t size, const uint16_t *freqs,
    const uint16_t *cdf, const std::pair<size_t, size_t> shape,
    const uint32_t tile_height, const uint32_t tile_width) {
    auto gpu_result = rans_compress_tiled_cuda<RansConfig8>(
        ws->internal, data, size, freqs, cdf, shape, tile_height, tile_width);

    const size_t height = shape.first;
    const size_t width = shape.second;
    // const uint32_t tile_height = gpu_result.tile_height;
    // const uint32_t tile_width = gpu_result.tile_width;

    // std::cout << "\n--- GPU Result Analysis ---" << std::endl;
    // std::cout << "Success Flag: " << (gpu_result.success ? "YES" : "NO")
    //           << std::endl;
    // std::cout << "Streams Allocated: " << gpu_result.num_streams <<
    // std::endl; std::cout << "Tile Size: " << tile_height << "x" << tile_width
    // << std::endl;

    // Total number of tiles
    uint32_t expected_num_tiles_k = (height + tile_height - 1) / tile_height;
    uint32_t expected_num_tiles_n = (width + tile_width - 1) / tile_width;

    if (gpu_result.num_streams !=
        expected_num_tiles_k * expected_num_tiles_n * tile_width) {
        std::cerr << "WARNING: Number of streams (" << gpu_result.num_streams
                  << ") does not match expected ("
                  << expected_num_tiles_k * expected_num_tiles_n * tile_width
                  << "). This may indicate a GPU kernel issue." << std::endl;
    }

    // Check that expected number to tiles matches what the GPU reports
    if (expected_num_tiles_k != gpu_result.num_tiles_k) {
        std::cerr << "WARNING: Number of tile rows (K) expected ("
                  << expected_num_tiles_k << ") does not match GPU result ("
                  << gpu_result.num_tiles_k
                  << "). This may indicate a GPU kernel issue." << std::endl;
    }

    if (expected_num_tiles_n != gpu_result.num_tiles_n) {
        std::cerr << "WARNING: Number of tile columns (N) expected ("
                  << expected_num_tiles_n << ") does not match GPU result ("
                  << gpu_result.num_tiles_n
                  << "). This may indicate a GPU kernel issue." << std::endl;
    }

    // 2. Download metadata
    std::vector<uint32_t> states_vec(gpu_result.num_streams);
    std::vector<uint32_t> sizes_vec(gpu_result.num_streams);
    std::vector<uint32_t> values_encoded_vec(gpu_result.num_streams);

    CUDA_CHECK(cudaMemcpy(states_vec.data(), gpu_result.final_states,
                          gpu_result.num_streams * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sizes_vec.data(), gpu_result.output_sizes,
                          gpu_result.num_streams * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(values_encoded_vec.data(), gpu_result.values_encoded,
                          gpu_result.num_streams * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    // Check for no zero-length streams, which would indicate a potential issue
    // with the GPU kernel
    for (size_t i = 0; i < sizes_vec.size(); ++i) {
        if (sizes_vec[i] == 0) {
            std::cerr
                << "WARNING: Stream " << i
                << " has zero length. This may indicate a GPU kernel issue."
                << std::endl;
        }
    }

    // Check no zero for end state
    for (size_t i = 0; i < states_vec.size(); ++i) {
        if (states_vec[i] == 0) {
            std::cerr
                << "WARNING: Stream " << i
                << " has zero end state. This may indicate a GPU kernel issue."
                << std::endl;
        }
    }

    // Check that values encoded matches expected output size
    uint64_t total_values_encoded = 0;
    for (size_t i = 0; i < values_encoded_vec.size(); ++i) {
        total_values_encoded += values_encoded_vec[i];
    }
    if (total_values_encoded != size) {
        std::cerr << "WARNING: Total values encoded (" << total_values_encoded
                  << ") does not match input size (" << size
                  << "). This may indicate a GPU kernel issue." << std::endl;
    }
    // std::cout << "Total Values Encoded: " << total_values_encoded
    //           << " / Input Size: " << size << std::endl;

    // ANALYZE SIZES
    uint32_t max_len = 0;
    uint32_t zero_count = 0;
    uint64_t total_compressed_bytes = 0;
    for (uint32_t sz : sizes_vec) {
        if (sz == 0)
            zero_count++;
        if (sz > max_len)
            max_len = sz;
        total_compressed_bytes += sz;
    }

    // std::cout << "Max Stream Length: " << max_len << std::endl;
    // std::cout << "Empty Streams: " << zero_count << " / "
    //           << gpu_result.num_streams << std::endl;
    // std::cout << "Total Bytes (Unpacked): " << total_compressed_bytes
    //           << std::endl;

    // size_t full_size = (size_t)stream_capacity * gpu_result.num_streams;
    // std::vector<uint8_t> raw_gpu_data(full_size);

    // CUDA_CHECK(cudaMemcpy(raw_gpu_data.data(), gpu_result.stream, full_size,
    //                       cudaMemcpyDeviceToHost));

    // 3. Download the raw, oversized GPU buffer
    size_t trimmed_size = (size_t)max_len * gpu_result.num_streams;
    std::vector<uint8_t> raw_gpu_data(trimmed_size);
    CUDA_CHECK(cudaMemcpy(raw_gpu_data.data(), gpu_result.stream, trimmed_size,
                          cudaMemcpyDeviceToHost));

    uint32_t num_tiles_k = (height + tile_height - 1) / tile_height;
    uint32_t num_tiles_n = (width + tile_width - 1) / tile_width;

    std::vector<uint8_t> tiled_stream_vec;
    std::vector<uint32_t> tile_offsets;
    std::vector<uint32_t> tile_max_lens(num_tiles_k * num_tiles_n, 0);

    // std::cout << "Starting Interleave. Grid: " << num_tiles_k << "x"
    //           << num_tiles_n << std::endl;

    for (uint32_t tk = 0; tk < num_tiles_k; ++tk) {
        for (uint32_t tn = 0; tn < num_tiles_n; ++tn) {
            uint32_t tile_id = tk * num_tiles_n + tn;
            tile_offsets.push_back((uint32_t)tiled_stream_vec.size());

            uint32_t local_max_len = 0;

            uint32_t tile_base_sid = tile_id * tile_width;

            // 1. Find the deepest stream in the tile
            for (uint32_t s = 0; s < tile_width; ++s) {
                uint32_t sid = tile_base_sid + s;
                if (sid < sizes_vec.size())
                    local_max_len = std::max(local_max_len, sizes_vec[sid]);
            }

            tile_max_lens[tile_id] = local_max_len;

            for (uint32_t byte_idx = 0; byte_idx < local_max_len; ++byte_idx) {
                for (uint32_t s = 0; s < tile_width; ++s) {
                    uint32_t sid = tile_base_sid + s;
                    uint32_t stream_len =
                        (sid < sizes_vec.size()) ? sizes_vec[sid] : 0;

                    // This calculates the "gap" at the beginning
                    uint32_t padding_prefix = local_max_len - stream_len;

                    if (byte_idx < padding_prefix) {
                        // Put zeros at the START of the block (low indices)
                        tiled_stream_vec.push_back(0);
                    } else {
                        // Real data starts later, ensuring Row 0 (last byte)
                        // is at the very LAST index of the tile block
                        // (byte_idx == local_max_len - 1)
                        uint32_t actual_idx = byte_idx - padding_prefix;
                        size_t src_idx =
                            (size_t)actual_idx * gpu_result.num_streams + sid;
                        tiled_stream_vec.push_back(raw_gpu_data[src_idx]);
                    }
                }
            }
            while (tiled_stream_vec.size() % 64 != 0)
                tiled_stream_vec.push_back(0);
        }
    }

    // for (uint32_t tk = 0; tk < num_tiles_k; ++tk) {
    //     for (uint32_t tn = 0; tn < num_tiles_n; ++tn) {
    //         uint32_t tile_id = tk * num_tiles_n + tn;
    //         tile_offsets.push_back((uint32_t)tiled_stream_vec.size());

    //         uint32_t local_max_len = 0;
    //         uint32_t tile_base_sid = tile_id * tile_width;

    //         // 1. Find the deepest stream in this tile group (G = tile_width)
    //         for (uint32_t s = 0; s < tile_width; ++s) {
    //             uint32_t sid = tile_base_sid + s;
    //             if (sid < sizes_vec.size())
    //                 local_max_len = std::max(local_max_len, sizes_vec[sid]);
    //         }

    //         tile_max_lens[tile_id] = local_max_len;

    //         // 2. Calculate where this tile starts in the raw GPU buffer
    //         // This assumes the GPU wrote: tile_id * (capacity * tile_width)
    //         size_t raw_tile_base =
    //             (size_t)tile_id * stream_capacity * tile_width;

    //         // 3. Interleave and Pad
    //         for (uint32_t byte_idx = 0; byte_idx < local_max_len; ++byte_idx)
    //         {
    //             for (uint32_t s = 0; s < tile_width; ++s) {
    //                 uint32_t sid = tile_base_sid + s;
    //                 uint32_t stream_len =
    //                     (sid < sizes_vec.size()) ? sizes_vec[sid] : 0;

    //                 uint32_t padding_prefix = local_max_len - stream_len;

    //                 if (byte_idx < padding_prefix) {
    //                     tiled_stream_vec.push_back(0);
    //                 } else {
    //                     uint32_t actual_idx = byte_idx - padding_prefix;

    //                     // KEY CHANGE: Indexing is now local to the tile
    //                     block
    //                     // byte_idx * tile_width + s
    //                     size_t src_idx =
    //                         raw_tile_base + (actual_idx * tile_width) + s;

    //                     tiled_stream_vec.push_back(raw_gpu_data[src_idx]);
    //                 }
    //             }
    //         }

    //         // 4. Align the tile to 64 bytes for GPU memory controller
    //         // friendliness
    //         while (tiled_stream_vec.size() % 64 != 0)
    //             tiled_stream_vec.push_back(0);
    //     }
    // }
    // Add final offset for end of last tile
    // tile_offsets.push_back((uint32_t)tiled_stream_vec.size());

    size_t stream_len = tiled_stream_vec.size();
    // std::cout << "Interleaving Success. Final Packed Size: " << stream_len
    //           << " bytes\n"
    //           << std::endl;

    // Check that we have equal amount of offsets and tiles
    if (tile_offsets.size() != num_tiles_k * num_tiles_n) {
        std::cerr << "CRITICAL ERROR: Number of tile offsets ("
                  << tile_offsets.size()
                  << ") does not match expected number of tiles ("
                  << num_tiles_k * num_tiles_n
                  << "). This indicates a logic error in offset calculation."
                  << std::endl;
        return {false};
    }

    std::vector<uint32_t> tables_vec(256);
    CUDA_CHECK(cudaMemcpy(tables_vec.data(), gpu_result.tables,
                          256 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    return {gpu_result.success,
            tiled_stream_vec,
            states_vec,
            sizes_vec,
            tables_vec,
            gpu_result.slot_to_sym,
            gpu_result.num_streams,
            stream_len,
            tile_offsets,
            tile_max_lens,
            num_tiles_k,
            num_tiles_n,
            tile_height,
            tile_width};
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
