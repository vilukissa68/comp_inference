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

RansManager::UncoalescedTiledCompressResult
RansManager::compress_tiled_uncoalesced(const uint8_t *data, size_t size,
                                        const uint16_t *freqs,
                                        const uint16_t *cdf,
                                        const std::pair<size_t, size_t> shape,
                                        uint32_t tile_height,
                                        uint32_t tile_width) {
    // Drop ILP2 from kernel call
    auto gpu_result = rans_compress_tiled_uncoalesced_cuda<RansConfig8>(
        ws->internal, data, size, freqs, cdf, shape, tile_height, tile_width);

    const size_t height = shape.first;
    const size_t width = shape.second;

    while (tile_height > height)
        tile_height /= 2;
    while (tile_width > width)
        tile_width /= 2;

    uint32_t expected_num_tiles_k = (height + tile_height - 1) / tile_height;
    uint32_t expected_num_tiles_n = (width + tile_width - 1) / tile_width;

    // FIX 1: ILP-2 removed, streams are exactly tiles * tile_width
    uint32_t expected_num_streams =
        expected_num_tiles_k * expected_num_tiles_n * tile_width;

    if (gpu_result.num_streams != expected_num_streams) {
        std::cerr << "WARNING: Number of streams (" << gpu_result.num_streams
                  << ") does not match expected (" << expected_num_streams
                  << "). This may indicate a GPU kernel issue." << std::endl;
    }

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

    for (size_t i = 0; i < states_vec.size(); ++i) {
        if (states_vec[i] == 0) {
            std::cerr
                << "WARNING: Stream " << i
                << " has zero end state. This may indicate a GPU kernel issue."
                << std::endl;
        }
    }

    uint64_t total_values_encoded = 0;
    for (size_t i = 0; i < values_encoded_vec.size(); ++i) {
        total_values_encoded += values_encoded_vec[i];
    }
    if (total_values_encoded != size) {
        std::cerr << "WARNING: Total values encoded (" << total_values_encoded
                  << ") does not match input size (" << size
                  << "). This may indicate a GPU kernel issue." << std::endl;
    }

    // ANALYZE SIZES
    uint32_t max_len = 0;
    uint64_t total_compressed_bytes = 0;
    for (uint32_t sz : sizes_vec) {
        if (sz > max_len)
            max_len = sz;
        total_compressed_bytes += sz; // Track exactly how many bytes we need
    }

    size_t trimmed_size = (size_t)max_len * gpu_result.num_streams;
    std::vector<uint8_t> raw_gpu_data(trimmed_size);
    CUDA_CHECK(cudaMemcpy(raw_gpu_data.data(), gpu_result.stream, trimmed_size,
                          cudaMemcpyDeviceToHost));

    uint32_t num_tiles_k = (height + tile_height - 1) / tile_height;
    uint32_t num_tiles_n = (width + tile_width - 1) / tile_width;

    std::vector<uint8_t> packed_stream_vec;
    packed_stream_vec.reserve(
        total_compressed_bytes); // Prevent reallocation overhead
    std::vector<uint32_t> stream_offsets(gpu_result.num_streams, 0);

    // --- FIX 2: DENSE PACKING ---
    // Instead of padding streams with zeros to match local_max_len,
    // we sequentially concatenate exactly the valid bytes from each stream.
    for (uint32_t sid = 0; sid < gpu_result.num_streams; ++sid) {
        // Record where this stream begins in the dense array
        stream_offsets[sid] = (uint32_t)packed_stream_vec.size();

        uint32_t stream_len = sizes_vec[sid];

        // Extract valid bytes from the strided GPU layout
        for (uint32_t byte_idx = 0; byte_idx < stream_len; ++byte_idx) {
            size_t src_idx = (size_t)byte_idx * gpu_result.num_streams + sid;
            packed_stream_vec.push_back(raw_gpu_data[src_idx]);
        }
    }

    size_t stream_len = packed_stream_vec.size();

    // Ensure we generated an offset for every stream
    if (stream_offsets.size() != gpu_result.num_streams) {
        std::cerr << "CRITICAL ERROR: Offset calculation failed." << std::endl;
        return {false};
    }

    std::vector<uint32_t> tables_vec(256);
    CUDA_CHECK(cudaMemcpy(tables_vec.data(), gpu_result.tables,
                          256 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    return {gpu_result.success,
            packed_stream_vec,
            states_vec,
            sizes_vec,
            tables_vec,
            gpu_result.slot_to_sym,
            gpu_result.num_streams,
            stream_len,
            stream_offsets, // <-- We now return stream_offsets
            num_tiles_k,
            num_tiles_n,
            tile_height,
            tile_width};
}

RansManager::TiledCompressResult
RansManager::compress_tiled_ilp2(const uint8_t *data, size_t size,
                                 const uint16_t *freqs, const uint16_t *cdf,
                                 const std::pair<size_t, size_t> shape,
                                 uint32_t tile_height, uint32_t tile_width) {
    auto gpu_result = rans_compress_tiled_cuda_ilp2<RansConfig8>(
        ws->internal, data, size, freqs, cdf, shape, tile_height, tile_width);

    const size_t height = shape.first;
    const size_t width = shape.second;

    // Check that tile height and width fit within the shape dimensions
    // If not decrease to the next power of two

    while (tile_height > height)
        tile_height /= 2;

    while (tile_width > width)
        tile_width /= 2;

    // Total number of tiles
    uint32_t expected_num_tiles_k = (height + tile_height - 1) / tile_height;
    uint32_t expected_num_tiles_n = (width + tile_width - 1) / tile_width;

    // FIX 1: ILP-2 doubles the streams per tile.
    uint32_t expected_num_streams =
        expected_num_tiles_k * expected_num_tiles_n * tile_width * 2;

    if (gpu_result.num_streams != expected_num_streams) {
        std::cerr << "WARNING: Number of streams (" << gpu_result.num_streams
                  << ") does not match expected (" << expected_num_streams
                  << "). This may indicate a GPU kernel issue." << std::endl;
    }

    // 2. Download metadata
    std::vector<uint32_t> states_vec(gpu_result.num_streams);
    std::vector<uint32_t> sizes_vec(gpu_result.num_streams);

    // NOTE: values_encoded_vec is intermediate debug data. Do NOT save this to
    // safetensors!
    std::vector<uint32_t> values_encoded_vec(gpu_result.num_streams);

    CUDA_CHECK(cudaMemcpy(states_vec.data(), gpu_result.final_states,
                          gpu_result.num_streams * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sizes_vec.data(), gpu_result.output_sizes,
                          gpu_result.num_streams * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    // FIX 2: Zero-length streams are EXPECTED for edge tiles in ILP-2.
    // We only warn if a stream is zero length AND it is NOT in the final K
    // tile.
    for (size_t i = 0; i < sizes_vec.size(); ++i) {
        if (sizes_vec[i] == 0) {
            uint32_t streams_per_tile = tile_width * 2;
            uint32_t global_tile_idx = i / streams_per_tile;
            // Assuming block mapping is (tile_n, tile_k) based on dim3
            // grid(num_tiles_n, num_tiles_k)
            uint32_t tile_k_idx = global_tile_idx / expected_num_tiles_n;

            if (tile_k_idx != expected_num_tiles_k - 1) {
                std::cerr
                    << "WARNING: Non-edge Stream " << i
                    << " has zero length. This indicates a GPU kernel issue."
                    << std::endl;
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(values_encoded_vec.data(), gpu_result.values_encoded,
                          gpu_result.num_streams * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

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

    uint32_t streams_per_tile = tile_width * 2;
    for (uint32_t tk = 0; tk < num_tiles_k; ++tk) {
        for (uint32_t tn = 0; tn < num_tiles_n; ++tn) {
            uint32_t tile_id = tk * num_tiles_n + tn;
            tile_offsets.push_back((uint32_t)tiled_stream_vec.size());

            uint32_t local_max_len = 0;
            // FIX 2: Correct base SID calculation
            uint32_t tile_base_sid = tile_id * streams_per_tile;

            // 1. Find the deepest stream in the tile across ALL ILP streams
            for (uint32_t s = 0; s < streams_per_tile; ++s) {
                uint32_t sid = tile_base_sid + s;
                if (sid < sizes_vec.size())
                    local_max_len = std::max(local_max_len, sizes_vec[sid]);
            }

            tile_max_lens[tile_id] = local_max_len;

            // 2. Interleave the bytes
            for (uint32_t byte_idx = 0; byte_idx < local_max_len; ++byte_idx) {
                // Iterate over all streams in this tile (Top Half A and Bottom
                // Half B)
                for (uint32_t s = 0; s < streams_per_tile; ++s) {
                    uint32_t sid = tile_base_sid + s;
                    uint32_t stream_len =
                        (sid < sizes_vec.size()) ? sizes_vec[sid] : 0;

                    uint32_t padding_prefix = local_max_len - stream_len;

                    if (byte_idx < padding_prefix) {
                        tiled_stream_vec.push_back(0);
                    } else {
                        uint32_t actual_idx = byte_idx - padding_prefix;
                        // src_idx uses the global num_streams (which is already
                        // doubled)
                        size_t src_idx =
                            (size_t)actual_idx * gpu_result.num_streams + sid;
                        tiled_stream_vec.push_back(raw_gpu_data[src_idx]);
                    }
                }
            }

            // Align tile to 64 bytes for coalesced memory access
            // while (tiled_stream_vec.size() % 64 != 0)
            //    tiled_stream_vec.push_back(0);
        }
    }
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

RansManager::TiledCompressResult
RansManager::compress_tiled(const uint8_t *data, size_t size,
                            const uint16_t *freqs, const uint16_t *cdf,
                            const std::pair<size_t, size_t> shape,
                            uint32_t tile_height, uint32_t tile_width) {
    auto gpu_result = rans_compress_tiled_cuda<RansConfig8>(
        ws->internal, data, size, freqs, cdf, shape, tile_height, tile_width);

    const size_t height = shape.first;
    const size_t width = shape.second;

    // Check that tile height and width fit within the shape dimensions
    while (tile_height > height)
        tile_height /= 2;

    while (tile_width > width)
        tile_width /= 2;

    // Total number of tiles
    uint32_t expected_num_tiles_k = (height + tile_height - 1) / tile_height;
    uint32_t expected_num_tiles_n = (width + tile_width - 1) / tile_width;

    // Standard baseline: 1 stream per tile column
    uint32_t expected_num_streams =
        expected_num_tiles_k * expected_num_tiles_n * tile_width;

    if (gpu_result.num_streams != expected_num_streams) {
        std::cerr << "WARNING: Number of streams (" << gpu_result.num_streams
                  << ") does not match expected (" << expected_num_streams
                  << "). This may indicate a GPU kernel issue." << std::endl;
    }

    // 2. Download metadata
    std::vector<uint32_t> states_vec(gpu_result.num_streams);
    std::vector<uint32_t> sizes_vec(gpu_result.num_streams);

    // NOTE: values_encoded_vec is intermediate debug data. Do NOT save this to
    // safetensors!
    std::vector<uint32_t> values_encoded_vec(gpu_result.num_streams);

    CUDA_CHECK(cudaMemcpy(states_vec.data(), gpu_result.final_states,
                          gpu_result.num_streams * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sizes_vec.data(), gpu_result.output_sizes,
                          gpu_result.num_streams * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

    // Zero-length streams are expected for edge tiles.
    // We only warn if a stream is zero length AND it is NOT in the final K
    // tile.
    for (size_t i = 0; i < sizes_vec.size(); ++i) {
        if (sizes_vec[i] == 0) {
            uint32_t streams_per_tile = tile_width;
            uint32_t global_tile_idx = i / streams_per_tile;
            // Assuming block mapping is (tile_n, tile_k) based on dim3
            // grid(num_tiles_n, num_tiles_k)
            uint32_t tile_k_idx = global_tile_idx / expected_num_tiles_n;

            if (tile_k_idx != expected_num_tiles_k - 1) {
                std::cerr
                    << "WARNING: Non-edge Stream " << i
                    << " has zero length. This indicates a GPU kernel issue."
                    << std::endl;
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(values_encoded_vec.data(), gpu_result.values_encoded,
                          gpu_result.num_streams * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));

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

    size_t trimmed_size = (size_t)max_len * gpu_result.num_streams;
    std::vector<uint8_t> raw_gpu_data(trimmed_size);
    CUDA_CHECK(cudaMemcpy(raw_gpu_data.data(), gpu_result.stream, trimmed_size,
                          cudaMemcpyDeviceToHost));

    uint32_t num_tiles_k = (height + tile_height - 1) / tile_height;
    uint32_t num_tiles_n = (width + tile_width - 1) / tile_width;

    std::vector<uint8_t> tiled_stream_vec;
    std::vector<uint32_t> tile_offsets;
    std::vector<uint32_t> tile_max_lens(num_tiles_k * num_tiles_n, 0);

    uint32_t streams_per_tile = tile_width;
    for (uint32_t tk = 0; tk < num_tiles_k; ++tk) {
        for (uint32_t tn = 0; tn < num_tiles_n; ++tn) {
            uint32_t tile_id = tk * num_tiles_n + tn;
            tile_offsets.push_back((uint32_t)tiled_stream_vec.size());

            uint32_t local_max_len = 0;
            uint32_t tile_base_sid = tile_id * streams_per_tile;

            // 1. Find the deepest stream in the tile
            for (uint32_t s = 0; s < streams_per_tile; ++s) {
                uint32_t sid = tile_base_sid + s;
                if (sid < sizes_vec.size())
                    local_max_len = std::max(local_max_len, sizes_vec[sid]);
            }

            tile_max_lens[tile_id] = local_max_len;

            // 2. Interleave the bytes
            for (uint32_t byte_idx = 0; byte_idx < local_max_len; ++byte_idx) {
                // Iterate over all streams in this tile
                for (uint32_t s = 0; s < streams_per_tile; ++s) {
                    uint32_t sid = tile_base_sid + s;
                    uint32_t stream_len =
                        (sid < sizes_vec.size()) ? sizes_vec[sid] : 0;

                    uint32_t padding_prefix = local_max_len - stream_len;

                    if (byte_idx < padding_prefix) {
                        tiled_stream_vec.push_back(0);
                    } else {
                        uint32_t actual_idx = byte_idx - padding_prefix;
                        size_t src_idx =
                            (size_t)actual_idx * gpu_result.num_streams + sid;
                        tiled_stream_vec.push_back(raw_gpu_data[src_idx]);
                    }
                }
            }
        }
    }

    size_t stream_len = tiled_stream_vec.size();

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
