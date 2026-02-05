#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector>

struct RansWorkspaceWrapper;

// Interface class for RANS compression/decompression
class RansManager {
  public:
    RansManager(size_t max_data_size_hint = 0);
    ~RansManager();

    // Prevent copying
    RansManager(const RansManager &) = delete;
    RansManager &operator=(const RansManager &) = delete;

    struct CompressResult {
        bool success;
        std::vector<uint8_t> stream;
        std::vector<uint32_t> states;
        std::vector<uint32_t> sizes;
        std::vector<uint32_t> tables;
        std::vector<uint8_t> slot_map;
        std::vector<uint64_t> checkpoints;
        uint32_t num_streams;
        size_t stream_len;
    };

    CompressResult compress(const uint8_t *data, size_t size,
                            const uint16_t *freqs, const uint16_t *cdf,
                            const std::pair<size_t, size_t> shape,
                            size_t min_block_size);

    float decompress(const uint8_t *stream, size_t stream_size,
                     const uint32_t *states, const uint32_t *sizes,
                     uint32_t num_streams, size_t output_size,
                     const uint16_t *freqs, const uint16_t *cdf,
                     uint8_t *output_buffer);

  private:
    RansWorkspaceWrapper *ws;
};

void decompress(const uint8_t *stream, size_t stream_size,
                const uint32_t *states, const uint32_t *sizes,
                uint32_t num_streams, size_t output_size,
                const uint32_t *tables, const uint8_t *slot_map,
                uint8_t *output);
