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

    struct TiledCompressResult {
        bool success;
        std::vector<uint8_t> stream;
        std::vector<uint32_t> states;
        std::vector<uint32_t> sizes;
        std::vector<uint32_t> tables;
        std::vector<uint8_t> slot_map;
        uint32_t num_streams;
        size_t stream_len;
        std::vector<uint32_t> tile_offsets;
        std::vector<uint32_t> tile_max_lens;
        uint32_t num_tiles_k; // Height (K / 128)
        uint32_t num_tiles_n; // Width (N / 32)
        uint32_t tile_height; // Tile height (K)
        uint32_t tile_width;  // Tile width (N)
    };

    struct UncoalescedTiledCompressResult {
        bool success;
        std::vector<uint8_t> stream;
        std::vector<uint32_t> states;
        std::vector<uint32_t> sizes;
        std::vector<uint32_t> tables;
        std::vector<uint8_t> slot_map;
        uint32_t num_streams;
        size_t stream_len;
        std::vector<uint32_t> stream_offsets;
        uint32_t num_tiles_k;
        uint32_t num_tiles_n;
        uint32_t tile_height;
        uint32_t tile_width;
    };

    TiledCompressResult
    compress_tiled(const uint8_t *data, size_t size, const uint16_t *freqs,
                   const uint16_t *cdf, const std::pair<size_t, size_t> shape,
                   uint32_t tile_height, uint32_t tile_width);
    TiledCompressResult
    compress_tiled_ilp2(const uint8_t *data, size_t size, const uint16_t *freqs,
                        const uint16_t *cdf,
                        const std::pair<size_t, size_t> shape,
                        uint32_t tile_height, uint32_t tile_width);
    UncoalescedTiledCompressResult
    compress_tiled_uncoalesced(const uint8_t *data, size_t size,
                               const uint16_t *freqs, const uint16_t *cdf,
                               const std::pair<size_t, size_t> shape,
                               uint32_t tile_height, uint32_t tile_width);
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
