#pragma once
#include <vector>
#include <cstdint>

struct RansWorkspaceWrapper; 

// Interface class for RANS compression/decompression
class RansManager {
public:
    RansManager(size_t max_data_size_hint = 0);
    ~RansManager();

    // Prevent copying
    RansManager(const RansManager&) = delete;
    RansManager& operator=(const RansManager&) = delete;

    struct CompressResult {
        std::vector<uint8_t> stream;
        std::vector<uint32_t> states;
        std::vector<uint32_t> sizes;
        uint32_t num_streams;
        size_t stream_len;
    };

    CompressResult compress(
        const uint8_t* data, size_t size,
        const uint16_t* freqs, const uint16_t* cdf
    );

    float decompress(
        const uint8_t* stream,
        size_t stream_size,
        const uint32_t* states,
        const uint32_t* sizes,
        uint32_t num_streams,
        size_t output_size,
        const uint16_t* freqs, const uint16_t* cdf,
        uint8_t* output_buffer,
    );

private:
    RansWorkspaceWrapper* ws;
};
