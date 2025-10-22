#include "wrapper.hpp"
#include <iostream>



void cuda_vector_add(const float *a, const float *b, float *c, int n) {
    std::cout << "In C++ wrapper function" << std::endl;
#if defined(USE_CUDA) && USE_CUDA
    cuda_vector_add_impl(a, b, c, n);
#else
    std::cerr << "CUDA is not enabled. Cannot perform vector addition."
              << std::endl;
#endif

    std::cout << "Exiting C++ wrapper function" << std::endl;
}

float simulate_decompression(const input_type *input, output_type *output,
                            uint64_t input_size, uint64_t output_size) {
    std::cout << "Simulating decompression in C++ wrapper function"
              << std::endl;
    std::cout << "Input size: " << input_size
              << ", Output size: " << output_size << std::endl;
#if defined(USE_CUDA) && USE_CUDA
    float decomp_time_ms = cuda_simulate_decompression(input, output, input_size, output_size);
#else
    std::cerr << "CUDA is not enabled. Cannot perform decompression." << std::endl;
#endif
    std::cout << "Exiting decompression simulation" << std::endl;
	return decomp_time_ms;
}
