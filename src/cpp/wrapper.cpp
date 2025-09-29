#include "wrapper.hpp"
#include <iostream>

void cuda_vector_add(const float *a, const float *b, float *c, int n) {
    std::cout << "In C++ wrapper function" << std::endl;
    cuda_vector_add_impl(a, b, c, n);
    std::cout << "Exiting C++ wrapper function" << std::endl;
}

void simulate_decompression(const input_type *input, output_type *output,
                            uint64_t input_size, uint64_t output_size) {
    std::cout << "Simulating decompression in C++ wrapper function"
              << std::endl;
    std::cout << "Input size: " << input_size
              << ", Output size: " << output_size << std::endl;
    cuda_simulate_decompression(input, output, input_size, output_size);

    std::cout << "Exiting decompression simulation" << std::endl;
}
