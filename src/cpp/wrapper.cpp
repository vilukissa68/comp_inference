#include "wrapper.hpp"
#include "../cuda/kernels.cuh"
#include <iostream>

void cuda_vector_add(const float *a, const float *b, float *c, int n) {
    std::cout << "In C++ wrapper function" << std::endl;
    cuda_vector_add_impl(a, b, c, n);
    std::cout << "Exiting C++ wrapper function" << std::endl;
}
