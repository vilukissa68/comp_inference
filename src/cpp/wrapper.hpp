#ifndef WRAPPER_H_
#define WRAPPER_H_
#include "../cuda/kernels.cuh"
#include <cstdint>
// C++ wrapper function that will call the CUDA implementation
void cuda_vector_add(const float *a, const float *b, float *c, int n);
void simulate_decompression(const input_type *input, output_type *output,
                            uint64_t input_size, uint64_t output_size);

#endif // WRAPPER_H_
