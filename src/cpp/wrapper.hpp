#ifndef WRAPPER_H_
#define WRAPPER_H_

// Include the CUDA header only if CUDA is enabled
#if defined(USE_CUDA) && USE_CUDA
#include "../cuda/kernels.cuh"
#endif

#include <cstdint>
typedef uint8_t output_type;
typedef uint8_t input_type;

// C++ wrapper function that will call the CUDA implementation
void cuda_vector_add(const float *a, const float *b, float *c, int n);
float simulate_decompression(const input_type *input, output_type *output,
                            uint64_t input_size, uint64_t output_size);

#endif // WRAPPER_H_
