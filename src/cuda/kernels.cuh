#ifndef KERNELS_CUH
#define KERNELS_CUH
#include <cuda_runtime.h>
#include <stdint.h>

typedef uint8_t output_type;
typedef uint8_t input_type;

// Kernel declaration
__global__ void addVectorsKernel(const float *a, const float *b, float *c,
                                 int n);
__global__ void simulate_decompression_kernel(const input_type *input,
                                              output_type *output, uint64_t n,
                                              uint64_t output_block_size);

// Host function that manages memory and launches the kernel
void cuda_vector_add_impl(const float *a, const float *b, float *c, int n);
float cuda_simulate_decompression(const input_type *input, output_type *output,
                                 uint64_t input_size, uint64_t output_size);

#endif // KERNELS_CUH
