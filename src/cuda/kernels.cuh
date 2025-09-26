#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>

// Kernel declaration
__global__ void addVectorsKernel(const float *a, const float *b, float *c,
                                 int n);

// Host function that manages memory and launches the kernel
void cuda_vector_add_impl(const float *a, const float *b, float *c, int n);

#endif // KERNELS_CUH
