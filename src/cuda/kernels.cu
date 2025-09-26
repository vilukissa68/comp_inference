#include "kernels.cuh"
#include <stdio.h>

__global__ void addVectorsKernel(const float *a, const float *b, float *c,
                                 int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void cuda_vector_add_impl(const float *a, const float *b, float *c, int n) {
    // Basic device checking
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Device count: %d\n", deviceCount);

    // Print input values for debugging
    printf("First values: a[0]=%f, b[0]=%f\n", a[0], b[0]);

    // Allocate device memory
    float *d_a = NULL, *d_b = NULL, *d_c = NULL;
    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMalloc((void **)&d_b, n * sizeof(float));
    cudaMalloc((void **)&d_c, n * sizeof(float));

    // Check allocations
    if (d_a == NULL || d_b == NULL || d_c == NULL) {
        printf("Memory allocation failed\n");
        return;
    }

    // Copy to device
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    printf("Launching kernel with %d blocks, %d threads\n", numBlocks,
           blockSize);
    addVectorsKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // Synchronize and check errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Kernel execution successful\n");
    }

    // Copy back to host
    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: c[0]=%f\n", c[0]);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
