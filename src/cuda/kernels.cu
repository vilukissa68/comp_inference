#include "kernels.cuh"
#include <stdio.h>

#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

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

__global__ void simulate_decompression_kernel(const input_type *input,
                                              output_type *output,
                                              uint64_t input_size,
                                              uint64_t output_size) {
    uint64_t idx =
        uint64_t(blockIdx.x) * uint64_t(blockDim.x) + uint64_t(threadIdx.x);
    if (idx < output_size) {
        if (idx < input_size) {
            output[idx] = static_cast<output_type>(input[idx]) + 1;
        } else {
            output[idx] = 2;
        }
    }
}

void cuda_simulate_decompression(const input_type *input, output_type *output,
                                 uint64_t input_size, uint64_t output_size) {
    int deviceId = 0; // Change this if you want to use a different GPU
    CHECK_CUDA(cudaSetDevice(deviceId));

    // Basic device checking
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    printf("Device count: %d\n", deviceCount);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    printf("maxGridDimX = %d, maxGridDimY = %d, maxGridDimZ = %d\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    // Allocate device memory
    input_type *d_input = nullptr;
    output_type *d_output = nullptr;

    float memcpy_time_ms = 0.0f;
    float decomp_time_ms = 0.0f;
    cudaEvent_t memcpy_start, memcpy_stop, decomp_start, decomp_stop;
    cudaEventCreate(&memcpy_start);
    cudaEventCreate(&memcpy_stop);
    cudaEventCreate(&decomp_start);
    cudaEventCreate(&decomp_stop);

    CHECK_CUDA(cudaMalloc((void **)&d_input, input_size * sizeof(input_type)));
    CHECK_CUDA(
        cudaMalloc((void **)&d_output, output_size * sizeof(output_type)));

    cudaEventRecord(memcpy_start);
    CHECK_CUDA(cudaMemcpy(d_input, input, input_size * sizeof(input_type),
                          cudaMemcpyHostToDevice));
    cudaEventRecord(memcpy_stop);
    cudaEventSynchronize(memcpy_stop);
    cudaEventElapsedTime(&memcpy_time_ms, memcpy_start, memcpy_stop);

    // Kernel launch configuration
    uint64_t threads_per_block = 256;
    uint64_t num_blocks =
        (output_size + threads_per_block - 1ULL) / threads_per_block;
    printf("Launching kernel with %llu blocks, %llu threads\n", num_blocks,
           threads_per_block);

    cudaEventRecord(decomp_start);
    simulate_decompression_kernel<<<num_blocks, threads_per_block>>>(
        d_input, d_output, input_size, output_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Launch error: %s\n", cudaGetErrorString(err));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    cudaEventRecord(decomp_stop);
    cudaEventSynchronize(decomp_stop);
    cudaEventElapsedTime(&decomp_time_ms, decomp_start, decomp_stop);

    printf("Decompression kernel time: %f ms\n", decomp_time_ms);
    printf("Memory copy time: %f ms\n", memcpy_time_ms);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Decompression kernel execution successful\n");
    }

    CHECK_CUDA(cudaMemcpy(output, d_output, output_size * sizeof(output_type),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_output);
}
