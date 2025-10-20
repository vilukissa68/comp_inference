#include "kernels.cuh"
#include <cassert>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

#define SHARED_MEM_SIZE 49152 // 48KB

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

// __global__ void simulate_decompression_kernel(const input_type *input,
//                                               output_type *output,
//                                               uint64_t input_size,
//                                               uint64_t output_size) {
//     uint64_t idx =
//         uint64_t(blockIdx.x) * uint64_t(blockDim.x) + uint64_t(threadIdx.x);
//     if (idx < output_size) {
//         if (idx < input_size) {
//             output[idx] = static_cast<output_type>(input[idx]) + 1;
//         } else {
//             output[idx] = 2;
//         }
//     }
// }

__global__ void simulate_decompression_kernel(const input_type *input,
                                              output_type *output,
                                              uint64_t input_size,
                                              uint64_t output_size) {
    extern __shared__ output_type shared_mem[]; // dynamic shared memory

    uint64_t globalIdx = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x; // index within shared memory for this thread

    // First, write into shared memory (if within valid bounds)
    if (globalIdx < output_size) {
        if (globalIdx < input_size) {
            shared_mem[localIdx] =
                static_cast<output_type>(input[globalIdx]) + 1;
        } else {
            shared_mem[localIdx] = 2;
        }
    } else {
        // Optional: you could write a dummy, or nothing (but must not skip
        // barrier mismatch) E.g. shared_mem[localIdx] = 0;
    }

    __syncthreads();

    // Now copy from shared memory back to global
    if (globalIdx < output_size) {
        output[globalIdx] = shared_mem[localIdx];
    }
}

__global__ void simulate_decompression_kernel_output_bound(
    const input_type *input, output_type *output, uint64_t input_size,
    uint64_t output_size) {
    // NOTE: This kernel assumes threads launched == output_size
    extern __shared__ output_type shared_mem[]; // dynamic shared memory

    uint64_t globalIdx = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x; // index within shared memory for this thread

    if (globalIdx < input_size) {
        shared_mem[localIdx] = static_cast<output_type>(input[globalIdx]) + 1;
    } else {
        shared_mem[localIdx] = 2;
    }

    __syncthreads();

    output[globalIdx] = shared_mem[localIdx];
}

void cuda_simulate_decompression(const input_type *input, output_type *output,
                                 uint64_t input_size, uint64_t output_size) {

    int deviceId = 1; // Change this if you want to use a different GPU
    CHECK_CUDA(cudaSetDevice(deviceId));

    cudaEvent_t e2e_start, e2e_stop;
    CHECK_CUDA(cudaEventCreate(&e2e_start));
    CHECK_CUDA(cudaEventCreate(&e2e_stop));
    CHECK_CUDA(cudaEventRecord(e2e_start));

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
    CHECK_CUDA(cudaEventCreate(&memcpy_start));
    CHECK_CUDA(cudaEventCreate(&memcpy_stop));
    CHECK_CUDA(cudaEventCreate(&decomp_start));
    CHECK_CUDA(cudaEventCreate(&decomp_stop));

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
    uint64_t threads_per_block = 256; // Start with a high number of threads
    uint64_t num_blocks =
        (output_size + threads_per_block - 1ULL) / threads_per_block;

    int max_shared = 0;
    cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock,
                           deviceId);
    printf("Max shared per block = %d bytes\n", max_shared);

    // compute the shared memory size (in bytes) per block
    size_t bytes_shared = threads_per_block * sizeof(output_type);

    if (bytes_shared > max_shared) {
        printf("Requested shared memory per block (%zu bytes) exceeds device "
               "maximum (%d bytes). Adjusting threads per block.\n",
               bytes_shared, max_shared);
        threads_per_block = max_shared / sizeof(output_type);
        bytes_shared = threads_per_block * sizeof(output_type);
        num_blocks =
            (output_size + threads_per_block - 1ULL) / threads_per_block;
    }

    printf(
        "Launching kernel with %llu blocks, %llu threads, %zu bytes shared\n",
        (unsigned long long)num_blocks, (unsigned long long)threads_per_block,
        bytes_shared);

    cudaEventRecord(decomp_start);
    // simulate_decompression_kernel<<<num_blocks, threads_per_block,
    //                                 bytes_shared>>>(d_input, d_output,
    //                                                 input_size, output_size);

    assert(num_blocks * threads_per_block == output_size &&
           "Threads launched must equal output size for this kernel");
    simulate_decompression_kernel_output_bound<<<num_blocks, threads_per_block,
                                                 bytes_shared>>>(
        d_input, d_output, input_size, output_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Launch error: %s\n", cudaGetErrorString(err));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(decomp_stop));
    CHECK_CUDA(cudaEventSynchronize(decomp_stop));
    CHECK_CUDA(
        cudaEventElapsedTime(&decomp_time_ms, decomp_start, decomp_stop));

    printf("Decompression kernel time: %f ms\n", decomp_time_ms);
    printf("Memory copy time: %f ms\n", memcpy_time_ms);
    CHECK_CUDA(cudaEventDestroy(memcpy_start));
    CHECK_CUDA(cudaEventDestroy(memcpy_stop));
    CHECK_CUDA(cudaEventDestroy(decomp_start));
    CHECK_CUDA(cudaEventDestroy(decomp_stop));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    } else {
        printf("Decompression kernel execution successful\n");
    }

    cudaEvent_t copyback_start, copyback_stop;
    float copyback_time_ms = 0.0f;
    CHECK_CUDA(cudaEventCreate(&copyback_start));
    CHECK_CUDA(cudaEventCreate(&copyback_stop));
    CHECK_CUDA(cudaEventRecord(copyback_start));
    CHECK_CUDA(cudaMemcpy(output, d_output, output_size * sizeof(output_type),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(copyback_stop));
    CHECK_CUDA(cudaEventSynchronize(copyback_stop));
    CHECK_CUDA(
        cudaEventElapsedTime(&copyback_time_ms, copyback_start, copyback_stop));
    printf("Copy back time: %f ms\n", copyback_time_ms);

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(e2e_stop));
    CHECK_CUDA(cudaEventSynchronize(e2e_stop));
    float e2e_time_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&e2e_time_ms, e2e_start, e2e_stop));
    printf("Total E2E time: %f ms\n", e2e_time_ms);
    CHECK_CUDA(cudaEventDestroy(e2e_start));
    CHECK_CUDA(cudaEventDestroy(e2e_stop));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
}
