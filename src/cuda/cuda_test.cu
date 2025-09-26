#include <stdio.h>

__global__ void testKernel() { printf("CUDA kernel works!\n"); }

int main() {
    printf("Testing CUDA... \n");
    testKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    printf("Kernel launched\n");
    return 0;
}
