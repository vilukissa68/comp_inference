#ifndef WRAPPER_H_
#define WRAPPER_H_

// C++ wrapper function that will call the CUDA implementation
void cuda_vector_add(const float *a, const float *b, float *c, int n);

#endif // WRAPPER_H_
