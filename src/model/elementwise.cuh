#pragma once
#include <cuda_runtime.h>
#include "../trait.cuh"
#include "../utils.cuh"

namespace {
template <typename T2>
__global__ void batched_add_kernel(int dim, const T2* a, const T2* b, T2* c) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    for (int i = col; i < dim; i += blockDim.x) {
        c[row * dim + i] = a[row * dim + i] + b[i];
    }
}

template <typename T2>
__global__ void elementwise_add_kernel(int dim, const T2* a, const T2* b, T2* c) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    for (int i = col; i < dim; i += blockDim.x) {
        c[row * dim + i] = a[row * dim + i] + b[row * dim + i];
    }
}

template <typename T>
__global__ void batched_mul_kernel(int dim, const T* a, const T* b, T* c) {
    int row = blockIdx.x;
    int col = threadIdx.x;
    T bv = b[row];
    for (int i = col; i < dim; i += blockDim.x) {
        c[row * dim + i] = a[row * dim + i] * bv;
    }
}
} // namespace

template <typename T>
void batched_add(const Stream& stream, int num_tokens, int dim, const T* a, const T* b, T* c) {
    using T2 = typename TypeTraits<T>::half2;
    batched_add_kernel<T2><<<num_tokens, 256, 0, stream.stream>>>(dim/2, (T2*)a, (T2*)b, (T2*)c);
}

template <typename T>
void elementwise_add(const Stream& stream, int num_tokens, int dim, const T* a, const T* b, T* c) {
    using T2 = typename TypeTraits<T>::half2;
    elementwise_add_kernel<T2><<<num_tokens, 256, 0, stream.stream>>>(dim/2, (T2*)a, (T2*)b, (T2*)c);
}

template <typename T>
void batched_mul(const Stream& stream, int num_tokens, int dim, const T* a, const T* b, T* c) {
    batched_mul_kernel<<<num_tokens, 128, 0, stream.stream>>>(dim, (T*)a, (T*)b, (T*)c);
}