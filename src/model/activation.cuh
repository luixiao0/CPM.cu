#pragma once
#include "../trait.cuh"
#include <cuda_runtime.h>

namespace {
template <typename T>
__global__ void gated_silu_interleaved_kernel(int dim, const T* src, T* tgt) {
    int row_offset = blockIdx.x * dim;
    int row_offset_2 = row_offset * 2;
    int col = blockIdx.y * 256 + threadIdx.x;
    int col2 = col + dim;
    if (col < dim) {
        float g = float(src[row_offset_2 + col]);
        float u = float(src[row_offset_2 + col2]);
        float s = 1.0f / (1.0f + expf(-g));
        tgt[row_offset + col] = T(g * s * u);
    }
}

template<typename T>
__global__ void gated_silu_kernel(int dim, const T* src, T* tgt) {
    int row_offset = blockIdx.x * dim;
    int col = blockIdx.y * 256 + threadIdx.x;
    if (col < dim) {
        float g = float(src[row_offset + col]);
        float u = float(tgt[row_offset + col]);
        float s = 1.0f / (1.0f + expf(-g));
        tgt[row_offset + col] = T(g * s * u);
    }
}

template<typename T>
__global__ void silu_kernel(int dim, const T* src, T* tgt) {
    int row_offset = blockIdx.x * dim;
    int col = blockIdx.y * 256 + threadIdx.x;
    if (col < dim) {
        float g = float(src[row_offset + col]);
        float s = 1.0f / (1.0f + expf(-g));
        tgt[row_offset + col] = T(g * s);
    }
}
}

template <typename T>
void gated_silu_interleaved(int num_tokens, int dim, const T* src, T* tgt) {
    gated_silu_interleaved_kernel<T><<<dim3(num_tokens, (dim+255)/256), 256, 0, calc_stream>>>(dim, src, tgt);
}

template <typename T>
void gated_silu(int num_tokens, int dim, const T* src, T* tgt) {
    gated_silu_kernel<T><<<dim3(num_tokens, (dim+255)/256), 256, 0, calc_stream>>>(dim, src, tgt);
}

template<typename T>
void silu(int num_tokens, int dim, const T* src, T* tgt) {
    silu_kernel<T><<<dim3(num_tokens, (dim+255)/256), 256, 0, calc_stream>>>(dim, src, tgt);
}

template <typename T>
void silu_inplace(int num_tokens, int dim, T* x) {
    silu(num_tokens, dim, x, x);
}