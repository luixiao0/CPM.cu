#pragma once
#include <cuda_runtime.h>
#include "../trait.cuh"
#include "../utils.cuh"

namespace {
template <typename T>
__global__ void rms_norm_kernel(int dim, const T* input, const T* weight, T* output, float eps) {
    __shared__ float shared_sum;
    __shared__ float warp_sum[8];
    int row = blockIdx.x;
    int col = threadIdx.x;
    float sum = 0.0f;
    for (int i = col; i < dim; i += blockDim.x) {
        float val = float(input[row * dim + i]);
        sum += val * val;
    }
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (col % 32 == 0) warp_sum[col / 32] = sum;
    __syncthreads();
    if (col < 8) {
        sum = warp_sum[col];
        sum += __shfl_down_sync(0x000000ff, sum, 4);
        sum += __shfl_down_sync(0x000000ff, sum, 2);
        sum += __shfl_down_sync(0x000000ff, sum, 1);
    }
    if (col == 0) {
        shared_sum = rsqrtf(sum / dim + eps);
    }
    __syncthreads();
    sum = shared_sum;
    for (int i = col; i < dim; i += blockDim.x) {
        output[row * dim + i] = T(sum * float(input[row * dim + i]) * float(weight[i]));
    }
}

template <typename T, typename T2>
__global__ void rms_norm_kernel_half2(int dim, const T2* input, const T2* weight, T2* output, float eps) {
    __shared__ float shared_sum;
    __shared__ float warp_sum[8];
    int row = blockIdx.x;
    int col = threadIdx.x;
    float sum1 = 0.0f, sum2 = 0.0f;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 val = input[row * dim + i];
        float val1 = float(val.x);
        float val2 = float(val.y);
        sum1 += val1 * val1;
        sum2 += val2 * val2;
    }
    float sum = sum1 + sum2;
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (col % 32 == 0) warp_sum[col / 32] = sum;
    __syncthreads();
    if (col < 8) {
        sum = warp_sum[col];
        sum += __shfl_down_sync(0x000000ff, sum, 4);
        sum += __shfl_down_sync(0x000000ff, sum, 2);
        sum += __shfl_down_sync(0x000000ff, sum, 1);
    }
    if (col == 0) {
        shared_sum = rsqrtf(sum / (2*dim) + eps);
    }
    __syncthreads();
    sum = shared_sum;
    for (int i = col; i < dim; i += blockDim.x) {
        T2 inp = input[row * dim + i];
        T2 w = weight[i];
        output[row * dim + i] = T2(
            T(sum * float(inp.x) * float(w.x)),
            T(sum * float(inp.y) * float(w.y))
        );
    }
}
}

template <typename T>
struct RMSNorm {
    int dim;
    float eps;
    T* weight;
    T* output;

    RMSNorm(int dim, float eps) {
        this->dim = dim;
        this->eps = eps;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(dim * sizeof(T));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        this->output = (T*)(memory->memory_pool + offset);
        return offset + num_tokens * dim * sizeof(T);
    }

    void load_to_storage(std::string name, void* ptr) {
        cudaMemcpy((void*)weight, ptr, dim * sizeof(T), cudaMemcpyHostToDevice);
    }

    void prefill(int32_t num_tokens, T* input, T* tgt=nullptr) {
        if (tgt == nullptr) tgt = this->output;
        // rms_norm_kernel<T><<<num_tokens, 256, 0, calc_stream>>>(dim, input, weight, this->output, eps);
        using T2 = typename TypeTraits<T>::half2;
        rms_norm_kernel_half2<T, T2><<<num_tokens, 256, 0, calc_stream>>>(dim/2, (T2*)input, (T2*)weight, (T2*)tgt, eps);
    }
};