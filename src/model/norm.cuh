#pragma once
#include <cuda_runtime.h>
#include "../trait.cuh"

template <typename T>
__global__ void rms_norm_kernel(int dim, const T* input, const T* weight, T* output, float eps) {
    __shared__ float shared_sum;
    int row = blockIdx.x;
    int col = threadIdx.x;
    float sum = 0.0f;
    for (int i = col; i < dim; i += blockDim.x) {
        float val = TypeTraits<T>::to_float(input[row * dim + i]);
        sum += val * val;
    }
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    if (col == 0) {
        shared_sum = rsqrt(sum / dim + eps);
    }
    __syncthreads();
    sum = shared_sum;
    for (int i = col; i < dim; i += blockDim.x) {
        output[row * dim + i] = TypeTraits<T>::from_float(sum * TypeTraits<T>::to_float(input[row * dim + i]) * TypeTraits<T>::to_float(weight[i]));
    }
}

template <typename T>
struct RMSNorm {
    virtual void init_storage(Memory* memory) = 0;
    virtual void load_to_storage(std::string name, void* ptr) = 0;
    virtual void prefill(int32_t num_tokens, T* input, T* output) = 0;
};

template <typename T>
struct RMSNormImpl : RMSNorm<T> {
    int dim;
    float eps;
    T* weight;

    RMSNormImpl(int dim, float eps) {
        this->dim = dim;
        this->eps = eps;
    }

    void init_storage(Memory* memory) {
        weight = (T*)memory->allocate_for_model(dim * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {
        cudaMemcpy((void*)weight, ptr, dim * sizeof(T), cudaMemcpyHostToDevice);
    }

    void prefill(int32_t num_tokens, T* input, T* output) {
        rms_norm_kernel<<<num_tokens, 32>>>(dim, input, weight, output, eps); // TODO 32, TODO float4, TODO shared memory for input
    }
};