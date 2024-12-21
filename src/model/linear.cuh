#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../trait.cuh"
#include "../utils.cuh"

namespace {
    template <typename T>
    __global__ void batched_add_kernel(int dim, const T* a, const T* b, T* c) {
        int row = blockIdx.x;
        int col = threadIdx.x;
        for (int i = col; i < dim; i += blockDim.x) {
            c[row * dim + i] = a[row * dim + i] + b[i];
        }
    }

    template <typename T2>
    __global__ void batched_add_kernel_half2(int dim, const T2* a, const T2* b, T2* c) {
        int row = blockIdx.x;
        int col = threadIdx.x;
        for (int i = col; i < dim; i += blockDim.x) {
            c[row * dim + i] = a[row * dim + i] + b[i];
        }
    }

    template <typename T>
    __global__ void elementwise_add_kernel(int dim, const T* a, const T* b, T* c) {
        int row = blockIdx.x;
        int col = threadIdx.x;
        for (int i = col; i < dim; i += blockDim.x) {
            c[row * dim + i] = a[row * dim + i] + b[row * dim + i];
        }
    }

    template <typename T2>
    __global__ void elementwise_add_kernel_half2(int dim, const T2* a, const T2* b, T2* c) {
        int row = blockIdx.x;
        int col = threadIdx.x;
        for (int i = col; i < dim; i += blockDim.x) {
            c[row * dim + i] = a[row * dim + i] + b[row * dim + i];
        }
    }
}

template <typename T, bool transposed=true>
void linear(int num_tokens, int dim_in, int dim_out, const T* input, const T* weight, T* output, bool inplace=false) {
    float alpha = 1.0f;
    float beta = inplace ? 1.0f : 0.0f;
    if constexpr (transposed) {
        cublasCheck(cublasGemmEx(cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            dim_out, num_tokens, dim_in,
            &alpha,
            weight, TypeTraits<T>::cublas_type(), dim_in,
            input, TypeTraits<T>::cublas_type(), dim_in,
            &beta,
            output, TypeTraits<T>::cublas_type(), dim_out,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ));
    } else {
        cublasCheck(cublasGemmEx(cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            dim_out, num_tokens, dim_in,
            &alpha,
            weight, TypeTraits<T>::cublas_type(), dim_out,
            input, TypeTraits<T>::cublas_type(), dim_in,
            &beta,
            output, TypeTraits<T>::cublas_type(), dim_out,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT
        ));
    }
}

template <typename T>
void batched_add(int num_tokens, int dim, const T* a, const T* b, T* c) {
    // batched_add_kernel<<<num_tokens, 256, 0, calc_stream>>>(dim, (T*)a, (T*)b, (T*)c);
    using T2 = typename TypeTraits<T>::half2;
    batched_add_kernel_half2<T2><<<num_tokens, 256, 0, calc_stream>>>(dim/2, (T2*)a, (T2*)b, (T2*)c);
}

template <typename T>
void elementwise_add(int num_tokens, int dim, const T* a, const T* b, T* c) {
    // elementwise_add_kernel<<<num_tokens, 256, 0, calc_stream>>>(dim, (T*)a, (T*)b, (T*)c);
    using T2 = typename TypeTraits<T>::half2;
    elementwise_add_kernel_half2<T2><<<num_tokens, 256, 0, calc_stream>>>(dim/2, (T2*)a, (T2*)b, (T2*)c);
}

template <typename T, bool transposed=true, bool has_bias=false>
struct Linear {
    int dim_in;
    int dim_out;
    T* output;
    T* weight;
    T* bias;

    Linear(int dim_in, int dim_out) {
        this->dim_in = dim_in;
        this->dim_out = dim_out;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(dim_in * dim_out * sizeof(T));
        if constexpr (has_bias) {
            bias = (T*)memory->allocate_for_model(dim_out * sizeof(T));
        }
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * dim_out * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("weight") != std::string::npos) {
            cudaMemcpy((void*)weight, ptr, dim_in * dim_out * sizeof(T), cudaMemcpyHostToDevice);
        } else if (name.find("bias") != std::string::npos) {
            cudaMemcpy((void*)bias, ptr, dim_out * sizeof(T), cudaMemcpyHostToDevice);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(int32_t num_tokens, T* input, T* tgt=nullptr, bool inplace=false) {
        if (tgt == nullptr) tgt = this->output;
        linear<T, transposed>(num_tokens, dim_in, dim_out, input, weight, tgt, inplace);
        if constexpr (has_bias) {
            batched_add<T>(num_tokens, dim_out, tgt, bias, tgt);
        }
    }
};