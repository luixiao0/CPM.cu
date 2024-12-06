#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../trait.cuh"
#include "../utils.cuh"

template <typename T, bool transposed, bool inplace=false>
void linear(int num_tokens, int dim_in, int dim_out, const T* input, const T* weight, T* output) {
    float alpha = 1.0f;
    float beta;
    if constexpr (inplace) {
        beta = 1.0f;
    } else {
        beta = 0.0f;
    }
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

template <typename T, bool transposed>
struct Linear {
    int dim_in;
    int dim_out;
    T* output;
    T* weight;

    Linear(int dim_in, int dim_out) {
        this->dim_in = dim_in;
        this->dim_out = dim_out;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(dim_in * dim_out * sizeof(T));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        this->output = (T*)(memory->memory_pool + offset);
        return offset + num_tokens * dim_out * sizeof(T);
    }

    void load_to_storage(std::string name, void* ptr) {
        cudaMemcpy((void*)weight, ptr, dim_in * dim_out * sizeof(T), cudaMemcpyHostToDevice);
    }

    void prefill(int32_t num_tokens, T* input, T* tgt=nullptr) {
        if (tgt == nullptr) tgt = this->output;
        linear<T, transposed>(num_tokens, dim_in, dim_out, input, weight, tgt);
    }
};