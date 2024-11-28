#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../trait.cuh"
#include "../utils.cuh"

template <typename T, bool transposed>
void linear(int num_tokens, int dim_in, int dim_out, T* input, T* weight, T* output) {
    float alpha = 1.0f;
    float beta = 0.0f;
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
struct Linear {
    T* output;
    virtual void init_weight_ptr(Memory* memory) = 0;
    virtual int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) = 0;
    virtual void load_to_storage(std::string name, void* ptr) = 0;
    virtual void prefill(int32_t num_tokens, T* input) = 0;
};

template <typename T, bool transposed>
struct LinearImpl : Linear<T> {
    int dim_in;
    int dim_out;
    T* weight;

    LinearImpl(int dim_in, int dim_out) {
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

    void prefill(int32_t num_tokens, T* input) {
        linear<T, transposed>(num_tokens, dim_in, dim_out, input, weight, this->output);
    }
};