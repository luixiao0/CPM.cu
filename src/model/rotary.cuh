#pragma once
#include <cuda_runtime.h>

namespace {
template<typename T>
__global__ void rotary_embedding(int num_heads, int num_heads_kv, int half_dim, float theta, const int* pos, T* q, T* k) {
    int tid = threadIdx.x;

    int p = pos[blockIdx.x];

    for (int i = tid; i < num_heads * half_dim; i += blockDim.x) {
        int row = i / half_dim;
        int col = i % half_dim;
        int offset = blockIdx.x * num_heads * half_dim * 2 + row * half_dim * 2; // TODO cache some constant
        float freq = p * powf(theta, -float(col) / half_dim);
        float cos_freq = cos(freq), sin_freq = sin(freq);
        float a = TypeTraits<T>::to_float(q[offset + col]);
        float b = TypeTraits<T>::to_float(q[offset + col + half_dim]);
        q[offset + col] = TypeTraits<T>::from_float(a * cos_freq - b * sin_freq);
        q[offset + col + half_dim] = TypeTraits<T>::from_float(a * sin_freq + b * cos_freq);
    }
    for (int i = tid; i < num_heads_kv * half_dim; i += blockDim.x) {
        int row = i / half_dim;
        int col = i % half_dim;
        int offset = blockIdx.x * num_heads_kv * half_dim * 2 + row * half_dim * 2; // TODO cache some constant
        float freq = p * powf(theta, -float(col) / half_dim);
        float cos_freq = cos(freq), sin_freq = sin(freq);
        float a = TypeTraits<T>::to_float(k[offset + col]);
        float b = TypeTraits<T>::to_float(k[offset + col + half_dim]);
        k[offset + col] = TypeTraits<T>::from_float(a * cos_freq - b * sin_freq);
        k[offset + col + half_dim] = TypeTraits<T>::from_float(a * sin_freq + b * cos_freq);
    }
}
}

template <typename T>
struct RotaryEmbedding {
    int half_dim;
    float theta;

    RotaryEmbedding(int head_dim, float theta) {
        this->half_dim = head_dim / 2;
        this->theta = theta;
    }

    void init_weight_ptr(Memory* memory) {}
    void init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {}
    void load_to_storage(std::string name, void* ptr) {}

    void prefill(int32_t num_tokens, int num_heads, int num_heads_kv, T* q, T* k, int32_t* position_ids) {
        rotary_embedding<<<num_tokens, 256>>>(num_heads, num_heads_kv, half_dim, theta, position_ids, q, k); // TODO 256, TODO float4
    }
};