#pragma once
#include <cuda_runtime.h>
#include "../utils.cuh"

namespace {
template<typename T>
__global__ void rotary_embedding(int num_heads, int num_heads_kv, int half_dim, float theta, const int* pos, T* q, T* k) {
    int tid = threadIdx.x;

    int p = pos[blockIdx.x];

    for (int i = tid; i < num_heads * half_dim; i += blockDim.x) {
        int row = i / half_dim;
        int col = i % half_dim;
        int offset = blockIdx.x * num_heads * half_dim * 2 + row * half_dim * 2;
        float freq = p * powf(theta, -float(col) / half_dim);
        T cos_freq = T(cos(freq)), sin_freq = T(sin(freq));
        T a = q[offset + col];
        T b = q[offset + col + half_dim];
        q[offset + col] = a * cos_freq - b * sin_freq;
        q[offset + col + half_dim] = a * sin_freq + b * cos_freq;
    }
    for (int i = tid; i < num_heads_kv * half_dim; i += blockDim.x) {
        int row = i / half_dim;
        int col = i % half_dim;
        int offset = blockIdx.x * num_heads_kv * half_dim * 2 + row * half_dim * 2;
        float freq = p * powf(theta, -float(col) / half_dim);
        T cos_freq = T(cos(freq)), sin_freq = T(sin(freq));
        T a = k[offset + col];
        T b = k[offset + col + half_dim];
        k[offset + col] = a * cos_freq - b * sin_freq;
        k[offset + col + half_dim] = a * sin_freq + b * cos_freq;
    }
}

template<typename T, typename T2>
__global__ void rotary_embedding_half2(int num_heads, int num_heads_kv, int half_dim, float theta, const int* pos, T2* q, T2* k) {
    int tid = threadIdx.x;

    int p = pos[blockIdx.x];

    for (int i = tid; i < num_heads * half_dim; i += blockDim.x) {
        int row = i / half_dim;
        int col = i % half_dim;
        int offset = blockIdx.x * num_heads * half_dim * 2 + row * half_dim * 2;
        float freq = p * powf(theta, -float(col) / half_dim);
        T cos_freq = T(cos(freq)), sin_freq = T(sin(freq));
        float freq1 = p * powf(theta, -float(col * 2 + 1) / (2 * half_dim));
        T cos_freq1 = T(cos(freq1)), sin_freq1 = T(sin(freq1));
        T2 cos_freq2 = T2(cos_freq, cos_freq1), sin_freq2 = T2(sin_freq, sin_freq1);
        T2 a = q[offset + col];
        T2 b = q[offset + col + half_dim];
        q[offset + col] = a * cos_freq2 - b * sin_freq2;
        q[offset + col + half_dim] = a * sin_freq2 + b * cos_freq2;
    }
    for (int i = tid; i < num_heads_kv * half_dim; i += blockDim.x) {
        int row = i / half_dim;
        int col = i % half_dim;
        int offset = blockIdx.x * num_heads_kv * half_dim * 2 + row * half_dim * 2;
        float freq = p * powf(theta, -float(col) / half_dim);
        T cos_freq = T(cos(freq)), sin_freq = T(sin(freq));
        float freq1 = p * powf(theta, -float(col * 2 + 1) / (2 * half_dim));
        T cos_freq1 = T(cos(freq1)), sin_freq1 = T(sin(freq1));
        T2 cos_freq2 = T2(cos_freq, cos_freq1), sin_freq2 = T2(sin_freq, sin_freq1);
        T2 a = k[offset + col];
        T2 b = k[offset + col + half_dim];
        k[offset + col] = a * cos_freq2 - b * sin_freq2;
        k[offset + col + half_dim] = a * sin_freq2 + b * cos_freq2;
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
        rotary_embedding<T><<<num_tokens, 256, 0, calc_stream>>>(num_heads, num_heads_kv, half_dim, theta, position_ids, q, k);
        // using T2 = typename TypeTraits<T>::half2;
        // rotary_embedding_half2<T, T2><<<num_tokens, 256, 0, calc_stream>>>(num_heads, num_heads_kv, half_dim/2, theta, position_ids, (T2*)q, (T2*)k);
    }
};