#pragma once
#include <cuda_runtime.h>

namespace {
template<typename T>
__global__ void rotary_embedding(int num_heads, int half_dim, float theta, const int* pos, T* hidden) {
    int col = threadIdx.x;
    int offset = int(blockIdx.x * half_dim * 2);

    float freq = pos[blockIdx.x / num_heads] * powf(theta, -float(col) / half_dim);
    float cos_freq = cos(freq), sin_freq = sin(freq);
    
    float a = TypeTraits<T>::to_float(hidden[offset + col]);
    float b = TypeTraits<T>::to_float(hidden[offset + col + half_dim]);
    hidden[offset + col] = TypeTraits<T>::from_float(a * cos_freq - b * sin_freq);
    hidden[offset + col + half_dim] = TypeTraits<T>::from_float(a * sin_freq + b * cos_freq);
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

    void init_weight_ptr(Memory* memory) {
        // Nothing
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        // Nothing (inplace operation)
        return -1;
    }

    void load_to_storage(std::string name, void* ptr) {
        // Nothing
    }

    void prefill(int32_t num_tokens, int num_heads, T* input, int32_t* position_ids) {
        rotary_embedding<<<num_tokens * num_heads, half_dim>>>(num_heads, half_dim, theta, position_ids, input); // TODO float4
    }
};