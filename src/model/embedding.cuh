#pragma once
#include <cuda_runtime.h>

template <typename T>
__global__ void embedding_kernel(int32_t num_cols, const int32_t* input, const T* weight, T* output) { // TODO add __restrict__
    int row = blockIdx.x;
    int col = threadIdx.x;
    int offset_output = row * num_cols;
    int offset_weight = input[row] * num_cols;
    for (int i = col; i < num_cols; i += blockDim.x) {
        output[offset_output + i] = weight[offset_weight + i];
    }
}

template <typename T>
struct Embedding {
    int vocab_size;
    int hidden_size;
    T* weight;
    T* output;

    Embedding(int vocab_size, int hidden_size) {
        this->vocab_size = vocab_size;
        this->hidden_size = hidden_size;
    }

    void init_weight_ptr(Memory* memory) {
        weight = (T*)memory->allocate_for_model(vocab_size * hidden_size * sizeof(T));
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        this->output = (T*)(memory->memory_pool + offset);
        return offset + num_tokens * hidden_size * sizeof(T);
    }

    void load_to_storage(std::string name, void* ptr) {
        cudaMemcpy((void*)weight, ptr, vocab_size * hidden_size * sizeof(T), cudaMemcpyHostToDevice);
    }

    void prefill(int32_t num_tokens, int32_t* input) {
        embedding_kernel<T><<<num_tokens, 256>>>(hidden_size, input, weight, this->output); // TODO float4, TODO adjust 256
    }
};