#pragma once
#include <cuda_runtime.h>
#include <stdio.h>

template <typename T>
__global__ void embedding_kernel(int32_t num_cols, int32_t* input, T* weight, T* output) {
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

    Embedding(int vocab_size, int hidden_size) {
        this->vocab_size = vocab_size;
        this->hidden_size = hidden_size;
    }

    void init_storage(Memory* memory) {
        weight = (T*)memory->allocate_for_model(vocab_size * hidden_size * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {
        cudaMemcpy((void*)weight, ptr, vocab_size * hidden_size * sizeof(T), cudaMemcpyHostToDevice);
    }

    void prefill(int32_t num_tokens, int32_t* input, T* output) {
        printf("embedding kernel %d\n", num_tokens);
        embedding_kernel<T><<<num_tokens, 256>>>(hidden_size, input, weight, output); // TODO float4, TODO adjust 256
        cudaDeviceSynchronize();
        printf("embedding kernel done\n");
    }
};