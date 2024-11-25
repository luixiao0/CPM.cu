#pragma once
#include "memory.cuh"
#include "embedding.cuh"
#include <cuda_runtime.h>

struct Model {
    virtual void init(
        int64_t memory_limit,
        void* memory_pool,
        int vocab_size,
        int num_hidden_layers,
        int hidden_size,
        int intermediate_size,
        int num_attention_heads,
        int num_key_value_heads,
        float rms_norm_eps,
        float rope_theta
    ) = 0;
    virtual void init_storage() = 0;
    virtual void load_to_storage(std::string name, void* ptr) = 0;
    virtual void prefill(int32_t num_tokens, int32_t* input, int32_t* output) = 0;
};

template <typename T>
struct ModelImpl : Model {
    Memory* memory;
    Embedding<T>* embedding;

    void init(
        int64_t memory_limit,
        void* memory_pool,
        int vocab_size,
        int num_hidden_layers,
        int hidden_size,
        int intermediate_size,
        int num_attention_heads,
        int num_key_value_heads,
        float rms_norm_eps,
        float rope_theta
    ) {
        this->memory = new Memory(memory_limit, memory_pool);
        this->embedding = new Embedding<T>(vocab_size, hidden_size);
    }

    void init_storage() {
        this->embedding->init_storage(this->memory);

        // this->memory->allocate_for_hidden_states(hidden_size);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 18) == "model.embed_tokens") {
            this->embedding->load_to_storage(name, ptr);
        } else if (name.substr(0, 10) == "model.norm") {
        } else if (name.substr(0, 7) == "lm_head") {
        } else if (name.substr(0, 12) == "model.layers") {
        } else {
            throw std::invalid_argument("Unsupported name");
        }
    }

    void prefill(int32_t num_tokens, int32_t* input, int32_t* output) {
        printf("model offset: %lld\n", this->memory->model_offset);
        this->embedding->prefill(num_tokens, input, (T*)(this->memory->memory_pool + this->memory->model_offset));
    }
};