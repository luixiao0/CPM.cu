#pragma once
#include "memory.cuh"
#include "embedding.cuh"
#include "norm.cuh"
#include "linear.cuh"
#include "layer.cuh"
#include "kvcache.cuh"

#include <algorithm>
#include <cuda_runtime.h>
#include <vector>
#include <regex>

struct Model {
    virtual void init_storage() = 0;
    virtual void load_to_storage(std::string name, void* ptr) = 0;
    virtual void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) = 0;
    virtual void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, void* output) = 0;
};

template <typename T>
struct ModelImpl : Model {
    Memory* memory;

    int num_hidden_layers;
    int hidden_size;
    int chunk_length;
    int max_output_length;

    KVCacheManager<T>* kv_cache;

    Embedding<T>* embedding;
    std::vector<Layer<T>*> layers;
    RMSNorm<T>* norm;
    Linear<T, true>* lm_head;

    ModelImpl(
        int64_t memory_limit,
        void* memory_pool,
        int vocab_size,
        int num_hidden_layers,
        int hidden_size,
        int intermediate_size,
        int num_attention_heads,
        int num_key_value_heads,
        int head_dim,
        float rms_norm_eps,
        float rope_theta,
        int chunk_length
    ) {
        this->num_hidden_layers = num_hidden_layers;
        this->hidden_size = hidden_size;
        this->chunk_length = chunk_length;
        
        memory = new Memory(memory_limit, memory_pool);

        kv_cache = new KVCacheManager<T>(num_hidden_layers, num_key_value_heads * head_dim);

        embedding = new Embedding<T>(vocab_size, hidden_size);
        for (int i = 0; i < num_hidden_layers; i++) {
            layers.push_back(new Layer<T>(hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps, rope_theta));
        }
        norm = new RMSNorm<T>(hidden_size, rms_norm_eps);
        lm_head = new Linear<T, true>(hidden_size, vocab_size);
    }

    void init_weight_ptr() {
        embedding->init_weight_ptr(memory);
        for (int i = 0; i < num_hidden_layers; i++) {
            layers[i]->init_weight_ptr(memory);
        }
        norm->init_weight_ptr(memory);
        lm_head->init_weight_ptr(memory);
    }

    void init_output_ptr() {
        int64_t embedding_end = embedding->init_output_ptr(memory, chunk_length, memory->model_offset);
        int64_t layer_end = 0;
        for (int i = 0; i < num_hidden_layers; i++) {
            layer_end = layers[i]->init_output_ptr(memory, chunk_length, embedding_end);
        }
        // norm and lm_head are not used in prefill
        int64_t norm_end = norm->init_output_ptr(memory, chunk_length, memory->model_offset);
        int64_t lm_head_end = lm_head->init_output_ptr(memory, 64, norm_end);

        memory->kv_cache_offset = std::max(layer_end, lm_head_end);
        this->max_output_length = kv_cache->init_output_ptr(memory, memory->kv_cache_offset);

        printf("model offset: %lld, kv_cache offset: %lld\n", memory->model_offset, memory->kv_cache_offset);
        printf("maximum possible output length: %d\n", max_output_length);
    }

    void init_storage() {
        init_weight_ptr();
        init_output_ptr();
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 18) == "model.embed_tokens") {
            embedding->load_to_storage(name, ptr);
        } else if (name.substr(0, 10) == "model.norm") {
            norm->load_to_storage(name, ptr);
        } else if (name.substr(0, 7) == "lm_head") {
            lm_head->load_to_storage(name, ptr);
        } else if (name.substr(0, 12) == "model.layers") { // e.g. model.layers.20.attn.q_proj.weight
            std::regex layer_regex("model\\.layers\\.(\\d+)\\.(.*)");
            std::smatch matches;
            if (std::regex_search(name, matches, layer_regex)) {
                int layer_idx = std::stoi(matches[1]);
                layers[layer_idx]->load_to_storage(matches[2], ptr);
            } else {
                throw std::invalid_argument("Unsupported name (layer_idx not found): " + name);
            }
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        this->embedding->prefill(num_tokens, input);
        for (int i = 0; i < num_hidden_layers; i++) {
            this->layers[i]->prefill(num_tokens, num_history_tokens, this->embedding->output, position_ids, this->kv_cache->caches[i]);
        }
        this->norm->prefill(num_tokens, this->embedding->output);
        this->lm_head->prefill(1, this->norm->output + (num_tokens - 1) * this->hidden_size, (T*)output);
    }

    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, void* output) {
        this->embedding->prefill(num_tokens, input);
        for (int i = 0; i < num_hidden_layers; i++) {
            this->layers[i]->decode(num_tokens, padded_length, this->embedding->output, position_ids, cache_length, this->kv_cache->caches[i]);
        }
        this->norm->prefill(num_tokens, this->embedding->output);
        this->lm_head->prefill(num_tokens, this->norm->output, (T*)output);
    }
};

