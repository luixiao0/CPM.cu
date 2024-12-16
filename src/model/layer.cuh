#pragma once
#include "norm.cuh"
#include "attn.cuh"
#include "ffn.cuh"
#include "kvcache.cuh"
#include <cuda_runtime.h>

template <typename T>
struct Layer {
    Attention<T> *attn;
    FFN<T> *ffn;

    Layer(int hidden_size, int intermediate_size, int num_attention_heads, int num_key_value_heads, int head_dim, float rms_norm_eps) {
        this->attn = new Attention<T>(hidden_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps);
        this->ffn = new GatedFFN<T>(hidden_size, intermediate_size, rms_norm_eps);
    }

    void init_weight_ptr(Memory* memory) {
        this->attn->init_weight_ptr(memory);
        this->ffn->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t attn_end = this->attn->init_output_ptr(memory, num_tokens, offset);
        int64_t ffn_end = this->ffn->init_output_ptr(memory, num_tokens, offset);
        return std::max(attn_end, ffn_end);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("attn") != std::string::npos || name.find("input_layernorm") != std::string::npos) {
            this->attn->load_to_storage(name, ptr);
        } else if (name.find("mlp") != std::string::npos || name.find("post_attention_layernorm") != std::string::npos) {
            this->ffn->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, T* input, int32_t* position_ids, KVCache<T>* kv_cache) {
        this->attn->prefill(num_tokens, num_history_tokens, input, position_ids, kv_cache);
        this->ffn->prefill(num_tokens, input);
    }

    void decode(int32_t num_tokens, int32_t padded_length, T* input, int32_t* position_ids, int32_t* cache_length, KVCache<T>* kv_cache) {
        this->attn->decode(num_tokens, padded_length, input, position_ids, cache_length, kv_cache);
        this->ffn->prefill(num_tokens, input);
    }
};
