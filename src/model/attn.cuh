#pragma once
#include "../trait.cuh"
#include "norm.cuh"
#include "linear.cuh"
#include "rotary.cuh"
#include "kvcache.cuh"
#include "../flash_attn/flash_api.hpp"
#include <cuda_runtime.h>

template <typename T>
struct Attention {
    int hidden_size;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    float rms_norm_eps;
    float rope_theta;

    RMSNorm<T> *attn_norm;
    Linear<T, true> *q_proj, *k_proj, *v_proj;
    Linear<T, false> *o_proj;
    RotaryEmbedding<T> *rotary_emb;

    T* attn_output;
    float *softmax_lse, *softmax_lse_accum, *oaccum;

    Attention(int hidden_size, int num_attention_heads, int num_key_value_heads, int head_dim, float rms_norm_eps, float rope_theta) {
        this->hidden_size = hidden_size;
        this->num_attention_heads = num_attention_heads;
        this->num_key_value_heads = num_key_value_heads;
        this->head_dim = head_dim;
        this->rms_norm_eps = rms_norm_eps;
        this->rope_theta = rope_theta;

        this->attn_norm = new RMSNorm<T>(hidden_size, rms_norm_eps);
        this->q_proj = new Linear<T, true>(hidden_size, num_attention_heads * head_dim);
        this->k_proj = new Linear<T, true>(hidden_size, num_key_value_heads * head_dim);
        this->v_proj = new Linear<T, true>(hidden_size, num_key_value_heads * head_dim);
        this->rotary_emb = new RotaryEmbedding<T>(head_dim, rope_theta);
        this->o_proj = new Linear<T, false>(hidden_size, num_attention_heads * head_dim);
    }

    void init_weight_ptr(Memory* memory) {
        this->attn_norm->init_weight_ptr(memory);
        this->q_proj->init_weight_ptr(memory);
        this->k_proj->init_weight_ptr(memory);
        this->v_proj->init_weight_ptr(memory);
        this->rotary_emb->init_weight_ptr(memory);
        this->o_proj->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t attn_norm_end = this->attn_norm->init_output_ptr(memory, num_tokens, offset);
        int64_t q_proj_end = this->q_proj->init_output_ptr(memory, num_tokens, attn_norm_end);
        int64_t k_proj_end = this->k_proj->init_output_ptr(memory, num_tokens, q_proj_end);
        int64_t v_proj_end = this->v_proj->init_output_ptr(memory, num_tokens, k_proj_end);
        
        // TODO suppose num_splits = 1 for encode and 4 for decode for now
        this->attn_output = (T*)(memory->memory_pool + offset);
        this->softmax_lse = (float*)(memory->memory_pool + v_proj_end);
        int64_t softmax_lse_end = v_proj_end + num_tokens * this->num_attention_heads * sizeof(float);
        this->softmax_lse_accum = (float*)(memory->memory_pool + softmax_lse_end);
        int64_t softmax_lse_accum_end = softmax_lse_end + num_tokens * this->num_attention_heads * sizeof(float);
        this->oaccum = (float*)(memory->memory_pool + softmax_lse_accum_end);
        int64_t oaccum_end = softmax_lse_accum_end + num_tokens * this->num_attention_heads * this->head_dim * sizeof(float);

        return oaccum_end;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("q_proj") != std::string::npos) {
            this->q_proj->load_to_storage(name, ptr);
        } else if (name.find("k_proj") != std::string::npos) {
            this->k_proj->load_to_storage(name, ptr);
        } else if (name.find("v_proj") != std::string::npos) {
            this->v_proj->load_to_storage(name, ptr);
        } else if (name.find("o_proj") != std::string::npos) {
            this->o_proj->load_to_storage(name, ptr);
        } else if (name.find("input_layernorm") != std::string::npos) {
            this->attn_norm->load_to_storage(name, ptr);
        } else if (name.find("rotary_emb") != std::string::npos) {
            this->rotary_emb->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(int32_t num_tokens, T* input, int32_t* position_ids, int32_t* cache_length, KVCache<T>* kv_cache) {
        this->attn_norm->prefill(num_tokens, input);
        this->q_proj->prefill(num_tokens, this->attn_norm->output);
        this->k_proj->prefill(num_tokens, this->attn_norm->output);
        this->v_proj->prefill(num_tokens, this->attn_norm->output);
        this->rotary_emb->prefill(num_tokens, this->num_attention_heads, this->q_proj->output, position_ids);
        this->rotary_emb->prefill(num_tokens, this->num_key_value_heads, this->k_proj->output, position_ids);

        mha_fwd_kvcache(
            TypeTraits<T>::type_code()==1,
            1,
            num_tokens,
            num_tokens, // TODO max kvcache length here
            num_tokens,
            this->num_attention_heads,
            this->num_key_value_heads,
            this->head_dim,
            this->q_proj->output,
            kv_cache->k_cache,
            kv_cache->v_cache,
            this->k_proj->output,
            this->v_proj->output,
            cache_length,
            this->attn_output,
            this->softmax_lse,
            this->softmax_lse_accum,
            this->oaccum,
            rsqrt(float(this->head_dim)),
            true,
            -1,
            -1,
            0,
            1,
            0 // TODO 0 for default stream
        );

        // flash attention and put output to attn_norm->output
        linear<T, false, /*inplace=*/true>(num_tokens, this->num_attention_heads * this->head_dim, this->hidden_size, this->attn_output, this->o_proj->weight, input);
    }
};