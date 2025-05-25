#pragma once
#include "../attn.cuh"
#include "minicpm4_kvcache.cuh"

template <typename T>
void debug_print(T* arr, int q, int k) {
    T* h_arr = new T[q * k];
    cudaMemcpy(h_arr, arr, q * k * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < q; i++) if (i <= 2 || i >= q-3) {
        float value;
        for (int j = 0; j < k; j++) {
            value = float(h_arr[i * k + j]);
            printf("%f ", value);
            if (isnan(value)) {
                printf("NaN detected, exiting.\n");
                exit(0);
            }
        }
        printf("\n");
    }
    // printf("head = 1\n");
    // for (int i = 0; i < q; i++) if (i <= 2 || i >= q-3) {
    //     for (int j = 0; j < k; j++) {
    //         float value = float(h_arr[q * k + i * k + j]);
    //         printf("%f ", value);
    //         if (isnan(value)) {
    //             printf("NaN detected, exiting.\n");
    //             exit(0);
    //         }
    //     }
    //     printf("\n");
    // }
}

template <typename T>
struct MiniCPM4Attention {
    int hidden_size;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    float rms_norm_eps;

    Norm<T> *attn_norm;
    Linear<T> *q_proj, *k_proj, *v_proj;
    Linear<T> *o_proj;
    T* output;

    T* attn_output;
    float *softmax_lse, *softmax_lse_accum, *oaccum;

    int sink_window_size;
    int block_window_size;

    MiniCPM4Attention(int hidden_size, int num_attention_heads, int num_key_value_heads, int head_dim, float rms_norm_eps, int sink_window_size, int block_window_size) {
        this->hidden_size = hidden_size;
        this->num_attention_heads = num_attention_heads;
        this->num_key_value_heads = num_key_value_heads;
        this->head_dim = head_dim;
        this->rms_norm_eps = rms_norm_eps;

        this->attn_norm = new RMSNorm<T>(hidden_size, rms_norm_eps);
        this->q_proj = new Linear<T>(hidden_size, num_attention_heads * head_dim);
        this->k_proj = new Linear<T>(hidden_size, num_key_value_heads * head_dim);
        this->v_proj = new Linear<T>(hidden_size, num_key_value_heads * head_dim);
        this->o_proj = new Linear<T>(hidden_size, num_attention_heads * head_dim);

        this->sink_window_size = sink_window_size;
        this->block_window_size = block_window_size;
    }

    void init_weight_ptr(Memory* memory) {
        this->attn_norm->init_weight_ptr(memory);
        this->q_proj->init_weight_ptr(memory);
        this->k_proj->init_weight_ptr(memory);
        this->v_proj->init_weight_ptr(memory);
        this->o_proj->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t attn_norm_end = this->attn_norm->init_output_ptr(memory, num_tokens, offset);
        int64_t q_proj_end = this->q_proj->init_output_ptr(memory, num_tokens, attn_norm_end);
        int64_t k_proj_end = this->k_proj->init_output_ptr(memory, num_tokens, q_proj_end);
        int64_t v_proj_end = this->v_proj->init_output_ptr(memory, num_tokens, k_proj_end);
        
        memory->allocate((void**)&this->attn_output, offset);
        int64_t softmax_lse_end = memory->allocate((void**)&this->softmax_lse, v_proj_end, num_tokens * this->num_attention_heads * sizeof(float)); // TODO minicpm4 support larger num_splits
        int64_t softmax_lse_accum_end = memory->allocate((void**)&this->softmax_lse_accum, softmax_lse_end, num_tokens * this->num_attention_heads * sizeof(float));
        int64_t oaccum_end = memory->allocate((void**)&this->oaccum, softmax_lse_accum_end, num_tokens * this->num_attention_heads * this->head_dim * sizeof(float));

        int64_t o_proj_end = this->o_proj->init_output_ptr(memory, num_tokens, v_proj_end);
        this->output = this->o_proj->output;

        return std::max(oaccum_end, o_proj_end);
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
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(const Stream& stream, int32_t num_tokens, int32_t num_history_tokens, T* input, T* prev_output, int32_t* position_ids, MiniCPM4KVCache<T>* kv_cache) {
        T* k_cache = kv_cache->offset_k(num_history_tokens);
        T* v_cache = kv_cache->offset_v(num_history_tokens);

        this->attn_norm->prefill(stream, num_tokens, input, prev_output);
        this->q_proj->prefill(stream, num_tokens, this->attn_norm->output);
        this->k_proj->prefill(stream, num_tokens, this->attn_norm->output, k_cache);
        this->v_proj->prefill(stream, num_tokens, this->attn_norm->output, v_cache);
        kv_cache->rotary_embedding->prefill(stream, num_tokens, this->num_attention_heads, this->num_key_value_heads, this->q_proj->output, k_cache, position_ids);

        if (num_history_tokens == 0) {
            kv_cache->init();
        } else {
            kv_cache->compress(stream);
        }

        uint64_t fakeblock; // TODO minicpm4 fake blockmask now
        uint64_t *blockmask = nullptr;
        if (kv_cache->c1_len > 0) {
            int q_round, k_round, out_len;
            mha_fwd_stage1(
                TypeTraits<T>::type_code()==1,
                1,
                num_tokens,
                kv_cache->c1_len,
                num_tokens,
                this->num_attention_heads,
                this->num_key_value_heads,
                this->head_dim,
                this->q_proj->output,
                kv_cache->c1_cache,
                kv_cache->c2_cache,
                nullptr,
                kv_cache->stage1_score,
                rsqrtf(float(this->head_dim)),
                false,
                -1,
                -1,
                0,
                stream.stream,
                q_round,
                k_round
            );
            maxpooling_func(
                stream.stream,
                kv_cache->stage1_score,
                kv_cache->pool_score,
                this->num_key_value_heads,
                num_tokens,
                q_round,
                k_round,
                kv_cache->next_kv_length,
                this->sink_window_size,
                this->block_window_size,
                out_len
            );
            kv_cache->topk_func->prefill(
                stream,
                this->num_key_value_heads*num_tokens,
                kv_cache->pool_score,
                out_len
            );
            topk_to_uint64_func(
                stream.stream,
                kv_cache->topk_func->topk_pos,
                kv_cache->blockmask,
                this->num_key_value_heads*num_tokens,
                kv_cache->topk_func->top,
                num_history_tokens+num_tokens // TODO minicpm4 decode should be padded length
            );
            // TODO minicpm4 delete these
            // printf("num_tokens: %d, q_round: %d, k_round: %d, num_history_tokens: %d, out_len: %d, prev_kv_length: %d, next_kv_length: %d, c1_len: %d, c2_len: %d\n", num_tokens, q_round, k_round, num_history_tokens, out_len, kv_cache->prev_kv_length, kv_cache->next_kv_length, kv_cache->c1_len, kv_cache->c2_len);
            // debug_print(kv_cache->stage1_score, 2*q_round, k_round);
            // debug_print(kv_cache->pool_score, 2*num_tokens, out_len);
            // debug_print(kv_cache->c1_cache, kv_cache->c1_len, kv_cache->dim);
            // printf("topk_pos\n");
            // debug_print(kv_cache->topk_func->topk_pos, 2*num_tokens, kv_cache->topk_func->top);
            // printf("topk_val\n");
            // debug_print(kv_cache->topk_func->topk_val, num_tokens, kv_cache->topk_func->top);
            blockmask = kv_cache->blockmask;
        }

        mha_fwd_kvcache(
            TypeTraits<T>::type_code()==1,
            1,
            num_tokens,
            num_history_tokens+num_tokens,
            num_tokens,
            this->num_attention_heads,
            this->num_key_value_heads,
            this->head_dim,
            this->q_proj->output,
            kv_cache->k_cache,
            kv_cache->v_cache,
            nullptr,
            Mask(nullptr),
            this->attn_output,
            this->softmax_lse,
            this->softmax_lse_accum,
            this->oaccum,
            rsqrtf(float(this->head_dim)),
            true,
            -1,
            -1,
            0,
            stream.stream,
            blockmask,
            3072 // TODO minicpm4 fake block_window_size now
        );

        // flash attention and put output to attn_norm->output
        this->o_proj->prefill(stream, num_tokens, this->attn_output);

        kv_cache->next_kv_length = kv_cache->next_kv_length + num_tokens;
    }

    void decode(const Stream& stream, int32_t num_tokens, int32_t padded_length, T* input, T* prev_output, int32_t* position_ids, int32_t* cache_length, const Mask& mask, MiniCPM4KVCache<T>* kv_cache) {
        this->attn_norm->prefill(stream, num_tokens, input, prev_output);
        T *q, *k, *v;
        int merge_dim_out = (this->num_attention_heads + 2 * this->num_key_value_heads) * this->head_dim;
        if (num_tokens > 1) {
            linear<T>(stream, num_tokens, this->hidden_size, merge_dim_out, this->attn_norm->output, this->q_proj->weight, this->v_proj->output);
            permute(stream, num_tokens, this->num_attention_heads * this->head_dim, this->num_key_value_heads * this->head_dim, this->v_proj->output, this->q_proj->output);
        } else {
            linear<T>(stream, num_tokens, this->hidden_size, merge_dim_out, this->attn_norm->output, this->q_proj->weight, this->q_proj->output);
        }
        q = this->q_proj->output;
        k = q + num_tokens * this->num_attention_heads * this->head_dim;
        v = k + num_tokens * this->num_key_value_heads * this->head_dim;
        kv_cache->rotary_embedding->prefill(stream, num_tokens, this->num_attention_heads, this->num_key_value_heads, q, k, position_ids);

        copy_to_kvcache(stream, num_tokens, k, v, kv_cache, cache_length);

        kv_cache->compress(stream);

        uint64_t fakeblock; // TODO minicpm4 fake blockmask now
        uint64_t *blockmask = &fakeblock;
        mha_fwd_kvcache(
            TypeTraits<T>::type_code()==1,
            1,
            num_tokens,
            padded_length,
            num_tokens,
            this->num_attention_heads,
            this->num_key_value_heads,
            this->head_dim,
            q,
            kv_cache->k_cache,
            kv_cache->v_cache,
            cache_length,
            mask,
            this->attn_output,
            this->softmax_lse,
            this->softmax_lse_accum,
            this->oaccum,
            rsqrtf(float(this->head_dim)),
            true,
            -1,
            -1,
            0,
            stream.stream,
            blockmask, // TODO minicpm4 fake blockmask now
            3072 // TODO minicpm4 fake block_window_size now
        );

        // flash attention and put output to attn_norm->output
        this->o_proj->prefill(stream, num_tokens, this->attn_output);

        kv_cache->next_kv_length = kv_cache->next_kv_length + 1; // TODO minicpm4 eagle verify should -1 + acceptlength
    }
};