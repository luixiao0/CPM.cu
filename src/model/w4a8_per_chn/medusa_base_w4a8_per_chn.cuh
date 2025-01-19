#pragma once
#include "w4a8_per_chn_model.cuh"
#include "../medusa.cuh"


template<typename T>
struct MedusaImplBaseW4A8PerChn : Model {
    int num_heads;
    int num_layers;

    W4A8PerChnModelImpl<T>* model;
    Linear<T>* rotation;
    std::vector<ResidualBlock<T>*> blocks;
    std::vector<Linear<T>*> lm_heads;

    T* last_token_hidden_state;
    int32_t *h_best, *d_best;

    T* tmp_kvcache;

    MedusaImplBaseW4A8PerChn(
        W4A8PerChnModelImpl<T>* model,
        int num_heads,
        int num_layers
    ) {
        this->model = model;
        this->num_heads = num_heads;
        this->num_layers = num_layers; // asserted in python that num_layers == 1

        this->rotation = new Linear<T>(model->hidden_size, model->hidden_size);
        for (int i = 0; i < num_heads; i++) {
            blocks.push_back(new ResidualBlock<T>(model->hidden_size, model->hidden_size));
            lm_heads.push_back(new Linear<T>(model->hidden_size, model->vocab_size));
        }
    }

    void init_weight_ptr(Memory* memory) {
        rotation->init_weight_ptr(memory);
        for (int i = 0; i < num_heads; i++) {
            blocks[i]->init_weight_ptr(memory);
            lm_heads[i]->init_weight_ptr(memory);
        }
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        offset = rotation->init_output_ptr(memory, num_tokens, offset);
        for (int i = 0; i < num_heads; i++) {
            offset = blocks[i]->init_output_ptr(memory, num_tokens, offset);
            // lm_head do not allocate, directly output
        } 
        offset = memory->allocate((void**)&d_best, offset, 2 * sizeof(int32_t));
        cudaMallocHost(&h_best, 2 * sizeof(int32_t));
        offset = memory->allocate((void**)&tmp_kvcache, offset, 64 * this->model->kv_caches->num_hidden_layers * 2 * this->model->kv_caches->dim * sizeof(T));
        return offset;
    }

    int init_storage() {
        this->model->init_weight_ptr(this->model->memory);
        this->init_weight_ptr(this->model->memory);
        int64_t offset = this->model->init_output_ptr(this->model->memory, this->model->chunk_length, this->model->memory->model_offset);
        int64_t kv_cache_offset = this->init_output_ptr(this->model->memory, 1, offset);
        this->model->init_kv_cache(kv_cache_offset);
        return this->model->kv_caches->budget;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 6) == "medusa") {
            if (name.find("rotation") != std::string::npos) {
                this->rotation->load_to_storage(name, ptr);
            } else {
                std::regex layer_regex("medusa\\.(\\d+)\\.(\\d+).*");
                std::smatch matches;
                if (std::regex_search(name, matches, layer_regex)) {
                    int head_idx = std::stoi(matches[1]);
                    int layer_idx = std::stoi(matches[2]);
                    if (layer_idx == 0) {
                        blocks[head_idx]->load_to_storage(name, ptr);
                    } else {
                        lm_heads[head_idx]->load_to_storage(name, ptr);
                    }
                }
            }
        } else {
            this->model->load_to_storage(name, ptr);
        }
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        this->model->prefill(num_tokens, num_history_tokens, input, position_ids, output);
        this->last_token_hidden_state = this->model->norm->output + (num_tokens - 1) * this->model->hidden_size;
    }

    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        this->model->decode(num_tokens, padded_length, input, position_ids, cache_length, mask_2d, output);
    }

    void draft(void* output) {
        rotation->prefill(1, this->last_token_hidden_state);
        for (int i = 0; i < num_heads; i++) {
            blocks[i]->prefill(1, this->rotation->output);
            lm_heads[i]->prefill(1, blocks[i]->output, (T*)output + i * this->model->vocab_size);
        }
    }

    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* attn_mask, int32_t* tree_parent) {
        verify_kernel<<<1, 64, 0, calc_stream>>>(num_tokens, pred, gt, position_ids, cache_length, attn_mask, tree_parent, d_best);
        cudaMemcpyAsync(h_best, d_best, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream);
        cudaStreamSynchronize(calc_stream);
        fix_kv_cache(h_best[0], this->model->kv_caches->num_hidden_layers * 2, this->model->kv_caches->dim, pred, gt, cache_length, this->model->kv_caches->d_flat_caches, this->tmp_kvcache);
        this->last_token_hidden_state = this->model->norm->output + h_best[1] * this->model->hidden_size;
        return h_best[0];
    }
};