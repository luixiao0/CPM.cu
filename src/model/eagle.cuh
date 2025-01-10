#pragma once
#include "tree_drafter.cuh"
#include "model.cuh"
#include "topk.cuh"
#include "layer.cuh"
#include "kvcache.cuh"
#include "norm.cuh"
#include "elementwise.cuh"

namespace {
}

template<typename T>
struct Skip : Norm<T> {
    int dim;

    Skip(int dim) {
        this->dim = dim;
    }

    void init_weight_ptr(Memory* memory) {}

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * dim * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {}

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output, T* tgt=nullptr) {
        if (tgt == nullptr) tgt = this->output;
        if (prev_output == nullptr) {
            cudaMemcpy(tgt, input, sizeof(T) * this->dim * num_tokens, cudaMemcpyDeviceToDevice);
        } else {
            elementwise_add(stream, num_tokens, this->dim, input, prev_output, tgt);
        }
    }
};

template<typename T>
struct EagleImpl : Model {
    int num_layers;
    int num_iter;
    int topk_per_iter;
    int tree_size;
    int total_tried;

    ModelImpl<T>* model;
    KVCacheManager<T>* kv_caches;
    std::vector<Layer<T>*> layers;
    Linear<T, true, true> *fc1;
    Linear<T> *fc2;
    functions::TopK<T>* topk_func;

    T* last_token_hidden_state;
    int remaining_unprocessed_hidden_state;
    int32_t* eagle_position_ids, *eagle_cache_length;

    int32_t *h_best, *d_best;    

    T* tmp_kvcache;

    EagleImpl(
        ModelImpl<T>* model,
        int num_layers,
        int num_iter,
        int topk_per_iter,
        int tree_size
    ) {
        this->model = model;
        this->num_layers = num_layers;
        this->num_iter = num_iter;
        this->topk_per_iter = topk_per_iter;
        this->tree_size = tree_size;
        this->total_tried = topk_per_iter * topk_per_iter * (num_iter - 1) + topk_per_iter;

        kv_caches = new KVCacheManager<T>(num_layers, this->model->num_key_value_heads, this->model->head_dim);
        fc1 = new Linear<T, true, true>(this->model->hidden_size, this->model->hidden_size);
        fc2 = new Linear<T>(this->model->hidden_size, this->model->hidden_size);
        for (int i = 0; i < num_layers; i++) {
            layers.push_back(new Layer<T>(this->model->hidden_size, this->model->intermediate_size, this->model->num_attention_heads, this->model->num_key_value_heads, this->model->head_dim, this->model->rms_norm_eps));
        }

        topk_func = new functions::TopK<T>(model->vocab_size, topk_per_iter);
    }

    void init_weight_ptr(Memory* memory) {
        fc1->init_weight_ptr(memory);
        fc2->init_weight_ptr(memory);
        for (int i = 0; i < num_layers; i++) {
            layers[i]->init_weight_ptr(memory);
        }
        layers[0]->attn->attn_norm = new Skip<T>(this->model->hidden_size);
        kv_caches->rotary_embedding = this->model->kv_caches->rotary_embedding;
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        offset = fc1->init_output_ptr(memory, num_tokens, offset);
        offset = fc2->init_output_ptr(memory, num_tokens, offset);
        int64_t layer_end = 0;
        for (int i = 0; i < num_layers; i++) {
            layer_end = layers[i]->init_output_ptr(memory, num_tokens, offset);
        }
        offset = layer_end;

        offset = topk_func->init_output_ptr(memory, this->total_tried, offset);

        offset = memory->allocate((void**)&eagle_position_ids, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&eagle_cache_length, offset, sizeof(int32_t));

        offset = memory->allocate((void**)&d_best, offset, 2 * sizeof(int32_t));
        cudaMallocHost(&h_best, 2 * sizeof(int32_t));
        offset = memory->allocate((void**)&tmp_kvcache, offset, 64 * this->model->kv_caches->num_hidden_layers * 2 * this->model->kv_caches->dim * sizeof(T));
        return offset;
    }

    int init_storage() {
        this->model->init_weight_ptr(this->model->memory);
        this->init_weight_ptr(this->model->memory);
        int64_t offset = this->model->init_output_ptr(this->model->memory, this->model->chunk_length, this->model->memory->model_offset);
        int64_t kv_cache_offset = this->init_output_ptr(this->model->memory, this->model->chunk_length, offset);
        float ratio = float(this->model->num_hidden_layers) / (this->model->num_hidden_layers + this->num_layers);
        kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, ratio);
        kv_caches->init_output_ptr(this->model->memory, kv_cache_offset);
        return min(kv_caches->budget + 1, this->model->kv_caches->budget);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 5) == "eagle") {
            if (name.substr(0, 9) == "eagle.fc1") {
                fc1->load_to_storage(name, ptr);
            } else if (name.substr(0, 9) == "eagle.fc2") {
                fc2->load_to_storage(name, ptr);
            } else {
                std::regex layer_regex("eagle\\.layers\\.(\\d+)\\.(.*)");
                std::smatch matches;
                if (std::regex_search(name, matches, layer_regex)) {
                    int layer_idx = std::stoi(matches[1]);
                    layers[layer_idx]->load_to_storage(matches[2], ptr);
                } else {
                    throw std::invalid_argument("Unsupported name (layer_idx not found): " + name);
                }
            }
        } else {
            this->model->load_to_storage(name, ptr);
        }
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        if (num_history_tokens > 0) {
            this->fc2->prefill(calc_stream, 1, this->last_token_hidden_state);
        }
        this->model->prefill(num_tokens, num_history_tokens, input, position_ids, output);
        this->last_token_hidden_state = this->model->norm->output + (num_tokens - 1) * this->model->hidden_size;
        this->remaining_unprocessed_hidden_state = 1;

        if (num_history_tokens > 0) {
            this->fc2->prefill(calc_stream, num_tokens-1, this->model->norm->output, this->fc2->output + this->model->hidden_size);
            this->model->embedding->prefill(calc_stream, num_tokens, input);
            this->fc1->prefill(calc_stream, num_tokens, this->model->embedding->output);
            cudaMemcpy(this->eagle_position_ids + 1, position_ids, sizeof(int32_t) * (num_tokens-1), cudaMemcpyDeviceToDevice);
            elementwise_add(calc_stream, num_tokens, this->model->hidden_size, this->fc1->output, this->fc2->output, this->fc1->output);
            T* layer_output = nullptr;
            for (int i = 0; i < num_layers; i++) {
                this->layers[i]->prefill(num_tokens, num_history_tokens-1, this->fc1->output, layer_output, this->eagle_position_ids, this->kv_caches->caches[i]);
                layer_output = this->layers[i]->output;
            }
            elementwise_add(calc_stream, num_tokens, this->model->hidden_size, this->fc1->output, layer_output, this->fc1->output);
        } else {
            this->fc2->prefill(calc_stream, num_tokens-1, this->model->norm->output);
            this->model->embedding->prefill(calc_stream, num_tokens-1, input + 1);
            this->fc1->prefill(calc_stream, num_tokens-1, this->model->embedding->output);
            elementwise_add(calc_stream, num_tokens, this->model->hidden_size, this->fc1->output, this->fc2->output, this->fc1->output);
            T* layer_output = nullptr;
            for (int i = 0; i < num_layers; i++) {
                this->layers[i]->prefill(num_tokens-1, num_history_tokens, this->fc1->output, layer_output, position_ids, this->kv_caches->caches[i]);
                layer_output = this->layers[i]->output;
            }
            elementwise_add(calc_stream, num_tokens-1, this->model->hidden_size, this->fc1->output, layer_output, this->fc1->output);
        }

        cudaMemcpy(this->eagle_position_ids, position_ids + (num_tokens-1), sizeof(int32_t), cudaMemcpyDeviceToDevice);
    }

    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        this->model->decode(num_tokens, padded_length, input, position_ids, cache_length, mask_2d, output);
    }

    void draft(int32_t* tree_draft_ids, int32_t* tree_position_ids, int32_t* cache_length) {
        // TODO
    }

    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, int32_t* tree_parent) {
        verify_draft(calc_stream, num_tokens, pred, gt, position_ids, cache_length, mask_2d, tree_parent, this->d_best);
        cudaMemcpyAsync(this->h_best, this->d_best, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);
        fix_kv_cache(calc_stream, h_best[0], this->model->kv_caches->num_hidden_layers * 2, this->model->kv_caches->dim, pred, gt, cache_length, this->model->kv_caches->d_flat_caches, this->tmp_kvcache);
        this->last_token_hidden_state = this->model->norm->output + h_best[1] * this->model->hidden_size;
        return h_best[0];
    }
};