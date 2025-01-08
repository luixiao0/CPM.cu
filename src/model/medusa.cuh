#pragma once
#include "model.cuh"

namespace {
__global__ void verify_kernel(int num_tokens, int32_t* pred, const int32_t* gt, const int32_t* position_ids, const int32_t* cache_length, const uint64_t* attn_mask, const int32_t* tree_parent, int32_t* d_best) {
    int i = threadIdx.x;

    __shared__ uint64_t s_correct_mask[2];
    uint64_t correct_mask = 1;
    if (0 < i && i < num_tokens && pred[i] == gt[tree_parent[i]]) correct_mask |= 1ULL << i;
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 16);
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 8);
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 4);
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 2);
    correct_mask |= __shfl_down_sync(0xffffffff, correct_mask, 1);
    if (i % 32 == 0) s_correct_mask[i / 32] = correct_mask;
    __syncthreads();
    if (i == 0) s_correct_mask[0] |= s_correct_mask[1];
    __syncthreads();
    correct_mask = s_correct_mask[0];

    __shared__ int32_t mx[64], mx_idx[64];
    int prefix_length = cache_length[0];
    if ((correct_mask & attn_mask[i]) == attn_mask[i]) {
        mx[i] = position_ids[i] - prefix_length + 1; mx_idx[i] = i;
    } else {
        mx[i] = 1; mx_idx[i] = 0;
    }
    for (int offset = 32; offset > 0; offset >>= 1) {
        if (i < offset && mx[i + offset] > mx[i]) {
            mx[i] = mx[i + offset];
            mx_idx[i] = mx_idx[i + offset];
        }
        __syncthreads();
    }
    if (i == 0) {
        d_best[0] = mx[0]; d_best[1] = mx_idx[0];
    }
    __syncthreads();

    int p = mx_idx[0];
    if (i < num_tokens && (attn_mask[p] >> i & 1)) {
        pred[position_ids[i] - prefix_length] = i;
    }
}

template<typename T>
__global__ void fix_kvcache_kernel_1(int num_caches, int dim, int32_t* pred, const int32_t* gt, const int32_t* cache_length, const T* const* flat_caches, float4* tmp_kvcache) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k = blockIdx.y;
    int prefix_length = cache_length[0];
    int real_i = pred[i] + prefix_length;
    float4* tmp = tmp_kvcache + i * num_caches * dim;
    const float4* flat = (const float4*)flat_caches[k];
    for (int d = j; d < dim; d += blockDim.x) {
        tmp[k * dim + d] = flat[real_i * dim + d];
    }
}

template<typename T>
__global__ void fix_kvcache_kernel_2(int num_caches, int dim, int32_t* pred, const int32_t* gt, const int32_t* cache_length, T** flat_caches, const float4* tmp_kvcache) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int k = blockIdx.y;
    int prefix_length = cache_length[0];
    int real_i = i + prefix_length;
    const float4* tmp = tmp_kvcache + i * num_caches * dim;
    float4* flat = (float4*)flat_caches[k];
    for (int d = j; d < dim; d += blockDim.x) {
        flat[real_i * dim + d] = tmp[k * dim + d];
    }
    if (j == 0 && k == 0) {
        pred[i] = gt[pred[i]];
    }
}
}

void verify_draft(const Stream& stream, int num_tokens, int32_t* pred, const int32_t* gt, const int32_t* position_ids, const int32_t* cache_length, const uint64_t* attn_mask, const int32_t* tree_parent, int32_t* best) {
    verify_kernel<<<1, num_tokens, 0, stream.stream>>>(num_tokens, pred, gt, position_ids, cache_length, attn_mask, tree_parent, best);
}

template<typename T>
void fix_kv_cache(const Stream& stream, int accept_length, int num_caches, int dim, int32_t* pred, const int32_t* gt, const int32_t* cache_length, T** flat_caches, T* tmp_kvcache) {
    fix_kvcache_kernel_1<T><<<dim3(accept_length, num_caches, 1), 256, 0, stream.stream>>>(num_caches, dim/(16/sizeof(T)), pred, gt, cache_length, flat_caches, (float4*)tmp_kvcache);
    fix_kvcache_kernel_2<T><<<dim3(accept_length, num_caches, 1), 256, 0, stream.stream>>>(num_caches, dim/(16/sizeof(T)), pred, gt, cache_length, flat_caches, (float4*)tmp_kvcache);
}

template<typename T>
struct ResidualBlock : Linear<T, /*transposed=*/true, /*bias=*/true> {
    ResidualBlock(int dim_in, int dim_out) : Linear<T, true, true>(dim_in, dim_out) {}

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* tgt=nullptr, bool inplace=false) {
        if (tgt == nullptr) tgt = this->output;
        Linear<T, true, true>::prefill(stream, num_tokens, input);
        silu_inplace<T>(stream, num_tokens, this->dim_out, this->output);
        elementwise_add<T>(stream, num_tokens, this->dim_out, this->output, input, tgt);
    }
};

template<typename T>
struct MedusaImpl : Model {
    int num_heads;
    int num_layers;

    ModelImpl<T>* model;
    std::vector<ResidualBlock<T>*> blocks;
    std::vector<Linear<T>*> lm_heads;

    T* last_token_hidden_state;
    int32_t *h_best, *d_best;

    T* tmp_kvcache;

    MedusaImpl(
        ModelImpl<T>* model,
        int num_heads,
        int num_layers
    ) {
        this->model = model;
        this->num_heads = num_heads;
        this->num_layers = num_layers; // asserted in python that num_layers == 1

        for (int i = 0; i < num_heads; i++) {
            blocks.push_back(new ResidualBlock<T>(model->hidden_size, model->hidden_size));
            lm_heads.push_back(new Linear<T>(model->hidden_size, model->vocab_size));
        }
    }

    void init_weight_ptr(Memory* memory) {
        for (int i = 0; i < num_heads; i++) {
            blocks[i]->init_weight_ptr(memory);
            lm_heads[i]->init_weight_ptr(memory);
        }
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
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
        for (int i = 0; i < num_heads; i++) {
            blocks[i]->prefill(calc_stream, 1, this->last_token_hidden_state);
            lm_heads[i]->prefill(calc_stream, 1, blocks[i]->output, (T*)output + i * this->model->vocab_size);
        }
    }

    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* attn_mask, int32_t* tree_parent) {
        verify_draft(calc_stream, num_tokens, pred, gt, position_ids, cache_length, attn_mask, tree_parent, this->d_best);
        cudaMemcpyAsync(this->h_best, this->d_best, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);
        fix_kv_cache(calc_stream, h_best[0], this->model->kv_caches->num_hidden_layers * 2, this->model->kv_caches->dim, pred, gt, cache_length, this->model->kv_caches->d_flat_caches, this->tmp_kvcache);
        this->last_token_hidden_state = this->model->norm->output + h_best[1] * this->model->hidden_size;
        return h_best[0];
    }
};