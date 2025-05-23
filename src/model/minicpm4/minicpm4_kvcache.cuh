#pragma once
#include "../kvcache.cuh"

namespace {
template <typename T>
__global__ void meanpooling_16_kernel(int left, int dim, T* compressed, const T* k_cache) {
    __shared__ T s[32][33];

    int idx = blockIdx.x + left;
    int orig_left = idx * 16;
    T* c = compressed + idx * dim;
    const T* k = k_cache + orig_left * dim;
    int i = threadIdx.x / 32;
    int j = threadIdx.x % 32;

    for (int offset = 0; offset < dim; offset += 32) {
        s[i][j] = k[i * dim + offset + j];
        __syncthreads();
        float v = s[j][i];
        v += __shfl_down_sync(0xffffffff, v, 16);
        v += __shfl_down_sync(0xffffffff, v, 8);
        v += __shfl_down_sync(0xffffffff, v, 4);
        v += __shfl_down_sync(0xffffffff, v, 2);
        v += __shfl_down_sync(0xffffffff, v, 1);
        if (j == 0) {
            c[offset + i] = T(v / 32.0f);
        }
    }
}

template <typename T>
__global__ void meanpooling_64_kernel(int left, int dim, T* compressed, const T* k_cache) {
    __shared__ T s[32][33];

    int idx = blockIdx.x + left;
    int orig_left = idx * 64;
    T* c = compressed + idx * dim;
    const T* k = k_cache + orig_left * dim;
    int i = threadIdx.x / 32;
    int j = threadIdx.x % 32;

    for (int offset = 0; offset < dim; offset += 32) {
        float v_sum[32] = {0};
        for (int offset_row = 0; offset_row < 128; offset_row += 32) {
            s[i][j] = k[(i + offset_row) * dim + offset + j];
            __syncthreads();
            float v = s[j][i];
            v += __shfl_down_sync(0xffffffff, v, 16);
            v += __shfl_down_sync(0xffffffff, v, 8);
            v += __shfl_down_sync(0xffffffff, v, 4);
            v += __shfl_down_sync(0xffffffff, v, 2);
            v += __shfl_down_sync(0xffffffff, v, 1);
            if (j == 0) {
                v_sum[i] += v;
            }
        }
        if (j == 0) {
            c[offset + i] = T(v_sum[i] / 128.0f);
        }
    }
}

template <typename T>
void meanpooling(const Stream& stream, int left, int right, int dim, T* compressed, const T* k_cache, int stride) {
    if (left == right) return;
    if (stride == 16) {
        meanpooling_16_kernel<<<right-left, 1024, 0, stream.stream>>>(left, dim, compressed, k_cache);
    } else if (stride == 64) {
        meanpooling_64_kernel<<<right-left, 1024, 0, stream.stream>>>(left, dim, compressed, k_cache);
    } else {
        throw std::runtime_error("Unsupported meanpooling stride: " + std::to_string(stride));
    }
}
}

template <typename T>
struct MiniCPM4KVCache : KVCache<T> {
    T *c1_cache, *c2_cache;
    int c1_stride, c2_stride;
    int prev_kv_length;
    int next_kv_length;

    MiniCPM4KVCache(int dim, RotaryEmbedding<T> *rotary_embedding) : KVCache<T>(dim, rotary_embedding) {
        c1_stride = 16;
        c2_stride = 64;
        assert(this->dim % 32 == 0);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int32_t num_c1, int32_t num_c2, int64_t offset) {
        offset = KVCache<T>::init_output_ptr(memory, num_tokens, offset);
        offset = memory->allocate((void**)&this->c1_cache, offset, num_c1 * this->dim * sizeof(T));
        offset = memory->allocate((void**)&this->c2_cache, offset, num_c2 * this->dim * sizeof(T));
        return offset;
    }

    T* offset_c1(int offset) { return c1_cache + offset * this->dim; }
    T* offset_c2(int offset) { return c2_cache + offset * this->dim; }

    void compress(const Stream& stream) {
        int prev_pos, cur_pos;
        prev_pos = max((this->prev_kv_length - c1_stride) / c1_stride, 0);
        cur_pos = max((this->next_kv_length - c1_stride) / c1_stride, 0);
        meanpooling(stream, prev_pos, cur_pos, this->dim, this->c1_cache, this->k_cache, c1_stride);
        prev_pos = max((this->prev_kv_length - c2_stride) / c2_stride, 0);
        cur_pos = max((this->next_kv_length - c2_stride) / c2_stride, 0);
        meanpooling(stream, prev_pos, cur_pos, this->dim, this->c2_cache, this->k_cache, c2_stride);
    }
};

template <typename T>
struct MiniCPM4KVCacheManager {
    int num_hidden_layers;
    int dim;
    int budget;
    int budget_c1, budget_c2;
    std::vector<MiniCPM4KVCache<T>*> caches;
    T **h_flat_caches, **d_flat_caches;
    RotaryEmbedding<T> *rotary_embedding;

    MiniCPM4KVCacheManager(int num_hidden_layers, int num_key_value_heads, int head_dim) {
        this->num_hidden_layers = num_hidden_layers;
        this->dim = num_key_value_heads * head_dim;
        this->rotary_embedding = new RotaryEmbedding<T>(head_dim);
    }

    void init_weight_ptr(Memory* memory) {
        this->rotary_embedding->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int64_t offset, float ratio=1.0) {
        offset = memory->allocate((void**)&this->d_flat_caches, offset, num_hidden_layers * 2 * sizeof(T*));

        budget = int64_t((memory->memory_limit - offset) * ratio) / (this->num_hidden_layers * 2 * this->dim * sizeof(T));
        for (int i = 0; i < this->num_hidden_layers; i++) {
            caches.push_back(new MiniCPM4KVCache<T>(this->dim, this->rotary_embedding));
        }
        budget_c2 = (int)(budget / 69.0); // 1 + 4 + 64
        budget_c1 = budget_c2 * 4;
        budget = budget_c1 * 16;
        for (int i = 0; i < this->num_hidden_layers; i++) {
            offset = caches[i]->init_output_ptr(memory, budget, budget_c1, budget_c2, offset);
        }
        this->h_flat_caches = new T*[num_hidden_layers * 2];
        for (int i = 0; i < num_hidden_layers; i++) {
            this->h_flat_caches[i * 2] = caches[i]->k_cache;
            this->h_flat_caches[i * 2 + 1] = caches[i]->v_cache;
        }
        cudaMemcpy(this->d_flat_caches, this->h_flat_caches, num_hidden_layers * 2 * sizeof(T*), cudaMemcpyHostToDevice);
        return offset;
    }
};
