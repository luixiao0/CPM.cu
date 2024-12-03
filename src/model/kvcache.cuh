#pragma once
#include "../trait.cuh"
#include <vector>
#include <cuda_runtime.h>

template <typename T>
struct KVCache {
    int dim;
    int budget;
    T *k_cache, *v_cache;
    
    KVCache(int dim, int budget) {
        this->dim = dim;
        this->budget = budget;
    }

    int64_t init_output_ptr(Memory* memory, int64_t offset) {
        k_cache = (T*)(memory->memory_pool + offset);
        offset += budget * dim * sizeof(T);
        v_cache = (T*)(memory->memory_pool + offset);
        offset += budget * dim * sizeof(T);
        return offset;
    }
};

template <typename T>
struct KVCacheManager {
    int num_hidden_layers;
    int dim;
    std::vector<KVCache<T>*> caches;

    KVCacheManager(int num_hidden_layers, int dim) {
        this->num_hidden_layers = num_hidden_layers;
        this->dim = dim;
    }

    int64_t init_output_ptr(Memory* memory, int64_t offset) {
        int64_t budget = memory->memory_limit - offset;
        budget /= this->num_hidden_layers * 2 * this->dim * sizeof(T);
        for (int i = 0; i < this->num_hidden_layers; i++) {
            caches.push_back(new KVCache<T>(this->dim, budget));
        }
        for (int i = 0; i < this->num_hidden_layers; i++) {
            offset = caches[i]->init_output_ptr(memory, offset);
        }
        return budget;
    }
};
