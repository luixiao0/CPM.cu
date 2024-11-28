#pragma once
#include <cuda_runtime.h>

struct Memory {
    int64_t memory_limit;
    uint8_t* memory_pool;
    int64_t model_offset;
    int64_t kv_cache_offset;

    Memory(int64_t memory_limit, void* memory_pool) {
        this->memory_limit = memory_limit;
        this->memory_pool = (uint8_t*)memory_pool;
        this->model_offset = 0;
    }

    void* allocate_for_model(size_t size) {
        model_offset = (model_offset + 15) / 16 * 16; // Align to 16 bytes
        uint8_t* ret = memory_pool + model_offset;
        model_offset += size;
        return (void*)ret;
    }
};