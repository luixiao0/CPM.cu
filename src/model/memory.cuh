#pragma once
#include "../utils.cuh"
#include <cuda_runtime.h>
#include "../signal_handler.cuh"

#define ALIGN_SIZE 16

struct Memory {
    int64_t memory_limit;
    uint8_t* memory_pool;
    int64_t model_offset;
    int64_t max_output_offset;

    Memory(int64_t memory_limit, void* memory_pool) {
        this->memory_limit = memory_limit;
        this->memory_pool = (uint8_t*)memory_pool;
        this->model_offset = 0;
        this->max_output_offset = 0;
    }

    void* allocate_for_model(size_t size) {
        uint8_t* ret = memory_pool + model_offset;
        model_offset += size;
        model_offset = ROUND_UP(model_offset, ALIGN_SIZE); // Align to 16 bytes
        return (void*)ret;
    }
#if 1
    void* allocate_for_model_cudaMalloc(size_t size) {
        void* ptr;
        size_t aligned_size = ROUND_UP(size, ALIGN_SIZE);
        cudaError_t err = cudaMalloc(&ptr, aligned_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "Warning: cudaMalloc failed: %s, size: %ld\n", cudaGetErrorString(err), size);
            print_stack_trace();
            return nullptr;
        }
        return ptr;
    }
#else
    void* allocate_for_model_cudaMalloc(size_t size) {
        uint8_t* ret = memory_pool + model_offset;
        model_offset += size;
        model_offset = ROUND_UP(model_offset, ALIGN_SIZE); // Align to 16 bytes
        return (void*)ret;
    }
#endif
    
#if 1
    int64_t allocate(void** ptr, int64_t offset, size_t size = 0) { // 0 for reuse previous allocated memory, just need start offset, return value is useless
        if (size == 0) {
            fprintf(stderr, "Warning: size is 0\n");
            print_stack_trace();
        }
        *ptr = memory_pool + offset;
        offset += size;
        offset = ROUND_UP(offset, ALIGN_SIZE); // Align to 16 bytes
        if (offset > this->max_output_offset) {
            this->max_output_offset = offset;
        }
        if (offset > this->memory_limit) {
            print_stack_trace();
            fprintf(stderr, "Warning: memory limit exceeded, offset: %ld, size: %ld, memory_limit: %ld\n", offset, size, this->memory_limit);
            return -1;
        }
        return offset;
    }
#else
    int64_t allocate(void** ptr, int64_t offset, size_t size = 0) { // 0 for reuse previous allocated memory, just need start offset, return value is useless
        if (size == 0) {
            fprintf(stderr, "Warning: size is 0\n");
            print_stack_trace();
            return -1;
        }
        
        size_t aligned_size = ROUND_UP(size, ALIGN_SIZE);
        cudaError_t err = cudaMalloc(ptr, aligned_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "Warning: cudaMalloc failed: %s, size: %ld\n", cudaGetErrorString(err), size);
            print_stack_trace();
            *ptr = nullptr;
            return -1;
        }
        
        // Update max_output_offset for tracking purposes
        offset += aligned_size;
        if (offset > this->max_output_offset) {
            this->max_output_offset = offset;
        }
        
        return offset;
    }
#endif
};