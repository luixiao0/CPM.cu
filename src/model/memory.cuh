#pragma once
#include "../utils.cuh"
#include <cuda_runtime.h>
#include "../signal_handler.cuh"

#define ALIGN_SIZE 16

struct Memory {
    int64_t memory_limit;
    uint8_t* memory_pool;
    int64_t model_offset;

    Memory(int64_t memory_limit) {
        this->memory_limit = memory_limit;
        this->model_offset = 0;
        
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&this->memory_pool), memory_limit);
        if (err != cudaSuccess) {
            print_stack_trace();
            fprintf(stderr, "\nError: cudaMalloc failed in Memory constructor: %s, size: %ld\n\n", cudaGetErrorString(err), memory_limit);
            this->memory_pool = nullptr;
        }
    }

    // 添加析构函数防止内存泄漏
    ~Memory() {
        if (memory_pool != nullptr) {
            cudaError_t err = cudaFree(memory_pool);
            if (err != cudaSuccess) {
                fprintf(stderr, "\nWarning: cudaFree failed in Memory destructor: %s\n\n", cudaGetErrorString(err));
            }
        }
    }

#ifdef DISABLE_MEMPOOL
    void* allocate_for_model(size_t size) {
        void* ptr;
        size_t aligned_size = ROUND_UP(size, ALIGN_SIZE);
        cudaError_t err = cudaMalloc(&ptr, aligned_size);
        if (err != cudaSuccess) {
            print_stack_trace();
            fprintf(stderr, "\nError: cudaMalloc failed: %s, size: %ld\n\n", cudaGetErrorString(err), size);
            return nullptr;
        }
        return ptr;
    }
    int64_t allocate(void** ptr, int64_t offset, size_t size = 0) { // 0 for reuse previous allocated memory, just need start offset, return value is useless
        if (size == 0) {
            print_stack_trace();
            fprintf(stderr, "\nError: size is 0\n\n");
            return -1;
        }
        
        size_t aligned_size = ROUND_UP(size, ALIGN_SIZE);
        cudaError_t err = cudaMalloc(ptr, aligned_size);
        if (err != cudaSuccess) {
            print_stack_trace();
            fprintf(stderr, "\nError: cudaMalloc failed: %s, size: %ld\n\n", cudaGetErrorString(err), size);
            *ptr = nullptr;
            return -1;
        }
        
        // Update max_output_offset for tracking purposes
        offset += aligned_size;
        return offset;
    }
#else
    void* allocate_for_model(size_t size) {
        uint8_t* ret = memory_pool + model_offset;
        model_offset += size;
        model_offset = ROUND_UP(model_offset, ALIGN_SIZE); // Align to 16 bytes
        if (model_offset > this->memory_limit) {
            print_stack_trace();
            fprintf(stderr, "\nError: memory limit exceeded, offset: %ld, size: %ld, memory_limit: %ld\n\n", model_offset, size, this->memory_limit);
            return nullptr;
        }
        return (void*)ret;
    }
    int64_t allocate(void** ptr, int64_t offset, size_t size = 0) { // 0 for reuse previous allocated memory, just need start offset, return value is useless
        if (size == 0) {
            print_stack_trace();
            fprintf(stderr, "\nError: size is 0\n\n");
            return -1;
        }
        *ptr = memory_pool + offset;
        offset += size;
        offset = ROUND_UP(offset, ALIGN_SIZE); // Align to 16 bytes
        if (offset > this->memory_limit) {
            print_stack_trace();
            fprintf(stderr, "\nError: memory limit exceeded, offset: %ld, size: %ld, memory_limit: %ld\n\n", offset, size, this->memory_limit);
            *ptr = nullptr;
            return -1;
        }
        return offset;
    }
#endif
};