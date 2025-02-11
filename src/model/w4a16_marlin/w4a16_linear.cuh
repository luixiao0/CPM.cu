#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../linear.cuh"
#include "../../qgemm/exllamav2/config.h"
#include "../../qgemm/exllamav2/cuda/q_matrix.cuh"
#include "../../qgemm/exllamav2/cuda/q_gemm.cuh"
#include "../../qgemm/marlin/marlin_cuda_kernel.cuh"

template <typename T, bool transposed=true, bool has_bias=false>
struct W4A16Linear {
    int dim_in;
    int dim_out;
    T* output;
    T* weight;
    T* bias;

    virtual void load_to_storage(std::string name, void* ptr) = 0;
    virtual void init_weight_ptr(Memory* memory) = 0;
    virtual int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) = 0;
    virtual void prefill(const Stream& stream, int32_t num_tokens, T* input, T* tgt=nullptr, bool inplace=false) = 0;
    virtual ~W4A16Linear() {}
};

template <typename T, bool transposed=true, bool has_bias=false>
struct QLinear : public W4A16Linear<T, transposed, has_bias> {

    uint32_t* weight; // device
    uint32_t* qzeros; // device
    uint16_t* perm; // device
    uint16_t* invperm; // device
    uint32_t* g_idx; // device or cpu?
    T* temp_dq; // device half
    T* qscales; // device
    QMatrix * qm;
    int group_size;
    int bits;
    int load_ctx;

    QLinear(int dim_in, int dim_out, int group_size, int bits) {
        this->dim_in = dim_in;
        this->dim_out = dim_out;
        this->group_size = group_size;
        this->bits = bits;
        this->load_ctx = 0;
    }

    ~QLinear() {
        free(g_idx);
    }

    void init_weight_ptr(Memory* memory) {

        const int per_int_size = 32 / this->bits;
        const int weight_size = this->dim_in / per_int_size * this->dim_out;
        const int qzeros_size = this->dim_in / this->group_size * this->dim_out / per_int_size;
        const int qscales_size = this->dim_in / this->group_size * this->dim_out;
        const int perm_size = this->dim_in / per_int_size * 8;
        const int g_idx_size = this->dim_in;

        weight = (uint32_t*)memory->allocate_for_model(weight_size * sizeof(uint32_t));
        qzeros = (uint32_t*)memory->allocate_for_model(qzeros_size * sizeof(uint32_t));
        qscales = (T*)memory->allocate_for_model(qscales_size * sizeof(T));
        perm = (uint16_t*)memory->allocate_for_model(perm_size * sizeof(uint16_t));
        cudaMemset(perm, 0, perm_size * sizeof(uint16_t));
        invperm = (uint16_t*)memory->allocate_for_model(perm_size * sizeof(uint16_t));
        cudaMemset(invperm, 0, perm_size * sizeof(uint16_t));
        g_idx = (uint32_t*) malloc(g_idx_size * sizeof(uint32_t));

        if constexpr (has_bias) {
            this->bias = (T*)memory->allocate_for_model(this->dim_out * sizeof(T));
        }
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        const int temp_dq_size = this->dim_in * this->dim_out;
        memory->allocate((void**)&this->temp_dq, offset, temp_dq_size * sizeof(T));

        return memory->allocate((void**)&this->output, offset + temp_dq_size * sizeof(T), num_tokens * this->dim_out * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("qweight") != std::string::npos) {
            const int per_int_size = 32 / this->bits;
            const int weight_size = this->dim_in / per_int_size * this->dim_out;
            cudaMemcpy((void*)this->weight, ptr, weight_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
            this->load_ctx++;
        } else if (name.find("bias") != std::string::npos) {
            if (has_bias) cudaMemcpy((void*)this->bias, ptr, this->dim_out * sizeof(T), cudaMemcpyHostToDevice);
        } else if (name.find("qzeros") != std::string::npos){
            const int per_int_size = 32 / this->bits;
            const int qzeros_size = this->dim_in / this->group_size * this->dim_out / per_int_size;
            cudaMemcpy((void*)this->qzeros, ptr, qzeros_size * sizeof(uint32_t), cudaMemcpyHostToDevice);
            this->load_ctx++;
        } else if (name.find("g_idx") != std::string::npos){
            memcpy(g_idx, ptr, this->dim_in * sizeof(uint32_t));
            this->load_ctx++;
        } else if (name.find("scales") != std::string::npos){
            const int qscales_size = this->dim_in / this->group_size * this->dim_out;
            cudaMemcpy((void*)this->qscales, ptr, qscales_size * sizeof(T), cudaMemcpyHostToDevice);
            this->load_ctx++;
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
        if (this->load_ctx == 4) {
            init_quant_matraix();
        }
    }


    void init_quant_matraix() {
        int currentDevice;
        cudaPointerAttributes attr;
        cudaError_t status = cudaPointerGetAttributes(&attr, this->weight);
        if (status == cudaSuccess) {
            currentDevice = attr.device;
        }

        qm = new QMatrix
        (
            currentDevice,
            this->dim_in,
            this->dim_out,
            this->dim_in / this->group_size,
            this->weight,
            this->perm,
            this->invperm,
            NULL,
            NULL,
            NULL,
            this->qzeros,
            this->qscales,
            this->g_idx,
            NULL
        );
    }

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* tgt=nullptr, bool inplace=false) {
        if (tgt == nullptr) tgt = this->output;
        // int currentDevice;
        // cudaPointerAttributes attr;
        // cudaError_t status = cudaPointerGetAttributes(&attr, this->weight);
        // if (status == cudaSuccess) {
        //     currentDevice = attr.device;
        // }
        // cudaSetDevice(currentDevice);

        gemm_half_q_half_cuda
        (
            stream.cublas_handle,
            input,
            this->qm,
            tgt,
            num_tokens, // m
            this->dim_out, // n
            this->dim_in, // k
            !inplace,
            this->temp_dq,
            false,
            stream.stream
        );
        if constexpr (has_bias) {
            batched_add<T>(stream, num_tokens, this->dim_out, tgt, this->bias, tgt);
        }
    }
};

template <typename T, bool transposed=true, bool has_bias=false>
struct MarlinLinear : public W4A16Linear<T, transposed, has_bias> {
    int32_t* B; // device
    T* s; // device
    int32_t* workspace; // device
    const int group_size = 128; // marlin only support group size 128

    MarlinLinear(int dim_in, int dim_out) {
        this->dim_in = dim_in;
        this->dim_out = dim_out;
    }

    void init_weight_ptr(Memory* memory) {
        const int B_size = this->dim_in * this->dim_out / 8;
        this->B = (int32_t*)memory->allocate_for_model(B_size * sizeof(int32_t));

        const int s_size = this->dim_in * this->dim_out / this->group_size;
        this->s = (T*)memory->allocate_for_model(s_size * sizeof(T));

        const int workspace_size = this->dim_out / 8;
        this->workspace = (int32_t*)memory->allocate_for_model(workspace_size * sizeof(int32_t));
        if constexpr (has_bias) {
            this->bias = (T*)memory->allocate_for_model(this->dim_out * sizeof(T));
        }
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        return memory->allocate((void**)&this->output, offset, num_tokens * this->dim_out * sizeof(T));
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find(".B") != std::string::npos) {
            const int B_size = this->dim_in * this->dim_out / 8;
            cudaMemcpy((void*)this->B, ptr, B_size * sizeof(int32_t), cudaMemcpyHostToDevice);
        } else if (name.find(".bias") != std::string::npos) {
            cudaMemcpy((void*)this->bias, ptr, this->dim_out * sizeof(T), cudaMemcpyHostToDevice);
        } else if (name.find(".s") != std::string::npos) {
            const int s_size = this->dim_in * this->dim_out / this->group_size;
            cudaMemcpy((void*)this->s, ptr, s_size * sizeof(T), cudaMemcpyHostToDevice);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }


    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* tgt=nullptr, bool inplace=false) {
        T* tgt_temp;
        if (tgt == nullptr) {
            tgt_temp = this->output;
            tgt = tgt_temp;
        } else if (inplace && tgt) {
            tgt_temp = this->output;
        }
        else if (!inplace && tgt) {
            tgt_temp = tgt;
        }

        int err = marlin_cuda(
            input,
            this->B,
            tgt_temp,
            this->s,
            num_tokens, this->dim_out, this->dim_in,
            this->workspace,
            this->group_size,
            0,
            stream.stream,
            -1, -1, -1, 16
        );
        assert(err == 0);
        if (inplace) {
            elementwise_add<T>(stream, num_tokens, this->dim_out, tgt, tgt_temp, tgt);
        }
        if constexpr (has_bias) {
            batched_add<T>(stream, num_tokens, this->dim_out, tgt, this->bias, tgt);
        }
    }
};

template <typename T, bool transposed=true, bool has_bias=false>
struct W4A16LinearFactory {
    static W4A16Linear<T, transposed, has_bias>* CreateLinear(int dim_in, int dim_out, int group_size, int bits, bool use_marlin) {
        if (use_marlin && group_size == 128 && bits == 4) {
            return new MarlinLinear<T, transposed, has_bias>(dim_in, dim_out); // Only support group size = 128, bits = 4
        } else if (!use_marlin && group_size != 0) {
            return new QLinear<T, transposed, has_bias>(dim_in, dim_out, group_size, bits);
        }
    }
};