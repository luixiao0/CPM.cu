#pragma once
#include "../trait.cuh"
#include "linear.cuh"
#include "norm.cuh"
#include <cuda_runtime.h>

template <typename T>
__global__ void inplace_add_kernel(int numel, T* tgt, const T* src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        tgt[idx] += src[idx];
    }
}

template <typename T>
void inplace_add(int numel, T* tgt, const T* src) {
    inplace_add_kernel<T><<<(numel+255)/256, 256>>>(numel, tgt, src); // TODO adjust 256, TODO float4
}

template <typename T>
__global__ void inplace_gated_silu_kernel(int numel, T* tgt, const T* src) { // gated_silu(tgt, src) = tgt * sigmoid(tgt) * src
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float v = TypeTraits<T>::to_float(tgt[idx]);
        float s = 1.0f / (1.0f + exp(-v));
        tgt[idx] = TypeTraits<T>::from_float(v * s * TypeTraits<T>::to_float(src[idx]));
    }
}

template <typename T>
void inplace_gated_silu(int numel, T* tgt, const T* src) {
    inplace_gated_silu_kernel<T><<<(numel+255)/256, 256>>>(numel, tgt, src); // TODO adjust 256, TODO float4
}

template <typename T>
__global__ void inplace_gated_silu_interleaved_kernel(int intermediate_size, T* tgt) {
    int row_offset = blockIdx.x * (2 * intermediate_size);
    int col = blockIdx.y * 256 + threadIdx.x;
    int col2 = col + intermediate_size;
    if (col < intermediate_size) {
        float v = TypeTraits<T>::to_float(tgt[row_offset + col]);
        float s = 1.0f / (1.0f + exp(-v));
        tgt[row_offset + col] = TypeTraits<T>::from_float(v * s * TypeTraits<T>::to_float(tgt[row_offset + col2]));
    }
}

template <typename T>
void inplace_gated_silu_interleaved(int num_tokens, int intermediate_size, T* tgt) {
    inplace_gated_silu_interleaved_kernel<T><<<dim3(num_tokens, (intermediate_size+255)/256), 256>>>(intermediate_size, tgt); // TODO adjust 256, TODO float4
}

template <typename T>
struct FFN {
    virtual void init_weight_ptr(Memory* memory) = 0;
    virtual int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) = 0;
    virtual void load_to_storage(std::string name, void* ptr) = 0;
    virtual void prefill(int32_t num_tokens, T* input) = 0;
};

template <typename T>
struct GatedFFN : FFN<T> {
    int hidden_size;
    int intermediate_size;
    float rms_norm_eps;

    RMSNorm<T> *ffn_norm;
    Linear<T, true> *gate_proj, *up_proj;
    Linear<T, false> *down_proj;

    GatedFFN(int hidden_size, int intermediate_size, float rms_norm_eps) {
        this->hidden_size = hidden_size;
        this->intermediate_size = intermediate_size;
        this->rms_norm_eps = rms_norm_eps;

        this->ffn_norm = new RMSNorm<T>(hidden_size, rms_norm_eps);
        this->gate_proj = new Linear<T, true>(hidden_size, intermediate_size);
        this->up_proj = new Linear<T, true>(hidden_size, intermediate_size);
        this->down_proj = new Linear<T, false>(intermediate_size, hidden_size);
    }

    void init_weight_ptr(Memory* memory) {
        this->ffn_norm->init_weight_ptr(memory);
        this->gate_proj->init_weight_ptr(memory);
        this->up_proj->init_weight_ptr(memory);
        this->down_proj->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t ffn_norm_end = this->ffn_norm->init_output_ptr(memory, num_tokens, offset);
        int64_t gate_proj_end = this->gate_proj->init_output_ptr(memory, num_tokens, ffn_norm_end);
        int64_t up_proj_end = this->up_proj->init_output_ptr(memory, num_tokens, gate_proj_end);
        int64_t down_proj_end = this->down_proj->init_output_ptr(memory, num_tokens, offset);
        return up_proj_end;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("gate_proj") != std::string::npos) {
            this->gate_proj->load_to_storage(name, ptr);
        } else if (name.find("up_proj") != std::string::npos) {
            this->up_proj->load_to_storage(name, ptr);
        } else if (name.find("down_proj") != std::string::npos) {
            this->down_proj->load_to_storage(name, ptr);
        } else if (name.find("post_attention_layernorm") != std::string::npos) {
            this->ffn_norm->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(int32_t num_tokens, T* input) {
        this->ffn_norm->prefill(num_tokens, input);
        this->gate_proj->prefill(num_tokens, this->ffn_norm->output);
        this->up_proj->prefill(num_tokens, this->ffn_norm->output);
        inplace_gated_silu<T>(num_tokens * this->intermediate_size, this->gate_proj->output, this->up_proj->output);
        this->down_proj->prefill(num_tokens, this->gate_proj->output);
        inplace_add<T>(num_tokens * this->hidden_size, input, this->down_proj->output);
    }
};

template <typename T>
struct FusedGatedFFN : GatedFFN<T> {
    FusedGatedFFN(int hidden_size, int intermediate_size, float rms_norm_eps) : GatedFFN<T>(hidden_size, intermediate_size, rms_norm_eps) {}

    void prefill(int32_t num_tokens, T* input) {
        this->ffn_norm->prefill(num_tokens, input);
        // this->gate_proj->prefill(num_tokens, this->ffn_norm->output);
        // this->up_proj->prefill(num_tokens, this->ffn_norm->output);
        linear<T, true>(num_tokens, this->hidden_size, this->intermediate_size*2, this->ffn_norm->output, this->gate_proj->weight, this->gate_proj->output);
        inplace_gated_silu_interleaved<T>(num_tokens, this->intermediate_size, this->gate_proj->output);
        linear<T, false, /*inplace=*/true>(num_tokens, this->intermediate_size, this->hidden_size, this->gate_proj->output, this->down_proj->weight, input);
    }
};