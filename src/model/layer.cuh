#pragma once
#include "norm.cuh"
#include "linear.cuh"
#include <cuda_runtime.h>

template <typename T>
__global__ void inplace_add_kernel(int numel, T* tgt, T* src) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        tgt[idx] += src[idx];
    }
}

template <typename T>
void inplace_add(int numel, T* tgt, T* src) {
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
struct Layer {
    Linear<T> *q_proj, *k_proj, *v_proj, *o_proj, *gate_proj, *up_proj, *down_proj;
    RMSNorm<T> *attn_norm, *ffn_norm;
    virtual void init_weight_ptr(Memory* memory) = 0;
    virtual int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) = 0;
    virtual void load_to_storage(std::string name, void* ptr) = 0;
    virtual void prefill(int32_t num_tokens, T* input) = 0;
};

template <typename T>
struct LayerImpl : Layer<T> {
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    float rms_norm_eps;
    float rope_theta;

    LayerImpl(int hidden_size, int intermediate_size, int num_attention_heads, int num_key_value_heads, int head_dim, float rms_norm_eps, float rope_theta) {
        this->hidden_size = hidden_size;
        this->intermediate_size = intermediate_size;
        this->num_attention_heads = num_attention_heads;
        this->num_key_value_heads = num_key_value_heads;
        this->head_dim = head_dim;
        this->rms_norm_eps = rms_norm_eps;
        this->rope_theta = rope_theta;

        this->attn_norm = new RMSNormImpl<T>(hidden_size, rms_norm_eps);
        this->ffn_norm = new RMSNormImpl<T>(hidden_size, rms_norm_eps);
        this->q_proj = new LinearImpl<T, true>(hidden_size, num_attention_heads * head_dim);
        this->k_proj = new LinearImpl<T, true>(hidden_size, num_key_value_heads * head_dim);
        this->v_proj = new LinearImpl<T, true>(hidden_size, num_key_value_heads * head_dim);
        this->o_proj = new LinearImpl<T, false>(hidden_size, num_attention_heads * head_dim);
        this->gate_proj = new LinearImpl<T, true>(hidden_size, intermediate_size);
        this->up_proj = new LinearImpl<T, true>(hidden_size, intermediate_size);
        this->down_proj = new LinearImpl<T, false>(hidden_size, intermediate_size);
    }

    void init_weight_ptr(Memory* memory) {
        this->attn_norm->init_weight_ptr(memory);
        this->q_proj->init_weight_ptr(memory);
        this->k_proj->init_weight_ptr(memory);
        this->v_proj->init_weight_ptr(memory);
        this->o_proj->init_weight_ptr(memory);
        this->ffn_norm->init_weight_ptr(memory);
        this->gate_proj->init_weight_ptr(memory);
        this->up_proj->init_weight_ptr(memory);
        this->down_proj->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t attn_norm_end = this->attn_norm->init_output_ptr(memory, num_tokens, offset);
        int64_t q_proj_end = this->q_proj->init_output_ptr(memory, num_tokens, attn_norm_end);
        int64_t k_proj_end = this->k_proj->init_output_ptr(memory, num_tokens, q_proj_end);
        int64_t v_proj_end = this->v_proj->init_output_ptr(memory, num_tokens, k_proj_end);
        int64_t o_proj_end = this->o_proj->init_output_ptr(memory, num_tokens, offset);

        int64_t ffn_norm_end = this->ffn_norm->init_output_ptr(memory, num_tokens, offset);
        int64_t gate_proj_end = this->gate_proj->init_output_ptr(memory, num_tokens, ffn_norm_end);
        int64_t up_proj_end = this->up_proj->init_output_ptr(memory, num_tokens, gate_proj_end);
        int64_t down_proj_end = this->down_proj->init_output_ptr(memory, num_tokens, offset);
        return up_proj_end;
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.find("q_proj") != std::string::npos) {
            this->q_proj->load_to_storage(name, ptr);
        } else if (name.find("k_proj") != std::string::npos) {
            this->k_proj->load_to_storage(name, ptr);
        } else if (name.find("v_proj") != std::string::npos) {
            this->v_proj->load_to_storage(name, ptr);
        } else if (name.find("o_proj") != std::string::npos) {
            this->o_proj->load_to_storage(name, ptr);
        } else if (name.find("gate_proj") != std::string::npos) {
            this->gate_proj->load_to_storage(name, ptr);
        } else if (name.find("up_proj") != std::string::npos) {
            this->up_proj->load_to_storage(name, ptr);
        } else if (name.find("down_proj") != std::string::npos) {
            this->down_proj->load_to_storage(name, ptr);
        } else if (name.find("input_layernorm") != std::string::npos) {
            this->attn_norm->load_to_storage(name, ptr);
        } else if (name.find("post_attention_layernorm") != std::string::npos) {
            this->ffn_norm->load_to_storage(name, ptr);
        } else if (name.find("rotary_emb") != std::string::npos) {
            // rotary_emb->load_to_storage(name, ptr);
        } else {
            throw std::invalid_argument("Unsupported name " + name);
        }
    }

    void prefill(int32_t num_tokens, T* input) {
        this->attn_norm->prefill(num_tokens, input);
        // this->q_proj->prefill(num_tokens, this->attn_norm->output);
        // this->k_proj->prefill(num_tokens, this->attn_norm->output);
        // this->v_proj->prefill(num_tokens, this->attn_norm->output);
        this->ffn_norm->prefill(num_tokens, input);
        // linear<T, true>(num_tokens, hidden_size, 2*intermediate_size, this->ffn_norm->output, this->gate_proj->weight, this->gate_proj->output);
        this->gate_proj->prefill(num_tokens, this->ffn_norm->output);
        this->up_proj->prefill(num_tokens, this->ffn_norm->output); // TODO merge gate & up by directly calling linear_kernel
        inplace_gated_silu<T>(num_tokens*intermediate_size, this->gate_proj->output, this->up_proj->output);
        // this->down_proj->prefill(num_tokens, this->gate_proj->output);
        // inplace_add<T>(num_tokens*hidden_size, input, this->down_proj->output); // TODO merge down & add by directly calling linear_kernel
    }
};