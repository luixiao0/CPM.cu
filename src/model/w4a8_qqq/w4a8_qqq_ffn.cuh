#pragma once
#include "../norm.cuh"
#include "../activation.cuh"
#include "quant_sfloat.cuh"
#include "w4a8_qqq_linear.cuh"
#include <cuda_runtime.h>


template <typename T>
struct W4A8QQQFFN {
    int hidden_size;
    int intermediate_size;
    float rms_norm_eps;

    Norm<T> *ffn_norm;
    
    QuantizerScalefloat<T> * gate_up_quantizer;
    W4A8QQQLinear<T> *gate_proj, *up_proj;
    QuantizerScalefloat<T> * down_quantizer;
    W4A8QQQLinear<T> *down_proj;

    T* output;
    T* gated_up;

    W4A8QQQFFN(int hidden_size, int intermediate_size, float rms_norm_eps) {
        this->hidden_size = hidden_size;
        this->intermediate_size = intermediate_size;
        this->rms_norm_eps = rms_norm_eps;

        this->ffn_norm = new RMSNorm<T>(hidden_size, rms_norm_eps);
        this->gate_up_quantizer = new QuantizerScalefloat<T>(hidden_size);
        this->gate_proj = new W4A8QQQLinear<T>(hidden_size, intermediate_size);
        this->up_proj = new W4A8QQQLinear<T>(hidden_size, intermediate_size);
        this->down_quantizer = new QuantizerScalefloat<T>(intermediate_size);
        this->down_proj = new W4A8QQQLinear<T>(intermediate_size, hidden_size);
    }

    void init_weight_ptr(Memory* memory) {
        this->ffn_norm->init_weight_ptr(memory);
        this->gate_proj->init_weight_ptr(memory);
        this->up_proj->init_weight_ptr(memory);
        this->down_proj->init_weight_ptr(memory);

        this->gate_proj->init_scale_ptr(memory);
        this->up_proj->init_scale_ptr(memory);
        this->down_proj->init_scale_ptr(memory);
        
        this->gate_proj->init_workspace_ptr(memory);
        this->up_proj->init_workspace_ptr(memory);
        this->down_proj->init_workspace_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t ffn_norm_end = this->ffn_norm->init_output_ptr(memory, num_tokens, offset);
        int64_t gate_proj_quant_end = this->gate_up_quantizer->init_output_ptr(memory, num_tokens, ffn_norm_end);
        int64_t gate_proj_end = this->gate_proj->init_output_ptr(memory, num_tokens, gate_proj_quant_end);
        int64_t up_proj_end = this->up_proj->init_output_ptr(memory, num_tokens, gate_proj_end);

        int64_t gate_proj_tmp_end = this->gate_proj->init_tmp_ptr(memory, num_tokens, up_proj_end);
        int64_t up_proj_tmp_end = this->up_proj->init_tmp_ptr(memory, num_tokens, gate_proj_tmp_end);

        int64_t gated_up_end = memory->allocate((void**)&this->gated_up, up_proj_tmp_end, num_tokens * intermediate_size * sizeof(T));

        int64_t down_proj_quant_end = this->down_quantizer->init_output_ptr(memory, num_tokens, gated_up_end);
        int64_t down_proj_end = this->down_proj->init_output_ptr(memory, num_tokens, down_proj_quant_end);
        this->output = this->down_proj->output;

        int64_t down_proj_tmp_end = this->down_proj->init_tmp_ptr(memory, num_tokens, down_proj_end);
        return down_proj_tmp_end;
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

    void prefill(const Stream& stream, int32_t num_tokens, T* input, T* prev_output) {
        this->ffn_norm->prefill(stream, num_tokens, input, prev_output);

        this->gate_up_quantizer->invoke(stream, this->ffn_norm->output, num_tokens);
        this->gate_proj->prefill(stream, num_tokens, this->gate_up_quantizer->output, this->gate_up_quantizer->output_scale);
        this->up_proj->prefill(stream, num_tokens, this->gate_up_quantizer->output, this->gate_up_quantizer->output_scale);

        // marlin_qqq_gemm(
        //     this->gate_up_quantizer->output,
        //     this->gate_proj->B,
        //     this->gate_up_quantizer->output_scale,
        //     this->gate_proj->s_channel,
        //     this->gate_proj->s_group,
        //     this->gate_proj->workspace, num_tokens,
        //     this->intermediate_size*2, this->hidden_size,
        //     this->gate_proj->group_size,
        //     this->gate_proj->c_tmp,
        //     this->gate_proj->output,
        //     stream.stream
        // );

        // gated_silu_interleaved<T>(stream, num_tokens, this->intermediate_size, this->gate_proj->output, this->gated_up);
        gated_silu<T>(stream, num_tokens, this->intermediate_size, this->gate_proj->output, this->up_proj->output);
        this->down_quantizer->invoke(stream, this->up_proj->output, num_tokens);

        // this->down_quantizer->invoke(stream, this->gated_up, num_tokens);
        this->down_proj->prefill(stream, num_tokens, this->down_quantizer->output, this->down_quantizer->output_scale);

    }

    void decode(const Stream& stream, int32_t num_tokens, T* input, T* prev_output) {
        prefill(stream, num_tokens, input, prev_output);
    }
};
