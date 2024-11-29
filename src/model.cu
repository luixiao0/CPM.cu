#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

#include "utils.cuh"
#include "trait.cuh"
#include "model/model.cuh"

Model* model;

void init_model(
    int64_t memory_limit,
    std::uintptr_t memory_pool,
    int vocab_size,
    int num_hidden_layers,
    int hidden_size,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    int head_dim,
    float rms_norm_eps,
    float rope_theta,
    int torch_dtype,
    int chunk_length
) {
    init_cublas();

    if (torch_dtype == 0) {
        std::cout << "Using float16 precision" << std::endl;
        model = new ModelImpl<__half>(
            memory_limit,
            reinterpret_cast<void*>(memory_pool),
            vocab_size,
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            chunk_length
        );
    } else if (torch_dtype == 1) {
        std::cout << "Using bfloat16 precision" << std::endl;
        model = new ModelImpl<__nv_bfloat16>(
            memory_limit,
            reinterpret_cast<void*>(memory_pool),
            vocab_size,
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            chunk_length
        );
    } else if (torch_dtype == 2) {
        std::cout << "Using float32 precision" << std::endl;
        model = new ModelImpl<float>(
            memory_limit,
            reinterpret_cast<void*>(memory_pool),
            vocab_size,
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rope_theta,
            chunk_length
        );
    } else {
        throw std::invalid_argument("Unsupported dtype");
    }

    model->init_storage();
}

void load_model(std::string name, std::uintptr_t param) {
    model->load_to_storage(name, reinterpret_cast<void*>(param));
}

void generate(int input_length, int chunk_length, int output_length, std::uintptr_t input, std::uintptr_t position_ids, std::uintptr_t output) {
    model->prefill(input_length, reinterpret_cast<int32_t*>(input), reinterpret_cast<int32_t*>(position_ids), reinterpret_cast<int32_t*>(output));
}

PYBIND11_MODULE(C, m) {
    m.def("init_model", &init_model, "Init model");
    m.def("load_model", &load_model, "Load model");
    m.def("generate", &generate, "Generate");
} 