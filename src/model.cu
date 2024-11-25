#include <cuda_runtime.h>
#include <torch/extension.h>

#include "trait.cuh"
#include "model/model.cuh"

Model* model;

void init_model(
    int64_t memory_limit,
    torch::Tensor memory_pool,
    int vocab_size,
    int num_hidden_layers,
    int hidden_size,
    int intermediate_size,
    int num_attention_heads,
    int num_key_value_heads,
    float rms_norm_eps,
    float rope_theta,
    torch::ScalarType torch_dtype
) {
    if (torch_dtype == torch::ScalarType::Half)
        model = new ModelImpl<__half>();
    else if (torch_dtype == torch::ScalarType::BFloat16)
        model = new ModelImpl<__nv_bfloat16>();
    else if (torch_dtype == torch::ScalarType::Float)
        model = new ModelImpl<float>();
    else
        throw std::invalid_argument("Unsupported dtype");
            
    model->init(
        memory_limit,
        (void*)memory_pool.data_ptr(),
        vocab_size,
        num_hidden_layers,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_key_value_heads,
        rms_norm_eps,
        rope_theta
    );
    model->init_storage();
}

void load_model(std::string name, torch::Tensor param) {
    void* ptr = param.data_ptr();
    model->load_to_storage(name, ptr);
}

void generate(torch::Tensor input, torch::Tensor output) {
    model->prefill(input.numel(), (int32_t*)input.data_ptr(), (int32_t*)output.data_ptr());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_model", &init_model, "Init model");
    m.def("load_model", &load_model, "Load model");
    m.def("generate", &generate, "Generate");
} 