#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

#include "utils.cuh"
#include "trait.cuh"
#include "model/model.cuh"
#include "model/medusa.cuh"
#include "model/eagle.cuh"
#include "model/spec_model.cuh"
#include "model/w8a8/w8a8_model.cuh"
#include "model/w8a8/medusa_base_w8a8.cuh"
#include "model/w8a8/eagle_base_w8a8.cuh"
#include "model/w4a8_per_chn/w4a8_per_chn_model.cuh"
#include "model/w4a8_per_chn/medusa_base_w4a8_per_chn.cuh"
#include "model/w4a8_per_chn/eagle_base_w4a8_per_chn.cuh"
#include "model/w4a8_per_chn/spec_w4a8_per_chn_model.cuh"
#include "model/w4a8_per_chn/w4a16_gptq_marlin_spec_w4a8_per_chn_model.cuh"
#include "model/w4a16_marlin/w4a16_marlin_model.cuh"
#include "model/w4a16_marlin/medusa_base_w4a16_marlin.cuh"
#include "model/w4a16_marlin/eagle_base_w4a16_marlin.cuh"
#include "model/w4a16_gptq_marlin/w4a16_gptq_marlin_model.cuh"
#include "model/w4a16_gptq_marlin/medusa_base_w4a16_gptq_marlin.cuh"
#include "model/w4a16_gptq_marlin/eagle_base_w4a16_gptq_marlin.cuh"
#include "model/w4a16_gptq_marlin/spec_w4a16_gptq_marlin_model.cuh"
#include "model/w4a8_per_group/w4a8_per_group_model.cuh"
#include "model/w4a8_qqq/w4a8_qqq_model.cuh"


#define DTYPE_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND == 0) {                              \
      using elem_type = __half;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = __nv_bfloat16; \
      return __VA_ARGS__();                  \
    }                                        \
  }()

Model* model;

void init_base_model(
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
    int torch_dtype,
    int chunk_length
) {
    init_resources();

    DTYPE_SWITCH(torch_dtype, [&] {
        model = new ModelImpl<elem_type>(
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
            chunk_length
        );
    });

}

void init_medusa_model(
    int num_heads,
    int num_layers,
    int topk_per_head,
    int tree_size,
    std::uintptr_t tree_indices,
    std::uintptr_t draft_position_ids,
    int torch_dtype
) {
    DTYPE_SWITCH(torch_dtype, [&] {
        model = new MedusaImpl<elem_type>(
            (ModelImpl<elem_type>*)model,
            num_heads,
            num_layers,
            topk_per_head,
            tree_size,
            reinterpret_cast<int32_t*>(tree_indices),
            reinterpret_cast<int32_t*>(draft_position_ids)
        );
    });
}

void init_eagle_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    DTYPE_SWITCH(torch_dtype, [&] {
        model = new EagleImpl<elem_type>(
            (ModelImpl<elem_type>*)model,
            num_layers,
            num_iter,
            topk_per_iter,
            tree_size
        );
    });
}

void init_spec_model(
    int draft_vocab_size,
    int draft_num_hidden_layers,
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    int draft_head_dim,
    float draft_rms_norm_eps,
    int num_iter,
    bool draft_cuda_graph,
    int torch_dtype
) {
    // TODO: different type 
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for spec model");
    }
    // DTYPE_SWITCH(torch_dtype, [&] {
    model = new SpecModelImpl<half>(
        (ModelImpl<half>*)model,
        draft_vocab_size,
        draft_num_hidden_layers,
        draft_hidden_size,
        draft_intermediate_size,
        draft_num_attention_heads,
        draft_num_key_value_heads,
        draft_head_dim,
        draft_rms_norm_eps,
        num_iter, 
        draft_cuda_graph
    );
    // });
}

void init_w8a8_base_model(
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
    int torch_dtype,
    int chunk_length
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    init_resources();

    model = new W8A8ModelImpl<half>(
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
        chunk_length
    );

}

void init_medusa_w8a8_model(
    int num_heads,
    int num_layers,
    int topk_per_head,
    int tree_size,
    std::uintptr_t tree_indices,
    std::uintptr_t draft_position_ids,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new MedusaImplBaseW8A8<half>(
        (W8A8ModelImpl<half>*)model,
        num_heads,
        num_layers,
        topk_per_head,
        tree_size,
        reinterpret_cast<int32_t*>(tree_indices),
        reinterpret_cast<int32_t*>(draft_position_ids)
    );
}

void init_eagle_w8a8_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new W8A8EagleImpl<half>(
        (W8A8ModelImpl<half>*)model,
        num_layers,
        num_iter,
        topk_per_iter,
        tree_size
    );
}

void init_w4a8_per_chn_base_model(
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
    int torch_dtype,
    int chunk_length
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    init_resources();

    model = new W4A8PerChnModelImpl<half>(
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
        chunk_length
    );
}

void init_medusa_w4a8_per_chn_model(
    int num_heads,
    int num_layers,
    int topk_per_head,
    int tree_size,
    std::uintptr_t tree_indices,
    std::uintptr_t draft_position_ids,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new MedusaImplBaseW4A8PerChn<half>(
        (W4A8PerChnModelImpl<half>*)model,
        num_heads,
        num_layers,
        topk_per_head,
        tree_size,
        reinterpret_cast<int32_t*>(tree_indices),
        reinterpret_cast<int32_t*>(draft_position_ids)
    );
}

void init_eagle_w4a8_per_chn_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new W4A8PerChnEagleImpl<half>(
        (W4A8PerChnModelImpl<half>*)model,
        num_layers,
        num_iter,
        topk_per_iter,
        tree_size
    );
}

void init_spec_w4a8_per_chn_model(
    int draft_vocab_size,
    int draft_num_hidden_layers,
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    int draft_head_dim,
    float draft_rms_norm_eps,
    int num_iter,
    bool draft_cuda_graph,
    int torch_dtype
) {
    // TODO: different type 
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for spec model");
    }
    // DTYPE_SWITCH(torch_dtype, [&] {
    model = new SpecW4A8PerChnModelImpl<half>(
        (W4A8PerChnModelImpl<half>*)model,
        draft_vocab_size,
        draft_num_hidden_layers,
        draft_hidden_size,
        draft_intermediate_size,
        draft_num_attention_heads,
        draft_num_key_value_heads,
        draft_head_dim,
        draft_rms_norm_eps,
        num_iter, 
        draft_cuda_graph
    );
    // });
}

void init_w4a16_gptq_marlin_spec_w4a8_per_chn_model(
    int draft_vocab_size,
    int draft_num_hidden_layers,
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    int draft_head_dim,
    float draft_rms_norm_eps,
    int num_iter,
    bool draft_cuda_graph,
    int torch_dtype
) {
    // TODO: different type 
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for spec model");
    }
    // DTYPE_SWITCH(torch_dtype, [&] {
    model = new W4A16GMSpecW4A8PCModelImpl<half>(
        (W4A8PerChnModelImpl<half>*)model,
        draft_vocab_size,
        draft_num_hidden_layers,
        draft_hidden_size,
        draft_intermediate_size,
        draft_num_attention_heads,
        draft_num_key_value_heads,
        draft_head_dim,
        draft_rms_norm_eps,
        num_iter, 
        draft_cuda_graph
    );
    // });
}

void init_w4a16_marlin_base_model(
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
    int torch_dtype,
    int chunk_length,
    int group_size,
    int bits,
    bool use_marlin
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    init_resources();

    model = new W4A16MarlinModelImpl<half>(
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
        chunk_length,
        group_size,
        bits,
        use_marlin
    );

}

void init_medusa_w4a16_marlin_model(
    int num_heads,
    int num_layers,
    int topk_per_head,
    int tree_size,
    std::uintptr_t tree_indices,
    std::uintptr_t draft_position_ids,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new MedusaImplBaseW4A16Marlin<half>(
        (W4A16MarlinModelImpl<half>*)model,
        num_heads,
        num_layers,
        topk_per_head,
        tree_size,
        reinterpret_cast<int32_t*>(tree_indices),
        reinterpret_cast<int32_t*>(draft_position_ids)
    );
}

void init_eagle_w4a16_marlin_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new EagleImplBaseW4A16Marlin<half>(
        (W4A16MarlinModelImpl<half>*)model,
        num_layers,
        num_iter,
        topk_per_iter,
        tree_size
    );
}


void init_w4a16_gptq_marlin_base_model(
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
    int torch_dtype,
    int chunk_length
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    init_resources();

    model = new W4A16GPTQMarlinModelImpl<half>(
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
        chunk_length
    );

}

void init_medusa_w4a16_gptq_marlin_model(
    int num_heads,
    int num_layers,
    int topk_per_head,
    int tree_size,
    std::uintptr_t tree_indices,
    std::uintptr_t draft_position_ids,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new MedusaImplBaseW4A16GPTQMarlin<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        num_heads,
        num_layers,
        topk_per_head,
        tree_size,
        reinterpret_cast<int32_t*>(tree_indices),
        reinterpret_cast<int32_t*>(draft_position_ids)
    );
}

void init_eagle_w4a16_gptq_marlin_model(
    int num_layers,
    int num_iter,
    int topk_per_iter,
    int tree_size,
    int torch_dtype
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    model = new EagleImplBaseW4A16GPTQMarlin<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        num_layers,
        num_iter,
        topk_per_iter,
        tree_size
    );
}

void init_spec_w4a16_gptq_marlin_model(
    int draft_vocab_size,
    int draft_num_hidden_layers,
    int draft_hidden_size,
    int draft_intermediate_size,
    int draft_num_attention_heads,
    int draft_num_key_value_heads,
    int draft_head_dim,
    float draft_rms_norm_eps,
    int num_iter,
    bool draft_cuda_graph,
    int torch_dtype
) {
    // TODO: different type 
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for spec model");
    }
    // DTYPE_SWITCH(torch_dtype, [&] {
    model = new SpecW4A16GPTQMarlinModelImpl<half>(
        (W4A16GPTQMarlinModelImpl<half>*)model,
        draft_vocab_size,
        draft_num_hidden_layers,
        draft_hidden_size,
        draft_intermediate_size,
        draft_num_attention_heads,
        draft_num_key_value_heads,
        draft_head_dim,
        draft_rms_norm_eps,
        num_iter, 
        draft_cuda_graph
    );
    // });
}


void init_w4a8_per_group_base_model(
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
    int torch_dtype,
    int chunk_length
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    init_resources();

    model = new W4A8PerGroupModelImpl<half>(
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
        chunk_length
    );
}


void init_w4a8_qqq_base_model(
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
    int torch_dtype,
    int chunk_length
) {
    if (torch_dtype != 0) {
        throw std::invalid_argument("Only half precision is supported for W8A8 model");
    }
    init_resources();

    model = new W4A8QQQModelImpl<half>(
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
        chunk_length
    );

}

int init_storage() {
    return model->init_storage();
}

void load_model(std::string name, std::uintptr_t param) {
    model->load_to_storage(name, reinterpret_cast<void*>(param));
}

void prefill(int input_length, int history_length, std::uintptr_t input, std::uintptr_t position_ids, std::uintptr_t output) {
    model->prefill(input_length, history_length, reinterpret_cast<int32_t*>(input), reinterpret_cast<int32_t*>(position_ids), (void*)(output));
}

void decode(int input_length, int padded_length, std::uintptr_t input, std::uintptr_t position_ids, std::uintptr_t cache_length, std::uintptr_t mask_2d, std::uintptr_t output, bool cuda_graph) {
    if (cuda_graph) {
        if (graphCreated_padding_length != padded_length || graphCreated_input_length != input_length) {
            if (graphExec != nullptr) {
                cudaGraphExecDestroy(graphExec);
                graphExec = nullptr;
            }
            if (graph != nullptr) {
                cudaGraphDestroy(graph);
                graph = nullptr;
            }
            cudaStreamBeginCapture(calc_stream.stream, cudaStreamCaptureModeGlobal);
            model->decode(input_length, padded_length, reinterpret_cast<int32_t*>(input), reinterpret_cast<int32_t*>(position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(mask_2d), reinterpret_cast<void*>(output));
            cudaStreamEndCapture(calc_stream.stream, &graph);
            cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
            graphCreated_padding_length = padded_length;
            graphCreated_input_length = input_length;
        }
        cudaGraphLaunch(graphExec, calc_stream.stream);
    } else {
        model->decode(input_length, padded_length, reinterpret_cast<int32_t*>(input), reinterpret_cast<int32_t*>(position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(mask_2d), reinterpret_cast<void*>(output));
    }
}

void draft(std::uintptr_t tree_draft_ids, std::uintptr_t tree_position_ids, std::uintptr_t cache_length, std::uintptr_t attn_mask, std::uintptr_t tree_parent) {
    model->draft(reinterpret_cast<int32_t*>(tree_draft_ids), reinterpret_cast<int32_t*>(tree_position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(attn_mask), reinterpret_cast<int32_t*>(tree_parent));
}

int verify_and_fix(int num_tokens, std::uintptr_t pred, std::uintptr_t gt, std::uintptr_t position_ids, std::uintptr_t cache_length, std::uintptr_t attn_mask, std::uintptr_t tree_parent) {
    return model->verify(num_tokens, reinterpret_cast<int32_t*>(pred), reinterpret_cast<int32_t*>(gt), reinterpret_cast<int32_t*>(position_ids), reinterpret_cast<int32_t*>(cache_length), reinterpret_cast<uint64_t*>(attn_mask), reinterpret_cast<int32_t*>(tree_parent));
}

PYBIND11_MODULE(C, m) {
    m.def("init_base_model", &init_base_model, "Init base model");
    m.def("init_medusa_model", &init_medusa_model, "Init medusa model");
    m.def("init_eagle_model", &init_eagle_model, "Init eagle model");
    m.def("init_spec_model", &init_spec_model, "Init spec model");
    m.def("init_w8a8_base_model", &init_w8a8_base_model, "Init W8A8 base model");
    m.def("init_medusa_w8a8_model", &init_medusa_w8a8_model, "Init medusa W8A8 model");
    m.def("init_eagle_w8a8_model", &init_eagle_w8a8_model, "Init eagle W8A8 model");
    m.def("init_w4a8_per_chn_base_model", &init_w4a8_per_chn_base_model, "Init W4A8 per channel base model");
    m.def("init_medusa_w4a8_per_chn_model", &init_medusa_w4a8_per_chn_model, "Init medusa W4A8 per channel model");
    m.def("init_eagle_w4a8_per_chn_model", &init_eagle_w4a8_per_chn_model, "Init eagle W4A8 per channel model");
    m.def("init_spec_w4a8_per_chn_model", &init_spec_w4a8_per_chn_model, "init spec W4A8 per channel model");
    //init_w4a16_gptq_marlin_spec_w4a8_per_chn_model
    m.def("init_w4a16_gptq_marlin_spec_w4a8_per_chn_model", &init_w4a16_gptq_marlin_spec_w4a8_per_chn_model, "init w4a16 spec W4A8 per channel model");
    m.def("init_w4a16_marlin_base_model", &init_w4a16_marlin_base_model, "Init W4A16 base model");
    m.def("init_medusa_w4a16_marlin_model", &init_medusa_w4a16_marlin_model, "Init medusa W4A16 model");
    m.def("init_eagle_w4a16_marlin_model", &init_eagle_w4a16_marlin_model, "Init eagle W4A16 model");
    m.def("init_w4a16_gptq_marlin_base_model", &init_w4a16_gptq_marlin_base_model, "Init W4A16 base model");
    m.def("init_medusa_w4a16_gptq_marlin_model", &init_medusa_w4a16_gptq_marlin_model, "Init medusa W4A16 model");
    m.def("init_eagle_w4a16_gptq_marlin_model", &init_eagle_w4a16_gptq_marlin_model, "Init eagle W4A16 model");
    m.def("init_spec_w4a16_gptq_marlin_model", &init_spec_w4a16_gptq_marlin_model, "Init spec W4A16 model");
    m.def("init_w4a8_per_group_base_model", &init_w4a8_per_group_base_model, "Init W4A8 per group base model");
    m.def("init_w4a8_qqq_base_model", &init_w4a8_qqq_base_model, "Init W4A8 per group base model");
    m.def("init_storage", &init_storage, "Init storage");
    m.def("load_model", &load_model, "Load model");
    m.def("prefill", &prefill, "Prefill");
    m.def("decode", &decode, "Decode");
    m.def("draft", &draft, "Draft");
    m.def("verify_and_fix", &verify_and_fix, "Verify and fix");
} 