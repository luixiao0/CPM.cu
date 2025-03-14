#pragma once
#include "w4a16_gptq_marlin_model.cuh"
#include "../eagle.cuh"
#include "../drafter.cuh"
#include "w4a16_gptq_marlin_layer.cuh"


template <typename T>
struct CascadeEagleSpecW4A16GPTQMarlinModelImplSep: Model {

    // eagle
    int ea_num_layers;
    int ea_num_iter;
    int ea_topk_per_iter;
    int ea_tree_size;
    int ea_total_tried;

    KVCacheManager<T>* ea_kv_caches;
    std::vector<Layer<T>*> ea_layers;
    Linear<T, true, true> *ea_fc1;
    Linear<T> *ea_fc2;
    functions::TopK<T>* ea_topk_func;
    functions::TopK<T>* ea_topk_func_2;

    T *ea_prev_hidden_state, *ea_prev_embed;
    int ea_num_prev, ea_num_history_tokens;
    int32_t *eagle_position_ids, *eagle_cache_length;
    int *eagle_original_length, eagle_padded_length;
    uint64_t *eagle_mask_2d, *ea_tmp_mask_2d;
    T* eagle_logits;
    T* ea_tired_history_val; int32_t* ea_tired_history_pos;
    int32_t* ea_tired_history_parent;
    bool ea_is_first_draft;
    

    int32_t *ea_h_best, *ea_d_best;    

    T* ea_tmp_kvcache;

    int32_t* ea_tree_draft_ids, *ea_tree_position_ids, *ea_tree_cache_length, *ea_tree_parent;
    // TODO: not need in sep version
    // uint64_t* ea_tree_attn_mask;

    // draft & target

    W4A16GPTQMarlinModelImpl<T>* draft_model;
    W4A16GPTQMarlinModelImpl<T>* model;

    // draft args
    int32_t *draft_input;
    int32_t *draft_position_ids, *draft_cache_length;
    int * host_draft_cache_length;
    int draft_padded_length; 
    T* draft_logits;
    bool is_first_draft;
    functions::TopK<T>* topk_func;
    int32_t *draft_tmp;
    int32_t *h_best, *d_best;    
    int num_prev, num_history_tokens;

    // draft mask always nullptr
    uint64_t* draft_mask_2d;   

    // graph
    bool draft_cuda_graph;
    int draft_graphCreated_padding_length;
    int draft_graphCreated_input_length;
    cudaGraph_t draft_graph;
    cudaGraphExec_t draft_graphExec;

    // cascade vars
    int cur_draft_length;
    int min_draft_length;
    T * draft_tmp_hidden_state;

    // debug args
    int forward_times;

    CascadeEagleSpecW4A16GPTQMarlinModelImplSep(
        W4A16GPTQMarlinModelImpl<T>* model,
        int draft_vocab_size,
        int draft_num_hidden_layers,
        int draft_hidden_size,
        int draft_intermediate_size,
        int draft_num_attention_heads,
        int draft_num_key_value_heads,
        int draft_head_dim,
        float draft_rms_norm_eps,
        // int num_iter,
        int min_draft_length,
        bool draft_cuda_graph,
        // eagle args
        int ea_num_layers,
        int ea_num_iter,
        int ea_topk_per_iter,
        int ea_tree_size
    ) {
        this->model = model;
        this->draft_model = new W4A16GPTQMarlinModelImpl<T>(
            0,
            nullptr,
            draft_vocab_size,
            draft_num_hidden_layers,
            draft_hidden_size,
            draft_intermediate_size,
            draft_num_attention_heads,
            draft_num_key_value_heads,
            draft_head_dim,
            draft_rms_norm_eps,
            this->model->chunk_length
        );

        // draft config
        this->draft_mask_2d = 0;
        topk_func = new functions::TopK<T>(model->vocab_size, 1); // greedy sample
        
        this->draft_cuda_graph = draft_cuda_graph;
        this->draft_graphCreated_padding_length = -1;
        this->draft_graphCreated_input_length = -1;
        this->draft_graph = nullptr;
        this->draft_graphExec = nullptr;
        
        this->min_draft_length = min_draft_length;

        // eagle config
        this->ea_num_layers = ea_num_layers;
        this->ea_num_iter = ea_num_iter;
        this->ea_topk_per_iter = ea_topk_per_iter;
        this->ea_tree_size = ea_tree_size;
        this->ea_total_tried = ea_topk_per_iter * ea_topk_per_iter * (ea_num_iter-1) +  ea_topk_per_iter;

        // ea model
        ea_kv_caches = new KVCacheManager<T>(ea_num_layers, this->draft_model->num_key_value_heads, this->draft_model->head_dim);
        ea_fc1 = new Linear<T, true, true>(this->draft_model->hidden_size, this->draft_model->hidden_size);
        ea_fc2 = new Linear<T>(this->draft_model->hidden_size, this->draft_model->hidden_size);
        for (int i = 0; i < ea_num_layers; i++) {
            ea_layers.push_back(new Layer<T>(this->draft_model->hidden_size, this->draft_model->intermediate_size, this->draft_model->num_attention_heads, this->draft_model->num_key_value_heads, this->draft_model->head_dim, this->draft_model->rms_norm_eps));
        }

        ea_topk_func = new functions::TopK<T>(this->draft_model->vocab_size, ea_topk_per_iter);
        ea_topk_func_2 = new functions::TopK<T>(ea_total_tried, this->ea_tree_size-1); // TODO current topk do not support k > 32

        // debug args
        this->forward_times = 0;
        
    }

    void init_weight_ptr(Memory* memory) {
        ea_fc1->init_weight_ptr(memory);
        ea_fc2->init_weight_ptr(memory);
        for (int i = 0; i < ea_num_layers; i++) {
            ea_layers[i]->init_weight_ptr(memory);
        }
        ea_layers[0]->attn->attn_norm = new Skip<T>(this->draft_model->hidden_size);
        ea_kv_caches->rotary_embedding = this->draft_model->kv_caches->rotary_embedding;
    }
    

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        // init eagle output
        offset = ea_fc1->init_output_ptr(memory, num_tokens, offset);
        offset = ea_fc2->init_output_ptr(memory, num_tokens, offset);
        int64_t layer_end = 0;
        for (int i = 0; i < ea_num_layers; i++) {
            layer_end = ea_layers[i]->init_output_ptr(memory, num_tokens, offset);
        }
        offset = layer_end;
        offset = memory->allocate((void**)&eagle_logits, offset, this->ea_topk_per_iter * this->draft_model->vocab_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_mask_2d, offset, this->ea_topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&ea_tmp_mask_2d, offset, this->ea_topk_per_iter * sizeof(uint64_t));
        offset = memory->allocate((void**)&ea_tired_history_val, offset, this->ea_total_tried * sizeof(T));
        offset = memory->allocate((void**)&ea_tired_history_pos, offset, this->ea_total_tried * sizeof(int32_t));
        offset = memory->allocate((void**)&ea_tired_history_parent, offset, this->ea_topk_per_iter * (this->ea_num_iter - 1) * sizeof(int32_t));
        cudaMallocHost(&eagle_original_length, sizeof(int32_t));

        offset = ea_topk_func->init_output_ptr(memory, this->ea_topk_per_iter, offset);
        offset = ea_topk_func_2->init_output_ptr(memory, 1, offset);

        offset = memory->allocate((void**)&ea_prev_hidden_state, offset, num_tokens * this->draft_model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&ea_prev_embed, offset, num_tokens * this->draft_model->hidden_size * sizeof(T));
        offset = memory->allocate((void**)&eagle_position_ids, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&eagle_cache_length, offset, sizeof(int32_t));

        offset = memory->allocate((void**)&ea_d_best, offset, 2 * sizeof(int32_t));
        cudaMallocHost(&ea_h_best, 2 * sizeof(int32_t));
        offset = memory->allocate((void**)&ea_tmp_kvcache, offset, 64 * this->draft_model->kv_caches->num_hidden_layers * 2 * this->draft_model->kv_caches->dim * sizeof(T));

        // to allocate ealge draft some states
        offset = memory->allocate((void**)&ea_tree_draft_ids, offset, this->ea_tree_size * sizeof(int32_t));
        offset = memory->allocate((void**)&ea_tree_position_ids, offset, this->ea_tree_size * sizeof(int32_t));
        offset = memory->allocate((void**)&ea_tree_cache_length, offset, sizeof(int32_t));
        offset = memory->allocate((void**)&ea_tree_parent, offset, this->ea_tree_size * sizeof(int32_t));
        
        // TODO: not need in sep version
        // offset = memory->allocate((void**)&ea_tree_attn_mask, offset, this->ea_tree_size * sizeof(uint64_t));


        // init draft output
        int64_t lm_head_end = this->draft_model->init_output_ptr(memory, num_tokens, offset);
        offset = lm_head_end;
        
        offset = memory->allocate((void**)&draft_input, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&draft_position_ids, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&draft_cache_length, offset, sizeof(int32_t));
        cudaMallocHost(&host_draft_cache_length, sizeof(int32_t));

        
        offset = memory->allocate((void**)&draft_logits, offset, 64 * this->draft_model->vocab_size * sizeof(T));
        offset = topk_func->init_output_ptr(memory, 64, offset);
        
        offset = memory->allocate((void**)&draft_tmp, offset, 16*sizeof(int32_t));
        offset = memory->allocate((void**)&d_best, offset, sizeof(int32_t));
        cudaMallocHost(&h_best, sizeof(int32_t));

        // cascade vars
        // TODO: update the token nums of draft tmp hidden states
        offset = memory->allocate((void**)&draft_tmp_hidden_state, offset, (this->min_draft_length + 8) * this->draft_model->hidden_size * sizeof(T));
        return offset;
    }

    int init_storage() {

        this->model->init_weight_ptr(this->model->memory);
        // this->init_weight_ptr(this->model->memory);
        this->draft_model->init_weight_ptr(this->model->memory);
        this->init_weight_ptr(this->model->memory);

        int64_t offset = this->model->init_output_ptr(this->model->memory, this->model->chunk_length, this->model->memory->model_offset);
        int64_t kv_cache_offset = init_output_ptr(this->model->memory, this->model->chunk_length, offset);

        int model_kv_size = (this->model->num_hidden_layers*this->model->num_key_value_heads*this->model->head_dim);
        int draft_kv_size = (this->draft_model->num_hidden_layers*this->draft_model->num_key_value_heads*this->draft_model->head_dim);
        int ea_kv_size = this->ea_num_layers * this->draft_model->num_key_value_heads * this->draft_model->head_dim;
        float ratio = float(model_kv_size)/float(model_kv_size + draft_kv_size + ea_kv_size);
        kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, ratio);
        ratio = float(draft_kv_size)/float(draft_kv_size + ea_kv_size);
        kv_cache_offset = this->draft_model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, ratio);
        this->ea_kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, 1.0);
        return min(min(this->draft_model->kv_caches->budget, this->model->kv_caches->budget), this->ea_kv_caches->budget + 1);
    }

    void load_to_storage(std::string name, void* ptr) {
        
        if (name.substr(0, 5) == "eagle") {
            if (name.substr(0, 9) == "eagle.fc1") {
                ea_fc1->load_to_storage(name, ptr);
            } else if (name.substr(0, 9) == "eagle.fc2") {
                ea_fc2->load_to_storage(name, ptr);
            } else {
                std::regex layer_regex("eagle\\.layers\\.(\\d+)\\.(.*)");
                std::smatch matches;
                if (std::regex_search(name, matches, layer_regex)) {
                    int layer_idx = std::stoi(matches[1]);
                    ea_layers[layer_idx]->load_to_storage(matches[2], ptr);
                } else {
                    throw std::invalid_argument("Unsupported name (layer_idx not found): " + name);
                }
            }
        } else if (name.substr(0, 5) == "draft"){
            std::string draft_name = name.substr(6);
            this->draft_model->load_to_storage(draft_name, ptr);
        } else {
            this->model->load_to_storage(name, ptr);
        }
    }


    
    // void draft_decode_with_graph_control(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
    //     if (this->draft_cuda_graph) {
    //         if (this->draft_graphCreated_padding_length != padded_length || this->draft_graphCreated_input_length != num_tokens) {
    //             if (this->draft_graphExec != nullptr) {
    //                 cudaGraphExecDestroy(this->draft_graphExec);
    //                 this->draft_graphExec = nullptr;
    //             }
    //             if (this->draft_graph != nullptr) {
    //                 cudaGraphDestroy(this->draft_graph);
    //                 this->draft_graph = nullptr;
    //             }
    //             cudaStreamBeginCapture(calc_stream.stream, cudaStreamCaptureModeGlobal);
    //             // this->draft_decode(num_tokens, padded_length, output);
    //             this->draft_model->decode(num_tokens, padded_length, input, position_ids, cache_length, mask_2d, output);
    //             cudaStreamEndCapture(calc_stream.stream, &(this->draft_graph));
    //             cudaGraphInstantiate(&(this->draft_graphExec), this->draft_graph, nullptr, nullptr, 0);
    //             this->draft_graphCreated_padding_length = padded_length;
    //             this->draft_graphCreated_input_length = num_tokens;
    //         }
    //         cudaGraphLaunch(this->draft_graphExec, calc_stream.stream);
    //     } else {
    //         // this->draft_decode(num_tokens, padded_length, output);
    //         this->draft_model->decode(num_tokens, padded_length, input, position_ids, cache_length, mask_2d, output);
    //     }
    // }

    void eagle_prefill(int num_history_tokens) {
        // TODO: adapt for cascade prefill
        cudaMemcpy(this->ea_prev_embed + (ea_num_prev - 1) * this->draft_model->hidden_size, this->draft_model->embedding->output, this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        // ****
        int check_size = 1;
        std::cout << "eagle prefill: " << std::endl;
        T* host_check_embed = new T[this->ea_num_prev * this->draft_model->hidden_size];
        cudaMemcpy(host_check_embed, this->ea_prev_embed, this->ea_num_prev * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToHost);
        std::cout << "eagle prev embed: " << std::endl;
        for (int i = 0; i < this->ea_num_prev; i++) {
            for (int j = 0; j < check_size; j++) {
                float value = __half2float(host_check_embed[i * this->draft_model->hidden_size + j]);
                std::cout << value << " ";
            }
            // std::cout << std::endl;
        }
        std::cout << std::endl;
        
        // print the prev_hidden_state
        std::cout << "eagle prev hidden state: " << std::endl;
        T* host_check_hidden_state = new T[this->ea_num_prev * this->draft_model->hidden_size];
        cudaMemcpy(host_check_hidden_state, this->ea_prev_hidden_state, this->ea_num_prev * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToHost);
        for (int i = 0; i < this->ea_num_prev; i++) {
            for (int j = 0; j < check_size; j++) {
                float value = __half2float(host_check_hidden_state[i * this->draft_model->hidden_size + j]);
                std::cout << value << " ";
            }
            // std::cout << std::endl;
        }
        std::cout << std::endl;

        delete[] host_check_embed;
        delete[] host_check_hidden_state;
        // ****


        this->ea_fc1->prefill(calc_stream, ea_num_prev, this->ea_prev_embed);
        this->ea_fc2->prefill(calc_stream, ea_num_prev, this->ea_prev_hidden_state);
        elementwise_add(calc_stream, ea_num_prev, this->draft_model->hidden_size, this->ea_fc1->output, this->ea_fc2->output, this->ea_fc2->output);
        T* layer_output = nullptr;

        // ******
        // TODO:debug output
        std::cout << "eagle prefill" << this->ea_num_prev << std::endl;
        int * host_eagle_position_ids = new int[this->ea_num_prev];
        cudaMemcpy(host_eagle_position_ids, this->eagle_position_ids, this->ea_num_prev * sizeof(int), cudaMemcpyDeviceToHost);
        
        std::cout << "eagle position ids: " << std::endl;
        for (int i = 0; i < this->ea_num_prev; i++) {
            std::cout << host_eagle_position_ids[i] << " ";
        }
        std::cout << std::endl;

        
        delete[] host_eagle_position_ids;
        // **********
        
        
        for (int i = 0; i < ea_num_layers; i++) {
            this->ea_layers[i]->prefill(num_prev, num_history_tokens, this->ea_fc2->output, layer_output, this->eagle_position_ids, this->ea_kv_caches->caches[i]);
            layer_output = this->ea_layers[i]->output;
        }
        elementwise_add(calc_stream, ea_num_prev, this->draft_model->hidden_size, this->ea_fc2->output, layer_output, this->ea_fc2->output);
    }

    void eagle_decode(int32_t* cache_length) {
        // ****
        int check_size = 1;
        std::cout << "eagle decode: " << std::endl;
        std::cout << "eagle num prev" << this->ea_num_prev << std::endl;
        T* host_check_embed = new T[this->ea_num_prev * this->draft_model->hidden_size];
        cudaMemcpy(host_check_embed, this->ea_prev_embed, this->ea_num_prev * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToHost);
        std::cout << "eagle prev embed: " << std::endl;
        for (int i = 0; i < this->ea_num_prev; i++) {
            for (int j = 0; j < check_size; j++) {
                float value = __half2float(host_check_embed[i * this->draft_model->hidden_size + j]);
                std::cout << value << " ";
            }
            // std::cout << std::endl;
        }
        std::cout << std::endl;
        
        // print the prev_hidden_state
        std::cout << "eagle prev hidden state: " << std::endl;
        T* host_check_hidden_state = new T[this->ea_num_prev * this->draft_model->hidden_size];
        cudaMemcpy(host_check_hidden_state, this->ea_prev_hidden_state, this->ea_num_prev * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToHost);
        for (int i = 0; i < this->ea_num_prev; i++) {
            for (int j = 0; j < check_size; j++) {
                float value = __half2float(host_check_hidden_state[i * this->draft_model->hidden_size + j]);
                std::cout << value << " ";
            }
            // std::cout << std::endl;
        }
        std::cout << std::endl;

        delete[] host_check_embed;
        delete[] host_check_hidden_state;
        // ****

        this->ea_fc1->prefill(calc_stream, ea_num_prev, this->ea_prev_embed);
        this->ea_fc2->prefill(calc_stream, ea_num_prev, this->ea_prev_hidden_state);
        elementwise_add(calc_stream, ea_num_prev, this->draft_model->hidden_size, this->ea_fc1->output, this->ea_fc2->output, this->ea_fc2->output);
        T* layer_output = nullptr;
        for (int i = 0; i < ea_num_layers; i++) {
            this->ea_layers[i]->decode(ea_num_prev, this->eagle_padded_length, this->ea_fc2->output, layer_output, this->eagle_position_ids, cache_length, Mask(nullptr), this->ea_kv_caches->caches[i]);
            layer_output = this->ea_layers[i]->output;
        }
        elementwise_add(calc_stream, ea_num_prev, this->draft_model->hidden_size, this->ea_fc2->output, layer_output, this->ea_fc2->output);
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        this->model->prefill(num_tokens, num_history_tokens, input, position_ids, output);
        // this->draft_model->embedding->prefill(calc_stream, num_tokens, input);
        if (num_history_tokens > 0) {
            this->draft_model->embedding->prefill(calc_stream, this->num_prev, this->draft_input);
            cudaMemcpy(this->ea_prev_embed, this->draft_model->embedding->output+this->draft_model->hidden_size, (this->num_prev-1) * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
            this->draft_model->prefill_embed(this->num_prev, this->num_history_tokens, this->draft_model->embedding->output, this->draft_position_ids, (void*)this->draft_logits);
            // this->draft_model->prefill(this->num_prev, this->num_history_tokens, this->draft_input, this->draft_position_ids, this->draft_logits);
            
            cudaMemcpy(this->ea_prev_hidden_state, this->draft_model->norm->output, this->num_prev * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
            
            this->draft_model->embedding->prefill(calc_stream, 1, input);
            this->eagle_prefill(this->ea_num_history_tokens);

            // this->draft_model->
        }
        
        cudaMemcpy(this->draft_input, input, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->draft_position_ids, position_ids, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->num_prev = num_tokens;
        this->num_history_tokens = num_history_tokens;
        this->is_first_draft = true;

        // eagle
        cudaMemcpy(this->eagle_position_ids, this->draft_position_ids, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->ea_num_prev = num_tokens;
        this->ea_num_history_tokens = num_history_tokens;
        this->ea_is_first_draft = true;
    }

    
    void draft_decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        this->draft_model->decode(num_tokens, padded_length, input, position_ids, cache_length, mask_2d, output);
    }
    
    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        this->model->decode(num_tokens, padded_length, input, position_ids, cache_length, nullptr, output);
    }

    
    // TODO: allocate eagle tree draft ids, ea_tree_position_ids and so on
    void draft_with_eagle(int32_t* ea_tree_draft_ids, int32_t* ea_tree_position_ids, int32_t* ea_cache_length, uint64_t* ea_tree_attn_mask, int32_t* ea_tree_parent) {
        cudaMemcpy(this->eagle_original_length, ea_cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
        this->eagle_padded_length = (this->eagle_original_length[0] + 256 - 1) / 128 * 128;
        if (this->ea_is_first_draft) {
            // prefill hidden states and embedding have been cpy
            this->draft_model->embedding->prefill(calc_stream, 1, ea_tree_draft_ids);
            this->eagle_prefill(this->ea_num_history_tokens);
        } else {
            // TODO: check the cache for eagle decode
            this->eagle_decode(ea_cache_length);
        }
        
        cudaMemcpy(this->eagle_cache_length, ea_cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->eagle_position_ids, ea_cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        repeat(calc_stream, ea_topk_per_iter, 1, 0, this->eagle_position_ids);

        { // d = 0
            this->draft_model->lm_head->prefill(calc_stream, 1, this->ea_fc2->output + (ea_num_prev - 1) * this->draft_model->hidden_size, this->eagle_logits);
            log_softmax(calc_stream, 1, this->draft_model->vocab_size, this->eagle_logits);
            this->ea_topk_func->prefill(calc_stream, 1, this->eagle_logits);
            cudaMemcpy(this->ea_topk_func_2->topk_pos, this->ea_topk_func->topk_pos, ea_topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_topk_func_2->topk_val, this->ea_topk_func->topk_val, ea_topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_tired_history_val, this->ea_topk_func->topk_val, ea_topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_tired_history_pos, this->ea_topk_func->topk_pos, ea_topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            repeat(calc_stream, ea_topk_per_iter, this->draft_model->hidden_size, ea_num_prev-1, this->ea_fc2->output, this->ea_fc1->output);
            init_tree(calc_stream, ea_topk_per_iter, this->eagle_mask_2d);
        }
        

        // ****
        // debug output
        int * host_iter_0_ids = new int[ea_topk_per_iter];
        cudaMemcpy(host_iter_0_ids, this->ea_topk_func_2->topk_pos, ea_topk_per_iter * sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "iter 0 ids: " << std::endl;
        for (int i = 0; i < ea_topk_per_iter; i++) {
            std::cout << host_iter_0_ids[i] << " ";
        }
        std::cout << std::endl;
        delete[] host_iter_0_ids;
        // ****

        for (int d = 1; d < this->ea_num_iter; ++d) {
            add(calc_stream, 1, this->eagle_cache_length, ea_topk_per_iter);
            this->draft_model->embedding->prefill(calc_stream, ea_topk_per_iter, this->ea_topk_func_2->topk_pos);
            this->ea_fc2->prefill(calc_stream, ea_topk_per_iter, this->ea_fc1->output);
            this->ea_fc1->prefill(calc_stream, ea_topk_per_iter, this->draft_model->embedding->output);
            elementwise_add(calc_stream, ea_topk_per_iter, this->draft_model->hidden_size, this->ea_fc1->output, this->ea_fc2->output, this->ea_fc2->output);
            T* layer_output = nullptr;
            for (int i = 0; i < ea_num_layers; i++) {
                this->ea_layers[i]->decode(ea_topk_per_iter, this->eagle_padded_length, this->ea_fc2->output, layer_output, this->eagle_position_ids, this->eagle_cache_length, Mask(eagle_mask_2d, ea_topk_per_iter, ea_topk_per_iter * d), this->ea_kv_caches->caches[i]);
                layer_output = this->ea_layers[i]->output;
            }
            elementwise_add(calc_stream, ea_topk_per_iter, this->draft_model->hidden_size, this->ea_fc2->output, layer_output, this->ea_fc2->output);
            add(calc_stream, ea_topk_per_iter, this->eagle_position_ids, 1);

            this->draft_model->lm_head->prefill(calc_stream, ea_topk_per_iter, this->ea_fc2->output, this->eagle_logits);
            log_softmax(calc_stream, ea_topk_per_iter, this->draft_model->vocab_size, this->eagle_logits);
            this->ea_topk_func->prefill(calc_stream, ea_topk_per_iter, this->eagle_logits);
            cumsum(calc_stream, ea_topk_per_iter, ea_topk_per_iter, this->ea_topk_func->topk_val, this->ea_topk_func_2->topk_val);
            cudaMemcpy(this->ea_tired_history_val + ea_topk_per_iter + (d - 1) * ea_topk_per_iter * ea_topk_per_iter, this->ea_topk_func->topk_val, ea_topk_per_iter * ea_topk_per_iter * sizeof(T), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_tired_history_pos + ea_topk_per_iter + (d - 1) * ea_topk_per_iter * ea_topk_per_iter, this->ea_topk_func->topk_pos, ea_topk_per_iter * ea_topk_per_iter * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            this->ea_topk_func_2->prefill(calc_stream, 1, this->ea_topk_func->topk_val, ea_topk_per_iter * ea_topk_per_iter, ea_topk_per_iter);

            cudaMemcpy(this->ea_tmp_mask_2d, this->eagle_mask_2d, ea_topk_per_iter * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
            set_parent(calc_stream, ea_topk_per_iter, this->ea_tired_history_parent + (d - 1) * ea_topk_per_iter, this->ea_topk_func_2->topk_pos, 10 + (d - 1) * ea_topk_per_iter * ea_topk_per_iter);
            update_tree(calc_stream, ea_topk_per_iter, ea_topk_per_iter * d, this->eagle_mask_2d, this->ea_tmp_mask_2d, this->ea_topk_func_2->topk_pos);
            remap_hidden(calc_stream, ea_topk_per_iter, this->draft_model->hidden_size, this->ea_topk_func_2->topk_pos, this->ea_fc2->output, this->ea_fc1->output, ea_topk_per_iter);
            remap_id(calc_stream, ea_topk_per_iter, this->ea_topk_func_2->topk_pos, this->ea_topk_func->topk_pos);
        }

        this->ea_topk_func_2->prefill(calc_stream, 1, this->ea_tired_history_val);

        // build tree
        build_dynamic_tree(calc_stream, this->ea_tree_size, this->eagle_original_length[0], this->ea_topk_per_iter, this->ea_tired_history_parent, this->ea_topk_func_2->topk_pos, ea_tree_position_ids, ea_tree_attn_mask, ea_tree_parent);
        remap_id(calc_stream, this->ea_tree_size-1, this->ea_topk_func_2->topk_pos, this->ea_tired_history_pos, ea_tree_draft_ids + 1);

        this->ea_is_first_draft = false;
    }

    void draft(int32_t *tree_draft_ids, int32_t *tree_position_ids, int32_t *cache_length, uint64_t*, int32_t*) {
        // reset cur draft length
        this->cur_draft_length = 0;
        if (this->is_first_draft) {
            // append tree draft ids to draft input
            cudaMemcpy(this->draft_input+this->num_prev, tree_draft_ids, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_position_ids+this->num_prev, tree_position_ids, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            this->num_prev += 1;
            
            this->draft_model->embedding->prefill(calc_stream, this->num_prev, this->draft_input);
            cudaMemcpy(this->ea_prev_embed, this->draft_model->embedding->output+ this->draft_model->hidden_size, (this->num_prev-1) * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

            this->draft_model->prefill_embed(this->num_prev, this->num_history_tokens, this->draft_model->embedding->output, this->draft_position_ids, (void*)this->draft_logits);

            // this->draft_model->prefill(this->num_prev, this->num_history_tokens, this->draft_input, this->draft_position_ids, (void*)this->draft_logits);

            // eagle prepare for draft_with_eagle function 
            // ea_is_first_draft is True
            cudaMemcpy(this->eagle_position_ids + (this->ea_num_prev), tree_position_ids, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            this->ea_num_prev = this->num_prev;
            // this->ea_num_history_tokens = this->num_history_tokens;
            
            // this->ea_prev_hidden_state = this->draft_model->norm->output;
            cudaMemcpy(this->ea_prev_hidden_state, this->draft_model->norm->output, this->num_prev * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
            
            this->topk_func->prefill(calc_stream, 1, this->draft_logits);
            

            // prepare for draft with eagle     
            cudaMemcpy(this->ea_tree_draft_ids, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_tree_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->ea_tree_cache_length, 1);
            // this->draft_with_eagle(this->ea_tree_draft_ids, this->ea_tree_position_ids, this->ea_tree_cache_length, this->ea_tree_attn_mask, this->ea_tree_parent);

            // TODO: prepare for the draft model decode
            // cudaMemcpy(this->host_draft_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(this->draft_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->draft_cache_length, 1);
            cudaMemcpy(this->host_draft_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);           
            // cudaMemcpy(this->draft_position_ids, tree_position_ids, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            // this->draft_padded_length = (this->host_draft_cache_length[0]+ 128 -1) / 128*128;
            // this->topk_func->prefill(calc_stream, 1, this->draft_logits);


            
        } else if (this->num_prev == 2){
            std::cout << "full accept" << std::endl;
            std::cout << "num_prev: " << this->num_prev << std::endl;
            std::cout << "ea_num_prev: " << this->ea_num_prev << std::endl;
            // TODO: delete print draft_cache_length
            int * host_draft_cache_length_print = new int[1];
            std::cout << "draft cache length: " << std::endl;
            cudaMemcpy(host_draft_cache_length_print, this->draft_cache_length, sizeof(int), cudaMemcpyDeviceToHost);
            std::cout << host_draft_cache_length_print[0] << std::endl;
            delete[] host_draft_cache_length_print;

            
            // print draft_position_ids
            int * host_draft_position_ids = new int[this->num_prev];
            cudaMemcpy(host_draft_position_ids, this->draft_position_ids, this->num_prev * sizeof(int), cudaMemcpyDeviceToHost);
            std::cout << "draft position ids: " << std::endl;
            for (int i = 0; i < this->num_prev; i++) {
                std::cout << host_draft_position_ids[i] << " ";
            }
            std::cout << std::endl;
            delete[] host_draft_position_ids;

            // print draft_input
            int * host_draft_input = new int[this->num_prev];
            cudaMemcpy(host_draft_input, this->draft_input, this->num_prev * sizeof(int), cudaMemcpyDeviceToHost);
            std::cout << "draft input: " << std::endl;
            for (int i = 0; i < this->num_prev; i++) {
                std::cout << host_draft_input[i] << " ";
            }
            std::cout << std::endl;
            delete[] host_draft_input;

            // this->draft_decode(this->num_prev, this->draft_padded_length, this->draft_logits);
            this->draft_model->embedding->prefill(calc_stream, this->num_prev, this->draft_input);
            cudaMemcpy(this->ea_prev_embed + (ea_num_prev)* this->draft_model->hidden_size, this->draft_model->embedding->output + this->draft_model->hidden_size, this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

            this->draft_model->decode_embed(this->num_prev, this->draft_padded_length, this->draft_model->embedding->output, this->draft_position_ids, this->draft_cache_length, nullptr, (void*)this->draft_logits);

            // this->draft_model->decode(this->num_prev, this->draft_padded_length, this->draft_input, this->draft_position_ids, this->draft_cache_length, nullptr, (void*)this->draft_logits);
            this->topk_func->prefill(calc_stream, 1, this->draft_logits+(this->draft_model->vocab_size));

            // add(calc_stream, 1, this->draft_position_ids, 1);

            //TODO: prepare for the eagle input
            
            cudaMemcpy(this->ea_prev_hidden_state + (ea_num_prev)* this->draft_model->hidden_size, this->draft_model->norm->output, this->num_prev * this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
            this->draft_model->embedding->prefill(calc_stream, 1, this->topk_func->topk_pos);
            cudaMemcpy(this->ea_prev_embed + (ea_num_prev+1)* this->draft_model->hidden_size, this->draft_model->embedding->output, this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
            this->ea_num_prev += this->num_prev;

            

            // prepare for draft with eagle
            cudaMemcpy(this->ea_tree_draft_ids, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_tree_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            // this->draft_with_eagle(this->ea_tree_draft_ids, this->ea_tree_position_ids, this->draft_cache_length, this->ea_tree_attn_mask, this->ea_tree_parent);
            
            
        } 
        else {
            // num_prev == 1
            // this->draft_decode_with_graph_control(this->num_prev, this->draft_padded_length, this->draft_input, this->draft_position_ids, this->draft_cache_length, nullptr, (void*)this->draft_logits);
            // TODO: remove decode in the future
            this->draft_model->decode(this->num_prev, this->draft_padded_length, this->draft_input, this->draft_position_ids, this->draft_cache_length, nullptr, (void*)this->draft_logits);
            this->topk_func->prefill(calc_stream, 1, this->draft_logits);
            
            // TODO: prepare for the eagle input
            cudaMemcpy(this->ea_prev_hidden_state + (ea_num_prev)*this->draft_model->hidden_size, this->draft_model->norm->output, this->num_prev * this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
            
            this->draft_model->embedding->prefill(calc_stream, 1, this->topk_func->topk_pos);
            cudaMemcpy(this->ea_prev_embed + (ea_num_prev)* this->draft_model->hidden_size, this->draft_model->embedding->output, this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
            this->ea_num_prev += this->num_prev; 

            // prepare for draft with eagle            
            cudaMemcpy(this->ea_tree_draft_ids, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->ea_tree_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);

            // this->draft_with_eagle(this->ea_tree_draft_ids, this->ea_tree_position_ids, this->draft_cache_length, this->ea_tree_attn_mask, this->ea_tree_parent);
        }

        // draft model has forward one time
        cudaMemcpy(this->draft_tmp, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->draft_tmp_hidden_state, this->draft_model->norm->output + (this->num_prev-1) * this->draft_model->hidden_size, this->draft_model->hidden_size*sizeof(T), cudaMemcpyDeviceToDevice);
        this->cur_draft_length += 1;
        
        // while (this->cur_draft_length < min_draft_length){
        //     // iter 0
        //     // TODO: draft input maybe no need
        //     // cudaMemcpy(this->draft_input, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        //     // eagle draft
        //     this->draft_with_eagle(
        //         this->ea_tree_draft_ids,
        //         this->ea_tree_position_ids,
        //         this->ea_tree_cache_length,
        //         this->ea_tree_attn_mask,
        //         this->ea_tree_parent
        //     );
        //     cudaMemcpy(this->draft_cache_length, this->ea_tree_cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        //     add(calc_stream, 1, this->draft_cache_length, this->ea_tree_size);
        //     this->host_draft_cache_length += this->ea_tree_size;
        //     this->draft_padded_length = (this->host_draft_cache_length[0]+ 128 -1) / 128*128;;
        //     // draft model verify eagle tree draft
        //     this->draft_model->decode(
        //         this->ea_tree_size,
        //         this->draft_padded_length,
        //         this->ea_tree_draft_ids,
        //         this->ea_tree_position_ids,
        //         this->draft_cache_length,
        //         this->ea_tree_attn_mask,
        //         (void*) this->draft_logits
        //     );
        //     this->topk_func->prefill(calc_stream, this->ea_tree_size, this->draft_logits);
            
        //     this->eagle_verify(
        //         this->ea_tree_size,
        //         this->ea_tree_draft_ids,
        //         this->topk_func->topk_pos,
        //         this->ea_tree_position_ids,
        //         this->draft_cache_length,
        //         this->ea_tree_attn_mask,
        //         this->ea_tree_parent
        //     );


        //     cudaMemcpy(this->draft_tmp_hidden_state + (this->cur_draft_length), this->ea_prev_hidden_states, this->ea_num_prev * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        //     // accept return to ea_h_best[0]
        //     cudaMemcpy(this->draft_tmp + this->cur_draft_length, this->ea_tree_draft_ids, this->ea_h_best[0]*sizeof(int32_t), cudaMemcpyDeviceToDevice);
        //     this->cur_draft_length += this->ea_h_best[0];
        //     cudaMemcpy(this->ea_tree_draft_ids, this->ea_tree_draft_ids + (this->ea_h_best[0]-1), sizeof(int32_t), cudaMemcpyDeviceToDevice);
        //     add(calc_stream, 1, this->ea_tree_cache_length, this->ea_h_best[0]);
            
        // }
        

        cudaMemcpy(tree_draft_ids + 1, this->draft_tmp, this->cur_draft_length*sizeof(int32_t), cudaMemcpyDeviceToDevice);
        // TODO: move to python
        // make_arange(calc_stream, this->cur_draft_length+1, cache_length, tree_position_ids);
        this->is_first_draft = false;
    }
    
    int draft_verify(int32_t ea_num_tokens, int32_t* ea_pred, int32_t* ea_gt, int32_t* ea_position_ids, int32_t* ea_cache_length, uint64_t* ea_mask_2d, int32_t* ea_tree_parent) {
        verify_draft(calc_stream, ea_num_tokens, ea_pred, ea_gt, ea_position_ids, ea_cache_length, ea_mask_2d, ea_tree_parent, this->ea_d_best);
    
        cudaMemcpyAsync(this->ea_h_best, this->ea_d_best, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);

        this->ea_num_prev = ea_h_best[0];
        remap_hidden(calc_stream, this->ea_num_prev, this->draft_model->hidden_size, ea_pred, this->draft_model->norm->output, this->ea_prev_hidden_state);

        fix_kv_cache(calc_stream, ea_h_best[0], this->draft_model->kv_caches->num_hidden_layers * 2, this->draft_model->kv_caches->dim, ea_pred, ea_gt, ea_cache_length, this->draft_model->kv_caches->d_flat_caches, this->ea_tmp_kvcache);

        this->draft_model->embedding->prefill(calc_stream, this->ea_num_prev, ea_pred);
        cudaMemcpy(this->ea_prev_embed, this->draft_model->embedding->output, this->ea_num_prev * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        make_arange(calc_stream, this->ea_num_prev, ea_cache_length, this->eagle_position_ids);

        // store the draft model hiddeen states
        std::cout << "cur draft length after verify" << this->cur_draft_length << std::endl;
        cudaMemcpy(this->draft_tmp_hidden_state + (this->cur_draft_length*this->draft_model->hidden_size), this->ea_prev_hidden_state, this->ea_num_prev * this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

        cudaMemcpy(this->draft_tmp + this->cur_draft_length, ea_pred, this->ea_num_prev * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->cur_draft_length += this->ea_num_prev;

        // TODO: delete print
        std::cout << "ea_num_prev after verify: " <<  this->ea_num_prev << std::endl;

        return ea_h_best[0];
    }

    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* attn_mask, int32_t* tree_parent) { 
        // *****
        // print the pred and gt
        int * host_pred = new int[num_tokens];
        int * host_gt = new int[num_tokens];
        
        cudaMemcpy(host_pred, pred, num_tokens * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(host_gt, gt, num_tokens * sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "pred: " << std::endl;
        for (int i = 0; i < num_tokens; i++) {
            std::cout << host_pred[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "gt: " << std::endl;
        for (int i = 0; i < num_tokens; i++) {
            std::cout << host_gt[i] << " ";
        }
        std::cout << std::endl;

        delete[] host_pred;
        delete[] host_gt;
        // *****
        


        verify_seq_draft(calc_stream, num_tokens, pred, gt, (uint16_t*)attn_mask, this->d_best);
        cudaMemcpyAsync(this->h_best, this->d_best, 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);
        
        std::cout << "h_best: " << this->h_best[0] << std::endl;

        // debug accept is less than original eagle length
        // this->h_best[0] = 3;
        
        if (h_best[0]==(this->cur_draft_length+1)) {
            // full accept   
            this->num_prev = 2;
            cudaMemcpy(this->draft_input, gt + (this->cur_draft_length-1), 2*sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->draft_cache_length, this->h_best[0]+1);
            make_arange(calc_stream, 2, cache_length, this->draft_position_ids);
            add(calc_stream, 2, this->draft_position_ids, this->cur_draft_length);
            cudaMemcpy(this->host_draft_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
            this->draft_padded_length = (this->host_draft_cache_length[0]+ 128 -1) / 128*128;
        } else {
            this->num_prev = 1;
            cudaMemcpy(this->draft_input, gt + (this->h_best[0]-1), sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->draft_cache_length, this->h_best[0]+1);
            cudaMemcpy(this->draft_position_ids, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->draft_position_ids, this->h_best[0]);
            cudaMemcpy(this->host_draft_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
            this->draft_padded_length = (this->host_draft_cache_length[0]+ 128 -1) / 128*128;

            // adapt eagle draft ptr
            // conidtion 1: eagle last start postion is larger than accept position
            if (host_draft_cache_length[0] > this->eagle_original_length[0] + 1) {
                std::cout << "host draft cache length: " << host_draft_cache_length[0] << std::endl;
                std::cout << "eagle original length: " << this->eagle_original_length[0] << std::endl;
                this->ea_num_prev = host_draft_cache_length[0] - (this->eagle_original_length[0] + 1);
                // keep ea_prev_hidden_state and update ea_prev_embed
                this->draft_model->embedding->prefill(calc_stream, 1, this->draft_input);
                cudaMemcpy(this->ea_prev_embed+(this->ea_num_prev-1), this->draft_model->embedding->output, this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);

                // print the last ea_prev_hidden_state
                T* host_check_hidden_state = new T[this->draft_model->hidden_size];
                cudaMemcpy(host_check_hidden_state, this->ea_prev_hidden_state + (this->ea_num_prev-1) * this->draft_model->hidden_size, this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToHost);

                std::cout << "ea num prev" << this->ea_num_prev << std::endl;
                
                std::cout << "eagle prev hidden state: " << std::endl;
                for (int j = 0; j < this->draft_model->hidden_size; j++) {
                    float value = __half2float(host_check_hidden_state[j]);
                    std::cout << value << " ";
                }
                std::cout << std::endl;

                // print the kepted draft model hidden state
                T* host_check_hidden_state_2 = new T[this->draft_model->hidden_size];
                cudaMemcpy(host_check_hidden_state_2, this->draft_tmp_hidden_state + (this->h_best[0]-1) *this->draft_model->hidden_size, this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToHost);

                std::cout << "draft tmp hidden state: " << std::endl;
                for (int j = 0; j < this->draft_model->hidden_size; j++) {
                    float value = __half2float(host_check_hidden_state_2[j]);
                    std::cout << value << " ";
                }
                std::cout << std::endl;

                delete[] host_check_hidden_state;
                delete[] host_check_hidden_state_2;
            } else {
                // condition 2: eagle last start position is less than accept position
                // index from the kepted draft model hidden state
                cudaMemcpy(this->ea_prev_hidden_state, this->draft_tmp_hidden_state + (h_best[0]-1) *this->draft_model->hidden_size, this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
                this->ea_num_prev = 1;
                this->draft_model->embedding->prefill(calc_stream, 1, this->draft_input);
                cudaMemcpy(this->ea_prev_embed, this->draft_model->embedding->output, this->draft_model->hidden_size * sizeof(T), cudaMemcpyDeviceToDevice);
                cudaMemcpy(this->eagle_position_ids, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
                add(calc_stream, 1, this->eagle_position_ids, this->h_best[0]-1);
            }
            
            
        }
        this->cur_draft_length = 0;
        return h_best[0];

    }
};