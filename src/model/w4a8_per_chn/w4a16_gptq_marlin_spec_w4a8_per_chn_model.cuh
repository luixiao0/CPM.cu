#pragma once
#include "w4a8_per_chn_model.cuh"
#include "../eagle.cuh"
#include "../drafter.cuh"
#include "../w4a16_gptq_marlin/w4a16_gptq_marlin_layer.cuh"


template <typename T>
struct W4A16GMSpecW4A8PCModelImpl: Model {

    int vocab_size;
    int num_hidden_layers;
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_key_value_heads;
    int head_dim;
    float rms_norm_eps;


    W4A8PerChnModelImpl<T>* model;
    KVCacheManager<T>* kv_caches;

    Embedding<T>* embedding;
    std::vector<W4A16GPTQMarlinLayer<T>*> layers;
    RMSNorm<T>* norm;
    Linear<T>* lm_head;

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
    int num_iter;
    int num_prev, num_history_tokens;

    // draft mask always nullptr
    uint64_t* draft_mask_2d;   

    // graph
    bool draft_cuda_graph;
    int draft_graphCreated_padding_length;
    int draft_graphCreated_input_length;
    cudaGraph_t draft_graph;
    cudaGraphExec_t draft_graphExec;

    W4A16GMSpecW4A8PCModelImpl(
        W4A8PerChnModelImpl<T>* model,
        int draft_vocab_size,
        int draft_num_hidden_layers,
        int draft_hidden_size,
        int draft_intermediate_size,
        int draft_num_attention_heads,
        int draft_num_key_value_heads,
        int draft_head_dim,
        float draft_rms_norm_eps,
        int num_iter,
        bool draft_cuda_graph
    ) {
        this->model = model;
        this->vocab_size = draft_vocab_size;
        this->num_hidden_layers = draft_num_hidden_layers;
        this->hidden_size = draft_hidden_size;
        this->intermediate_size = draft_intermediate_size;
        this->num_attention_heads = draft_num_attention_heads;
        this->num_key_value_heads = draft_num_key_value_heads;
        this->head_dim = draft_head_dim;
        this->rms_norm_eps = draft_rms_norm_eps;

        this->num_iter = num_iter;

        this->draft_mask_2d = 0;
        
        kv_caches = new KVCacheManager<T>(num_hidden_layers, num_key_value_heads, head_dim);

        embedding = new Embedding<T>(vocab_size, hidden_size);
        for (int i = 0; i < num_hidden_layers; i++) {
            layers.push_back(new W4A16GPTQMarlinLayer<T>(hidden_size, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, rms_norm_eps));
        }
        norm = new RMSNorm<T>(hidden_size, rms_norm_eps);
        lm_head = new Linear<T>(hidden_size, vocab_size);

        topk_func = new functions::TopK<T>(model->vocab_size, 1); // greedy sample
        
        this->draft_cuda_graph = draft_cuda_graph;
        this->draft_graphCreated_padding_length = -1;
        this->draft_graphCreated_input_length = -1;
        this->draft_graph = nullptr;
        this->draft_graphExec = nullptr;
    }

    void init_weight_ptr(Memory* memory) {
        embedding->init_weight_ptr(memory);
        for (int i = 0; i < num_hidden_layers; i++) {
            layers[i]->init_weight_ptr(memory);
        }
        norm->init_weight_ptr(memory);
        lm_head->init_weight_ptr(memory);
        kv_caches->init_weight_ptr(memory);
    }

    int64_t init_output_ptr(Memory* memory, int32_t num_tokens, int64_t offset) {
        int64_t embedding_end = embedding->init_output_ptr(memory, num_tokens, offset);
        int64_t layer_end = 0;
        for (int i = 0; i < num_hidden_layers; i++) {
            layer_end = layers[i]->init_output_ptr(memory, num_tokens, embedding_end);
        }
        // norm and lm_head are not used in prefill
        int64_t norm_end = norm->init_output_ptr(memory, num_tokens, layer_end);
        int64_t lm_head_end = lm_head->init_output_ptr(memory, 64, norm_end);
        offset = lm_head_end;
        
        offset = memory->allocate((void**)&draft_input, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&draft_position_ids, offset, num_tokens * sizeof(int32_t));
        offset = memory->allocate((void**)&draft_cache_length, offset, sizeof(int32_t));
        cudaMallocHost(&host_draft_cache_length, sizeof(int32_t));

        
        offset = memory->allocate((void**)&draft_logits, offset, 64 * vocab_size * sizeof(T));
        offset = topk_func->init_output_ptr(memory, 1, offset);
        
        offset = memory->allocate((void**)&draft_tmp, offset, 16*sizeof(int32_t));
        offset = memory->allocate((void**)&d_best, offset, sizeof(int32_t));
        cudaMallocHost(&h_best, sizeof(int32_t));
        return offset;
    }

    int init_storage() {
        this->model->init_weight_ptr(this->model->memory);
        this->init_weight_ptr(this->model->memory);

        int64_t offset = this->model->init_output_ptr(this->model->memory, this->model->chunk_length, this->model->memory->model_offset);
        int64_t kv_cache_offset = init_output_ptr(this->model->memory, this->model->chunk_length, offset);

        int model_kv_size = (this->model->num_hidden_layers*this->model->num_key_value_heads*this->model->head_dim);
        int draft_kv_size = (this->num_hidden_layers*this->num_key_value_heads*this->head_dim);
        float ratio = float(model_kv_size)/float(model_kv_size + draft_kv_size);
        kv_cache_offset = this->model->kv_caches->init_output_ptr(this->model->memory, kv_cache_offset, ratio);
        kv_caches->init_output_ptr(this->model->memory, kv_cache_offset);
        return min(kv_caches->budget, this->model->kv_caches->budget);
    }

    void load_to_storage(std::string name, void* ptr) {
        if (name.substr(0, 5) == "draft"){
            if (name.substr(0, 24) == "draft.model.embed_tokens") {
                embedding->load_to_storage(name, ptr);
            } else if (name.substr(0, 16) == "draft.model.norm") {
                norm->load_to_storage(name, ptr);
            } else if (name.substr(0, 13) == "draft.lm_head") {
                lm_head->load_to_storage(name, ptr);
            } else if (name.find("rotary_emb") != std::string::npos) {
                kv_caches->rotary_embedding->load_to_storage(name, ptr);
            } else if (name.substr(0, 18) == "draft.model.layers") { // e.g. draft.model.layers.20.attn.q_proj.weight
                std::regex layer_regex("draft\\.model\\.layers\\.(\\d+)\\.(.*)");
                std::smatch matches;
                if (std::regex_search(name, matches, layer_regex)) {
                    int layer_idx = std::stoi(matches[1]);
                    layers[layer_idx]->load_to_storage(matches[2], ptr);
                } else {
                    throw std::invalid_argument("Unsupported name (layer_idx not found): " + name);
                }
            } else {
                throw std::invalid_argument("Unsupported name " + name);
            }
        } else {
            this->model->load_to_storage(name, ptr);
        }
    }

    void draft_prefill_embed(int32_t num_tokens, int32_t num_history_tokens, T* embed, int32_t* position_ids) {
        T* layer_output = nullptr;
        for (int i = 0; i < num_hidden_layers; i++) {
            this->layers[i]->prefill(num_tokens, num_history_tokens, embed, layer_output, position_ids, this->kv_caches->caches[i]);
            layer_output = this->layers[i]->output;
        }
        // TODO : remove norm and lm_head prefill
        this->norm->prefill(calc_stream, num_tokens, embed, layer_output);
        this->lm_head->prefill(calc_stream, 1, this->norm->output + (num_tokens - 1) * hidden_size);
    }

    void draft_prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids) {
        this->embedding->prefill(calc_stream, num_tokens, input);
        this->draft_prefill_embed(num_tokens, num_history_tokens, this->embedding->output, position_ids);
    }

    void draft_decode_embed(int32_t num_tokens, int32_t padded_length, T* embed, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        Mask mask(mask_2d, num_tokens, num_tokens);
        T* layer_output = nullptr;
        for (int i = 0; i < num_hidden_layers; i++) {
            this->layers[i]->decode(num_tokens, padded_length, this->embedding->output, layer_output, position_ids, cache_length, mask, this->kv_caches->caches[i]);
            layer_output = this->layers[i]->output;
        }
        this->norm->prefill(calc_stream, num_tokens, this->embedding->output, layer_output);
        this->lm_head->prefill(calc_stream, num_tokens, this->norm->output, (T*) output);
    }

    void draft_decode(int32_t num_tokens, int32_t padded_length, void* output) {
        this->embedding->prefill(calc_stream, num_tokens, this->draft_input);
        this->draft_decode_embed(num_tokens, padded_length, this->embedding->output, this->draft_position_ids, this->draft_cache_length, this->draft_mask_2d, output);
    }
    
    void draft_decode_with_graph_control(int32_t num_tokens, int32_t padded_length, void* output) {
        if (this->draft_cuda_graph) {
            if (this->draft_graphCreated_padding_length != padded_length || this->draft_graphCreated_input_length != num_tokens) {
                if (this->draft_graphExec != nullptr) {
                    cudaGraphExecDestroy(this->draft_graphExec);
                    this->draft_graphExec = nullptr;
                }
                if (this->draft_graph != nullptr) {
                    cudaGraphDestroy(this->draft_graph);
                    this->draft_graph = nullptr;
                }
                cudaStreamBeginCapture(calc_stream.stream, cudaStreamCaptureModeGlobal);
                this->draft_decode(num_tokens, padded_length, output);
                cudaStreamEndCapture(calc_stream.stream, &(this->draft_graph));
                cudaGraphInstantiate(&(this->draft_graphExec), this->draft_graph, nullptr, nullptr, 0);
                this->draft_graphCreated_padding_length = padded_length;
                this->draft_graphCreated_input_length = num_tokens;
            }
            cudaGraphLaunch(this->draft_graphExec, calc_stream.stream);
        } else {
            this->draft_decode(num_tokens, padded_length, output);
        }
    }

    void prefill(int32_t num_tokens, int32_t num_history_tokens, int32_t* input, int32_t* position_ids, void* output) {
        this->model->prefill(num_tokens, num_history_tokens, input, position_ids, output);
        // this->draft_prefill(num_tokens, num_history_tokens, input, position_ids);
        if (num_history_tokens > 0) {
            this->draft_prefill(this->num_prev, this->num_history_tokens, this->draft_input, this->draft_position_ids);
        }
        
        cudaMemcpy(this->draft_input, input, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->draft_position_ids, position_ids, num_tokens * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->num_prev = num_tokens;
        this->num_history_tokens = num_history_tokens;
        this->is_first_draft = true;
    }

    void decode(int32_t num_tokens, int32_t padded_length, int32_t* input, int32_t* position_ids, int32_t* cache_length, uint64_t* mask_2d, void* output) {
        this->model->decode(num_tokens, padded_length, input, position_ids, cache_length, nullptr, output);
    }

    void draft(int32_t *tree_draft_ids, int32_t *tree_position_ids, int32_t *cache_length, uint64_t*, int32_t*) {
        if (this->is_first_draft) {
            this->draft_prefill(this->num_prev, this->num_history_tokens, this->draft_input, this->draft_position_ids);
        }

        cudaMemcpy(this->host_draft_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);

        cudaMemcpy(this->draft_input, tree_draft_ids, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->draft_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(this->draft_position_ids, tree_position_ids, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        this->draft_padded_length = (this->host_draft_cache_length[0]+ 128 -1) / 128*128;;
        
        // iter 0
        {
            this->draft_decode_with_graph_control(1, this->draft_padded_length, (void*) this->draft_logits);
            // update input_ids
            // log_softmax(calc_stream, 1, this->vocab_size, this->draft_logits);
            this->topk_func->prefill(calc_stream, 1, this->draft_logits);
            cudaMemcpy(this->draft_input, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            // update draft tmp
            cudaMemcpy(this->draft_tmp, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        }

        for (int d = 1; d < this->num_iter; ++d){
            add(calc_stream, 1, this->draft_cache_length, 1);
            add(calc_stream, 1, this->draft_position_ids, 1);

            this->host_draft_cache_length[0] += 1;
            this->draft_padded_length = (this->host_draft_cache_length[0]+ 128 -1) / 128*128;;
            this->draft_decode_with_graph_control(1, this->draft_padded_length, (void*) this->draft_logits);
            this->topk_func->prefill(calc_stream, 1, this->draft_logits);
            cudaMemcpy(this->draft_input, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_tmp + d, this->topk_func->topk_pos, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        }

        cudaMemcpy(tree_draft_ids + 1, this->draft_tmp, num_iter*sizeof(int32_t), cudaMemcpyDeviceToDevice);
        make_arange(calc_stream, this->num_iter, cache_length, tree_position_ids+1);
        this->is_first_draft = false;
    }

    int verify(int32_t num_tokens, int32_t* pred, int32_t* gt, int32_t* position_ids, int32_t* cache_length, uint64_t* attn_mask, int32_t* tree_parent) { 
        verify_seq_draft(calc_stream, num_tokens, pred, gt, (uint16_t*)attn_mask, this->d_best);
        cudaMemcpyAsync(this->h_best, this->d_best, 1 * sizeof(int32_t), cudaMemcpyDeviceToHost, calc_stream.stream);
        cudaStreamSynchronize(calc_stream.stream);
        
        if (h_best[0]==(num_iter+1)) {
            // full accept   
            cudaMemcpy(this->draft_input, pred + num_iter, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpy(this->draft_cache_length, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->draft_cache_length, this->h_best[0]);
            cudaMemcpy(this->draft_position_ids, cache_length, sizeof(int32_t), cudaMemcpyDeviceToDevice);
            add(calc_stream, 1, this->draft_position_ids, num_iter);

            cudaMemcpy(this->host_draft_cache_length, this->draft_cache_length, sizeof(int32_t), cudaMemcpyDeviceToHost);
            this->draft_padded_length = (this->host_draft_cache_length[0]+ 128 -1) / 128*128;;
            this->draft_decode_with_graph_control(1, this->draft_padded_length, (void*) this->draft_logits);
        }

        return h_best[0];

    }
};