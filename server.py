from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import torch
import json
import time
import asyncio
import os
import argparse
from cpmcu.llm import LLM
from cpmcu.llm_w4a16_gptq_marlin import W4A16GPTQMarlinLLM
from cpmcu.speculative import LLM_with_eagle
from cpmcu.speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import W4A16GPTQMarlinLLM_with_eagle
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download

# Configuration
DEFAULT_CONFIG = {
    'test_minicpm4': True,
    'use_stream': True,
    'apply_eagle': True,
    'apply_quant': True,
    'apply_sparse': True,
    'apply_eagle_quant': True,
    "minicpm4_yarn": True,  # for long context test
    'frspec_vocab_size': 32768,
    'eagle_window_size': 8 * 128,
    'eagle_num_iter': 2,
    'eagle_topk_per_iter': 10,
    'eagle_tree_size': 12,
    'apply_compress_lse': True,
    'sink_window_size': 1,
    'block_window_size': 8,
    'sparse_topk_k': 64,
    "sparse_switch": 1,
    'num_generate': 256,
    'chunk_length': 2048,
    'memory_limit': 0.9,
    'cuda_graph': True,
    'dtype': torch.float16,
    'use_terminators': True,
    "temperature": 0.0,
    "random_seed": None,
    # Demo config
    'use_enter': False,
    'use_decode_enter': False,
}

# RoPE configuration
ROPE_CONFIG = {
    "rope_scaling": {
        "rope_type": "longrope",
        "long_factor": [0.9977997200264581, 1.014658295992452, 1.0349680404997148, 1.059429246056193, 1.0888815016813513, 1.1243301355211495, 1.166977103606075, 1.2182568066927284, 1.2798772354275727, 1.3538666751582975, 1.4426259039919596, 1.5489853358570191, 1.6762658237220625, 1.8283407612492941, 2.0096956085876183, 2.225478927469756, 2.481536379650452, 2.784415934557119, 3.1413289096347365, 3.560047844772632, 4.048719380066383, 4.752651957515948, 5.590913044973868, 6.584005926629993, 7.7532214876576155, 9.119754865903639, 10.704443927019176, 12.524994176518703, 14.59739595363613, 16.93214476166354, 19.53823297353041, 22.417131025031697, 25.568260840911098, 28.991144156566317, 32.68408069090375, 36.65174474170465, 40.90396065611201, 45.4664008671033, 50.37147343433591, 55.6804490772103, 61.470816952306556, 67.8622707390618, 75.00516023410414, 83.11898235973767, 92.50044360202462, 103.57086856690864, 116.9492274587385, 118.16074567836519, 119.18497548708795, 120.04810876261652, 120.77352815196981, 121.38182790207875, 121.89094985353891, 122.31638758099915, 122.6714244963338, 122.9673822552567, 123.21386397019609, 123.41898278254268, 123.58957065488238, 123.73136519024158, 123.84917421274221, 123.94701903496814, 124.02825801299717, 124.09569231686116],
        "short_factor": [0.9977997200264581, 1.014658295992452, 1.0349680404997148, 1.059429246056193, 1.0888815016813513, 1.1243301355211495, 1.166977103606075, 1.2182568066927284, 1.2798772354275727, 1.3538666751582975, 1.4426259039919596, 1.5489853358570191, 1.6762658237220625, 1.8283407612492941, 2.0096956085876183, 2.225478927469756, 2.481536379650452, 2.784415934557119, 3.1413289096347365, 3.560047844772632, 4.048719380066383, 4.752651957515948, 5.590913044973868, 6.584005926629993, 7.7532214876576155, 9.119754865903639, 10.704443927019176, 12.524994176518703, 14.59739595363613, 16.93214476166354, 19.53823297353041, 22.417131025031697, 25.568260840911098, 28.991144156566317, 32.68408069090375, 36.65174474170465, 40.90396065611201, 45.4664008671033, 50.37147343433591, 55.6804490772103, 61.470816952306556, 67.8622707390618, 75.00516023410414, 83.11898235973767, 92.50044360202462, 103.57086856690864, 116.9492274587385, 118.16074567836519, 119.18497548708795, 120.04810876261652, 120.77352815196981, 121.38182790207875, 121.89094985353891, 122.31638758099915, 122.6714244963338, 122.9673822552567, 123.21386397019609, 123.41898278254268, 123.58957065488238, 123.73136519024158, 123.84917421274221, 123.94701903496814, 124.02825801299717, 124.09569231686116],
        "original_max_position_embeddings": 32768
    }
}

# InfLLM v2 configuration
INFLLM_V2_CONFIG = {
    "sparse_config": {
        "kernel_size": 32,
        "kernel_stride": 16,
        "init_blocks": 1,
        "block_size": 64,
        "window_size": 2048,
        "topk": 64,
        "use_nope": False,
        "dense_len": 8192
    }
}

# Pydantic models for API
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = 256
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "organization"
    permission: List[Dict] = []

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

# Global variables
app = FastAPI()
model = None
tokenizer = None

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def check_or_download_model(path):
    if os.path.exists(path):
        return path
    else:
        cache_dir = snapshot_download(path)
        return cache_dir

def get_model_paths(path_prefix, config):
    if config['test_minicpm4']:
        if config['apply_eagle_quant']:
            eagle_path = f"{path_prefix}/MiniCPM4-8B-Eagle-FRSpec-QAT-cpmcu"
        else:
            eagle_path = f"{path_prefix}/MiniCPM4-8B-Eagle-FRSpec"
    else:
        eagle_path = f"{path_prefix}/EAGLE-LLaMA3-Instruct-8B"
    
    if not config['apply_quant']:
        if config['test_minicpm4']:
            base_path = f"{path_prefix}/MiniCPM4-8B"
        else:
            base_path = f"{path_prefix}/Meta-Llama-3-8B-Instruct"
    else:
        base_path = f"{path_prefix}/MiniCPM4-8B-marlin-cpmcu"

    eagle_path = check_or_download_model(eagle_path)
    base_path = check_or_download_model(base_path)
    
    return eagle_path, base_path

def create_model(eagle_path, base_path, config):
    common_kwargs = {
        'dtype': config['dtype'],
        'chunk_length': config['chunk_length'],
        'cuda_graph': config['cuda_graph'],
        'apply_sparse': config['apply_sparse'],
        'sink_window_size': config['sink_window_size'],
        'block_window_size': config['block_window_size'],
        'sparse_topk_k': config['sparse_topk_k'],
        'sparse_switch': config['sparse_switch'],
        'apply_compress_lse': config['apply_compress_lse'],
        'memory_limit': config['memory_limit'],
        'temperature': config['temperature'],
        'random_seed': config['random_seed'],
        'minicpm4_yarn': config['minicpm4_yarn'],
        'use_enter': config['use_enter'],
        'use_decode_enter': config['use_decode_enter']
    }
    
    eagle_kwargs = {
        'num_iter': config['eagle_num_iter'],
        'topk_per_iter': config['eagle_topk_per_iter'],
        'tree_size': config['eagle_tree_size'],
        'eagle_window_size': config['eagle_window_size'],
        'frspec_vocab_size': config['frspec_vocab_size'],
        'apply_eagle_quant': config['apply_eagle_quant'],
        'use_rope': config['test_minicpm4'],
        'use_input_norm': config['test_minicpm4'],
        'use_attn_norm': config['test_minicpm4']
    }
    
    if config['apply_quant']:
        if config['apply_eagle']:
            return W4A16GPTQMarlinLLM_with_eagle(eagle_path, base_path, **common_kwargs, **eagle_kwargs)
        else:
            return W4A16GPTQMarlinLLM(base_path, **common_kwargs)
    else:
        if config['apply_eagle']:
            return LLM_with_eagle(eagle_path, base_path, **common_kwargs, **eagle_kwargs)
        else:
            return LLM(base_path, **common_kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description='Start the CPM.cu API server')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port to bind the server to')
    parser.add_argument('--enable-rope', action='store_true',
                      help='Enable LongRoPE for extended context length (131072 tokens)')
    parser.add_argument('--rope-max-position', type=int, default=131072,
                      help='Maximum position embeddings for RoPE (default: 131072)')
    
    # Add InfLLM v2 arguments
    parser.add_argument('--enable-infllm', action='store_true',
                      help='Enable InfLLM v2 sparse attention mechanism')
    parser.add_argument('--kernel-size', type=int, default=32,
                      help='Semantic kernel size for InfLLM v2 (default: 32)')
    parser.add_argument('--kernel-stride', type=int, default=16,
                      help='Stride between adjacent kernels for InfLLM v2 (default: 16)')
    parser.add_argument('--init-blocks', type=int, default=1,
                      help='Number of initial blocks each query token attends to (default: 1)')
    parser.add_argument('--block-size', type=int, default=64,
                      help='Block size for key-value blocks (default: 64)')
    parser.add_argument('--window-size', type=int, default=2048,
                      help='Size of local sliding window (default: 2048)')
    parser.add_argument('--topk', type=int, default=64,
                      help='Number of most relevant key-value blocks to use (default: 64)')
    parser.add_argument('--use-nope', action='store_true',
                      help='Use NOPE technique in block selection')
    parser.add_argument('--dense-len', type=int, default=8192,
                      help='Length threshold for switching to sparse attention (default: 8192)')
    return parser.parse_args()

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    
    # Get command line arguments
    args = parse_args()
    
    # Initialize model and tokenizer
    config = DEFAULT_CONFIG.copy()
    eagle_path, base_path = get_model_paths("openbmb", config)
    
    # Load and apply model configuration
    model_config = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
    
    # Apply RoPE configuration if enabled
    if args.enable_rope:
        print(f"Enabling LongRoPE with max position {args.rope_max_position}")
        rope_config = ROPE_CONFIG.copy()
        rope_config["rope_scaling"]["original_max_position_embeddings"] = args.rope_max_position
        model_config.rope_scaling = rope_config["rope_scaling"]
        print("RoPE configuration applied successfully")
    
    # Apply InfLLM v2 configuration if enabled
    if args.enable_infllm:
        print("Enabling InfLLM v2 sparse attention mechanism")
        sparse_config = INFLLM_V2_CONFIG["sparse_config"].copy()
        sparse_config.update({
            "kernel_size": args.kernel_size,
            "kernel_stride": args.kernel_stride,
            "init_blocks": args.init_blocks,
            "block_size": args.block_size,
            "window_size": args.window_size,
            "topk": args.topk,
            "use_nope": args.use_nope,
            "dense_len": args.dense_len
        })
        model_config.sparse_config = sparse_config
        print("InfLLM v2 configuration applied successfully")
    
    # Initialize tokenizer with the configuration
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, config=model_config)
    model = create_model(eagle_path, base_path, config)
    
    # Initialize model
    model.init_storage()
    if config['apply_eagle'] and config['frspec_vocab_size'] > 0:
        fr_path = f'{eagle_path}/freq_{config["frspec_vocab_size"]}.pt'
        if os.path.exists(fr_path):
            with open(fr_path, 'rb') as f:
                token_id_remap = torch.tensor(torch.load(f, weights_only=True), dtype=torch.int32, device="cpu")
        else:
            cache_dir = snapshot_download(
                os.path.dirname(fr_path),
                ignore_patterns=["*.bin", "*.safetensors"],
            )
            file_path = os.path.join(
                cache_dir, os.path.basename(fr_path)
            )
            token_id_remap = torch.tensor(torch.load(file_path, weights_only=True), dtype=torch.int32, device="cpu")
        model._load("token_id_remap", token_id_remap, cls="eagle")
    model.load_from_hf()

@app.get("/v1/models")
async def list_models():
    return ModelList(data=[
        ModelCard(id="minicpm4-8b"),
        ModelCard(id="minicpm4-8b-eagle"),
        ModelCard(id="minicpm4-8b-quant"),
        ModelCard(id="minicpm4-8b-eagle-quant"),
    ])

def create_chat_completion_chunk(text: str, index: int = 0, finish_reason: Optional[str] = None):
    choice_data = {
        "index": index,
        "delta": {"content": text},
        "finish_reason": finish_reason
    }
    
    response = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "minicpm4-8b",
        "choices": [choice_data]
    }
    
    return f"data: {json.dumps(response)}\n\n"

async def generate_stream_response(messages: List[Message], max_tokens: int):
    # Prepare input
    prompt = tokenizer.apply_chat_template([m.dict() for m in messages], tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
    
    # Generate
    try:
        for result in model.generate(input_ids, max_tokens, teminators=[tokenizer.eos_token_id], use_stream=True):
            text = result['text']
            is_finished = result['is_finished']
            
            if is_finished:
                yield create_chat_completion_chunk(text, finish_reason="stop")
                break
            else:
                yield create_chat_completion_chunk(text)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        if request.stream:
            return StreamingResponse(
                generate_stream_response(request.messages, request.max_tokens),
                media_type="text/event-stream"
            )
        else:
            # Prepare input
            prompt = tokenizer.apply_chat_template([m.dict() for m in messages], tokenize=False, add_generation_prompt=True)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
            
            # Generate
            tokens, decode_time, prefill_time = model.generate(
                input_ids, 
                request.max_tokens, 
                teminators=[tokenizer.eos_token_id], 
                use_stream=False
            )
            
            generated_text = tokenizer.decode(tokens)
            
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": input_ids.shape[1],
                    "completion_tokens": len(tokens),
                    "total_tokens": input_ids.shape[1] + len(tokens)
                }
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port) 