import torch
from llamacu.llama import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "../../models/MiniCPM-1B-sft-llama-format"
prompt = "Hello, world!"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).cuda()
input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
num_tokens = input_ids.numel()
with torch.no_grad():
    input_embeds = model.get_input_embeddings()(input_ids)
    # print(input_embeds)
    for layer in model.model.layers:
        # attn_norm = layer.input_layernorm(input_embeds)
        # q = layer.self_attn.q_proj(attn_norm)
        # k = layer.self_attn.k_proj(attn_norm)
        # v = layer.self_attn.v_proj(attn_norm)
        # pos = torch.arange(num_tokens, dtype=torch.int32, device=v.device).view(1, -1)
        # cos, sin = layer.self_attn.rotary_emb(v, pos)
        # from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
        # q = q.view(1, 5, 24, 64).transpose(1, 2)
        # k = k.view(1, 5, 8, 64).transpose(1, 2)
        # v = v.view(1, 5, 8, 64).transpose(1, 2)
        # q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # q = q.transpose(1, 2).reshape(1, 5, 24, 64)
        # k = k.transpose(1, 2).reshape(1, 5, 8, 64)
        # v = v.transpose(1, 2).reshape(1, 5, 8, 64)
        # k_cache = torch.empty_like(k)
        # v_cache = torch.empty_like(v)
        # from flash_attn import flash_attn_with_kvcache
        # attn_output, softmax_lse = flash_attn_with_kvcache(
        #     q,
        #     k_cache, v_cache,
        #     k, v,
        #     causal=True,
        #     cache_seqlens=0,
        #     num_splits=1,
        #     return_softmax_lse=True
        # )
        # o = layer.self_attn.o_proj(attn_output.view(1, 5, -1))
        # input_embeds = o + input_embeds
        
        ffn_norm = layer.post_attention_layernorm(input_embeds)
        gate = layer.mlp.gate_proj(ffn_norm)
        up = layer.mlp.up_proj(ffn_norm)
        gate = torch.nn.functional.silu(gate) * up
        down = layer.mlp.down_proj(gate)
        input_embeds = down + input_embeds
        break
    # print(attn_norm)
    # print(q.transpose(1,2).reshape(5, -1))
    # print(k.transpose(1,2).reshape(5, -1))
    # print(v.transpose(1,2).reshape(5, -1))
    # print(k_cache.view(5, -1))
    # print(v_cache.view(5, -1))
    # print(attn_output.reshape(5, -1))
    # print(softmax_lse.view(5, -1))
    print(input_embeds.view(5, -1))
    # print(ffn_norm)
    print(gate)
    # print(up)
    # print(down)
    # print(input_embeds)

llm = LLM(path, dtype=dtype)
llm.load_from_hf()

model_offset = 2946100224//2
kvcache_offset = 2968120320//2
cache_length = 611962

llm.generate(prompt)
chunk_size = 1024
qs = chunk_size*llm.config.hidden_size
ks = chunk_size*llm.config.num_key_value_heads*llm.config.head_dim
cs = cache_length*llm.config.num_key_value_heads*llm.config.head_dim
our_input_embeds = llm.memory_pool.view(dtype)[model_offset:model_offset+qs]
our_attn_norm = llm.memory_pool.view(dtype)[model_offset+qs:model_offset+qs*2]
our_q = llm.memory_pool.view(dtype)[model_offset+qs*2:model_offset+qs*3]
our_k = llm.memory_pool.view(dtype)[model_offset+qs*3:model_offset+qs*3+ks]
our_v = llm.memory_pool.view(dtype)[model_offset+qs*3+ks:model_offset+qs*3+ks*2]
our_attn_output = llm.memory_pool.view(dtype)[model_offset+qs:model_offset+qs*2]
our_softmax_lse = llm.memory_pool.view(dtype)[model_offset+qs*3+ks*2:model_offset+qs*3+ks*2+chunk_size*llm.config.num_attention_heads*2].view(torch.float32)
our_ffn_norm = llm.memory_pool.view(dtype)[model_offset+chunk_size*llm.config.hidden_size:model_offset+chunk_size*llm.config.hidden_size*2]
our_gate = llm.memory_pool.view(dtype)[model_offset+chunk_size*llm.config.hidden_size*2:model_offset+chunk_size*llm.config.hidden_size*2+chunk_size*llm.config.intermediate_size]
our_up = llm.memory_pool.view(dtype)[model_offset+chunk_size*llm.config.hidden_size*2+chunk_size*llm.config.intermediate_size:model_offset+chunk_size*llm.config.hidden_size*2+chunk_size*llm.config.intermediate_size*2]
our_k_cache = llm.memory_pool.view(dtype)[kvcache_offset:kvcache_offset+cs]
our_v_cache = llm.memory_pool.view(dtype)[kvcache_offset+cs:kvcache_offset+cs*2]
print(our_input_embeds.view(chunk_size,-1)[:5])
# print(our_attn_norm.view(chunk_size,-1)[:5])
# print(our_q.view(chunk_size,-1)[:5])
# print(our_k.view(chunk_size,-1)[:5])
# print(our_v.view(chunk_size,-1)[:5])
# print(our_k_cache.view(cache_length,-1)[:5])
# print(our_v_cache.view(cache_length,-1)[:5])
# print(our_attn_output.view(chunk_size,-1)[:5])
# print(our_softmax_lse.view(chunk_size,-1)[:5])
print(our_ffn_norm.view(chunk_size,-1)[:5])
# print(our_gate.view(chunk_size,-1)[:5])
# print(our_up.view(chunk_size,-1)[:5])
