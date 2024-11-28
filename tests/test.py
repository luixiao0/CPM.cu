import torch
from llamacu.llama import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "../../models/MiniCPM-1B-sft-llama-format"
prompt = "Hello, world!"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
input_ids = tokenizer.encode(prompt, return_tensors="pt")
num_tokens = input_ids.numel()
with torch.no_grad():
    input_embeds = model.get_input_embeddings()(input_ids)
    print(input_embeds)
    for layer in model.model.layers:
        # attn_norm = layer.input_layernorm(input_embeds)
        # q = layer.self_attn.q_proj(attn_norm)
        # k = layer.self_attn.k_proj(attn_norm)
        # v = layer.self_attn.v_proj(attn_norm)
        ffn_norm = layer.post_attention_layernorm(input_embeds)
        gate = layer.mlp.gate_proj(ffn_norm)
        up = layer.mlp.up_proj(ffn_norm)
        gate = torch.nn.functional.silu(gate) * up
        # down = layer.mlp.down_proj(gate)
        # input_embeds = down + input_embeds
        break
    # print(q)
    # print(k)
    # print(v)
    print(ffn_norm)
    print(gate)
    print(up)

llm = LLM(path, dtype=dtype)
llm.load_from_hf()

model_offset = 2946100224//2

llm.generate(prompt)
chunk_size = 1024
our_input_embeds = llm.memory_pool.view(dtype)[model_offset:model_offset+chunk_size*llm.config.hidden_size]
# our_attn_norm = llm.memory_pool.view(dtype)[model_offset+chunk_size*llm.config.hidden_size:model_offset+chunk_size*llm.config.hidden_size*2]
# our_q = llm.memory_pool.view(dtype)[model_offset+chunk_size*llm.config.hidden_size*2:model_offset+chunk_size*llm.config.hidden_size*3]
# our_k = llm.memory_pool.view(dtype)[model_offset+chunk_size*llm.config.hidden_size*3:model_offset+chunk_size*llm.config.hidden_size*3+chunk_size*llm.config.num_key_value_heads*llm.config.head_dim]
# our_v = llm.memory_pool.view(dtype)[model_offset+chunk_size*llm.config.hidden_size*3+chunk_size*llm.config.num_key_value_heads*llm.config.head_dim:model_offset+chunk_size*llm.config.hidden_size*3+chunk_size*llm.config.num_key_value_heads*llm.config.head_dim*2]
our_ffn_norm = llm.memory_pool.view(dtype)[model_offset+chunk_size*llm.config.hidden_size:model_offset+chunk_size*llm.config.hidden_size*2]
our_gate = llm.memory_pool.view(dtype)[model_offset+chunk_size*llm.config.hidden_size*2:model_offset+chunk_size*llm.config.hidden_size*2+chunk_size*llm.config.intermediate_size]
our_up = llm.memory_pool.view(dtype)[model_offset+chunk_size*llm.config.hidden_size*2+chunk_size*llm.config.intermediate_size:model_offset+chunk_size*llm.config.hidden_size*2+chunk_size*llm.config.intermediate_size*2]
print(our_input_embeds.view(chunk_size,-1)[:5])
# print(our_attn_norm.view(chunk_size,-1)[:5])
# print(our_q.view(chunk_size,-1)[:5])
# print(our_k.view(chunk_size,-1)[:5])
# print(our_v.view(chunk_size,-1)[:5])
print(our_ffn_norm.view(chunk_size,-1)[:5])
print(our_gate.view(chunk_size,-1)[:6])
print(our_up.view(chunk_size,-1)[:6])