import torch
from llamacu.llama import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "../../models/MiniCPM-1B-sft-llama-format"
prompt = "Hello, world!"
dtype = torch.float32

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
input_ids = tokenizer.encode(prompt, return_tensors="pt")
with torch.no_grad():
    input_embeds = model.get_input_embeddings()(input_ids)
    norm = model.model.norm(input_embeds)
    output = norm @ model.lm_head.weight.T
    print(input_embeds)
    print(norm)
    print(output)

llm = LLM(path, dtype=dtype)
llm.load_from_hf()

model_offset = 225609216

llm.generate(prompt)
our_input_embeds = llm.memory_pool.view(dtype)[model_offset:model_offset+llm.config.hidden_size*5]
our_norm = llm.memory_pool.view(dtype)[model_offset+llm.config.hidden_size*5:model_offset+llm.config.hidden_size*5*2]
our_output = llm.memory_pool.view(dtype)[model_offset+llm.config.hidden_size*5*2:model_offset+llm.config.hidden_size*5*2+llm.config.vocab_size*5]
print(our_input_embeds)
print(our_norm)
print(our_output)
