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
    last_hidden = model.model(input_ids)[0]
    print(last_hidden.view(5, -1))

llm = LLM(path, dtype=dtype)
llm.load_from_hf()

model_offset = 2946100224//2
kvcache_offset = 2968120320//2

llm.generate(prompt)
chunk_size = 1024
qs = chunk_size*llm.config.hidden_size
our_last_hidden = llm.memory_pool.view(dtype)[model_offset:model_offset+qs].view(chunk_size,-1)[:5]
print(our_last_hidden)