import torch
from llamacu.llama import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

# path = "../../models/MiniCPM-1B-sft-llama-format"
path = "../../models/Llama-2-7b-hf"
dtype = torch.float16
cuda_graph = True

prompt = "Beijing is the"
tokenizer = AutoTokenizer.from_pretrained(path)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
num_tokens = input_ids.numel()
num_generate = 100

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

llm = LLM(path, dtype=dtype, memory_limit=0.4)
llm.load_from_hf()

print(llm.generate(input_ids))

logits = llm.prefill(input_ids, position_ids)
token = logits[0].argmax(dim=-1)
print(tokenizer.decode(token.item()))

input_ids = torch.tensor([[token.item()]], dtype=torch.int32, device="cuda")
position_ids = torch.tensor([[num_tokens]], dtype=torch.int32, device="cuda")
cache_length = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")

logits = llm.decode(input_ids, position_ids, cache_length)
token = logits[0].argmax(dim=-1)
print(tokenizer.decode(token.item()))

for i in range(1, num_generate):
    input_ids[0][0] = token.item()
    position_ids[0][0] = num_tokens + i
    cache_length[0] = num_tokens + i

    logits = llm.decode(input_ids, position_ids, cache_length)
    token = logits[0].argmax(dim=-1)
    print(tokenizer.decode(token.item()))
