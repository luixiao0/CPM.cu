import torch
from llamacu.llama import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from triton.testing import do_bench

path = "../../models/MiniCPM-1B-sft-llama-format"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(path)
num_tokens = 1024
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).cuda()
input_ids = torch.randint(0, 32000, (1, num_tokens,), dtype=torch.int32).cuda()
num_tokens = input_ids.numel()
with torch.no_grad():
    last_hidden = model.model(input_ids)[0]
    if True:
        time = do_bench(lambda: model.model(input_ids), warmup=10, rep=1000)
        print(f"bench {time} ms")
    a = last_hidden.view(num_tokens, -1)
    print(a)

llm = LLM(path, dtype=dtype)
llm.load_from_hf()

model_offset = 2946100224//2
kvcache_offset = 2968120320//2

output_ids = torch.empty_like(input_ids)
position_ids = torch.arange(input_ids.numel(), dtype=torch.int32, device="cuda")
cache_length = torch.tensor([0], dtype=torch.int32, device="cuda")
llm.prefill(input_ids, position_ids, cache_length, output_ids)
if True:
    our_time = do_bench(lambda: llm.prefill(input_ids, position_ids, cache_length, output_ids), warmup=10, rep=1000)
    print(f"bench {our_time} ms")
our_last_hidden = llm.memory_pool.view(dtype)[model_offset:model_offset+num_tokens*llm.config.hidden_size].view(num_tokens,-1)
b = our_last_hidden
print(b)

print((a-b).abs().mean())
print(f"baseline prefill: {num_tokens / time * 1000} tok/s")
print(f"our prefill: {num_tokens / our_time * 1000} tok/s")