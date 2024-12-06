import torch
from llamacu.llama import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from triton.testing import do_bench

# path = "../../models/MiniCPM-1B-sft-llama-format"
path = "../../models/Llama-2-7b-hf"
dtype = torch.bfloat16
Bench = True
cuda_graph = False

# seed
torch.manual_seed(0)

# prefill
num_tokens = 1024
chunk_length = num_tokens #//2 if want to test chunking
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype).cuda()
input_ids = torch.randint(0, 32000, (1, num_tokens,), dtype=torch.int32).cuda()
position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)
num_tokens = input_ids.numel()
with torch.no_grad():
    torch.cuda.nvtx.range_push("baseline prefill")
    last_hidden, past_key_values = model.model(input_ids, position_ids=position_ids, use_cache=True, return_dict=False)
    torch.cuda.nvtx.range_pop()
    if Bench:
        time = do_bench(lambda: model.model(input_ids, position_ids=position_ids, use_cache=True, return_dict=False), warmup=10, rep=1000)
    last_hidden = last_hidden.view(num_tokens, -1)

llm = LLM(path, dtype=dtype, chunk_length=chunk_length, memory_limit=0.4)
llm.load_from_hf()

our_last_hidden = torch.empty((chunk_length, llm.config.hidden_size), dtype=dtype, device="cuda")
torch.cuda.nvtx.range_push("our prefill")
llm.prefill(input_ids, position_ids, our_last_hidden)
torch.cuda.nvtx.range_pop()
if Bench:
    our_time = do_bench(lambda: llm.prefill(input_ids, position_ids, our_last_hidden), warmup=10, rep=1000)

print(last_hidden)
print(our_last_hidden)
print(f"prefill diff: {(last_hidden[-chunk_length:]-our_last_hidden).abs().mean()}")
print(f"baseline prefill: {num_tokens / time * 1000} tok/s")
print(f"our prefill: {num_tokens / our_time * 1000} tok/s")

# decode
num_verify = 1
input_ids = torch.randint(0, 32000, (1, num_verify), dtype=torch.int32, device="cuda")
position_ids = torch.tensor([[num_tokens]], dtype=torch.int32, device="cuda")
cache_length = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")

torch.cuda.nvtx.range_push("baseline decode")
last_hidden = model.model(input_ids, position_ids=position_ids, use_cache=True, past_key_values=past_key_values, return_dict=False)[0]
torch.cuda.nvtx.range_pop()
if Bench:
    time = do_bench(lambda: model.model(input_ids, position_ids=position_ids, use_cache=True, past_key_values=past_key_values, return_dict=False), warmup=10, rep=1000)

torch.cuda.nvtx.range_push("our decode")
llm.decode(input_ids, position_ids, cache_length, our_last_hidden, cuda_graph=cuda_graph)
torch.cuda.nvtx.range_pop()
if Bench:
    our_time = do_bench(lambda: llm.decode(input_ids, position_ids, cache_length, our_last_hidden, cuda_graph=cuda_graph), warmup=10, rep=1000)

print(f"decode diff: {(last_hidden-our_last_hidden[:1]).abs().mean()}")
print(f"baseline decode: {num_verify / time * 1000} tok/s")
print(f"our decode: {num_verify / our_time * 1000} tok/s")
