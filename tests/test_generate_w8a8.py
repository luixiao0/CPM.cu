import torch
from llamacu.llama_w8a8 import W8A8LLM
from llamacu.medusa_w8a8 import W8A8LLM_with_medusa
from transformers import AutoTokenizer
from triton.testing import do_bench

# path = "../../models/MiniCPM-1B-sft-llama-format"
# path = "../../models/Llama-2-7b-hf"
path = "/home/zhangyudi/checkpoints/neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8"
medusa_path = "/home/zhangyudi/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full"
dtype = torch.float16
cuda_graph = True
num_generate = 100
use_medusa = True
Bench = True

prompt = "Beijing is the"
tokenizer = AutoTokenizer.from_pretrained(path)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
num_tokens = input_ids.numel()
num_generate = 100

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

if use_medusa:
    llm = W8A8LLM_with_medusa(medusa_path, path, dtype=dtype, memory_limit=0.9, medusa_num_heads=3, medusa_choices='mc_sim_7b_61')
    our_generate = lambda: llm.generate(input_ids, num_generate)
else:
    llm = W8A8LLM(path, dtype=dtype, memory_limit=0.9)
    our_generate = lambda: llm.generate(input_ids, num_generate)

llm.init_storage()
llm.load_from_hf()

print(tokenizer.decode(our_generate()[0]))
if Bench:
    print("decode speed:", f"{num_generate / do_bench(our_generate, warmup=10, rep=1000) * 1000} tok/s")
