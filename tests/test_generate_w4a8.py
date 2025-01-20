import os
import torch
from llamacu.llama_w4a8_per_chn import W4A8PerChnLLM
from llamacu.medusa_w4a8_per_chn import W4A8PerChnLLM_with_medusa
# from llamacu.medusa_w8a8 import W8A8LLM_with_medusa
from transformers import AutoTokenizer
from triton.testing import do_bench
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# path = "../../models/MiniCPM-1B-sft-llama-format"
# path = "../../models/Llama-2-7b-hf"
path = "/home/ydzhang/checkpoints/deepcompress/Meta-Llama-3-8B-Instruct-w4a8-gchn-pileval"
medusa_path = "/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full-rotation"
dtype = torch.float16
cuda_graph = True
num_generate = 100
use_medusa = True
Bench = True

prompt = "Beijing is the"
tokenizer = AutoTokenizer.from_pretrained(path)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
num_tokens = input_ids.numel()
num_generate = 512

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

if use_medusa:
    llm = W4A8PerChnLLM_with_medusa(medusa_path, path, dtype=dtype, memory_limit=0.8, medusa_num_heads=3, medusa_choices='mc_sim_7b_61')
    our_generate = lambda: llm.generate(input_ids, num_generate)
else:
    llm = W4A8PerChnLLM(path, dtype=dtype, memory_limit=0.8, cuda_graph=cuda_graph)
    our_generate = lambda: llm.generate(input_ids, num_generate)

llm.init_storage()
llm.load_from_hf()

gen_result = our_generate()
if use_medusa:
    print(tokenizer.decode(gen_result[0]))
    print("mean acc:", np.mean(gen_result[1]))
else:
    print(tokenizer.decode(gen_result))
if Bench:
    print("decode speed:", f"{num_generate / do_bench(our_generate, warmup=10, rep=1000) * 1000} tok/s")
