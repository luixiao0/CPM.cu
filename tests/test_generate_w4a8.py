import os
import torch
from llamacu.llama_w4a8_per_chn import W4A8PerChnLLM
from llamacu.speculative.medusa_base_w4a8_per_chn import W4A8PerChnLLM_with_medusa
from llamacu.speculative.medusa_choices import *
# from llamacu.medusa_w8a8 import W8A8LLM_with_medusa
from transformers import AutoTokenizer
from triton.testing import do_bench

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# path = "../../models/MiniCPM-1B-sft-llama-format"
# path = "../../models/Llama-2-7b-hf"
path = "/home/ydzhang/checkpoints/deepcompress/Meta-Llama-3-8B-Instruct-w4a8-gchn-pileval"
medusa_path = "/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full-rotation"
dtype = torch.float16
cuda_graph = True
num_generate = 256
use_medusa = True
Bench = True

prompt = "Beijing is the"
tokenizer = AutoTokenizer.from_pretrained(path)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
num_tokens = input_ids.numel()

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

if use_medusa:
    llm = W4A8PerChnLLM_with_medusa(medusa_path, path, dtype=dtype, memory_limit=0.8, medusa_num_heads=3, medusa_choices=eval('mc_sim_7b_31'), cuda_graph=cuda_graph)
    our_generate = lambda: llm.generate(input_ids, num_generate)
else:
    llm = W4A8PerChnLLM(path, dtype=dtype, memory_limit=0.8, cuda_graph=cuda_graph)
    our_generate = lambda: llm.generate(input_ids, num_generate)

llm.init_storage()
llm.load_from_hf()

gen_result = our_generate()
if use_medusa:
    import numpy as np
    m_output = our_generate()
    print(tokenizer.decode(m_output[0]))
    print(len(m_output[0]))
    print("acc:", m_output[1])
    print("Mean acc:", np.mean(m_output[1]))
else:
    print(tokenizer.decode(gen_result))
if Bench:
    print("decode speed:", f"{num_generate / do_bench(our_generate, warmup=10, rep=1000) * 1000} tok/s")
