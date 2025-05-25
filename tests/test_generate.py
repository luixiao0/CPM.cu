import torch
from llamacu.llama import LLM
from llamacu.llama_w4a16_gptq_marlin import W4A16GPTQMarlinLLM
from llamacu.speculative import LLM_with_eagle
from llamacu.speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import W4A16GPTQMarlinLLM_with_eagle
from transformers import AutoTokenizer
import time
import numpy as np

sink_window_size = 1
block_window_size = 32
sparse_topk = 32 
apply_sparse = True
quant = False
# path = "/DATA/disk0/zhaoweilun/minicpm4/models/minicpm4_mupformat"
path = "/DATA/disk0/zhaoweilun/minicpm4/models/minicpm4_mupformat_transposed"
# path = "/DATA/disk0/zhaoweilun/minicpm4/models/minicpm4_marlin"
# path = "/home/test/test01/zwl/models/Meta-Llama-3-8B-Instruct"
# path = "/home/test/test01/zwl/models/Meta-Llama-3-8B-Instruct-GPTQ-Marlin"
eagle_path = ""
dtype = torch.float16
cuda_graph = True
chunk_length = 2048 # TODO minicpm4 change this to 1024 and test correctness
num_generate = 128
model_type = "base"

def make_input(digits, a = 2500, b = 4000):
    head = "There is a pass key hidden in the context. Find it and remember it. I will quiz you about it later. "
    before = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * a
    needle = f"The pass key is {digits}. Remember it. The pass key is {digits}"
    after = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * b
    query = "Now, give me the exact number of the pass key. The pass key is "
    return head + before + needle + after + query
# prompt = make_input(681725493, 2000, 4000) # 120k
prompt = make_input(681725493, 1000, 2000) # 60k
# prompt = make_input(681725493, 500, 1000) # 30k
# prompt = make_input(681725493, 400, 800) # 24k
# prompt = make_input(681725493, 300, 600) # 18k
# prompt = make_input(681725493, 200, 400) # 12k
# prompt = make_input(681725493, 200, 300) # 10k
# prompt = make_input(681725493, 100, 300) # 8k
# prompt = make_input(681725493, 100, 200) # 6k
# prompt = "Beijing is the"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
num_tokens = input_ids.numel()

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

teminators = [tokenizer.eos_token_id]

if quant:
    if model_type == "eagle":
        llm = W4A16GPTQMarlinLLM_with_eagle(eagle_path, path, dtype=dtype, memory_limit=0.8, num_iter=3, tree_size=30, chunk_length=chunk_length, cuda_graph=cuda_graph)
        our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)
    else:
        llm = W4A16GPTQMarlinLLM(path, dtype=dtype, memory_limit=0.8, chunk_length=chunk_length, cuda_graph=cuda_graph)
        our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)
else:
    if model_type == "eagle":
        llm = LLM_with_eagle(eagle_path, path, dtype=dtype, memory_limit=0.8, num_iter=3, tree_size=30, chunk_length=chunk_length, cuda_graph=cuda_graph)
        our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)
    else:
        llm = LLM(path, dtype=dtype, memory_limit=0.9, chunk_length=chunk_length, cuda_graph=cuda_graph, apply_sparse=apply_sparse, sink_window_size=sink_window_size, block_window_size=block_window_size, sparse_topk_k=sparse_topk)
        our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)

llm.init_storage()
llm.load_from_hf()

torch.cuda.synchronize()
st = time.time()
gen_result = our_generate()
torch.cuda.synchronize()
et = time.time()
decode_time = gen_result[-1]
prefill_time = et - st - decode_time

if model_type == "medusa" or model_type == "eagle":
    print(tokenizer.decode(gen_result[0]).strip())
    print("Mean acc:", np.mean(gen_result[1]))
else:
    print(tokenizer.decode(gen_result[0]).strip())

print("prefill length:", input_ids.shape[1])
print("prefill time:", prefill_time)
print("prefill tokens/s:", input_ids.shape[1] / prefill_time)
print("decode length:", len(gen_result[0]))
print("decode time:", decode_time)
print("decode tokens/s:", len(gen_result[0]) / decode_time)
