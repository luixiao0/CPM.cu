import torch
from llamacu.llama import LLM
from llamacu.llama_w4a16_gptq_marlin import W4A16GPTQMarlinLLM
from llamacu.speculative import LLM_with_eagle
from llamacu.speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import W4A16GPTQMarlinLLM_with_eagle
from transformers import AutoTokenizer
import time
import numpy as np

test_minicpm4 = True
apply_eagle = True
apply_quant = True
apply_sparse = True

num_generate = 128
sink_window_size = 1
# block_window_size = 2048
# sparse_topk_k = 0
block_window_size = 32
sparse_topk_k = 64
eagle_window_size = 16

if not test_minicpm4:
    print(f"test_minicpm4 is False, set apply_sparse to False")
    apply_sparse = False

model_type = "base" if not apply_eagle else "eagle"
dtype = torch.float16
cuda_graph = False
chunk_length = 2048 # TODO minicpm4 change this to 1024 and test correctness

if test_minicpm4:
    eagle_path = "/data1/liyx/eagle_0526/job_35949"
    # eagle_path = "/data1/liyx/eagle_0526/job_35949_llamaformat"
else:
    eagle_path = "/data1/liyx/Models/EAGLE-LLaMA3-Instruct-8B"

if not apply_quant:
    if test_minicpm4:   
        # path = "/DATA/disk0/zhaoweilun/minicpm4/models/minicpm4_mupformat"
        path = "/data1/liyx/eagle_0526/job_33952_step_17300"
        # path = "/data1/liyx/eagle_0526/job_33952_step_17300_llamaformat"
    else:
        path = "/data1/liyx/Models/Meta-Llama-3-8B-Instruct"
else:
    path = "/DATA/disk0/zhaoweilun/minicpm4/models/minicpm4_marlin"

def make_input(digits, a = 2500, b = 4000):
    head = "There is a pass key hidden in the context. Find it and remember it. I will quiz you about it later. "
    before = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * a
    needle = f"The pass key is {digits}. Remember it. The pass key is {digits}"
    after = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * b
    query = "Now, give me the exact number of the pass key. The pass key is "
    return head + before + needle + after + query

prompt = None
prompt_content = "北京有哪些好玩的地方"
with open("prompt.txt", "r") as f:
    prompt_content = f.read()

prompt = make_input(681725493, 2000, 4000) # 120k
# prompt = make_input(681725493, 1500, 3000) # 90k
# prompt = make_input(681725493, 1000, 2000) # 60k
# prompt = make_input(681725493, 500, 1000) # 30k
# prompt = make_input(681725493, 10, 50)
# prompt = "Beijing is the"
# prompt = prompt_content

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

if prompt is None:
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt_content}], tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
print(f"input_ids: {input_ids}")
print(f"Input token number is {input_ids.shape[1]}")
num_tokens = input_ids.numel()

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

teminators = [tokenizer.eos_token_id]

if apply_quant:
    if model_type == "eagle":
        llm = W4A16GPTQMarlinLLM_with_eagle(eagle_path, path, dtype=dtype, memory_limit=0.5, num_iter=1, topk_per_iter=4, tree_size=4, chunk_length=chunk_length, cuda_graph=cuda_graph, eagle_window_size=eagle_window_size, use_rope=test_minicpm4, use_input_norm=test_minicpm4, use_attn_norm=test_minicpm4, apply_sparse=apply_sparse, sink_window_size=sink_window_size, block_window_size=block_window_size, sparse_topk_k=sparse_topk_k)
        our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)
    else:
        llm = W4A16GPTQMarlinLLM(path, dtype=dtype, memory_limit=0.45, chunk_length=chunk_length, cuda_graph=cuda_graph, apply_sparse=apply_sparse, sink_window_size=sink_window_size, block_window_size=block_window_size, sparse_topk_k=sparse_topk_k)
        our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)
else:
    if model_type == "eagle":
        llm = LLM_with_eagle(eagle_path, path, dtype=dtype, memory_limit=0.9, num_iter=1, topk_per_iter=4, tree_size=4, chunk_length=chunk_length, cuda_graph=cuda_graph, eagle_window_size=eagle_window_size, use_rope=test_minicpm4, use_input_norm=test_minicpm4, use_attn_norm=test_minicpm4, apply_sparse=apply_sparse, sink_window_size=sink_window_size, block_window_size=block_window_size, sparse_topk_k=sparse_topk_k)
        # llm = LLM(path, dtype=dtype, memory_limit=0.9, chunk_length=chunk_length, cuda_graph=cuda_graph, sink_window_size=sink_window_size, block_window_size=block_window_size, sparse_topk_k=sparse_topk_k)
        our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)
    else:
        llm = LLM(path, dtype=dtype, memory_limit=0.9, chunk_length=chunk_length, cuda_graph=cuda_graph, apply_sparse=apply_sparse, sink_window_size=sink_window_size, block_window_size=block_window_size, sparse_topk_k=sparse_topk_k)
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

print("\n[gen_result]")
if model_type == "medusa" or model_type == "eagle":
    print(tokenizer.decode(gen_result[0]).strip())
    print("Mean acc:", np.mean(gen_result[1]))
else:
    print(tokenizer.decode(gen_result[0]).strip())
print("\n")

print(f"prefill length: {input_ids.shape[1]}")
print(f"prefill time: {prefill_time:.2f} s")
print(f"prefill tokens/s: {input_ids.shape[1] / prefill_time:.2f}")
print(f"decode length: {len(gen_result[0])}")
print(f"decode time: {decode_time:.2f} s")
print(f"decode tokens/s: {len(gen_result[0]) / decode_time:.2f}")

llm.print_perf_summary()
