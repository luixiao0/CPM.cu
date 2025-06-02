import torch
from llamacu.llama import LLM
from llamacu.llama_w4a16_gptq_marlin import W4A16GPTQMarlinLLM
from llamacu.speculative import LLM_with_eagle
from llamacu.speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import W4A16GPTQMarlinLLM_with_eagle
from transformers import AutoTokenizer
import time
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate text using LLM models')
parser.add_argument('--path_prefix', '-pp', type=str, default='/cache/copys/217/data1/liyx/Models', 
                    help='Path prefix for model directories (default: /cache/copys/217/data1/liyx/Models)')
args = parser.parse_args()

test_minicpm4 = True
apply_eagle = True
apply_quant = True # TODO: eagle+quant+sparse memcheck failed at build_dynamic_tree, only quant memcheck get incorrect result
apply_sparse = True # TODO: Maybe lead to illegal memory access

apply_compress_lse = True
num_generate = 256
sink_window_size = 1
# block_window_size = 2048
# sparse_topk_k = 0
block_window_size = 32
sparse_topk_k = 64
eagle_window_size = 64 * 128
frspec_vocab_size = 8192
chunk_length = 384 # TODO minicpm4 change this to 1024 and test correctness
cuda_graph = False

if not test_minicpm4:
    print(f"test_minicpm4 is False, set apply_sparse to False")
    apply_sparse = False

dtype = torch.float16
model_type = "base" if not apply_eagle else "eagle"

path_prefix = args.path_prefix
if test_minicpm4:
    eagle_path = f"{path_prefix}/job_35949"
    # eagle_path = "/data1/liyx/job_35949_llamaformat"
else:
    eagle_path = f"{path_prefix}/EAGLE-LLaMA3-Instruct-8B"

if not apply_quant:
    if test_minicpm4:   
        # path = "/DATA/disk0/zhaoweilun/minicpm4/models/minicpm4_mupformat"
        path = f"{path_prefix}/job_33952_step_17300"
        # path = f"{path_prefix}/job_33952_step_17300_llamaformat"
    else:
        path = f"{path_prefix}/Meta-Llama-3-8B-Instruct"
else:
    path = f"{path_prefix}/minicpm4_marlin"

def make_input(digits, a = 2500, b = 4000):
    head = "There is a pass key hidden in the context. Find it and remember it. I will quiz you about it later. "
    before = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * a
    needle = f"The pass key is {digits}. Remember it. The pass key is {digits}"
    after = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * b
    query = "Now, give me the exact number of the pass key. The pass key is "
    return head + before + needle + after + query

prompt = None
prompt_content = "北京有哪些好玩的地方"
# with open("prompt.txt", "r") as f:
#     prompt_content = f.read()

prompt = make_input(681725493, 2000, 4000) # 120k
# prompt = make_input(681725493, 1500, 3000) # 90k
# prompt = make_input(681725493, 1000, 2000) # 60k
# prompt = make_input(681725493, 500, 1000) # 30k
# prompt = make_input(681725493, 250, 500) # 15k
# prompt = make_input(681725493, 150, 250) # 8k
# prompt = make_input(681725493, 10, 50)
# prompt = "Beijing is the"

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

if prompt is None:
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt_content}], tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
print(f"Input token number is {input_ids.shape[1]}")
print(f"input_ids: {input_ids}")
num_tokens = input_ids.numel()

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

# teminators = [tokenizer.eos_token_id]
teminators = []

if apply_quant:
    if model_type == "eagle":
        llm = W4A16GPTQMarlinLLM_with_eagle(eagle_path, path, dtype=dtype, memory_limit=0.7, num_iter=2, topk_per_iter=16, tree_size=32, chunk_length=chunk_length, cuda_graph=cuda_graph, eagle_window_size=eagle_window_size, frspec_vocab_size=frspec_vocab_size, use_rope=test_minicpm4, use_input_norm=test_minicpm4, use_attn_norm=test_minicpm4, apply_sparse=apply_sparse, sink_window_size=sink_window_size, block_window_size=block_window_size, sparse_topk_k=sparse_topk_k, apply_compress_lse=apply_compress_lse)
        our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)
    else:
        llm = W4A16GPTQMarlinLLM(path, dtype=dtype, memory_limit=0.7, chunk_length=chunk_length, cuda_graph=cuda_graph, apply_sparse=apply_sparse, sink_window_size=sink_window_size, block_window_size=block_window_size, sparse_topk_k=sparse_topk_k, apply_compress_lse=apply_compress_lse)
        our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)
else:
    if model_type == "eagle":
        llm = LLM_with_eagle(eagle_path, path, dtype=dtype, memory_limit=0.9, num_iter=2, topk_per_iter=16, tree_size=32, chunk_length=chunk_length, cuda_graph=cuda_graph, eagle_window_size=eagle_window_size, frspec_vocab_size=frspec_vocab_size, use_rope=test_minicpm4, use_input_norm=test_minicpm4, use_attn_norm=test_minicpm4, apply_sparse=apply_sparse, sink_window_size=sink_window_size, block_window_size=block_window_size, sparse_topk_k=sparse_topk_k, apply_compress_lse=apply_compress_lse)
        our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)
    else:
        llm = LLM(path, dtype=dtype, memory_limit=0.9, chunk_length=chunk_length, cuda_graph=cuda_graph, apply_sparse=apply_sparse, sink_window_size=sink_window_size, block_window_size=block_window_size, sparse_topk_k=sparse_topk_k, apply_compress_lse=apply_compress_lse)
        our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)

llm.init_storage()
if model_type == "eagle" and frspec_vocab_size > 0:
    with open(f'fr_index/MiniCPM4-8B/freq_{frspec_vocab_size}.pt', 'rb') as f:
        token_id_remap = torch.tensor(torch.load(f, weights_only=True), dtype=torch.int32, device="cpu")
    llm._load("token_id_remap", token_id_remap, cls="eagle")
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
