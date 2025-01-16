import torch
from llamacu.llama import LLM
from llamacu.speculative import LLM_with_medusa, LLM_with_eagle
from transformers import AutoTokenizer
from triton.testing import do_bench

# path = "../../models/vicuna-7b-v1.3"
# medusa_path = "../../models/medusa/medusa-vicuna-7b-v1.3"
# eagle_path = "../../models/eagle-vicuna-7b-v1.3"
path = "../../models/Meta-Llama-3-8B-Instruct"
medusa_path = ""
eagle_path = "../../models/EAGLE-LLaMA3-Instruct-8B"
dtype = torch.float16
cuda_graph = True
num_generate = 100
model_type = ["base", "medusa", "eagle"][2]
Bench = True

prompt = "Beijing is the"
tokenizer = AutoTokenizer.from_pretrained(path)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
num_tokens = input_ids.numel()
num_generate = 100

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

if model_type == "medusa":
    llm = LLM_with_medusa(medusa_path, path, dtype=dtype, memory_limit=0.4)
    our_generate = lambda: llm.generate(input_ids, num_generate, output_avg_accept_length=True)
elif model_type == "eagle":
    llm = LLM_with_eagle(eagle_path, path, dtype=dtype, memory_limit=0.4)
    our_generate = lambda: llm.generate(input_ids, num_generate, output_avg_accept_length=True)
else:
    llm = LLM(path, dtype=dtype, memory_limit=0.4)
    our_generate = lambda: llm.generate(input_ids, num_generate)

llm.init_storage()
llm.load_from_hf()

if Bench:
    print("decode speed:", f"{num_generate / do_bench(our_generate, warmup=10, rep=1000) * 1000} tok/s")

print(our_generate())