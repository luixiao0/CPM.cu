import torch
from llamacu.llama import LLM
from llamacu.medusa import LLM_with_medusa
from transformers import AutoTokenizer
from triton.testing import do_bench

# path = "../../models/MiniCPM-1B-sft-llama-format"
# path = "../../models/Llama-2-7b-hf"
path = "../../models/vicuna-7b-v1.3"
medusa_path = "../../models/medusa/medusa-vicuna-7b-v1.3"
dtype = torch.float16
cuda_graph = True
num_generate = 100
use_medusa = True

prompt = "Beijing is the"
tokenizer = AutoTokenizer.from_pretrained(path)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
num_tokens = input_ids.numel()
num_generate = 100

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

if use_medusa:
    llm = LLM_with_medusa(medusa_path, path, dtype=dtype, memory_limit=0.4)
    our_generate = lambda: llm.generate(input_ids, num_generate, output_avg_accept_length=True)
else:
    llm = LLM(path, dtype=dtype, memory_limit=0.4)
    our_generate = lambda: llm.generate(input_ids, num_generate)

llm.init_storage()
llm.load_from_hf()

print(our_generate())
print("our generate:", f"{num_generate / do_bench(our_generate, warmup=10, rep=1000) * 1000} tok/s")
