import torch
from llamacu.std_llama import std_LLM
# from llamacu.medusa import LLM_with_medusa
from transformers import AutoTokenizer
from triton.testing import do_bench

# path = "../../models/MiniCPM-1B-sft-llama-format"
# path = "../../models/Llama-2-7b-hf"
# path = "/home/ydzhang/checkpoints/GPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-marlin"
# path = "/home/ydzhang/checkpoints/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ"
path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-4bit-128g-pileval-mse"
medusa_path = "../../models/medusa/medusa-vicuna-7b-v1.3"
dtype = torch.float16
cuda_graph = True
num_generate = 300
use_medusa = False
Bench = True

messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]

question = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."

messages.append({
                "role": "user",
                "content": question
            })
tokenizer = AutoTokenizer.from_pretrained(path)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
num_tokens = input_ids.numel()
num_generate = 100

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

if use_medusa:
    llm = LLM_with_medusa(medusa_path, path, dtype=dtype, memory_limit=0.4)
    our_generate = lambda: llm.generate(input_ids, num_generate, output_avg_accept_length=True)
else:
    llm = std_LLM(path, dtype=dtype, memory_limit=0.9, group_size=128, bits=4, use_marlin=True)
    our_generate = lambda: llm.generate(input_ids, num_generate)

llm.init_storage()
llm.load_from_hf()
gen_result = our_generate()
gen_result1 = our_generate()
print(gen_result)
print("=================================")
print(gen_result1)
print("=================================")
print(gen_result == gen_result1)
# print(our_generate())
# if Bench:
#     print("decode speed:", f"{num_generate / do_bench(our_generate, warmup=10, rep=1000) * 1000} tok/s")
