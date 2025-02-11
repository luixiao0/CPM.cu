import os
import torch
from llamacu.llama_w4a16_gptq_marlin import W4A16GPTQMarlinLLM
from llamacu.speculative.medusa_base_w4a16_gptq_marlin import W4A16GPTQMarlinLLM_with_medusa
from llamacu.speculative.medusa_choices import *
from llamacu.speculative.eagle_base_w4a16_marlin_gptq import W4A16GPTQMarlinLLM_with_eagle
# from llamacu.medusa_w8a8 import W8A8LLM_with_medusa
from transformers import AutoTokenizer
from triton.testing import do_bench

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# path = "../../models/MiniCPM-1B-sft-llama-format"
# path = "../../models/Llama-2-7b-hf"
# path = "/home/ydzhang/checkpoints/deepcompress/Meta-Llama-3-8B-Instruct-w4a8-gchn-pileval"
# path = "/home/ydzhang/checkpoints/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ"
path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-mse-gptq_marlin"
# path = "/home/ydzhang/checkpoints/GPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-marlin"
# path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-4bit-128g-pileval-mse-marlin"
# path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-4bit-128g-pileval-mse"
# quantized_model_dir = "/home/ydzhang/checkpoints/GPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-marlin"
medusa_path = "/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full"
eagle_path = "/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-8B"
# eagle_path = "/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-70B"
dtype = torch.float16
cuda_graph = True
num_generate = 512
model_type = ["base", "medusa", "eagle"][1]
Bench = True

# prompt = "Beijing is the"

messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]

question = "Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point."
messages.append({
                "role": "user",
                "content": question
            })

tokenizer = AutoTokenizer.from_pretrained(path)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda().int()
num_tokens = input_ids.numel()
print(num_tokens)

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)
teminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

if model_type == "medusa":
    llm = W4A16GPTQMarlinLLM_with_medusa(medusa_path, path, dtype=dtype, memory_limit=0.8, medusa_num_heads=3, medusa_choices=eval('mc_sim_7b_63_3head'), cuda_graph=cuda_graph)
    our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)
elif model_type == "eagle":
    llm = W4A16GPTQMarlinLLM_with_eagle(eagle_path, path, dtype=dtype, memory_limit=0.8, num_iter=6, tree_size=60, cuda_graph=cuda_graph)
    our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)
else:
    llm = W4A16GPTQMarlinLLM(path, dtype=dtype, memory_limit=0.5, cuda_graph=cuda_graph)
    our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)

llm.init_storage()
llm.load_from_hf()
print("Model loaded")

gen_result = our_generate()
gen_result1 = our_generate()
if model_type == "eagle" or model_type == "medusa":
    import numpy as np
    m_output = our_generate()
    print(tokenizer.decode(m_output[0]))
    print(len(m_output[0]))
    print("acc:", m_output[1])
    print("Mean acc:", np.mean(m_output[1]))
else:
    out_str = tokenizer.decode(gen_result)
    print(out_str)
    print("=========================================")
    out_str1 = tokenizer.decode(gen_result1)
    print(out_str1)
    print("=========================================")
    print(out_str1 == out_str)
# if Bench:
#     print("decode speed:", f"{num_generate / do_bench(our_generate, warmup=10, rep=1000) * 1000} tok/s")
