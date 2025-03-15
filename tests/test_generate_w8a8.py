import torch
from llamacu.llama_w8a8 import W8A8LLM
from llamacu.speculative.medusa_base_quant.medusa_base_w8a8 import W8A8LLM_with_medusa
from llamacu.speculative.medusa_choices import *
from llamacu.speculative.eagle_base_quant.eagle_base_w8a8 import W8A8LLM_with_eagle
from transformers import AutoTokenizer
from triton.testing import do_bench
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# path = "../../models/MiniCPM-1B-sft-llama-format"
# path = "../../models/Llama-2-7b-hf"
path = "/home/ydzhang/checkpoints/deepcompress/Meta-Llama-3-8B-Instruct-w8a8-gchn-pileval-neuralmagic"
medusa_path = "/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full"
eagle_path = "/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-8B"
dtype = torch.float16
cuda_graph = True
num_generate = 512
model_type = ["base", "medusa", "eagle"][2]
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
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()
num_tokens = input_ids.numel()
num_generate = 512

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

teminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

if model_type == "medusa":
    llm = W8A8LLM_with_medusa(medusa_path, path, dtype=dtype, memory_limit=0.8, medusa_num_heads=3, medusa_choices=eval('mc_sim_7b_31'))
    our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)
elif model_type == "eagle":
    llm = W8A8LLM_with_eagle(eagle_path, path, dtype=dtype, memory_limit=0.4)
    our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)
else:
    llm = W8A8LLM(path, dtype=dtype, memory_limit=0.8)
    our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)

llm.init_storage()
llm.load_from_hf()

gen_result = our_generate()
gen_result1 = our_generate()

if model_type == "medusa" or model_type == "eagle":
    import numpy as np
    m_output = our_generate()
    print(tokenizer.decode(m_output[0]))
    print("Mean acc:", np.mean(m_output[1]))
else:
    # print(tokenizer.decode(our_generate()))
    out_str = tokenizer.decode(gen_result[0])
    print(out_str)
    print("=========================================")
    out_str1 = tokenizer.decode(gen_result1[0])
    print(out_str1)
    print("=========================================")
    print(out_str1 == out_str)
