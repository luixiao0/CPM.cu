import os
import torch
from llamacu.speculative.cascade_eagle3_spec_quant.csc_eagle3_w4a16_gm_spec_w4a16_gm_ea3_head import CascadeEagle3W4A16GMSpecW4A16GMEa3Head
# from llamacu.medusa_w8a8 import W8A8LLM_with_medusa
from transformers import AutoTokenizer
from triton.testing import do_bench
from fastchat.model import get_conversation_template

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# path = "../../models/MiniCPM-1B-sft-llama-format"
# path = "../../models/Llama-2-7b-hf"
# path = "/home/ydzhang/checkpoints/deepcompress/Meta-Llama-3-8B-Instruct-w4a8-gchn-pileval"
# path = "/home/ydzhang/checkpoints/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ"
draft_path = "/home/ydzhang/checkpoints/GPTQModle_vllm_merge/Meta-Llama-3.1-8B-Instruct-4bit-128g-pileval_fl_128_2048_desc_static"
# path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-mse-gptq_marlin_merge"
path = "/home/ydzhang/checkpoints/GPTQModle_vllm_merge/Meta-Llama-3.3-70B-Instruct-4bit-128g-pileval_fl_128_2048_desc_static"
# path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-4bit-128g-pileval-mse-gptq_marlin"
# path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-4bit-128g-pileval-mse-gptq_marlin_merge"
# path = "/home/ydzhang/checkpoints/GPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-marlin"
# path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-4bit-128g-pileval-mse-marlin"
# path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-4bit-128g-pileval-mse"
# quantized_model_dir = "/home/ydzhang/checkpoints/GPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-marlin"
# medusa_path = "/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full"
eagle_path = "/home/ydzhang/checkpoints/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B"
# eagle_path = "/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-70B"
dtype = torch.float16
cuda_graph = True
draft_cuda_graph = True
num_generate = 1024
model_type = ["base", "medusa", "eagle"][1]
Bench = True

# prompt = "Beijing is the"

messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]

# question = "Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point."
question = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
messages.append({
                "role": "user",
                "content": question
            })



tokenizer = AutoTokenizer.from_pretrained(path)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# conv = get_conversation_template("llama-3")
# conv = get_conversation_template("vicuna")

# conv.message = []
# conv.append_message(conv.roles[0], question)
# conv.append_message(conv.roles[1], None)
# prompt = conv.get_prompt()
input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda().int()
print("input ids")
print(input_ids)
num_tokens = input_ids.numel()
print(num_tokens)

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)
teminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]


llm = CascadeEagle3W4A16GMSpecW4A16GMEa3Head(
    drafter_path=draft_path, 
    base_path=path,
    min_draft_length=6,
    draft_cuda_graph=draft_cuda_graph,
    tree_path=eagle_path,
    ea_num_iter=3,
    ea_topk_per_iter=10,
    tree_size=30,
    dtype=dtype, 
    memory_limit=0.8, 
    draft_model_start=False,
    cuda_graph=cuda_graph)
our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)

llm.init_storage()
llm.load_from_hf()
print("Model loaded")

gen_result = our_generate()

messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]

question = "Draft a professional email seeking your supervisor's feedback on the 'Quarterly Financial Report' you prepared. Ask specifically about the data analysis, presentation style, and the clarity of conclusions drawn. Keep the email short and to the point."
# question = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
messages.append({
                "role": "user",
                "content": question
            })

tokenizer = AutoTokenizer.from_pretrained(path)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda().int()
gen_result1 = llm.generate(input_ids, num_generate, teminators=teminators)

print("=========================================")
print("output ids")
print(gen_result[0])
out_str = tokenizer.decode(gen_result[0])
print(out_str)
print("acc:", gen_result[1])
import numpy as np
print("Mean acc:", np.mean(gen_result[1]))
print(f"ea accept avg:", np.mean(gen_result[-1].cpu().numpy()))
print("=========================================")





print("output ids")
print(gen_result1[0])
out_str = tokenizer.decode(gen_result1[0])
print(out_str)
print("acc:", gen_result1[1])
import numpy as np
print("Mean acc:", np.mean(gen_result1[1]))
print(f"ea accept avg:", np.mean(gen_result1[-1].cpu().numpy()))
# gen_result1 = our_generate()
# if model_type == "eagle" or model_type == "medusa":
#     import numpy as np
#     m_output = our_generate()
#     print(tokenizer.decode(m_output[0]))
#     print(len(m_output[0]))
#     print("acc:", m_output[1])
#     print("Mean acc:", np.mean(m_output[1]))
# else:
#     print("output ids")
#     print(gen_result[0])
#     out_str = tokenizer.decode(gen_result[0])
#     print(out_str)
#     print("=========================================")
#     out_str1 = tokenizer.decode(gen_result1[0])
#     print(out_str1)
#     print("=========================================")
#     print(out_str1 == out_str)
# if Bench:
#     print("decode speed:", f"{num_generate / do_bench(our_generate, warmup=10, rep=1000) * 1000} tok/s")
