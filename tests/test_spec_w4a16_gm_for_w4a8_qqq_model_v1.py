import os
import torch
from llamacu.llama_w4a8_qqq import W4A8QQQLLM
from llamacu.speculative.spec_quant.spec_w4a16_gm_for_w4a8_qqq_model import W4A16GMSpecW4A8QQQ
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from fastchat.model import get_conversation_template

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# path = "/home/ydzhang/checkpoints/lmsys/vicuna-7b-v1.3"
# path = "/home/ydzhang/checkpoints/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GPTQ-gptq_marlin"
# draft_path = "/home/ydzhang/checkpoints/QQQ/Meta-Llama-3-8B-Instruct-rotation-gptq-mse-pile/Meta-Llama-3-8B-Instruct-merge"
draft_path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-rotation-4bit-128g-pileval-mse-desc-static_group_default_merge"
# path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-mse-gptq_marlin_merge"
# path = "/home/ydzhang/checkpoints/QQQ/Meta-Llama-3-70B-Instruct-rotation-gptq-pile/Meta-Llama-3-70B-Instruct-merge"
path = "/home/ydzhang/checkpoints/QQQ/Meta-Llama-3-70B-Instruct-rotation-gptq-pile-g128/Meta-Llama-3-70B-Instruct-merge"

model_type = ["base", "spec"][1]

dtype = torch.float16

cuda_graph = True
draft_cuda_graph = True
num_tokens = 1024
num_verify = 1

# messages = [
#             {"role": "system",
#              "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
#         ]

messages = [
            {"role": "system",
             "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."},
        ]
question = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
messages.append({
                "role": "user",
                "content": question
            })


chunk_length = num_tokens #//2 if want to test chunking


torch.manual_seed(0)

tokenizer = AutoTokenizer.from_pretrained(path)

# conv = get_conversation_template("llama-3")
# conv = get_conversation_template("vicuna")

# conv.message = []
# conv.append_message(conv.roles[0], question)
# conv.append_message(conv.roles[1], None)
# prompt = conv.get_prompt()
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()[:, :num_tokens]
input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda().int()

teminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
position_ids = torch.arange(num_tokens).cuda().int().view(1, num_tokens)

num_tokens = input_ids.numel()

if model_type == "spec":
    
    llm = W4A16GMSpecW4A8QQQ(drafter_path=draft_path, base_path=path, draft_num=6, dtype=dtype, chunk_length=chunk_length, memory_limit=0.8, cuda_graph=cuda_graph , draft_cuda_graph= draft_cuda_graph)
else:
    llm = W4A8QQQLLM(path, dtype=dtype, memory_limit=0.5)

llm.init_storage()
llm.load_from_hf()
num_generate = 1024
# teminators = [tokenizer.eos_token_id]

our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)


if model_type == "base":
    b_output = our_generate()
    print(tokenizer.decode(b_output[0]))
else:
    m_output = our_generate()
    print(tokenizer.decode(m_output[0]))
    print("Mean acc:", np.mean(m_output[1]))
    print("====")
    m_output_1 = our_generate()
    print(tokenizer.decode(m_output_1[0]))
    print(m_output_1[1])
    print("Mean acc:", np.mean(m_output_1[1]))
    print(f"same:{tokenizer.decode(m_output[0])==tokenizer.decode(m_output_1[0])}")
    print("acc", m_output[1])