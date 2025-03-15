import os
import torch
from llamacu.llama import LLM
from llamacu.speculative.spec_model import LLM_with_spec
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from fastchat.model import get_conversation_template

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# path = "/home/ydzhang/checkpoints/lmsys/vicuna-7b-v1.3"
path = "/home/ydzhang/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct"

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

question = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."


chunk_length = num_tokens #//2 if want to test chunking


torch.manual_seed(0)

tokenizer = AutoTokenizer.from_pretrained(path)

conv = get_conversation_template("llama-3")
# conv = get_conversation_template("vicuna")

conv.message = []
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()[:, :num_tokens]

position_ids = torch.arange(num_tokens).cuda().int().view(1, num_tokens)

num_tokens = input_ids.numel()

if model_type == "spec":
    
    llm = LLM_with_spec(drafter_type='draft', drafter_path=path, base_path=path, draft_num=6, dtype=dtype, chunk_length=chunk_length, memory_limit=0.6 , cuda_graph=cuda_graph , draft_cuda_graph= draft_cuda_graph)
else:
    llm = LLM(path, dtype=dtype, memory_limit=0.6)

llm.init_storage()
llm.load_from_hf()
num_generate = 1024
teminators = [tokenizer.eos_token_id]

our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)


if model_type == "base":
    b_output = our_generate()
    print(tokenizer.decode(b_output))
else:
    m_output = our_generate()
    print(tokenizer.decode(m_output[0]))
    print("Mean acc:", np.mean(m_output[1]))