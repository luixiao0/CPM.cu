import torch
from llamacu.llama import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "../../models/MiniCPM-1B-sft-llama-format"
prompt = "Hello, world!"

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)
input_ids = tokenizer.encode(prompt, return_tensors="pt")
input_embeds = model.get_input_embeddings()(input_ids)
print(input_embeds @ model.lm_head.weight.T)

llm = LLM(path)
llm.load_from_hf()

model_offset = 225607680

llm.generate(prompt)
x = llm.memory_pool.view(torch.bfloat16)[model_offset+llm.config.hidden_size*5:model_offset+llm.config.hidden_size*5+llm.config.vocab_size*5]
print(x)
