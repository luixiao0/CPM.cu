import torch
from llamacu.llama import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM

path = "../../models/MiniCPM-1B-sft-llama-format"
prompt = "Hello, world!"

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path)
input_ids = tokenizer.encode(prompt, return_tensors="pt")
input_embeds = model.get_input_embeddings()(input_ids)
print(input_ids)
print(input_embeds)

llm = LLM(path)
llm.load_from_hf()

model_offset = 112803840

print(llm.memory_pool.view(torch.bfloat16)[:model_offset])
print(model.model.embed_tokens.weight)

llm.generate(prompt)
print(llm.memory_pool.view(torch.bfloat16)[model_offset:model_offset+1536*5])
