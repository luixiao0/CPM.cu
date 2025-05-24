import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import numpy as np

# ==================== 配置参数 ====================
# 模型配置
model_path = "/DATA/disk0/zhaoweilun/minicpm4/models/job_32175.step_3000"  # 模型路径或名称
dtype = torch.float16
use_4bit = False
use_8bit = True
trust_remote_code = True

# 生成配置
max_new_tokens = 128
temperature = 0.7
top_p = 0.9
do_sample = True

def make_input(digits, a = 2500, b = 4000):
    """生成针在草堆测试的输入文本"""
    head = "There is a pass key hidden in the context. Find it and remember it. I will quiz you about it later. "
    before = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * a
    needle = f"The pass key is {digits}. Remember it. The pass key is {digits}"
    after = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * b
    query = "Now, give me the exact number of the pass key. The pass key is "
    return head + before + needle + after + query

# prompt = make_input(681725493, 2000, 4000) # 120k
# prompt = make_input(681725493, 1000, 2000) # 60k
prompt = make_input(681725493, 500, 1000) # 30k
# prompt = "Beijing is the"

def load_model_and_tokenizer():
    """加载模型和分词器"""
    print(f"Loading model from: {model_path}")
    
    # 配置量化
    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        print("Using 4-bit quantization")
    elif use_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        print("Using 8-bit quantization")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=trust_remote_code
    )
    
    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        quantization_config=quantization_config,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
    )
    
    print(f"Model loaded successfully! Device: {model.device}")
    return model, tokenizer

# ==================== 主程序 ====================

print(f"Prompt length: {len(prompt)} characters")

# 加载模型和分词器
model, tokenizer = load_model_and_tokenizer()

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
input_ids = inputs.input_ids.to(model.device)
attention_mask = inputs.attention_mask.to(model.device)

input_length = input_ids.shape[1]
print(f"Input length: {input_length} tokens")

# 设置终止符
terminators = [tokenizer.eos_token_id]
if hasattr(tokenizer, 'im_end_id'):
    terminators.append(tokenizer.im_end_id)

# 生成参数
generation_kwargs = {
    "max_new_tokens": max_new_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "do_sample": do_sample,
    "eos_token_id": terminators,
    "pad_token_id": tokenizer.pad_token_id,
    "attention_mask": attention_mask,
    "use_cache": True
}

# 预热
print("Warming up...")
with torch.no_grad():
    _ = model.generate(input_ids, max_new_tokens=10, do_sample=False)

torch.cuda.synchronize()

# 正式生成测试
print("Starting generation...")
start_time = time.time()

with torch.no_grad():
    generated_ids = model.generate(
        input_ids,
        **generation_kwargs
    )

torch.cuda.synchronize()
end_time = time.time()

# 解码和分析结果
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
new_tokens = generated_ids[0][input_length:]
generated_length = len(new_tokens)

total_time = end_time - start_time
tokens_per_second = generated_length / total_time if total_time > 0 else 0

# 打印结果
print("\n" + "="*50)
print("GENERATION RESULTS")
print("="*50)

# 生成的文本预览（最后200个字符）
print("Generated text preview:")
print("-" * 30)
print(generated_text[-500:])
print("-" * 30)

print(f"\nInput length: {input_length} tokens")
print(f"Generated length: {generated_length} tokens")
print(f"Total time: {total_time:.2f} seconds")
print(f"Generation speed: {tokens_per_second:.2f} tokens/second")

# GPU 内存使用情况
if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU memory allocated: {memory_allocated:.2f} GB")
    print(f"GPU memory reserved: {memory_reserved:.2f} GB")

print("="*50) 