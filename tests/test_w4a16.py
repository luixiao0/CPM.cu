from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging
import torch
import os
from triton.testing import do_bench
from llamacu.llama_w4a16_marlin import W4A16MarlinLLM

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

quantized_model_dir = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-mse"
# quantized_model_dir = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-4bit-128g-pileval-mse"
quantized_model_dir_marlin = "/data1/fanruikai/Llama-2-7B-Chat-GPTQ-Marlin"

dtype = torch.float16
Bench = True
cuda_graph = True
num_tokens = 1024
num_verify = 1
chunk_length = num_tokens #//2 if want to test chunking
use_marlin = True
# seed
torch.manual_seed(0)

# prefill
tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_marlin=use_marlin)

input_ids = torch.randint(0, 32000, (1, num_tokens,), dtype=torch.int32).cuda()

position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

llm = W4A16MarlinLLM(quantized_model_dir, dtype=dtype, chunk_length=chunk_length, memory_limit=0.3, cuda_graph=cuda_graph, group_size=128, use_marlin=use_marlin)
llm.init_storage()
llm.load_from_hf()

torch.cuda.nvtx.range_push("our prefill")
our_prefill = lambda: llm.prefill(input_ids, position_ids)
our_logits = our_prefill()
torch.cuda.nvtx.range_pop()
if Bench:
    our_time = do_bench(our_prefill, warmup=10, rep=1000)

with torch.no_grad():
    torch.cuda.nvtx.range_push("baseline prefill")
    baseline_prefill = lambda: model(input_ids, position_ids=position_ids, use_cache=True, return_dict=False)
    logits, past_key_values = baseline_prefill()
    torch.cuda.nvtx.range_pop()
    if Bench:
        time = do_bench(baseline_prefill, warmup=10, rep=1000)

print(logits)
print(our_logits)
print(f"prefill diff: {(logits[0][-1] -our_logits).abs().mean()}")
print(f"baseline prefill: {num_tokens / time * 1000} tok/s")
print(f"our prefill: {num_tokens / our_time * 1000} tok/s")

input_ids = torch.randint(0, 32000, (1, num_verify), dtype=torch.int32, device="cuda")
position_ids = torch.tensor([[num_tokens] * num_verify], dtype=torch.int32, device="cuda")
cache_length = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
if num_verify > 1:
    mask_2d = torch.randint(0, 2, (num_verify, num_verify), dtype=torch.int32, device="cuda")
    mask_2d = mask_2d & torch.tril(torch.ones((num_verify, num_verify), dtype=torch.int32, device="cuda"), diagonal=0) # 1 means visible

    from llamacu.speculative.tree_drafter import pack_mask
    mask_2d_packed = pack_mask(mask_2d)
else:
    mask_2d = None
    mask_2d_packed = None

with torch.no_grad():
    torch.cuda.nvtx.range_push("baseline decode")
    baseline_decode = lambda: model(input_ids, position_ids=position_ids, use_cache=True, past_key_values=past_key_values, return_dict=False)[0]
    logits = baseline_decode()
    torch.cuda.nvtx.range_pop()
    if Bench:
        time = do_bench(baseline_decode, warmup=10, rep=1000)

torch.cuda.nvtx.range_push("our decode")
our_decode = lambda: llm.decode(input_ids, position_ids, cache_length, mask_2d=mask_2d_packed)
our_logits = our_decode()
torch.cuda.nvtx.range_pop()
if Bench:
    our_time = do_bench(our_decode, warmup=10, rep=1000)

print(our_logits)
print(logits)
print(f"baseline decode: {num_verify / time * 1000} tok/s")
print(f"decode diff: {(logits-our_logits).abs().mean()}")
print(f"our decode: {num_verify / our_time * 1000} tok/s")
