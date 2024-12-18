import os
import torch
from llamacu.llama import LLM
from transformers import AutoTokenizer, AutoModelForCausalLM
from triton.testing import do_bench

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

path = "../../models/MiniCPM-1B-sft-llama-format"
# path = "../../models/MiniCPM-1B-sft-llama-format-gptq-1016-v2dataset-wolmhead-perchannel-desc-true"
# path = "../../models/Llama-2-7b-hf"
# path = "../../models/TheBloke/Llama-2-7B-Chat-GPTQ"
# path = "../../models/Meta-Llama-3-8B"
# path = "/home/test/testdata/models/Llama-3.2-1B-instruct"
# path = "../../models/Mistral-7B-Instruct-v0.2"
dtype = torch.float16
Bench = True
cuda_graph = False
num_tokens = 1024
num_verify = 64
prompt = "Once upon a time, in a vast and enchanted forest, there lived a young girl named Luna who possessed an extraordinary gift - she could communicate with the ancient trees that surrounded her humble cottage. These weren't ordinary trees; they were the guardians of countless secrets spanning centuries. Luna's parents had disappeared mysteriously when she was just a child, leaving her in the care of her grandmother, a wise woman known throughout the region for her healing abilities and deep connection to nature. Every morning, Luna would wake at dawn to tend to her grandmother's herb garden, where they grew plants for their healing potions. The morning dew would glisten on the leaves like tiny diamonds, and the first rays of sunlight would filter through the dense canopy above. As she worked, the trees would whisper to her, sharing stories of travelers who had passed through their woods, of magical creatures that dwelled in the deepest parts of the forest, and of ancient battles long forgotten by human memory. One particularly misty morning, as Luna was collecting mushrooms near the oldest oak tree in the forest, she heard a different kind of whisper - urgent and troubled. The trees spoke of a darkness spreading from the Northern Mountains, a malevolent force slowly poisoning the land. Animals were fleeing their homes, and the very essence of magic that sustained the forest was beginning to fade. Luna knew she had to act. With her grandmother's blessing and armed with only her knowledge of herbs and her ability to communicate with the trees, she set out on a perilous journey to the Northern Mountains. Along the way, she encountered many challenges: treacherous ravines that required careful navigation, mysterious creatures that tested her courage, and ancient riddles that demanded clever solutions. But she also found unexpected allies - a wise old owl who became her guide, a mischievous fox who helped her outsmart dangerous predators, and a young dragon who had been separated from its family. As Luna ventured deeper into unknown territories, she discovered that the darkness was emanating from an ancient artifact, long ago buried by her own ancestors to protect the world from its corrupting influence. Someone had unearthed it, not understanding its terrible power. Through her journey, Luna learned that her parents hadn't simply disappeared - they had been the last guardians of this secret, and their disappearance was connected to their attempts to prevent the artifact from being discovered. With the help of her newfound friends and the wisdom of the trees that stretched even into these distant lands, Luna faced the ultimate challenge. She had to use all her knowledge, courage, and the deep magic of nature itself to seal away the darkness once again. The task seemed impossible for one young girl, but Luna wasn't alone - she had the strength of the entire forest behind her, the ancient wisdom of her grandmother's teachings, and the unwavering support of her loyal companions. In a climactic confrontation that tested every skill she had learned, Luna managed to return the artifact to its resting place, this time protected by even stronger magical barriers. The darkness receded, and life began to return to the poisoned lands. Most importantly, she discovered the truth about her parents - they were alive, trapped in a magical sleep to prevent the artifact's location from being discovered. With the danger passed, Luna was able to wake them, leading to a joyful reunion. Returning home as a hero, Luna realized that her greatest adventure had only just begun. She now understood her true purpose - to become a guardian of nature's secrets, like her parents before her, and to protect the delicate balance between the human world and the ancient magic that sustains it. Her story became legend, whispered by the trees to future generations, teaching them about courage, wisdom, and the power of staying true to oneself even in the darkest of times. And so, Luna continued to live in her forest home, growing stronger and wiser with each passing season, ready to face whatever new challenges might arise, knowing that she was never truly alone as long as she could hear the whispers of the trees." * 10
chunk_length = num_tokens #//2 if want to test chunking

# seed
torch.manual_seed(0)

# prefill
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map="auto")
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()[:, :num_tokens]
# input_ids = torch.randint(0, 32000, (1, num_tokens,), dtype=torch.int32).cuda()
position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)
num_tokens = input_ids.numel()
with torch.no_grad():
    torch.cuda.nvtx.range_push("baseline prefill")
    baseline_prefill = lambda: model(input_ids, position_ids=position_ids, use_cache=True, return_dict=False)
    logits, past_key_values = baseline_prefill()
    torch.cuda.nvtx.range_pop()
    if Bench:
        time = do_bench(baseline_prefill, warmup=10, rep=1000)
    logits = logits[0, -1:].view(-1)

llm = LLM(path, dtype=dtype, chunk_length=chunk_length, memory_limit=0.4, cuda_graph=cuda_graph)
llm.load_from_hf()

torch.cuda.nvtx.range_push("our prefill")
our_prefill = lambda: llm.prefill(input_ids, position_ids)
our_logits = our_prefill()
torch.cuda.nvtx.range_pop()
if Bench:
    our_time = do_bench(our_prefill, warmup=10, rep=1000)
our_logits = our_logits.view(-1)

print(logits)
print(our_logits)
print(f"prefill diff: {(logits-our_logits).abs().mean()}")
print(f"baseline prefill: {num_tokens / time * 1000} tok/s")
print(f"our prefill: {num_tokens / our_time * 1000} tok/s")

# decode
input_ids = torch.randint(0, 32000, (1, num_verify), dtype=torch.int32, device="cuda")
position_ids = torch.tensor([[num_tokens] * num_verify], dtype=torch.int32, device="cuda")
cache_length = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")
if num_verify > 1:
    mask_2d = torch.randint(0, 2, (num_verify, num_verify), dtype=torch.int32, device="cuda")
    mask_2d = mask_2d & torch.tril(torch.ones((num_verify, num_verify), dtype=torch.int32, device="cuda"), diagonal=0) # 1 means visible
    print(mask_2d)
else:
    mask_2d = None

with torch.no_grad():
    torch.cuda.nvtx.range_push("baseline decode")
    baseline_decode = lambda: model(input_ids, position_ids=position_ids, use_cache=True, past_key_values=past_key_values, mask_2d=mask_2d, return_dict=False)[0]
    logits = baseline_decode()
    torch.cuda.nvtx.range_pop()
    if Bench:
        time = do_bench(baseline_decode, warmup=10, rep=1000)

torch.cuda.nvtx.range_push("our decode")
our_decode = lambda: llm.decode(input_ids, position_ids, cache_length, mask_2d=mask_2d)
our_logits = our_decode()
torch.cuda.nvtx.range_pop()
if Bench:
    our_time = do_bench(our_decode, warmup=10, rep=1000)

print(logits)
print(our_logits)
print(f"decode diff: {(logits-our_logits).abs().mean()}")
print(f"baseline decode: {num_verify / time * 1000} tok/s")
print(f"our decode: {num_verify / our_time * 1000} tok/s")
