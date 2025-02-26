import argparse
import torch
from llamacu.llama_w4a8_qqq import W4A8QQQLLM
from transformers import AutoTokenizer
from QQQ.utils import (
    get_model_architecture,
    get_model_config,
    setup_seed,
)
from QQQ.gptq.models import get_quantized_model_class
from triton.testing import do_bench




# if __name__ == "__main__":
# setup_seed(42)
path = "/home/ydzhang/checkpoints/HandH1998/QQQ-Llama-3-8b"
cuda_graph = False

config = get_model_config("/home/ydzhang/checkpoints/HandH1998/QQQ-Llama-3-8b")
quant_config = config.quantization_config
# NOTE(HandH1998): delete quantization_config to avoid getting into transformers' quantization method validation,
# as transformers doesn't support qqq for now
del config.quantization_config
model_type = get_model_architecture(config)
quant_model_class = get_quantized_model_class(model_type)
model = quant_model_class.from_pretrained(
    "/home/ydzhang/checkpoints/HandH1998/QQQ-Llama-3-8b",
    config=config,
    quant_config=quant_config,
    device_map="sequential",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(
    "/home/ydzhang/checkpoints/HandH1998/QQQ-Llama-3-8b",
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token


merge_path = "/home/ydzhang/checkpoints/HandH1998/QQQ-Llama-3-8b-merge"
# merge_path = "/home/ydzhang/checkpoints/HandH1998/QQQ-Llama-3-8b-merge"
# llamacu
llm = W4A8QQQLLM(merge_path, dtype=torch.float16, memory_limit=0.6, cuda_graph=cuda_graph)
llm.init_storage()
llm.load_from_hf()

# prompt = "Beijing is"
dtype = torch.float16
Bench = True
# cuda_graph = True
num_tokens = 1024
num_verify = 64
prompt = "Once upon a time, in a vast and enchanted forest, there lived a young girl named Luna who possessed an extraordinary gift - she could communicate with the ancient trees that surrounded her humble cottage. These weren't ordinary trees; they were the guardians of countless secrets spanning centuries. Luna's parents had disappeared mysteriously when she was just a child, leaving her in the care of her grandmother, a wise woman known throughout the region for her healing abilities and deep connection to nature. Every morning, Luna would wake at dawn to tend to her grandmother's herb garden, where they grew plants for their healing potions. The morning dew would glisten on the leaves like tiny diamonds, and the first rays of sunlight would filter through the dense canopy above. As she worked, the trees would whisper to her, sharing stories of travelers who had passed through their woods, of magical creatures that dwelled in the deepest parts of the forest, and of ancient battles long forgotten by human memory. One particularly misty morning, as Luna was collecting mushrooms near the oldest oak tree in the forest, she heard a different kind of whisper - urgent and troubled. The trees spoke of a darkness spreading from the Northern Mountains, a malevolent force slowly poisoning the land. Animals were fleeing their homes, and the very essence of magic that sustained the forest was beginning to fade. Luna knew she had to act. With her grandmother's blessing and armed with only her knowledge of herbs and her ability to communicate with the trees, she set out on a perilous journey to the Northern Mountains. Along the way, she encountered many challenges: treacherous ravines that required careful navigation, mysterious creatures that tested her courage, and ancient riddles that demanded clever solutions. But she also found unexpected allies - a wise old owl who became her guide, a mischievous fox who helped her outsmart dangerous predators, and a young dragon who had been separated from its family. As Luna ventured deeper into unknown territories, she discovered that the darkness was emanating from an ancient artifact, long ago buried by her own ancestors to protect the world from its corrupting influence. Someone had unearthed it, not understanding its terrible power. Through her journey, Luna learned that her parents hadn't simply disappeared - they had been the last guardians of this secret, and their disappearance was connected to their attempts to prevent the artifact from being discovered. With the help of her newfound friends and the wisdom of the trees that stretched even into these distant lands, Luna faced the ultimate challenge. She had to use all her knowledge, courage, and the deep magic of nature itself to seal away the darkness once again. The task seemed impossible for one young girl, but Luna wasn't alone - she had the strength of the entire forest behind her, the ancient wisdom of her grandmother's teachings, and the unwavering support of her loyal companions. In a climactic confrontation that tested every skill she had learned, Luna managed to return the artifact to its resting place, this time protected by even stronger magical barriers. The darkness receded, and life began to return to the poisoned lands. Most importantly, she discovered the truth about her parents - they were alive, trapped in a magical sleep to prevent the artifact's location from being discovered. With the danger passed, Luna was able to wake them, leading to a joyful reunion. Returning home as a hero, Luna realized that her greatest adventure had only just begun. She now understood her true purpose - to become a guardian of nature's secrets, like her parents before her, and to protect the delicate balance between the human world and the ancient magic that sustains it. Her story became legend, whispered by the trees to future generations, teaching them about courage, wisdom, and the power of staying true to oneself even in the darkest of times. And so, Luna continued to live in her forest home, growing stronger and wiser with each passing season, ready to face whatever new challenges might arise, knowing that she was never truly alone as long as she could hear the whispers of the trees." * 10
chunk_length = num_tokens #//2 if want to test chunking

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda().int()[:, :num_tokens]
position_ids = torch.arange(num_tokens, dtype=torch.int32, device="cuda").view(1, num_tokens)

with torch.no_grad():
    baseline_prefill = lambda: model(input_ids, position_ids=position_ids, use_cache=True, return_dict=False)
    logits, past_key_values = baseline_prefill()
    logits = logits[0, -1:].view(-1)
    # logits = logits[0, -1:].view(-1)
# outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

our_prefill = lambda: llm.prefill(input_ids, position_ids)
our_logits = our_prefill()
our_logits = our_logits.view(-1)

print(logits)
print(our_logits)
print(f"prefill diff: {(logits-our_logits).abs().mean()}")
print(f"qqq index: {torch.argmax(logits)}")
print(f"our index: {torch.argmax(our_logits)}")


# decode

input_ids = torch.randint(0, 32000, (1, num_verify), dtype=torch.int32, device="cuda")
# input_ids = torch.argmax(logits).view(1, 1).cuda().int()
position_ids = torch.tensor([[num_tokens] * num_verify], dtype=torch.int32, device="cuda")
cache_length = torch.tensor([num_tokens], dtype=torch.int32, device="cuda")

with torch.no_grad():
    torch.cuda.nvtx.range_push("baseline decode")
    baseline_decode = lambda: model(input_ids, position_ids=position_ids, use_cache=True, past_key_values=past_key_values, return_dict=False)[0]
    logits = baseline_decode()

our_decode = lambda: llm.decode(input_ids, position_ids, cache_length)
our_logits = our_decode()
# if Bench:
#     our_time = do_bench(our_decode, warmup=10, rep=1000)

print(logits)
print(our_logits)
# torch.save(our_logits, "./tmp/debug_states/w4a8_qqq_1.pt")
print(f"decode diff: {(logits-our_logits).abs().mean()}")
print(f"qqq index: {torch.argmax(logits)}")
print(f"our index: {torch.argmax(our_logits)}")
# print(f"baseline decode: {num_verify / time * 1000} tok/s")
# print(f"our decode: {num_verify / our_time * 1000} tok/s")
# print(f"our decode: {num_verify / our_time * 1000} tok/s")