import os
import torch
from llamacu.llama_w4a16_gptq_marlin import W4A16GPTQMarlinLLM
from llamacu.speculative.medusa_base_quant.medusa_base_w4a16_gptq_marlin import W4A16GPTQMarlinLLM_with_medusa
from llamacu.speculative.medusa_choices import *
from llamacu.speculative.eagle_base_quant.eagle_base_w4a16_marlin_gptq import W4A16GPTQMarlinLLM_with_eagle
from llamacu.speculative.cascade_spec_quant.csc_eagle_w4a16_gm_spec_w4a16_gm_sep import CascadeEagleW4A16GMSpecW4A16GMSep
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
draft_path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-mse-gptq_marlin_merge"
# path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-mse-gptq_marlin_merge"
path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-4bit-128g-pileval-mse-gptq_marlin_merge"
# path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-4bit-128g-pileval-mse-gptq_marlin"
# path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-4bit-128g-pileval-mse-gptq_marlin_merge"
# path = "/home/ydzhang/checkpoints/GPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-marlin"
# path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-4bit-128g-pileval-mse-marlin"
# path = "/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-70B-Instruct-4bit-128g-pileval-mse"
# quantized_model_dir = "/home/ydzhang/checkpoints/GPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-marlin"
medusa_path = "/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full"
eagle_path = "/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-8B"
# eagle_path = "/home/ydzhang/checkpoints/yuhuili/EAGLE-LLaMA3-Instruct-70B"
dtype = torch.float16
cuda_graph = False
num_generate = 1024
model_type = ["base", "medusa", "eagle"][1]
Bench = True

# prompt = "Beijing is the"

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


# answer = """**Aloha Spirit: Unforgettable Cultural Experiences and Must-See Attractions in Hawaii**

# As I stepped off the plane in Honolulu, the warm tropical air enveloped me, and I knew that my trip to Hawaii was going to be an unforgettable adventure. The Aloha State is more than just a pretty face – it's a rich tapestry of culture, history, and natural beauty that will leave you enchanted and inspired. In this blog post, I'll share my favorite cultural experiences and must-see attractions from my recent trip to Hawaii, and why you should add this incredible destination to your bucket list.

# **Immersing in Hawaiian Culture**

# My journey began with a visit to the Bishop Museum, the largest museum in Hawaii and a treasure trove of Hawaiian history and culture. The museum's extensive collections and interactive exhibits provided a fascinating glimpse into the lives of Hawaii's ancient people, from their Polynesian roots to the modern-day struggles and triumphs of the Native Hawaiian community.

# Next, I headed to a traditional Hawaiian luau, where I was treated to a sumptuous feast of local delicacies, including kalua pig, poke, and haupia (coconut milk dessert). The highlight of the evening was the mesmerizing hula performance, which transported me to a bygone era of Hawaiian royalty and storytelling.

# **Must-See Attractions**

# No trip to Hawaii would be complete without visiting some of its breathtaking natural wonders. The iconic Waikiki Beach was a must-visit, with its white sand and crystal-clear waters that shimmered like diamonds in the sunlight. I spent a lazy morning soaking up the sun, snorkeling, and simply enjoying the laid-back atmosphere.

# Another unmissable attraction was the Diamond Head State Monument, an ancient volcanic crater and one of Hawaii's most recognizable landmarks. The moderately challenging hike to the summit was rewarded with panoramic views of Honolulu, the ocean, and the surrounding landscape.

# **Exploring the Islands**

# I also had the opportunity to explore the neighboring islands of Oahu and Maui. On Oahu, I visited the USS Arizona Memorial at Pearl Harbor, a poignant tribute to the lives lost during the historic attack. The memorial's tranquil atmosphere and thought-provoking exhibits left a lasting impression on me.

# On Maui, I embarked on a scenic drive along the Road to Hana, a 52-mile stretch of coastline that winds through lush rainforests, past waterfalls, and to secluded beaches. The journey was as breathtaking as the destinations, with stops at the Waianapanapa State Park and the charming town of Hana.

# **Insider Tips**

# * Be sure to try some of Hawaii's unique local flavors, such as shave ice, laulau, and spam musubi (a popular snack made with grilled spam and rice).
# * Respect Hawaiian culture by learning a few basic phrases, such as "aloha" (hello), "mahalo" (thank you), and "braddah" (brother).
# * Consider visiting during the shoulder season (April to June or September to November) for smaller crowds and lower prices.

# **Conclusion**

# Hawaii is a destination that will capture your heart and imagination. From its rich cultural heritage to its stunning natural beauty, there's something for everyone in the Aloha State. Whether you're a history buff, a foodie, or an adventure-seeker, Hawaii has the power to inspire and transform. So pack your bags, grab your sunscreen, and get ready to experience the ultimate tropical getaway.

# **Mahalo** (thank you) for joining me on this journey to Hawaii. I hope this blog post has inspired you to plan your own unforgettable adventure in the Aloha State."""
# messages.append({
#                 "role": "assistant",
#                 "content": answer,
#             })

# q1 = "Rewrite your previous response. Start every sentence with the letter A."

# messages.append({
#                 "role": "user",
#                 "content": q1
#             })

# answer1 = """**Aloha Spirit: Unforgettable Cultural Experiences and Must-See Attractions in Hawaii**

# Astonishingly, my trip to Hawaii was an adventure that I will always treasure. Amazingly, the warm tropical air enveloped me as I stepped off the plane in Honolulu, and I knew that my journey was going to be an unforgettable experience. Admirably, the Aloha State is more than just a pretty face – it's a rich tapestry of culture, history, and natural beauty that will leave you enchanted and inspired.

# Astride the Bishop Museum, the largest museum in Hawaii and a treasure trove of Hawaiian history and culture, I began my journey. Amazingly, the museum's extensive collections and interactive exhibits provided a fascinating glimpse into the lives of Hawaii's ancient people, from their Polynesian roots to the modern-day struggles and triumphs of the Native Hawaiian community.

# Astonishingly, my next stop was a traditional Hawaiian luau, where I was treated to a sumptuous feast of local delicacies, including kalua pig, poke, and haupia (coconut milk dessert). Amazingly, the highlight of the evening was the mesmerizing hula performance, which transported me to a bygone era of Hawaiian royalty and storytelling.

# Astride the iconic Waikiki Beach, I spent a lazy morning soaking up the sun, snorkeling, and simply enjoying the laid-back atmosphere. Amazingly, the beach's white sand and crystal-clear waters shimmered like diamonds in the sunlight.

# Astride the Diamond Head State Monument, an ancient volcanic crater and one of Hawaii's most recognizable landmarks, I embarked on a moderately challenging hike to the summit. Amazingly, the panoramic views of Honolulu, the ocean, and the surrounding landscape left me in awe.

# Astride the neighboring islands of Oahu and Maui, I explored the natural beauty and cultural heritage of the Aloha State. Amazingly, the USS Arizona Memorial at Pearl Harbor, a poignant tribute to the lives lost during the historic attack, left a lasting impression on me.

# Astride the Road to Hana, a 52-mile stretch of coastline on Maui, I embarked on a scenic drive that wound through lush rainforests, past waterfalls, and to secluded beaches. Amazingly, the journey was as breathtaking as the destinations, with stops at the Waianapanapa State Park and the charming town of Hana.

# **Aloha** to the Aloha State, where the spirit of aloha (love, compassion, and kindness) is alive and well. Amazingly, this destination has the power to inspire and transform, and I hope that this blog post has inspired you to plan your own unforgettable adventure in Hawaii."""

# messages.append({
#                 "role": "assistant",
#                 "content": answer1,
#             })

# q2 = "Rewrite your previous response. Start every sentence with the letter B."

# messages.append({
#                 "role": "user",
#                 "content": q2
#             })

# question = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."

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


llm = CascadeEagleW4A16GMSpecW4A16GMSep(
    drafter_path=draft_path, 
    base_path=path,
    min_draft_length=6,
    draft_cuda_graph=False,
    tree_path=eagle_path,
    ea_num_iter=3,
    ea_topk_per_iter=10,
    tree_size=30,
    dtype=dtype, 
    memory_limit=0.8, 
    cuda_graph=cuda_graph)
our_generate = lambda: llm.generate(input_ids, num_generate, teminators=teminators)

llm.init_storage()
llm.load_from_hf()
print("Model loaded")

gen_result = our_generate()
gen_result1 = our_generate()
print("=========================================")
print("output ids")
print(gen_result[0])
out_str = tokenizer.decode(gen_result[0])
print(out_str)
print("acc:", gen_result[1])
import numpy as np
print("Mean acc:", np.mean(gen_result[1]))
print("=========================================")
print("output ids")
print(gen_result1[0])
out_str = tokenizer.decode(gen_result1[0])
print(out_str)
print("acc:", gen_result1[1])
import numpy as np
print("Mean acc:", np.mean(gen_result1[1]))
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
