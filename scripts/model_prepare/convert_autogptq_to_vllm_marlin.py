import torch
from safetensors.torch import load_file, save_file
import re
from typing import List
from vllm import _custom_ops as ops

model_path = '/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-mse/gptq_model-4bit-128g.safetensors'
output_path = '/home/ydzhang/checkpoints/AutoGPTQ/Meta-Llama-3-8B-Instruct-4bit-128g-pileval-mse-gptq_marlin/model_gptq_marlin.safetensors'
autogptq_weigths = load_file(model_path)

gptq_convert_dict = {
    "model.layers.{}.self_attn.q_proj.qweight": ["model.layers.{}.self_attn.q_proj.scales", "model.layers.{}.self_attn.q_proj.g_idx", "model.layers.{}.self_attn.q_proj.qzeros"], 
    "model.layers.{}.self_attn.k_proj.qweight":["model.layers.{}.self_attn.k_proj.scales", "model.layers.{}.self_attn.k_proj.g_idx", "model.layers.{}.self_attn.k_proj.qzeros"],
    "model.layers.{}.self_attn.v_proj.qweight":["model.layers.{}.self_attn.v_proj.scales", "model.layers.{}.self_attn.v_proj.g_idx", "model.layers.{}.self_attn.v_proj.qzeros"],
    "model.layers.{}.self_attn.o_proj.qweight":["model.layers.{}.self_attn.o_proj.scales", "model.layers.{}.self_attn.o_proj.g_idx", "model.layers.{}.self_attn.o_proj.qzeros"],
    "model.layers.{}.mlp.gate_proj.qweight":["model.layers.{}.mlp.gate_proj.scales", "model.layers.{}.mlp.gate_proj.g_idx", "model.layers.{}.mlp.gate_proj.qzeros"],
    "model.layers.{}.mlp.up_proj.qweight": ["model.layers.{}.mlp.up_proj.scales", "model.layers.{}.mlp.up_proj.g_idx", "model.layers.{}.mlp.up_proj.qzeros"],
    "model.layers.{}.mlp.down_proj.qweight": ["model.layers.{}.mlp.down_proj.scales", "model.layers.{}.mlp.down_proj.g_idx", "model.layers.{}.mlp.down_proj.qzeros"],
}

vllm_checkpoints = {}
processed_keys = set()

def get_scale_perms():
    scale_perm: List[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: List[int] = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(s: torch.Tensor, size_k: int, size_n: int,
                          group_size: int) -> torch.Tensor:

    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s

for gptq_key in autogptq_weigths:
    if gptq_key in processed_keys:
        continue
    elif "layers" in gptq_key:
        abstract_key = re.sub(r'(\d+)', '{}', gptq_key)
        layer_num = re.search(r'\d+', gptq_key).group(0)
        if abstract_key in gptq_convert_dict:
            if abstract_key.endswith('qweight'):
                x = autogptq_weigths[gptq_key].clone().cuda()
                shape_0 = x.shape[0]*8
                shape_1 = x.shape[1]
                x.data = ops.gptq_marlin_repack(x.data.contiguous(),
                        perm=torch.Tensor([]).to( device="cuda", dtype=torch.int32),
                        size_k=shape_0,
                        size_n=shape_1,
                        num_bits=4)
                vllm_checkpoints[gptq_key] = x.cpu()


                for q_keys in gptq_convert_dict[abstract_key]:
                    if q_keys.endswith("scales"):
                        scales_x = autogptq_weigths[q_keys.format(layer_num)].clone().cuda()
                        scales_x.data = marlin_permute_scales(scales_x.data.contiguous(),
                                size_k=shape_0,
                                size_n=shape_1,
                                group_size=128)
                        vllm_checkpoints[q_keys.format(layer_num)] = scales_x.cpu()
                    processed_keys.add(q_keys.format(layer_num))
        elif "post_attention_layernorm" in gptq_key or "input_layernorm" in gptq_key:
            vllm_checkpoints[gptq_key] = autogptq_weigths[gptq_key].clone()
    else:  
        vllm_checkpoints[gptq_key] = autogptq_weigths[gptq_key].clone()

save_file(vllm_checkpoints, output_path)