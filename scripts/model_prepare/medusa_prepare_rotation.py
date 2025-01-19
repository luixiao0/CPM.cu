import re
import torch
from safetensors.torch import load_file,save_file

if __name__=="__main__":
    medusa_file = "/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa/medusa_lm_head.safetensors"

    medusa_ckpt = load_file(medusa_file)
    llama3_ckpt = load_file("/home/ydzhang/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/model-00004-of-00004.safetensors")

    llama3_head_weight = llama3_ckpt["lm_head.weight"]
    
    new_medusa_ckpt = {}

    # load rotation
    def transform_rms_norm_and_rotation(norm_weight, rotation_weight):
        """Fuse the weight multiplication of rms norm into the next adjacent linear modules.

        Args:
            norm (`nn.LayerNorm` or `RMSNorm`):
                normalization module.
            next_modules (`Iterable[nn.Linear]`):
                modules after the normalization module.
        """
        ln_w = norm_weight.to(dtype=torch.float64)

        dtype = rotation_weight.dtype
        fc_w = rotation_weight.to(dtype=torch.float64)
        ln_w = ln_w.to(fc_w.device)
        rotation_weight_norm = (fc_w * ln_w.unsqueeze(1)).to(dtype=dtype)
        return rotation_weight_norm

    rotation_weights = torch.load("/home/ydzhang/project/spec_decoding/deepcompressor/runs/llm/cache/quant/rotation/hadamard/llama-3-8b-instruct.pt")
    rms_norm_weights = llama3_ckpt["model.norm.weight"]

    rotation_weights_norm = transform_rms_norm_and_rotation(rms_norm_weights, rotation_weights)
    new_medusa_ckpt['rotation.weight'] = rotation_weights_norm

    # pattern = re.compile(r'^(\d+)\.0\.linear\.weight$')
    # for key in medusa_ckpt:
    #     match = pattern.match(key)
    #     if match:
    #         index = int(match.group(1))
    #         new_medusa_ckpt[key] = llama3_head_weight[index]
    new_medusa_ckpt['0.0.linear.bias'] = medusa_ckpt['0.0.linear.bias'].clone().detach()
    new_medusa_ckpt['0.0.linear.weight'] = medusa_ckpt['0.0.linear.weight'].clone().detach()
    new_medusa_ckpt['0.1.weight'] = llama3_head_weight.clone().detach()

    new_medusa_ckpt['1.0.linear.bias'] = medusa_ckpt['1.0.linear.bias'].clone().detach()
    new_medusa_ckpt['1.0.linear.weight'] = medusa_ckpt['1.0.linear.weight'].clone().detach()
    new_medusa_ckpt['1.1.weight'] = llama3_head_weight.clone().detach()

    new_medusa_ckpt['2.0.linear.bias'] = medusa_ckpt['2.0.linear.bias'].clone().detach()
    new_medusa_ckpt['2.0.linear.weight'] = medusa_ckpt['2.0.linear.weight'].clone().detach()
    new_medusa_ckpt['2.1.weight'] = llama3_head_weight.clone().detach()

    new_medusa_file = "/home/ydzhang/checkpoints/predibase/Meta-Llama-3-8B-Instruct-medusa-full-rotation/medusa_lm_head.safetensors"

    save_file(new_medusa_ckpt, new_medusa_file)