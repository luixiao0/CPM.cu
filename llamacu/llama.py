import torch
from . import C

from transformers import AutoTokenizer, AutoConfig

class LLM(torch.nn.Module):
    def __init__(self,
                 path: str, # hf model path
                 memory_limit: float = 0.9):
        super().__init__()

        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.config = AutoConfig.from_pretrained(path)

        self.memory_limit = int(torch.cuda.get_device_properties(0).total_memory * memory_limit)
        self.memory_pool = torch.nn.Parameter(torch.empty(self.memory_limit, dtype=torch.uint8, device="cuda"), requires_grad=False)

        C.init_model(
            self.memory_limit,
            self.memory_pool,
            self.config.vocab_size,
            self.config.num_hidden_layers,
            self.config.hidden_size,
            self.config.intermediate_size,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.rms_norm_eps,
            self.config.rope_theta,
            self.config.torch_dtype,
        )

    def load_from_hf(self):
        with torch.no_grad():
            ckpt = torch.load(self.path+"/pytorch_model.bin", map_location="cpu")
            for name, param in ckpt.items():
                if 'gate_proj' in name or 'up_proj' in name:
                    param = param.transpose(0, 1)
                C.load_model(name, param.contiguous().to(self.config.torch_dtype).view(-1))

    def generate(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(torch.int32).cuda()
        output_ids = torch.empty_like(input_ids)
        C.generate(input_ids, output_ids)
        return output_ids
