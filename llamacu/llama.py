from . import C

import os, json
import torch
from transformers import AutoTokenizer, AutoConfig
from triton.testing import do_bench

dtype_map = {
    torch.float16: 0,
    torch.bfloat16: 1,
}

def dtype_to_int(dtype):
    ret = dtype_map.get(dtype, -1)
    if ret == -1:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return ret

class LLM(torch.nn.Module):
    def __init__(self,
                 path: str, # hf model path
                 memory_limit: float = 0.8,
                 chunk_length: int = 1024,
                 output_length: int = 32,
                 dtype: torch.dtype = None,
    ):
        super().__init__()

        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.config = AutoConfig.from_pretrained(path)
        self.dtype = dtype if dtype is not None else self.config.torch_dtype

        self.memory_limit = int(torch.cuda.get_device_properties(0).total_memory * memory_limit)
        self.memory_pool = torch.nn.Parameter(torch.empty(self.memory_limit, dtype=torch.uint8, device="cuda"), requires_grad=False)

        self.chunk_length = chunk_length
        self.output_length = output_length

        C.init_model(
            self.memory_limit,
            self.memory_pool.data.data_ptr(),
            self.config.vocab_size,
            self.config.num_hidden_layers,
            self.config.hidden_size,
            self.config.intermediate_size,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            self.config.rms_norm_eps,
            self.config.rope_theta,
            dtype_to_int(self.dtype),
            self.chunk_length,
        )

    def _load(self, name, param):
        if 'o_proj' in name or 'down_proj' in name:
            param = param.transpose(0, 1)

        param = param.contiguous().to(self.dtype)
        C.load_model(name, param.data_ptr())

    def load_from_hf(self):
        with torch.no_grad():
            if os.path.exists(os.path.join(self.path, "pytorch_model.bin")):
                ckpt = torch.load(os.path.join(self.path, "pytorch_model.bin"), map_location="cpu")
                for name, param in ckpt.items():
                    self._load(name, param)
            elif os.path.exists(os.path.join(self.path, "pytorch_model.bin.index.json")):
                with open(os.path.join(self.path, "pytorch_model.bin.index.json"), "r") as f:
                    file_list = set(json.load(f)["weight_map"].values())
                for file in file_list:
                    ckpt = torch.load(os.path.join(self.path, file), map_location="cpu")
                    for name, param in ckpt.items():
                        self._load(name, param)
            else:
                raise NotImplementedError(f"Unsupported checkpoint format for {self.path}")

    def prefill(self, input_ids, position_ids, output_ids):
        for i in range(0, input_ids.numel(), self.chunk_length):
            torch.cuda.nvtx.range_push(f"chunk from {i}")
            C.prefill(min(input_ids.numel() - i, self.chunk_length), i, input_ids.view(-1)[i:].data_ptr(), position_ids.view(-1)[i:].data_ptr(), output_ids.view(-1)[i:].data_ptr())
            torch.cuda.nvtx.range_pop()
        return output_ids

    def decode(self, input_ids, position_ids, cache_length, output_ids, cuda_graph=False):
        torch.cuda.nvtx.range_push(f"decode")
        C.decode(input_ids.numel(), input_ids.data_ptr(), position_ids.data_ptr(), cache_length.data_ptr(), output_ids.data_ptr(), cuda_graph)
        torch.cuda.nvtx.range_pop()
        return output_ids
