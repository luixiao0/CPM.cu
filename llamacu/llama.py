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
                 cuda_graph: bool = False,
    ):
        super().__init__()

        self.path = path
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.config = AutoConfig.from_pretrained(path)
        self.dtype = dtype if dtype is not None else self.config.torch_dtype
        self.cuda_graph = cuda_graph

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

        self.logits = torch.empty((64, self.config.vocab_size), dtype=self.dtype, device="cuda")

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

    def prefill(self, input_ids, position_ids):
        assert input_ids.dtype == torch.int32
        for i in range(0, input_ids.numel(), self.chunk_length):
            torch.cuda.nvtx.range_push(f"chunk from {i}")
            C.prefill(
                min(input_ids.numel() - i, self.chunk_length), i,
                input_ids.view(-1)[i:].data_ptr(), position_ids.view(-1)[i:].data_ptr(),
                self.logits.data_ptr()
            )
            torch.cuda.nvtx.range_pop()
        return self.logits[:1].clone()

    def decode(self, input_ids, position_ids, cache_length):
        assert input_ids.dtype == torch.int32
        assert position_ids.dtype == torch.int32
        assert cache_length.dtype == torch.int32

        torch.cuda.nvtx.range_push(f"decode")
        cache_length += input_ids.numel() # temparary add for convinience in flash_attn
        padded_length = (cache_length[0].item() + 128 - 1) // 128 * 128
        C.decode(
            input_ids.numel(), padded_length,
            input_ids.data_ptr(), position_ids.data_ptr(), cache_length.data_ptr(),
            self.logits.data_ptr(),
            self.cuda_graph
        )
        cache_length -= input_ids.numel()
        torch.cuda.nvtx.range_pop()
        return self.logits[:input_ids.numel()].clone()

    def generate(self, input_ids, generation_length=100):
        assert input_ids.dtype == torch.int32

        tokens = []

        prefix_length = input_ids.numel()
        position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
        logits = self.prefill(input_ids, position_ids)
        token = logits[0].argmax(dim=-1).item()
        tokens.append(token)

        input_ids = torch.tensor([[0]], dtype=torch.int32, device="cuda") # TODO move these to c and capture in graph
        position_ids = torch.tensor([[0]], dtype=torch.int32, device="cuda") # TODO move these to c and capture in graph
        cache_length = torch.tensor([0], dtype=torch.int32, device="cuda") # TODO move these to c and capture in graph
        for i in range(generation_length):
            input_ids[0][0] = token # TODO move these to c and capture in graph
            position_ids[0][0] = prefix_length + i # TODO move these to c and capture in graph
            cache_length[0] = prefix_length + i # TODO move these to c and capture in graph

            logits = self.decode(input_ids, position_ids, cache_length)
            token = logits[0].argmax(dim=-1).item() # TODO move these to c and capture in graph
            tokens.append(token)
        return self.tokenizer.decode(tokens)
