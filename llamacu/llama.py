from . import C

import os, json
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from safetensors.torch import load_file

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
        if not hasattr(self.config, "head_dim"):
            self.config.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.max_total_length = C.init_model(
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
            dtype_to_int(self.dtype),
            self.chunk_length,
        )

        self.logits = torch.empty((64, self.config.vocab_size), dtype=self.dtype, device="cuda")

    def _load(self, name, param, dtype=None):
        if dtype is None:
            if 'rotary_emb' in name:
                dtype = torch.float32
            else:
                dtype = self.dtype

        if 'gate_up_proj' in name:
            self._load(name.replace("gate_up_proj", "gate_proj"), param[:param.shape[0]//2], dtype)
            self._load(name.replace("gate_up_proj", "up_proj"), param[param.shape[0]//2:])
        elif 'qkv_proj' in name:
            self._load(name.replace("qkv_proj", "q_proj"), param[:self.config.num_attention_heads * self.config.head_dim])
            self._load(name.replace("qkv_proj", "k_proj"), param[self.config.num_attention_heads * self.config.head_dim:(self.config.num_attention_heads + self.config.num_key_value_heads) * self.config.head_dim])
            self._load(name.replace("qkv_proj", "v_proj"), param[(self.config.num_attention_heads + self.config.num_key_value_heads) * self.config.head_dim:])
        else:
            param = param.contiguous().to(dtype)
            C.load_model(name, param.data_ptr())

        if "embed_tokens" in name and hasattr(self.config, "tie_word_embeddings") and self.config.tie_word_embeddings:
            self._load("lm_head", param)

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
            elif os.path.exists(os.path.join(self.path, "model.safetensors")):
                ckpt = load_file(os.path.join(self.path, "model.safetensors"))
                for name, param in ckpt.items():
                    self._load(name, param)
            elif os.path.exists(os.path.join(self.path, "model.safetensors.index.json")):
                with open(os.path.join(self.path, "model.safetensors.index.json"), "r") as f:
                    file_list = set(json.load(f)["weight_map"].values())
                for file in file_list:
                    ckpt = load_file(os.path.join(self.path, file))
                    for name, param in ckpt.items():
                        self._load(name, param)
            else:
                raise NotImplementedError(f"Unsupported checkpoint format for {self.path}")

            # rope
            if hasattr(self.config, "rope_scaling") and self.config.rope_scaling is not None:
                rope_type = self.config.rope_scaling.get("rope_type", self.config.rope_scaling.get("type"))
            else:
                rope_type = "default"
            # TODO only support "default", "llama3" or "longrope" with long_factor=short_factor
            inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](self.config, "cpu", seq_len=self.max_total_length)
            # attention_scaling = torch.tensor([attention_scaling], dtype=torch.float32, device="cpu")
            self._load("model.rotary_emb.inv_freq", inv_freq, dtype=torch.float32)
            # self._load("model.rotary_emb.attention_scaling", attention_scaling, dtype=torch.float32)

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

    def decode(self, input_ids, position_ids, cache_length, mask_2d = None):
        assert input_ids.dtype == torch.int32
        assert position_ids.dtype == torch.int32
        assert cache_length.dtype == torch.int32
        if mask_2d is not None:
            assert mask_2d.dtype == torch.int32
            assert input_ids.numel() == mask_2d.shape[0]

        torch.cuda.nvtx.range_push(f"decode")
        cache_length += input_ids.numel() # temparary add for convinience in flash_attn
        padded_length = (cache_length[0].item() + 128 - 1) // 128 * 128
        C.decode(
            input_ids.numel(), padded_length,
            input_ids.data_ptr(), position_ids.data_ptr(), cache_length.data_ptr(),
            mask_2d.data_ptr() if mask_2d is not None else 0,
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
