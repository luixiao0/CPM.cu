from .. import C
from transformers import AutoTokenizer, AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from safetensors.torch import load_file
import os, json, glob
from ..llama_w4a8_per_chn import W4A8PerChnLLM

import torch

def pack_draft_mask(mask_2d):
    '''
    for static masks, pack them into a uint64 per row
    '''
    mask_2d_packed = torch.zeros((mask_2d.shape[0]), dtype=torch.uint16, device="cuda")
    for i in range(mask_2d.shape[0]):
        mask_1 = 0
        for j in range(i + 1):
            mask_1 |= (mask_2d[i][j].item() << j )
        mask_2d_packed[i] = mask_1
    mask_2d_packed = mask_2d_packed.view(torch.uint16).view(-1)
    return mask_2d_packed


class W4A8PerChnLLM_with_W4A16FMspec(W4A8PerChnLLM):
    def __init__(self,
                 drafter_path: str,
                 base_path: str,
                 draft_num: int,
                 draft_cuda_graph: bool,
                  **kwargs):
        super().__init__(base_path, **kwargs)
        
        self.drafter_type = 'draft'
        self.drafter_path = drafter_path
        self.drafter_tokenizer = AutoTokenizer.from_pretrained(drafter_path)
        self.drafter_config = AutoConfig.from_pretrained(drafter_path)

        self.draft_num = draft_num
        self.draft_ids = torch.empty((self.draft_num+1), dtype=torch.int32, device="cuda")
        self.draft_position_ids = torch.empty((self.draft_num+1), dtype=torch.int32, device="cuda")
        self.draft_gt_ids = torch.empty((self.draft_num+1), dtype=torch.int32, device="cuda")
        self.draft_attn_mask = pack_draft_mask(
            torch.tril(torch.ones(draft_num+1, draft_num+1, dtype=torch.bool)).to("cuda")
        )
        self.draft_parent = torch.tensor([], dtype=torch.int32, device="cuda")
        self.cache_length = torch.tensor([0], dtype=torch.int32, device="cuda")


        self.draft_prefill_logits = torch.empty((64, self.config.vocab_size), dtype=self.dtype, device="cuda")
        # self.draft_prefill_logits = torch.empty((64, self.config.hidden_size), dtype=self.dtype, device="cuda")
        # self.logits = torch.empty((64, self.config.hidden_size), dtype=self.dtype, device="cuda")
        self.draft_cuda_graph = draft_cuda_graph
        
        C.init_w4a16_gptq_marlin_spec_w4a8_per_chn_model(
            self.drafter_config.vocab_size,
            self.drafter_config.num_hidden_layers,
            self.drafter_config.hidden_size,
            self.drafter_config.intermediate_size,
            self.drafter_config.num_attention_heads,
            self.drafter_config.num_key_value_heads,
            self.drafter_config.head_dim,
            self.drafter_config.rms_norm_eps,
            self.draft_num,
            self.draft_cuda_graph,
            0,
        )
    
    def _load(self, name, param, dtype=None, cls=None):
        if cls == self.drafter_type:
            if dtype is None:
                if 'rotary_emb' in name:
                    dtype = torch.float32
                else:
                    dtype = self.dtype

            if 'gate_up_proj' in name:
                self._load(name.replace("gate_up_proj", "gate_proj"), param[:param.shape[0]//2], dtype, cls=cls)
                self._load(name.replace("gate_up_proj", "up_proj"), param[param.shape[0]//2:], cls=cls)
            elif 'qkv_proj' in name:
                self._load(name.replace("qkv_proj", "q_proj"), param[:self.config.num_attention_heads * self.config.head_dim], cls=cls)
                self._load(name.replace("qkv_proj", "k_proj"), param[self.config.num_attention_heads * self.config.head_dim:(self.config.num_attention_heads + self.config.num_key_value_heads) * self.config.head_dim], cls=cls)
                self._load(name.replace("qkv_proj", "v_proj"), param[(self.config.num_attention_heads + self.config.num_key_value_heads) * self.config.head_dim:], cls=cls)
            else:
                param = param.contiguous()
                if param.dtype not in [torch.int8, torch.int16, torch.int32]:
                    param = param.to(dtype)
                C.load_model(f"{cls}.{name}", param.data_ptr())

            if "embed_tokens" in name and hasattr(self.config, "tie_word_embeddings") and self.config.tie_word_embeddings:
                self._load("lm_head", param, cls)
        else:
            super()._load(name, param, dtype)
    
    def _load_from_ckpt(self, path, cls=None):
        supported_suffix_1 = ["bin.index.json", "safetensors.index.json"]
        supported_suffix_2 = ["bin", "safetensors", "pt"]
        file = None
        for suffix in supported_suffix_1:
            files = glob.glob(os.path.join(path, f"*.{suffix}"))
            if len(files) > 1:
                raise ValueError(f"Multiple files with suffix {suffix} found in {path}")
            elif len(files) == 1:
                file = files[0]
                break
        else:
            for suffix in supported_suffix_2:
                files = glob.glob(os.path.join(path, f"*.{suffix}"))
                if len(files) > 1:
                    print(files)
                    if path + "/model_gptq_marlin.safetensors" in files:
                            file = path + "/model_gptq_marlin.safetensors"
                    else:
                        raise ValueError(f"Autogptq models not found in {path}")
                    break
                elif len(files) == 1:
                    file = files[0]
                    break
            else:
                raise ValueError(f"No supported checkpoint file found in {path}, supported suffixes: {supported_suffix_1 + supported_suffix_2}")

        if file.endswith(".index.json"):
            with open(file, "r") as f:
                file_list = set(json.load(f)["weight_map"].values())
            file_list = [os.path.join(path, file) for file in file_list]
        else:
            file_list = [file]

        for file in file_list:
            print(f"load from {file}")
            if file.endswith(".bin") or file.endswith(".pt"):
                ckpt = torch.load(file, map_location="cpu")
            elif file.endswith(".safetensors"):
                ckpt = load_file(file)
            for name, param in ckpt.items():
                self._load(name, param, cls=cls)
    
    
    def load_from_hf(self):
        with torch.no_grad():
            self._load_from_ckpt(self.drafter_path, cls=self.drafter_type)
            # rope
            if hasattr(self.drafter_config, "rope_scaling") and self.drafter_config.rope_scaling is not None:
                draft_rope_type = self.drafter_config.rope_scaling.get("rope_type", self.drafter_config.rope_scaling.get("type"))
            else:
                draft_rope_type = "default"
            # TODO only support "default", "llama3" or "longrope" with long_factor=short_factor
            draft_inv_freq, draft_attention_scaling = ROPE_INIT_FUNCTIONS[draft_rope_type](self.drafter_config, "cpu", seq_len=self.max_total_length)
            # attention_scaling = torch.tensor([attention_scaling], dtype=torch.float32, device="cpu")
            self._load(f"{self.drafter_type}.model.rotary_emb.inv_freq", draft_inv_freq, dtype=torch.float32, cls=self.drafter_type)
            # self._load("model.rotary_emb.attention_scaling", attention_scaling, dtype=torch.float32)
        # super().load_from_hf()
        with torch.no_grad():
            super()._load_from_ckpt(self.path)

            # rope
            if hasattr(self.config, "rope_scaling") and self.config.rope_scaling is not None:
                rope_type = self.config.rope_scaling.get("rope_type", self.config.rope_scaling.get("type"))
            else:
                rope_type = "default"
            # TODO only support "default", "llama3" or "longrope" with long_factor=short_factor
            inv_freq, attention_scaling = ROPE_INIT_FUNCTIONS[rope_type](self.config, "cpu", seq_len=self.max_total_length)
            # attention_scaling = torch.tensor([attention_scaling], dtype=torch.float32, device="cpu")
            super()._load("model.rotary_emb.inv_freq", inv_freq, dtype=torch.float32)
            # self._load("model.rotary_emb.attention_scaling", attention_scaling, dtype=torch.float32)
    
    

    def generate(self, input_ids, generation_length=100, teminators=[]):
        assert input_ids.dtype == torch.int32

        prefix_length = input_ids.numel()
        position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
        logits = self.prefill(input_ids, position_ids)
        self.draft_ids[:1].copy_(logits[0].argmax(dim=-1))

        tokens = torch.empty((generation_length), dtype=torch.int32, device="cuda")
        tokens[0].copy_(self.draft_ids[0])
        accept_lengths = []
        i = 0
        model_step = 0
        terminal = False
        while i < generation_length-1 and not terminal:
            self.cache_length[0] = prefix_length + i
            self.draft_position_ids[0] = prefix_length + i

            torch.cuda.nvtx.range_push(f"draft")
            self.cache_length += 1
            C.draft(self.draft_ids.data_ptr(), self.draft_position_ids.data_ptr(), self.cache_length.data_ptr(), self.draft_attn_mask.data_ptr(), self.draft_parent.data_ptr())
            self.cache_length -= 1
            torch.cuda.nvtx.range_pop()

            
            logits = self.decode(self.draft_ids, self.draft_position_ids, self.cache_length, mask_2d=self.draft_attn_mask)
            self.draft_gt_ids.copy_(logits.argmax(dim=-1))

            torch.cuda.nvtx.range_push(f"verify")
            accept_length = C.verify_and_fix(
                self.draft_ids.numel(), self.draft_ids.data_ptr(), self.draft_gt_ids.data_ptr(),
                self.draft_position_ids.data_ptr(), self.cache_length.data_ptr(),
                self.draft_attn_mask.data_ptr(), self.draft_parent.data_ptr()
            )
            torch.cuda.nvtx.range_pop()

            model_step += 1
            accept_lengths.append(accept_length)
            for temin in teminators:
                if temin in self.draft_gt_ids[:accept_length]:
                    terminal = True
            append_length = min(accept_length, generation_length - 1 - i)
            tokens[1+i:1+i+append_length].copy_(self.draft_gt_ids[:append_length])
            self.draft_ids[0] = self.draft_gt_ids[accept_length - 1]
            i += accept_length

        tokens = tokens[:1+i].tolist()
        return tokens, accept_lengths, model_step