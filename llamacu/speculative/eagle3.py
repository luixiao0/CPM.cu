from .. import C
from .tree_drafter import LLM_with_tree_drafter

import torch
from transformers import PretrainedConfig

class Eagle3Config(PretrainedConfig):
    def __init__(
        self,
        draft_vocab_size=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.draft_vocab_size = draft_vocab_size

class LLM_with_eagle3(LLM_with_tree_drafter):
    def __init__(self,
                 eagle_path,
                 base_path,
                 num_iter=6,
                 topk_per_iter=10,
                 tree_size=60,
                 **kwargs):
        super().__init__(
            "eagle3", eagle_path, base_path,
            tree_size = tree_size,
            **kwargs
        )

        self.eagle_path = eagle_path
        self.eagle_config = Eagle3Config.from_pretrained(eagle_path)

        C.init_eagle3_model(
            self.eagle_config.draft_vocab_size,
            num_iter,
            topk_per_iter,
            self.tree_size,
            self.dtype_int,
        )

    def _load(self, name, param, dtype=None, cls=None):
        if cls == self.drafter_type:
            if dtype is None:
                dtype = self.dtype
            
            if 'd2t' in name:
                param = param.contiguous().to(torch.int)
            elif 't2d' not in name:
                param = param.contiguous().to(dtype)

            if 'embed_tokens' in name:
                return
            if 'fc' in name:
                if 'weight' in name:
                    split_dim = param.shape[-1] // 3
                    param1 = param[..., :split_dim].contiguous()
                    param2 = param[..., split_dim: split_dim*2].contiguous()
                    param3 = param[..., split_dim*2:].contiguous()
                    C.load_model(f"{cls}.{name.replace('fc', 'fc1')}", param1.data_ptr())
                    C.load_model(f"{cls}.{name.replace('fc', 'fc2')}", param2.data_ptr())
                    C.load_model(f"{cls}.{name.replace('fc', 'fc3')}", param3.data_ptr())
                else: # bias
                    C.load_model(f"{cls}.{name.replace('fc', 'fc1')}", param.data_ptr())
            else:
                C.load_model(f"{cls}.{name}", param.data_ptr())
        else:
            super()._load(name, param, dtype)
