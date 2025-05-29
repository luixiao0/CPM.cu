from ... import C
from ..eagle import EagleConfig
from ..tree_drafter_base_quant.tree_drafter_w4a16_gptq_marlin import W4A16GPTQMarlinLLM_with_tree_drafter

import math


class W4A16GPTQMarlinLLM_with_eagle(W4A16GPTQMarlinLLM_with_tree_drafter):
    def __init__(self,
                 eagle_path,
                 base_path,
                 num_iter=6,
                 topk_per_iter=10,
                 tree_size=60,
                 apply_eagle_quant: bool=False,
                 use_rope: bool=False,
                 use_input_norm: bool=False,
                 use_attn_norm: bool=False,
                 use_rotation: bool=False,
                 **kwargs):
        super().__init__(
            "eagle", eagle_path, base_path,
            tree_size = tree_size,
            **kwargs
        )

        self.eagle_path = eagle_path
        self.eagle_config = EagleConfig.from_pretrained(eagle_path)
        # Ensure presence consistency and equality for scale_depth, dim_model_base, and scale_emb
        for attr in ("scale_depth", "dim_model_base", "scale_emb"):
            base_has = hasattr(self.config, attr)
            eagle_has = hasattr(self.eagle_config, attr)
            assert base_has == eagle_has, f"{attr} presence mismatch between base and eagle config"
            if base_has:
                assert getattr(self.config, attr) == getattr(self.eagle_config, attr), f"{attr} in base config and eagle config should be the same"
        scale_residual = self.config.scale_depth / math.sqrt(self.config.num_hidden_layers + 1) if hasattr(self.config, "scale_depth") else 1.0
        print(f"eagle scale_residual: {scale_residual}")
        self.use_rotation = use_rotation

        if not apply_eagle_quant:
            if not use_rope and not use_input_norm and not use_attn_norm:
                if not use_rotation:
                    C.init_eagle_model(
                        self.eagle_config.eagle_num_layers,
                        num_iter,
                        topk_per_iter,
                        self.tree_size,
                        self.dtype_int
                    )
                else:
                    C.init_eagle_w4a16_gptq_marlin_rot_model(
                        self.eagle_config.eagle_num_layers,
                        num_iter,
                        topk_per_iter,
                        self.tree_size,
                        self.dtype_int
                    )
            else:
                C.init_minicpm4_eagle_model(
                    self.eagle_config.eagle_num_layers,
                    num_iter,
                    topk_per_iter,
                    self.tree_size,
                    self.dtype_int,
                    scale_residual,
                    use_input_norm, 
                    use_attn_norm
                )
        else:
            pass

    def _load(self, name, param, dtype=None, cls=None):
        if cls == self.drafter_type:
            if dtype is None:
                dtype = self.dtype
            param = param.contiguous().to(dtype)
            if (not self.use_rotation) and 'embed_tokens' in name:
                return
            if 'fc' in name:
                if 'weight' in name:
                    param1 = param[..., :param.shape[-1] // 2].contiguous()
                    param2 = param[..., param.shape[-1] // 2:].contiguous()
                    C.load_model(f"{cls}.{name.replace('fc', 'fc1')}", param1.data_ptr())
                    C.load_model(f"{cls}.{name.replace('fc', 'fc2')}", param2.data_ptr())
                else: # bias
                    C.load_model(f"{cls}.{name.replace('fc', 'fc1')}", param.data_ptr())
            else:
                C.load_model(f"{cls}.{name}", param.data_ptr())
        else:
            super()._load(name, param, dtype)
