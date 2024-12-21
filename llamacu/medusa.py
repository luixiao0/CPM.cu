from . import C
from .llama import LLM

import torch
from transformers import AutoConfig

# /////////// BELOW PART COMES FROM https://github.com/FasterDecoding/Medusa ///////////

mc_sim_7b_63 = [[0], [0, 0], [1], [0, 1], [2], [0, 0, 0], [1, 0], [0, 2], [3], [0, 3], [4], [0, 4], [2, 0], [0, 5], [0, 0, 1], [5], [0, 6], [6], [0, 7], [0, 1, 0], [1, 1], [7], [0, 8], [0, 0, 2], [3, 0], [0, 9], [8], [9], [1, 0, 0], [0, 2, 0], [1, 2], [0, 0, 3], [4, 0], [2, 1], [0, 0, 4], [0, 0, 5], [0, 0, 0, 0], [0, 1, 1], [0, 0, 6], [0, 3, 0], [5, 0], [1, 3], [0, 0, 7], [0, 0, 8], [0, 0, 9], [6, 0], [0, 4, 0], [1, 4], [7, 0], [0, 1, 2], [2, 0, 0], [3, 1], [2, 2], [8, 0], [0, 5, 0], [1, 5], [1, 0, 1], [0, 2, 1], [9, 0], [0, 6, 0], [0, 0, 0, 1], [1, 6], [0, 7, 0]]

import torch
import torch.nn.functional as F

TOPK=10 # topk for sparse tree (10 is a placeholder and it is sufficient)

def pad_path(path, length, pad_value=-2):
    """
    Pad the given path list with a specific value up to a specified length.
    
    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.
    
    Returns:
    - list: A new list based on the original path but padded to the desired length.
    
    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]
    
    Note:
    If the given path is already longer than the specified length, 
    then no padding occurs, and the original path is returned.
    """
    
    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))

def generate_medusa_buffers(medusa_choices, device="cuda"):
    """
    Generate buffers for the Medusa structure based on the provided choices.
    
    Parameters:
    - medusa_choices (list): A nested list representing tree in the Medusa structure.
    - device (str): Device to which the tensors should be moved. Default is "cuda".
    
    Returns:
    - dict: A dictionary containing buffers related to the Medusa structure.
    """

    # Sort the medusa_choices based on their lengths and then their values
    sorted_medusa_choices = sorted(medusa_choices, key=lambda x: (len(x), x))
    medusa_len = len(sorted_medusa_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_medusa_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    
    # Create the attention mask for Medusa
    medusa_attn_mask = torch.eye(medusa_len, medusa_len)
    medusa_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            # retrieve ancestor position
            if len(cur_medusa_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_medusa_choice) - 1):
                ancestor_idx.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]) + 1)
            medusa_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]

    # Generate tree indices for the Medusa structure
    medusa_tree_indices = torch.zeros(medusa_len, dtype=torch.long)
    medusa_tree_indices[0] = 0
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_medusa_choice = sorted_medusa_choices[start + j]
            medusa_tree_indices[start + j + 1] = cur_medusa_choice[-1] + TOPK * i + 1
        start += depth_counts[i]

    # Generate position IDs for the Medusa structure
    medusa_position_ids = torch.zeros(medusa_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        medusa_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    # Generate retrieval indices for Medusa structure verification
    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_medusa_choices)):
        cur_medusa_choice = sorted_medusa_choices[-i-1]
        retrieve_indice = []
        if cur_medusa_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_medusa_choice)):
                retrieve_indice.append(sorted_medusa_choices.index(cur_medusa_choice[:c+1]))
                retrieve_paths.append(cur_medusa_choice[:c+1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices], dim=1)

    # Aggregate the generated buffers into a dictionary
    medusa_buffers = {
        "medusa_attn_mask": medusa_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": medusa_tree_indices,
        "medusa_position_ids": medusa_position_ids,
        "retrieve_indices": retrieve_indices,
        }
    
    # Move the tensors in the dictionary to the specified device
    medusa_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v,  device=device)
        for k, v in medusa_buffers.items()
    }
    return medusa_buffers

from transformers import PretrainedConfig
class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path

# /////////// ABOVE PART COMES FROM https://github.com/FasterDecoding/Medusa ///////////

def pack_mask(mask_2d):
    mask_2d_packed = torch.zeros((mask_2d.shape[0], 2), dtype=torch.uint32, device="cuda")
    for i in range(mask_2d.shape[0]):
        mask_1 = 0
        mask_2 = 0
        for j in range(i + 1):
            if j < 32:
                mask_1 |= (mask_2d[i][j].item() << j)
            else:
                mask_2 |= (mask_2d[i][j].item() << (j - 32))
        mask_2d_packed[i][0] = mask_1
        mask_2d_packed[i][1] = mask_2
    mask_2d_packed = mask_2d_packed.view(torch.uint64)
    return mask_2d_packed

class LLM_with_medusa(LLM):
    def __init__(self,
                 medusa_path,
                 base_path,
                 **kwargs):
        super().__init__(base_path, **kwargs)

        self.medusa_path = medusa_path
        self.medusa_config = MedusaConfig.from_pretrained(medusa_path)

        self.medusa_choices = mc_sim_7b_63
        self.medusa_config.medusa_num_heads = 4 # TODO 4 for mc_sim_7b_63

        self.medusa_topk = TOPK
        medusa_buffers = generate_medusa_buffers(self.medusa_choices)
        self.medusa_tree_indices = medusa_buffers["tree_indices"][1:] - 1
        self.medusa_attn_mask = pack_mask(medusa_buffers["medusa_attn_mask"][0][0].to(torch.int32))
        self.medusa_position_ids = medusa_buffers["medusa_position_ids"]
        self.medusa_tree_parent = torch.tensor([-1] * 64, dtype=torch.int32, device="cuda")
        for i in range(1, 64):
            for j in reversed(range(i)):
                if medusa_buffers["medusa_attn_mask"][0][0][i][j] == 1:
                    self.medusa_tree_parent[i] = j
                    break
            else:
                assert False, f"No parent found for {i}"

        assert self.medusa_config.medusa_num_layers == 1, "Currently only supports 1 layer"

        C.init_medusa_model(
            self.medusa_config.medusa_num_heads,
            self.medusa_config.medusa_num_layers,
            self.dtype_int,
        )

        self.medusa_logits = torch.empty((self.medusa_config.medusa_num_heads, self.config.vocab_size), dtype=self.dtype, device="cuda")

    def _load(self, name, param, dtype=None, cls=None):
        if cls == "medusa":
            if dtype is None:
                dtype = self.dtype
            param = param.contiguous().to(dtype)
            if int(name.split(".")[0]) < self.medusa_config.medusa_num_heads:
                C.load_model(f"{cls}.{name}", param.data_ptr())
        else:
            super()._load(name, param, dtype)

    def load_from_hf(self):
        self._load_from_ckpt(self.medusa_path, cls="medusa")
        super().load_from_hf()

    def generate(self, input_ids, generation_length=100, teminators=[]):
        assert input_ids.dtype == torch.int32

        prefix_length = input_ids.numel()
        position_ids = torch.arange(prefix_length, dtype=torch.int32, device="cuda")
        logits = self.prefill(input_ids, position_ids)
        token = logits[0].argmax(dim=-1).item()

        tokens = [token]
        accept_lengths = []
        if not hasattr(self, "input_ids"):
            self.input_ids = torch.tensor([0]*64, dtype=torch.int32, device="cuda")
            self.position_ids = torch.tensor([0]*64, dtype=torch.int32, device="cuda")
            self.cache_length = torch.tensor([0], dtype=torch.int32, device="cuda")
            self.gt = torch.tensor([0]*64, dtype=torch.int32, device="cuda")
        i = 0
        model_step = 0
        terminal = False
        while i < generation_length and not terminal:
            torch.cuda.nvtx.range_push(f"medusa_draft")
            C.draft(self.medusa_logits.data_ptr())
            topk = self.medusa_logits.topk(self.medusa_topk, dim=-1).indices.view(-1)

            self.input_ids[0] = token
            self.input_ids[1:] = topk[self.medusa_tree_indices]
            self.position_ids.copy_((prefix_length + i) + self.medusa_position_ids)
            self.cache_length[0] = prefix_length + i
            torch.cuda.nvtx.range_pop()

            logits = self.decode(self.input_ids, self.position_ids, self.cache_length, mask_2d=self.medusa_attn_mask)
            self.gt.copy_(logits.argmax(dim=-1))

            torch.cuda.nvtx.range_push(f"medusa_verify")
            accept_length = C.verify(
                self.input_ids.numel(), self.input_ids.data_ptr(), self.gt.data_ptr(),
                self.position_ids.data_ptr(), self.cache_length.data_ptr(),
                self.medusa_attn_mask.data_ptr(), self.medusa_tree_parent.data_ptr()
            )
            torch.cuda.nvtx.range_pop()

            i += accept_length
            model_step += 1
            accept_lengths.append(accept_length)
            tokens.extend(self.input_ids[:accept_length].tolist())
            token = tokens[-1]
            
            for temin in teminators:
                if temin in self.input_ids[:accept_length]:
                    terminal = True
            
        tokens = tokens[:generation_length]
        
        return tokens, accept_lengths, model_step