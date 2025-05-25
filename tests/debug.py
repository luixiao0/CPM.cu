import torch
import torch.nn.functional as F

def load_file(file_name):
    with open(file_name, 'r') as f:
        n, m = map(int, f.readline().split())
        arr = torch.zeros(n, m)
        for i in range(n):
            arr[i] = torch.tensor(list(map(float, f.readline().split())))
    return arr

def load_mask(file_name):
    arr = []
    with open(file_name, 'r') as f:
        f.readline()
        for line in f:
            arr.append(line)
    return arr

q = load_file("q.txt")
k_cache = load_file("k_cache.txt")
v_cache = load_file("v_cache.txt")
c1_cache = load_file("c1_cache.txt")
stage1_score = load_file("stage1_score.txt")
pool_score = load_file("pool_score.txt")
topk_pos = load_file("topk.txt")
blockmask = load_mask("blockmask.txt")

def naive_attention(q, k, causal=False):
    # 计算 attention
    k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
    # 计算 attention
    attn = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
    if causal:
        causal_mask = torch.zeros(q.shape[1], k.shape[1], device=q.device).bool()
        for i in range(q.shape[1]):
            for j in range(k.shape[1]):
                if i >= (j * 16 + 32 - 1):
                    causal_mask[i, j] = True
        attn = attn.masked_fill(~causal_mask, -float('inf'))
    score = F.softmax(attn, dim=-1)
    score = score.reshape(2, 16, q.shape[1], k.shape[1]).sum(dim=1)
    return score

c1_cache_ref = F.avg_pool1d(
    k_cache.transpose(0, 1),
    kernel_size=32,
    stride=16,
).transpose(0, 1)

print((c1_cache_ref - c1_cache).abs().max())

stage1_ref = naive_attention(
    q.reshape(q.shape[0], 16, 2, -1).transpose(1, 2).reshape(q.shape[0], 32, -1).transpose(0, 1).contiguous(),
    c1_cache_ref.view(c1_cache.shape[0], 2, -1).transpose(0, 1).contiguous(),
)
stage1_score = stage1_score.view(2, -1, stage1_score.shape[-1])

print((stage1_ref - stage1_score[:, :stage1_ref.shape[1], :stage1_ref.shape[2]]).abs().max())

pool_ref = F.max_pool1d(
    stage1_ref,
    kernel_size=5,
    stride=4,
    padding=1
)[..., :pool_score.shape[-1]]
pool_score = pool_score.view(2, -1, pool_score.shape[-1])
mask = (pool_score == float('inf')) | (pool_score == float('-inf'))
print((pool_ref[~mask] - pool_score[~mask]).abs().max())
pool_ref[pool_score == float('inf')] = float('inf')
pool_ref[pool_score == float('-inf')] = float('-inf')

topk_ref = pool_ref.topk(
    k=32,
    dim=-1
).indices
topk_pos = topk_pos.view(2, -1, topk_pos.shape[-1]).int()
# Get the indices where topk_ref and topk_pos differ
diff_mask = (topk_ref != topk_pos)

# For each differing position, check if the corresponding values in pool_ref are equal
for i in range(2):  # Loop over the 2 dimensions
    for j in range(diff_mask[i].shape[0]):  # Loop over rows
        for k in range(diff_mask[i].shape[1]):  # Loop over columns
            if diff_mask[i,j,k]:
                ref_idx = topk_ref[i,j,k]
                pos_idx = topk_pos[i,j,k]
                assert torch.isclose(pool_ref[i,j,ref_idx], pool_ref[i,j,pos_idx], rtol=1e-2, atol=1e-2), \
                    f"Values differ at position ({i},{j}): {pool_ref[i,j,ref_idx]} != {pool_ref[i,j,pos_idx]}"

from IPython import embed; embed()