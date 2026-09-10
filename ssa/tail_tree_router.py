"""Per-token queries into SSA's existing append-only center/radius block tree.

Only completed past blocks enter the tree. Each query is independent (no pooling
with future query tokens). Fixed-beam search is approximate routing, not an
attention-mass certificate. This deliberately simple adapter returns the block
ids consumed by hybrid_tail_attention; it does not read archived values.
"""
import torch


@torch.no_grad()
def tail_tree_routes(q, k, *, block=64, top_blocks=2, beam=32, fanout=4):
    from .cascade_router import CausalTree, DEV

    b, h, n, d = q.shape
    if str(q.device).split(":")[0] != DEV:
        raise ValueError("CausalTree backend device must match the input")
    if k.shape[0] != b or k.shape[2:] != (n, d) or h % k.shape[1]:
        raise ValueError("requires aligned GQA query/key tensors")
    if block < 1 or top_blocks < 1 or beam < top_blocks:
        raise ValueError("invalid tree routing configuration")
    result = torch.full((b, h, n, top_blocks), -1, device=q.device, dtype=torch.long)
    group = h // k.shape[1]
    for bi in range(b):
        for hk in range(k.shape[1]):
            tree = CausalTree(d, block=block, sub=block, n_hint=((n + block - 1) // block) * block,
                             top_c=top_blocks, search_k=top_blocks, tree_beam=beam,
                             tree_fanout=fanout, outlier_rate=0, outlier_cap=0)
            first = hk * group
            for start in range(block, n, block):
                tree.append(k[bi, hk, start - block:start].detach())
                tree._commit()
                end = min(start + block, n)
                queries = q[bi, first:first + group, start:end].reshape(-1, d).float()
                scores, ids = tree._search_committed(queries)
                ids = ids.masked_fill(~torch.isfinite(scores), -1)
                count = min(top_blocks, ids.shape[-1])
                result[bi, first:first + group, start:end, :count] = ids[:, :count].reshape(group, end - start, count)
    return result
