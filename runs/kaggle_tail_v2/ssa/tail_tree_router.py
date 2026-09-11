"""Per-token queries into SSA's existing append-only center/radius block tree.

Only completed past blocks enter the tree. Each query is independent (no pooling
with future query tokens). Fixed-beam search is approximate routing, not an
attention-mass certificate. This deliberately simple adapter returns the block
ids consumed by hybrid_tail_attention; it does not read archived values.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def batched_tail_tree_routes(q, k, *, block=64, top_blocks=2, beam=32, fanout=4,
                             query_chunk=256, return_stats=False):
    """Batch independent causal prefix forests; never search a future-bearing node.

    All summaries can be built in parallel, but each query starts from the exact
    base-fanout decomposition of its completed prefix. Future subtrees do not
    enter that frontier. Beam pruning is approximate, as in CausalTree. Storage
    O(n*d/block); tested node slots O(n*heads*beam*fanout*log(n/block)).
    """
    from .cascade_router import _outward_nonnegative, _outward_score_cap
    b, h, n, d = q.shape
    if k.shape[0] != b or k.shape[2:] != (n, d) or h % k.shape[1]:
        raise ValueError("requires aligned GQA query/key tensors")
    if block < 1 or top_blocks < 1 or beam < top_blocks or fanout < 2 or query_chunk < 1:
        raise ValueError("invalid tree configuration")
    nb = (n + block - 1) // block
    capacity, height = 1, 0
    while capacity < nb:
        capacity *= fanout; height += 1
    leaf_base = (capacity - 1) // (fanout - 1)
    total = leaf_base + capacity
    centers = k.new_zeros(b, k.shape[1], total, d, dtype=torch.float32)
    radii = k.new_zeros(b, k.shape[1], total, dtype=torch.float32)
    means = F.pad(k.float(), (0, 0, 0, nb * block - n)).reshape(b, k.shape[1], nb, block, d).mean(-2)
    centers[:, :, leaf_base:leaf_base + nb] = means
    for depth in range(height - 1, -1, -1):
        count = fanout**depth
        first = (count - 1) // (fanout - 1)
        child_first = fanout * first + 1
        children = centers[:, :, child_first:child_first + count * fanout].reshape(b, k.shape[1], count, fanout, d)
        center = children.mean(-2)
        child_r = radii[:, :, child_first:child_first + count * fanout].reshape(b, k.shape[1], count, fanout)
        radius = _outward_nonnegative((children - center[..., None, :]).norm(dim=-1) + child_r, d).amax(-1)
        centers[:, :, first:first + count] = center
        radii[:, :, first:first + count] = torch.nextafter(radius, torch.full_like(radius, torch.inf))
    result = torch.full((b, h, n, top_blocks), -1, device=q.device, dtype=torch.long)
    bi = torch.arange(b, device=q.device)[:, None, None, None]
    hi = (torch.arange(h, device=q.device) // (h // k.shape[1]))[None, :, None, None]
    children_offset = torch.arange(fanout, device=q.device)
    node_slots = leaf_scores = 0
    for start in range(0, n, query_chunk):
        end = min(start + query_chunk, n)
        qc = q[:, :, start:end].float()
        prefix = torch.arange(start, end, device=q.device) // block
        frontier = []
        for level in range(height + 1):
            width = fanout**level
            digit = (prefix // width) % fanout
            local = (prefix // (width * fanout))[:, None] * fanout + children_offset[None, :fanout - 1]
            offset = (fanout**(height - level) - 1) // (fanout - 1)
            frontier.append((local + offset).masked_fill(children_offset[None, :fanout - 1] >= digit[:, None], -1))
        nodes = torch.cat(frontier, -1)[None, None].expand(b, h, -1, -1)

        def prune(ids):
            nonlocal node_slots
            c = centers[bi, hi, ids.clamp_min(0)]
            r = radii[bi, hi, ids.clamp_min(0)]
            upper = _outward_score_cap(qc.reshape(-1, d), c.reshape(-1, ids.shape[-1], d), r.reshape(-1, ids.shape[-1])).reshape_as(r)
            upper = upper.masked_fill(ids < 0, -torch.inf)
            node_slots += ids.numel()
            rank = upper.argsort(dim=-1, descending=True, stable=True)[..., :beam]
            return ids.gather(-1, rank)

        nodes = prune(nodes)
        for _ in range(height):
            internal = (nodes >= 0) & (nodes < leaf_base)
            children = fanout * nodes[..., None] + 1 + children_offset
            children = children.masked_fill(~internal[..., None], -1)
            children[..., 0] = torch.where(internal, children[..., 0], nodes)
            nodes = prune(children.flatten(-2))
        scores = (qc[..., None, :] * centers[bi, hi, nodes.clamp_min(0)]).sum(-1)
        scores = scores.masked_fill(nodes < leaf_base, -torch.inf)
        leaf_scores += nodes.numel()
        rank = scores.argsort(dim=-1, descending=True, stable=True)[..., :top_blocks]
        ids = nodes.gather(-1, rank) - leaf_base
        ids = ids.masked_fill(~torch.isfinite(scores.gather(-1, rank)), -1)
        result[:, :, start:end, :ids.shape[-1]] = ids
    stats = {"node_bound_slots": node_slots, "final_routing_score_slots": leaf_scores,
             "tree_summary_scalars": centers.numel() + radii.numel(), "height": height,
             "beam": beam, "fanout": fanout}
    return (result, stats) if return_stats else result


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
