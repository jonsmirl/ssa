"""Causal GQA sparse attention with a fixed-size positive tail state.

Complete-model reference implementation. Global block means route two completed
blocks; current-block keys are exact and causally masked. Cell counts and value
sums approximate unopened keys. Centers are fitted only from the first block;
that block is read densely until it is complete. Thus no future key affects an
earlier route, state, or output. The only trainable additions are log mass gains.

This implementation batches prefix states for training, gathers selected keys,
and never builds n-by-n scores. A flat block router still costs O(n^2/block).
It establishes quality and differentiability, not a subquadratic router/kernel.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def hybrid_tail_attention(q, k, v, *, block=64, top_blocks=2, cells=16,
                          log_gain=0.0, use_tail=True, query_chunk=32, routing_blocks=None):
    """q[B,H,n,d], k/v[B,Hkv,n,dv], return [B,H,n,dv].

    `routing_blocks[B,H,n,r]` supplies SSA/CCC/tree parent-block ids; negative
    padding is allowed, repeats are removed, and out-of-prefix entries are
    discarded before adding the current block. This bypasses flat routing.
    log_gain is scalar or a per-query-head tensor. No certificate is asserted.
    """
    batch, heads, n, d = q.shape
    if k.shape[2] != n or v.shape[:3] != k.shape[:3] or heads % k.shape[1]:
        raise ValueError("requires aligned, unpadded GQA self-attention")
    if block < 1 or not 1 <= cells <= block or top_blocks < 1 or query_chunk < 1:
        raise ValueError("invalid attention configuration")
    if n <= block:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=heads != k.shape[1])
    dtype = q.dtype
    q, k, v = q.float(), k.float(), v.float()
    group = heads // k.shape[1]
    headmap = torch.arange(heads, device=q.device) // group
    nb = (n + block - 1) // block
    if routing_blocks is None:
        padded_k = F.pad(k, (0, 0, 0, nb * block - n))
        means = padded_k.reshape(batch, k.shape[1], nb, block, d).mean(-2)
    elif (routing_blocks.shape[:3] != (batch, heads, n) or routing_blocks.ndim != 4
          or routing_blocks.dtype not in (torch.int32, torch.int64)):
        raise ValueError("routing_blocks must be integer [batch, heads, queries, routes]")
    centers = k[:, :, torch.linspace(0, block - 1, cells, device=q.device).long()].detach()
    if use_tail:
        # Fitting/assigning is hard, held fixed by autograd; values and masses
        # remain differentiable. Online serving needs just the final state.
        assignment = torch.cdist(k.detach(), centers).argmin(-1)
        features = F.one_hot(assignment, cells).to(q.dtype)
        counts = features.cumsum(-2)
        sums = (features[..., None] * v[..., None, :]).cumsum(2)
    outputs = []
    scale = d**-0.5
    gain = torch.as_tensor(log_gain, device=q.device, dtype=q.dtype)
    if gain.ndim == 1:
        gain = gain[None, :, None, None]
    for start in range(0, n, query_chunk):
        end = min(start + query_chunk, n)
        qc = q[:, :, start:end]
        positions = torch.arange(start, end, device=q.device)
        current = positions // block
        if routing_blocks is None:
            scores = torch.einsum("bhqd,bhcd->bhqc", qc, means[:, headmap])
            scores = scores.masked_fill(torch.arange(nb, device=q.device)[None, None, None] >= current[None, None, :, None], -torch.inf)
            route = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :min(top_blocks, nb)]
            route_valid = torch.isfinite(scores.gather(-1, route))
        else:
            route = routing_blocks[:, :, start:end].long()
            route_valid = (route >= 0) & (route < current[None, None, :, None])
            rank = torch.arange(route.shape[-1], device=q.device)
            earlier = rank[:, None] > rank[None, :]
            repeated = ((route[..., :, None] == route[..., None, :]) & earlier).any(-1)
            route_valid = route_valid & ~repeated
            route = route.clamp(0, nb - 1)
        local = current[None, None, :, None].expand(batch, heads, -1, -1)
        blocks = torch.cat((route, local), -1)
        block_valid = torch.cat((route_valid, torch.ones_like(local, dtype=torch.bool)), -1)
        ids = (blocks[..., None] * block + torch.arange(block, device=q.device)).flatten(-2)
        valid = block_valid[..., None].expand(-1, -1, -1, -1, block).flatten(-2)
        valid = valid & (ids <= positions[None, None, :, None]) & (ids < n)
        safe_ids = ids.clamp(0, n - 1)
        bi = torch.arange(batch, device=q.device)[:, None, None, None]
        hi = headmap[None, :, None, None]
        sk = k[bi, hi, safe_ids]
        sv = v[bi, hi, safe_ids]
        logits = (qc[..., None, :] * sk).sum(-1) * scale
        logits = logits.masked_fill(~valid, -torch.inf)
        log_z = logits.logsumexp(-1)
        sparse = (logits.softmax(-1)[..., None] * sv).sum(-2)
        if not use_tail:
            outputs.append(sparse)
            continue
        selected_cells = assignment[bi, hi, safe_ids]
        selected_counts = qc.new_zeros(batch, heads, end - start, cells)
        selected_counts = selected_counts.scatter_add(-1, selected_cells, valid.to(q.dtype))
        selected_sums = qc.new_zeros(batch, heads, end - start, cells, v.shape[-1])
        selected_sums = selected_sums.scatter_add(-2, selected_cells[..., None].expand_as(sv), sv * valid[..., None])
        remain_counts = (counts[:, headmap, start:end] - selected_counts).clamp_min(0)
        remain_sums = sums[:, headmap, start:end] - selected_sums
        cell_logits = torch.einsum("bhqd,bhcd->bhqc", qc, centers[:, headmap]) * scale + gain
        log_mass = (cell_logits + remain_counts.clamp_min(1).log()).masked_fill(remain_counts <= 0, -torch.inf)
        # For queries inside the first block, all keys are already selected;
        # masking guarantees future-derived centers cannot contribute.
        log_mass = log_mass.masked_fill(positions[None, None, :, None] < block, -torch.inf)
        weight = torch.cat((log_z[..., None], log_mass), -1).softmax(-1)
        tail_mean = remain_sums / remain_counts.clamp_min(1)[..., None]
        output = weight[..., :1] * sparse + (weight[..., 1:, None] * tail_mean).sum(-2)
        outputs.append(output)
    return torch.cat(outputs, dim=2).to(dtype)
