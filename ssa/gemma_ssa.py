"""
Gemma-4 SSA attention — a frozen-swap of cumulant-routed sparse selection into the FULL
(global) attention layers only, via the transformers pluggable attention interface.

Why the interface, not a global SDPA monkeypatch: `Gemma4TextAttention.forward` calls
`ALL_ATTENTION_FUNCTIONS.get_interface(...)(module, q, k, v, mask, scaling=..., sliding_window=...)`
*after* q_norm/k_norm + RoPE + transpose, so a registered fn receives exactly the post-norm/RoPE
`q (b,hq,n,d)`, `k/v (b,hkv,n,d)` the model attends on, and `module.is_sliding` says per layer
whether to route. We route ONLY the 5 full-attention layers (head_dim 512, K=V on this model);
the 25 sliding-window layers fall through to the stock kernel untouched.

This is the analytic SSA used for the frozen-swap QUALITY measurement: it forms the scores and
adds a block-selection mask (exact softmax over the selected key-blocks). It is not the fast
kernel (that's ssa_kernel.py) — here we measure ppl/recall vs budget, not wall-clock.

The routing score is the second cumulant  r_c(q) = <q,mu_c> + (beta/2) q^T diag(Sigma_c) q
(beta default 2.0 — the measured optimum). budget_frac = fraction of causally-visible blocks kept
(plus a local window); budget_frac >= 1 ==> all blocks kept ==> bit-exact dense (the sanity gate).

Run the unit tests:  pytest ssa/tests/test_gemma_ssa.py
Install on a model:  from ssa.gemma_ssa import install_ssa; install_ssa(model, budget_frac=0.25)
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F

NEG = float("-inf")


@dataclass
class SSAConfig:
    block: int = 256          # key-block size (positions per block)
    budget_frac: float = 0.25 # fraction of causally-visible blocks to keep (>=1 -> dense)
    top_c: int | None = None  # absolute #blocks to keep; overrides budget_frac if set
    local_w: int = 1          # also always keep the query's own block + this many preceding
    beta: float = 2.0         # cumulant routing temperature (measured optimum ~2)
    route_full_only: bool = True  # only sparsify full-attention layers (is_sliding == False)


# module-level config the registered interface reads; the kappa-sweep driver mutates this.
CFG = SSAConfig()
# fallback for sliding layers (and full layers when route is disabled); set at install time.
_FALLBACK = None


def repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def _block_stats(k: torch.Tensor, block: int):
    """Per-block mean and (diagonal, population) variance of keys.
    k: (b,h,n,d) -> mu,var: (b,h,nb,d); nb blocks, last padded+masked."""
    b, h, n, d = k.shape
    nb = (n + block - 1) // block
    pad = nb * block - n
    kp = F.pad(k, (0, 0, 0, pad))                                  # (b,h,nb*block,d)
    valid = torch.ones(b, h, nb * block, 1, device=k.device, dtype=k.dtype)
    if pad:
        valid[:, :, n:, :] = 0
    kb = kp.view(b, h, nb, block, d)
    vb = valid.view(b, h, nb, block, 1)
    cnt = vb.sum(3).clamp(min=1.0)                                 # (b,h,nb,1)
    mu = (kb * vb).sum(3) / cnt                                    # (b,h,nb,d)
    var = (kb * kb * vb).sum(3) / cnt - mu * mu                    # diagonal Sigma
    return mu, var.clamp(min=0.0), nb


def _selection_mask(q, k, cfg: SSAConfig):
    """Additive (0/-inf) mask (b,hq,n,n): for each query position, keep keys whose block is
    (a) in the top-`budget` by cumulant score, or (b) in the local window / own block; and causal."""
    b, hq, n, d = q.shape
    block = cfg.block
    mu, var, nb = _block_stats(k, block)                          # (b,hq,nb,d)
    # cumulant routing score r: (b,hq,n,nb)
    r = (torch.einsum("bhqd,bhcd->bhqc", q, mu)
         + 0.5 * cfg.beta * torch.einsum("bhqd,bhcd->bhqc", q * q, var))

    qpos = torch.arange(n, device=q.device)
    cidx = torch.arange(nb, device=q.device)
    block_start = cidx * block                                    # (nb,)
    # CAUSAL ROUTING: only route to blocks that are FULLY in the past, so a block's mean/var never
    # includes a key at a position > i. The query's own (partial) block is covered by the local
    # window below, not by the routing score — otherwise the selection decision leaks future tokens.
    blk_has_causal = block_start[None, :] <= qpos[:, None]        # (n,nb) block has >=1 key <= i
    blk_routable = (block_start + block - 1)[None, :] <= qpos[:, None]   # block fully <= i
    r = r.masked_fill(~blk_routable[None, None], NEG)

    # how many blocks to keep, per query (budget on the routable blocks)
    nvis = blk_routable.sum(-1)                                   # (n,)
    if cfg.top_c is not None:
        keep = torch.full_like(nvis, cfg.top_c)
    else:
        keep = torch.clamp((cfg.budget_frac * nvis).ceil().long(), min=1)
    top = int(keep.max().clamp(max=nb).item())

    sel = torch.zeros(b, hq, n, nb, dtype=torch.bool, device=q.device)
    idx = r.topk(top, dim=-1).indices                            # (b,hq,n,top)
    # respect per-query keep count: rank-mask the topk picks beyond keep[i]
    ranks = torch.arange(top, device=q.device)
    keep_mask = ranks[None, :] < keep[:, None]                   # (n,top)
    sel.scatter_(-1, idx, keep_mask[None, None].expand(b, hq, n, top))

    own = (qpos // block)                                         # (n,)
    local = (cidx[None, :] <= own[:, None]) & (cidx[None, :] >= (own[:, None] - cfg.local_w))
    sel |= local[None, None]
    sel &= blk_has_causal[None, None]                           # never select a wholly-future block

    blk_id = (qpos // block)                                      # key position -> block id
    key_block_sel = sel[..., blk_id]                             # (b,hq,n_q,n_k)
    causal = qpos[None, :] <= qpos[:, None]                      # (n_q,n_k)
    allow = key_block_sel & causal[None, None]
    return torch.where(allow, 0.0, torch.tensor(NEG, device=q.device))


def ssa_attention_forward(module, query, key, value, attention_mask=None,
                          scaling=None, dropout=0.0, **kwargs):
    """transformers attention-interface fn. query (b,hq,n,d); key/value (b,hkv,n,d).
    Returns (attn_output (b,n,hq,d), attn_weights=None)."""
    cfg = CFG
    is_sliding = bool(getattr(module, "is_sliding", False))
    if (cfg.route_full_only and is_sliding) and _FALLBACK is not None:
        return _FALLBACK(module, query, key, value, attention_mask,
                         scaling=scaling, dropout=dropout, **kwargs)

    groups = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
    k = repeat_kv(key, groups)
    v = repeat_kv(value, groups)
    if scaling is None:
        scaling = getattr(module, "scaling", 1.0)

    scores = torch.matmul(query, k.transpose(-1, -2)) * scaling   # (b,hq,n,n)
    n = query.shape[2]
    qpos = torch.arange(n, device=query.device)
    causal = (qpos[None, :] <= qpos[:, None])                     # (n,n)
    scores = scores.masked_fill(~causal[None, None], NEG)

    dense = (cfg.top_c is not None and cfg.top_c >= ((n + cfg.block - 1) // cfg.block)) \
        or (cfg.top_c is None and cfg.budget_frac >= 1.0)
    if not dense:
        scores = scores + _selection_mask(query, k, cfg)

    w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
    out = torch.matmul(w, v)                                      # (b,hq,n,d)
    return out.transpose(1, 2).contiguous(), None


def install_ssa(model, **cfg_kwargs):
    """Register SSA as the model's attention implementation. Sliding layers fall back to the
    model's prior kernel; full layers route. Validate the wiring with the GPU smoke test —
    the routing math itself is unit-tested in test_gemma_ssa.py."""
    global _FALLBACK
    for k, val in cfg_kwargs.items():
        setattr(CFG, k, val)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    prior = getattr(model.config, "_attn_implementation", None) or "sdpa"
    try:
        _FALLBACK = ALL_ATTENTION_FUNCTIONS.get_interface(prior)
    except Exception:
        _FALLBACK = ALL_ATTENTION_FUNCTIONS.get("sdpa")
    ALL_ATTENTION_FUNCTIONS["ssa"] = ssa_attention_forward
    model.config._attn_implementation = "ssa"
    if hasattr(model.config, "get_text_config"):
        try:
            model.config.get_text_config()._attn_implementation = "ssa"
        except Exception:
            pass
    return model
