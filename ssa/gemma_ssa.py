"""
Gemma-4 SSA attention — a frozen-swap of cumulant-routed sparse selection into the FULL
(global) attention layers only, via the transformers pluggable attention interface.

Why the interface, not a global SDPA monkeypatch: `Gemma4TextAttention.forward` calls
`ALL_ATTENTION_FUNCTIONS.get_interface(...)(module, q, k, v, mask, scaling=..., sliding_window=...)`
*after* q_norm/k_norm + RoPE + transpose, so a registered fn receives exactly the post-norm/RoPE
`q (b,hq,n,d)`, `k/v (b,hkv,n,d)` the model attends on, and `module.is_sliding` says per layer
whether to route. We route ONLY the 5 full-attention layers (head_dim 256, K=V on this model);
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
    edgeworth: bool = False    # add the (diagonal) 3rd-cumulant/skew term to routing (outlier detector)
    dense_layers: tuple = ()   # layer_idx values to leave DENSE (bypass selection) — e.g. the worst router
    impl: str = "analytic"     # "analytic" = O(n²) score+mask (quality); "flex" = the fused kernel (speed)
    share_route_from: int | None = None  # donor layer: compute the selection once here, reuse above it
    share_below: str = "per_layer"       # layers below the donor: "per_layer" | "dense"
    proj_path: str | None = None         # a trained RoutingProjection (.pt) — route in the shared low-dim space


# module-level config the registered interface reads; the kappa-sweep driver mutates this.
CFG = SSAConfig()
# fallback for sliding layers (and full layers when route is disabled); set at install time.
_FALLBACK = None
# trained routing projection (loaded from CFG.proj_path at install), and the donor-layer BlockMask cache.
_PROJ = None
_SHARE = {"sig": None, "bm": None}
# measured router wall-clock accumulator (longctx_share.py reads this to get route_ms / prefill_ms).
ROUTE_MS = 0.0


def repeat_kv(x: torch.Tensor, n: int) -> torch.Tensor:
    if n == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n, s, d).reshape(b, h * n, s, d)


def _block_stats(k: torch.Tensor, block: int):
    """Per-block diagonal mean / variance / 3rd-central-moment of keys.
    k: (b,h,n,d) -> mu,var,m3: (b,h,nb,d); nb blocks, last padded+masked."""
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
    cen = (kb - mu.unsqueeze(3)) * vb                              # centred, padded->0
    var = (cen * cen).sum(3) / cnt                                 # diagonal Sigma
    m3 = (cen ** 3).sum(3) / cnt                                   # diagonal 3rd central moment (skew)
    return mu, var.clamp(min=0.0), m3, nb


def _selection_mask(q, k, cfg: SSAConfig, qpos=None, kpos=None):
    """Additive (0/-inf) mask (b,hq,q_len,kv_len). `qpos` = absolute query positions (q_len,),
    `kpos` = key positions (kv_len,); defaults assume the queries are the last q_len of a left-aligned
    sequence (correct for prefill AND KV-cache decoding). Keeps keys whose block is (a) top-budget by
    cumulant score among FULLY-PAST blocks, or (b) in the local window / own block; and causal."""
    b, hq, q_len, d = q.shape
    kv_len = k.shape[2]
    block = cfg.block
    if kpos is None:
        kpos = torch.arange(kv_len, device=q.device)
    if qpos is None:
        qpos = torch.arange(kv_len - q_len, kv_len, device=q.device)
    mu, var, m3, nb = _block_stats(k, block)                      # blocks over kv_len
    # cumulant routing score r: (b,hq,q_len,nb)
    r = (torch.einsum("bhqd,bhcd->bhqc", q, mu)
         + 0.5 * cfg.beta * torch.einsum("bhqd,bhcd->bhqc", q * q, var))
    if cfg.edgeworth:                                             # 3rd-cumulant (skew) term: rewards
        r = r + (cfg.beta ** 2 / 6.0) * torch.einsum("bhqd,bhcd->bhqc", q ** 3, m3)  # outlier blocks

    cidx = torch.arange(nb, device=q.device)
    block_start = cidx * block                                    # (nb,)
    # CAUSAL ROUTING: only route to blocks FULLY in the past relative to each query's ABSOLUTE
    # position, so a block's mean/var never includes a key after the query. The query's own/partial
    # block is covered by the local window, not by the routing score.
    blk_has_causal = block_start[None, :] <= qpos[:, None]        # (q_len,nb) block has >=1 key <= pos
    blk_routable = (block_start + block - 1)[None, :] <= qpos[:, None]   # block fully <= query pos
    r = r.masked_fill(~blk_routable[None, None], NEG)

    # how many blocks to keep, per query (budget on the routable blocks)
    nvis = blk_routable.sum(-1)                                   # (q_len,)
    if cfg.top_c is not None:
        keep = torch.full_like(nvis, cfg.top_c)
    else:
        keep = torch.clamp((cfg.budget_frac * nvis).ceil().long(), min=1)
    top = int(keep.max().clamp(max=nb).item())

    sel = torch.zeros(b, hq, q_len, nb, dtype=torch.bool, device=q.device)
    idx = r.topk(top, dim=-1).indices                            # (b,hq,q_len,top)
    # respect per-query keep count: rank-mask the topk picks beyond keep[i]
    ranks = torch.arange(top, device=q.device)
    keep_mask = ranks[None, :] < keep[:, None]                   # (q_len,top)
    sel.scatter_(-1, idx, keep_mask[None, None].expand(b, hq, q_len, top))

    own = qpos // block                                          # (q_len,)
    local = (cidx[None, :] <= own[:, None]) & (cidx[None, :] >= (own[:, None] - cfg.local_w))
    sel |= local[None, None]
    sel &= blk_has_causal[None, None]                           # never select a wholly-future block

    blk_id = kpos // block                                       # key position -> block id (kv_len,)
    key_block_sel = sel[..., blk_id]                             # (b,hq,q_len,kv_len)
    causal = qpos[:, None] >= kpos[None, :]                      # (q_len,kv_len)
    allow = key_block_sel & causal[None, None]
    return torch.where(allow, 0.0, torch.tensor(NEG, device=q.device))


def _flex_mask(query, k, cfg, dense, proj, n):
    """Route -> BlockMask (the selection). Timed as the router; shareable across layers (same shape)."""
    import time
    from ssa import ssa_kernel as K
    from torch.nn.attention.flex_attention import BlockMask
    global ROUTE_MS
    blk = cfg.block
    pad = (-n) % blk
    N = n + pad
    qq = F.pad(query, (0, 0, 0, pad)) if pad else query
    kk = F.pad(k, (0, 0, 0, pad)) if pad else k
    if query.is_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    kv_num, kv_idx, _ = K.block_route_budget(
        qq, kk, blk, budget_frac=(1.0 if dense else cfg.budget_frac),
        top_c=(None if dense else cfg.top_c), local=cfg.local_w, beta=cfg.beta,
        edgeworth=cfg.edgeworth, n_real=n, sub=None, proj=proj)

    def mm(bb, hh, qi, kv):
        return (kv <= qi) & (kv < n)                                 # token-causal AND drop pad keys

    bm = BlockMask.from_kv_blocks(kv_num, kv_idx, BLOCK_SIZE=blk, mask_mod=mm, seq_lengths=(N, N))
    if query.is_cuda:
        torch.cuda.synchronize()
    ROUTE_MS += (time.time() - t0) * 1000
    return bm, N, pad


def _flex_forward(query, k, v, cfg, scaling, dense=False, proj=None, bm_pad=None):
    """The fused block-sparse kernel path (prefill only). `bm_pad` = a (BlockMask, N, pad) reused from the
    donor layer (cross-layer sharing); else route here. Returns (out (b,hq,n,d), (bm,N,pad))."""
    from ssa import ssa_kernel as K
    b, hq, n, d = query.shape
    if bm_pad is None:
        bm_pad = _flex_mask(query, k, cfg, dense, proj, n)
    bm, N, pad = bm_pad
    if pad:
        query, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (query, k, v))
    out = K._flex(query, k, v, block_mask=bm, scale=scaling)
    return out[:, :, :n], bm_pad


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

    q_len = query.shape[2]
    kv_len = k.shape[2]
    nb_layer = (kv_len + cfg.block - 1) // cfg.block
    force_dense = getattr(module, "layer_idx", None) in cfg.dense_layers
    is_dense = force_dense or (cfg.top_c is not None and cfg.top_c >= nb_layer) \
        or (cfg.top_c is None and cfg.budget_frac >= 1.0)
    # the fused kernel path — prefill-shaped CUDA calls only; decode (q_len==1), chunked prefill
    # (q_len != kv_len), and CPU fall through to the analytic path below (already decode-correct).
    if cfg.impl == "flex" and query.is_cuda and q_len == kv_len and q_len > cfg.block:
        li = getattr(module, "layer_idx", None)
        donor = cfg.share_route_from
        bm_pad = None
        if donor is not None and li is not None:
            if li < donor:                                          # below the donor: route per-layer (honest cost)
                is_dense = is_dense or cfg.share_below == "dense"
            elif li > donor:                                        # consumer: reuse the donor's selection
                sig = (tuple(query.shape), kv_len, str(query.device))
                if _SHARE["sig"] == sig:
                    bm_pad = _SHARE["bm"]
        out, made = _flex_forward(query, k, v, cfg, scaling, dense=is_dense, proj=_PROJ, bm_pad=bm_pad)
        if donor is not None and li == donor:                      # donor stashes its selection for consumers
            _SHARE["sig"] = (tuple(query.shape), kv_len, str(query.device))
            _SHARE["bm"] = made
        return out.transpose(1, 2).contiguous(), None
    # absolute query positions: prefer cache_position (KV-cache incremental decode); else assume the
    # queries are the last q_len of a left-aligned sequence. kv_len != q_len during generation.
    cpos = kwargs.get("cache_position")
    qpos = cpos.to(query.device).long() if cpos is not None \
        else torch.arange(kv_len - q_len, kv_len, device=query.device)
    kpos = torch.arange(kv_len, device=query.device)

    scores = torch.matmul(query, k.transpose(-1, -2)) * scaling   # (b,hq,q_len,kv_len)
    causal = qpos[:, None] >= kpos[None, :]                       # (q_len,kv_len)
    scores = scores.masked_fill(~causal[None, None], NEG)

    if not is_dense:
        scores = scores + _selection_mask(query, k, cfg, qpos, kpos)

    w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v.dtype)
    out = torch.matmul(w, v)                                      # (b,hq,q_len,d)
    return out.transpose(1, 2).contiguous(), None


def install_ssa(model, **cfg_kwargs):
    """Register SSA as the model's attention implementation. Sliding layers fall back to the
    model's prior kernel; full layers route. Validate the wiring with the GPU smoke test —
    the routing math itself is unit-tested in test_gemma_ssa.py."""
    global _FALLBACK, _PROJ
    for k, val in cfg_kwargs.items():
        setattr(CFG, k, val)
    _SHARE["sig"] = None; _SHARE["bm"] = None
    if CFG.proj_path:
        from ssa.routing_space import RoutingProjection
        _PROJ = RoutingProjection.load(CFG.proj_path)
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
