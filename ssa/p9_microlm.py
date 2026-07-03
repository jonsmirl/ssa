"""
P9 — the trained micro-LM comparison: does a LEARNED write gate close the compression gap P8 measured?

P8 measured the zero-attention (compression) corner with hand-built, untrained memories: write-time
compression cannot serve read-time-only relevance (recall 0.10 vs selection's 1.00). P8 could not reach the
TRAINING-dependent half of the recipe — a learned write policy + an auxiliary future-prediction objective
("rethink the objective function"). This module trains a small LM whose token-mixer is swappable between the
two corners, at matched state, to measure how much training narrows the gap.

Four token-mixers, bracketing the trilemma's two corners (all `attn_fn(q,k,v)->o`, q,k,v = (B,H,n,dh)):
  - `dense`    : full causal softmax attention — the SELECTION upper bound (κ=n).
  - `ssa`      : block-cumulant top-c selection + exact masked softmax (`ssa_swap.ssa_masked`) — the
                 selection corner at budget κ.
  - `deltanet` : the COMPRESSION corner — a differentiable causal scan of the delta rule
                 S_t = S_{t-1} + β_t(v_t − S_{t-1}k_t)k_tᵀ, o_t = S_t q̂_t, with an optional LEARNED per-token
                 write gate β_t = sigmoid(w_g·k_t) (the trained write policy).
  - `linear`   : additive write S += v kᵀ, no gate — the simplest compression write (no erase, no gate).

Plus a JEPA-style future-prediction auxiliary loss (`FuturePredictor` + stop-grad future-summary target).

Reuses `ssa_demo.Block` (the pre-LN block + its q/k/v projections) and `ssa_swap.ssa_masked`. Honest scope:
still synthetic MQAR, but TRAINED end-to-end — the half P8 could not measure.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ssa.ssa_demo import Block
from ssa.ssa_swap import ssa_masked

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _norm(x):
    return F.normalize(x, dim=-1, eps=1e-6)


# -- token mixers (attn_fn: (q,k,v)->o, all (B,H,n,dh)) --------------------------------------------

def dense_mix(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def ssa_mix(q, k, v, block=8, top_c=8, local=1):
    return ssa_masked(q, k, v, block=block, top_c=top_c, local=local)


def deltanet_mix(q, k, v, gate=None, beta=1.0):
    """Differentiable causal delta-rule scan (the compression corner). `gate` (a Linear dh->1) makes the
    per-token write strength LEARNED; else the constant `beta`. Sequential over n (BPTT), batched over (B,H)."""
    B, H, n, dh = q.shape
    kk = _norm(k); qq = _norm(q)
    betas = torch.sigmoid(gate(k)) if gate is not None else None       # (B,H,n,1) learned write strength
    # fold (B,H) into one batch dim so the per-step ops are single batched bmm calls (fewer kernel launches)
    K = kk.reshape(B * H, n, dh); Q = qq.reshape(B * H, n, dh); V = v.reshape(B * H, n, dh)
    Bt = betas.reshape(B * H, n, 1) if gate is not None else None
    S = q.new_zeros(B * H, dh, dh)
    outs = []
    for t in range(n):
        kt = K[:, t:t + 1]; vt = V[:, t:t + 1]; qt = Q[:, t:t + 1]     # (BH,1,dh)
        corr = vt - torch.bmm(kt, S)                                   # v − k S  (row-vec convention: o = q S)
        bt = Bt[:, t:t + 1] if gate is not None else beta              # (BH,1,1) or scalar
        S = S + bt * torch.bmm(kt.transpose(1, 2), corr)               # + β kᵀ(v − kS)
        outs.append(torch.bmm(qt, S))                                  # read o = q̂ S
    return torch.stack(outs, dim=1).reshape(B, H, n, dh)


def linear_mix(q, k, v):
    """Vanilla additive linear attention (causal) — the simplest compression write (no erase/gate). S_t = Σ_{s≤t} v_s k̂_sᵀ,
    o_t = S_t q̂_t = Σ_{s≤t} v_s ⟨k̂_s, q̂_t⟩ (cumulative, no gate, no erase)."""
    kk = _norm(k); qq = _norm(q)
    kv = torch.einsum('bhsi,bhsj->bhsij', v, kk)                       # per-step outer products
    S = kv.cumsum(2)                                                   # causal prefix state
    return torch.einsum('bhtij,bhtj->bhti', S, qq)


# -- the future-prediction auxiliary loss (JEPA-lite) ---------------------------------------------

class FuturePredictor(nn.Module):
    """Predict a stop-grad summary of future hidden states from the current one — the 'rethink the objective'
    ingredient: it pressures the write policy to retain information whose payoff is in the future."""

    def __init__(self, d, hidden=None):
        super().__init__()
        hidden = hidden or 2 * d
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, h):
        return self.net(h)


def jepa_loss(predictor, h, window=8):
    """1 − cos(predictor(h_t), stop_grad(mean of h over [t+1, t+window]))."""
    B, N, d = h.shape
    tgt = torch.zeros_like(h)
    cnt = h.new_zeros(N)
    for w in range(1, window + 1):
        if w < N:
            tgt[:, :N - w] += h[:, w:].detach()
            cnt[:N - w] += 1
    valid = cnt > 0
    tgt = tgt / cnt.clamp(min=1)[None, :, None]
    pred = predictor(h)
    return (1 - F.cosine_similarity(pred[:, valid], tgt[:, valid], dim=-1)).mean()


# -- the model ------------------------------------------------------------------------------------

class P9Model(nn.Module):
    """A micro-LM (reusing `ssa_demo.Block`) with a swappable token-mixer, an optional learned DeltaNet gate,
    and an optional JEPA head. `forward(ids, return_hidden)` exposes hidden states for the aux loss."""

    def __init__(self, vocab, d=128, n_layer=2, n_head=4, max_len=1024, mixer="dense",
                 ssa_block=8, ssa_top_c=8, ssa_local=1, delta_gate=True, jepa=False, jepa_window=8):
        super().__init__()
        self.d, self.n_head, self.dh = d, n_head, d // n_head
        self.mixer = mixer
        self.ssa_cfg = dict(block=ssa_block, top_c=ssa_top_c, local=ssa_local)
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        # learned per-layer DeltaNet write gate (key -> scalar write strength), INITIALIZED OPEN
        # (bias high ⇒ β≈1, "write everything") so it defaults to the strong constant-β policy and can only
        # deviate where selective writing helps — the standard open-forget-gate init.
        self.gates = None
        if mixer == "deltanet" and delta_gate:
            self.gates = nn.ModuleList([nn.Linear(self.dh, 1) for _ in range(n_layer)])
            for g in self.gates:
                nn.init.zeros_(g.weight); nn.init.constant_(g.bias, 3.0)   # sigmoid(3)≈0.95
        self.predictor = FuturePredictor(d) if jepa else None
        self.jepa_window = jepa_window

    def _mixer_fn(self, i):
        if self.mixer == "dense":
            return dense_mix
        if self.mixer == "ssa":
            return lambda q, k, v: ssa_mix(q, k, v, **self.ssa_cfg)
        if self.mixer == "linear":
            return linear_mix
        if self.mixer == "deltanet":
            g = self.gates[i] if self.gates is not None else None
            return lambda q, k, v, g=g: deltanet_mix(q, k, v, gate=g)
        raise ValueError(self.mixer)

    def forward(self, ids, return_hidden=False):
        B, N = ids.shape
        x = self.tok(ids) + self.pos(torch.arange(N, device=ids.device))[None]
        for i, blk in enumerate(self.blocks):
            x = blk(x, self._mixer_fn(i))
        h = self.lnf(x)
        logits = self.head(h)
        return (logits, h) if return_hidden else logits


# -- training + eval (the ssa_checkpoint loop + the aux loss) --------------------------------------

def train_model(model, task, steps, n_pairs, n_queries=1, lr=6e-4, bs=48, warmup=100,
                aux_weight=0.0, seed=0, log_every=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, 0.5 * (1 + math.cos(math.pi * s / steps))))
    model.train()
    need_h = aux_weight > 0 and model.predictor is not None
    for s in range(steps):
        ids, tgt = task.batch(bs, n_pairs, n_queries, device=DEV)
        out = model(ids, return_hidden=need_h)
        logits, h = out if need_h else (out, None)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100)
        if need_h:
            loss = loss + aux_weight * jepa_loss(model.predictor, h, model.jepa_window)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if log_every and s % log_every == 0:
            print(f"    step {s:>4} loss {loss.item():.4f}", flush=True)
    return model


@torch.no_grad()
def recall(model, task, n_pairs, n_queries=1, bs=32, trials=4, needle=None):
    model.eval()
    hit = tot = 0
    for _ in range(trials):
        ids, tgt = task.batch(bs, n_pairs, n_queries, needle_dist=needle, device=DEV)
        logits = model(ids)
        m = tgt != -100
        hit += (logits.argmax(-1)[m] == tgt[m]).sum().item(); tot += int(m.sum().item())
    return hit / max(tot, 1)
