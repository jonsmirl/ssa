"""
The shared low-dimensional ROUTING SPACE for the Certified Causal Cascade.

P2 reported "low-rank routing is a bust (5–14% block-selection agreement)" — but it tested an UNTRAINED
random projection on full-rank random keys. This trains one, and asks the real question: does a learned
d_r-dim projection preserve which key-blocks the cascade selects? Three arms — untrained random (the P2
control), PCA (unsupervised), and trained — evaluated by block-selection Jaccard vs full-d routing, on
synthetic geometries (Rung 1) and real Qwen keys (Rung 2).

The routing metric matched is the cascade's own: s(q_block, k_block) = max_{sub∈k_block} ⟨q̄, μ_sub⟩
(sub-block max-pool centroid). The projection is asymmetric linear W_q, W_k ∈ R^{d×d_r} (routing is the
rank-d_r bilinear form W_q W_kᵀ) — linear so the rebuttal to P2 is fair and the map fuses into the
summary GEMM. Primary loss: listwise KL between the full-space and routing-space block-score
distributions over causally-past blocks (targets selection, budget-agnostic). Ablation: top-κ hinge.

Run:  python3 -m ssa.routing_space                  # -> paper/figures/routing_space.json + runs/*.pt
"""
from __future__ import annotations
import json
import math
import os
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NEG = float("-inf")


class RoutingProjection:
    """Asymmetric linear routing map. Satisfies the CausalCascade(proj=) contract via project_q/project_k."""

    def __init__(self, W_q, W_k, meta=None):
        self.W_q = W_q.to(DEV).float()
        self.W_k = W_k.to(DEV).float()
        self.meta = meta or {}

    def project_q(self, q):
        return q @ self.W_q

    def project_k(self, k):
        return k @ self.W_k

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"W_q": self.W_q.cpu(), "W_k": self.W_k.cpu(), "meta": self.meta}, path)
        return path

    @staticmethod
    def load(path, device=DEV):
        d = torch.load(path, map_location=device, weights_only=False)
        return RoutingProjection(d["W_q"], d["W_k"], d.get("meta", {}))


# -- block scores (the cascade routing metric: sub-block max-pool centroid) --------------------------

def block_means_qk(Q, K, block=128, sub=32):
    """(seq,d) full-precision Q,K -> query-block means qb (nb,d) and key sub-block means ks (nb*spb,d)."""
    n, d = K.shape
    nb, spb = n // block, block // sub
    qb = Q[:nb * block].view(nb, block, d).mean(1)
    ks = K[:nb * spb * sub].view(nb * spb, sub, d).mean(1)
    return qb, ks, nb, spb


def block_scores(qb, ks, nb, spb, W_q=None, W_k=None):
    """(nb, nb) causal-agnostic block scores: max over a key block's sub-means of ⟨q̄, μ_sub⟩."""
    q = qb if W_q is None else qb @ W_q
    k = ks if W_k is None else ks @ W_k
    return (q @ k.T).view(nb, nb, spb).amax(-1)


def _causal(nb, device):
    qi = torch.arange(nb, device=device)
    return qi[:, None] > qi[None, :]                                # key block strictly before query block


# -- losses -----------------------------------------------------------------------------------------

def listwise_kl(s_full, s_route, T=2.0):
    """KL(softmax(s_full/T) ‖ softmax(s_route/T)) over causally-past blocks, meaned over query-blocks.
    Uses a finite mask (-1e9) not -inf so all-past-empty rows (block 0) don't produce NaN gradients."""
    nb = s_full.shape[0]
    causal = _causal(nb, s_full.device)
    add = torch.where(causal, 0.0, -1e9)
    lp_f = torch.log_softmax((s_full + add) / T, dim=1)
    lp_r = torch.log_softmax((s_route + add) / T, dim=1)
    term = torch.where(causal, lp_f.exp() * (lp_f - lp_r), torch.zeros_like(lp_f))
    valid = causal.any(1)
    return term.sum(1)[valid].mean()


def topk_hinge(s_full, s_route, kappa, margin=0.1):
    """Margin between the κ-th true block and the best non-top-κ block, in routing space."""
    nb = s_full.shape[0]
    causal = _causal(nb, s_full.device)
    sf = s_full.masked_fill(~causal, NEG)
    sr = s_route.masked_fill(~causal, NEG)
    loss, cnt = 0.0, 0
    for i in range(1, nb):
        kk = min(kappa, i)
        top = sf[i, :i].topk(kk).indices
        mask = torch.zeros(i, dtype=torch.bool, device=s_full.device)
        mask[top] = True
        true_min = sr[i, :i][mask].min()
        false_max = sr[i, :i][~mask].max() if (~mask).any() else torch.tensor(NEG, device=s_full.device)
        loss = loss + torch.clamp(margin + false_max - true_min, min=0.0)
        cnt += 1
    return loss / max(1, cnt)


# -- projections ------------------------------------------------------------------------------------

def random_proj(d, d_r, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(d, d_r, generator=g, device=DEV) / math.sqrt(d)


def pca_init(K_pool, d_r):
    """Top-d_r principal directions of the pooled (centered) keys -> (d, d_r)."""
    X = K_pool.to(DEV).float()
    X = X - X.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return Vh[:d_r].T.contiguous()


def train_projection(samples, d, d_r, loss="kl", steps=3000, lr=1e-3, T=2.0, kappa=8,
                     init="pca", seed=0, log_every=0):
    """Train W_q, W_k on a list of (qb, ks, nb, spb) samples (precomputed block/sub means). Returns
    a RoutingProjection. `init`: 'pca' (from pooled ks) or 'random'."""
    torch.manual_seed(seed)
    if init == "pca":
        pool = torch.cat([s[1] for s in samples], 0)
        W0 = pca_init(pool, d_r)
        Wq = W0.clone().requires_grad_(True)
        Wk = W0.clone().requires_grad_(True)
    else:
        Wq = random_proj(d, d_r, seed).clone().requires_grad_(True)
        Wk = random_proj(d, d_r, seed + 1).clone().requires_grad_(True)
    # precompute full-space targets (frozen)
    fulls = [block_scores(qb, ks, nb, spb).detach() for (qb, ks, nb, spb) in samples]
    opt = torch.optim.Adam([Wq, Wk], lr=lr)
    for it in range(steps):
        opt.zero_grad()
        tot = 0.0
        for (qb, ks, nb, spb), sf in zip(samples, fulls):
            sr = block_scores(qb, ks, nb, spb, Wq, Wk)
            tot = tot + (listwise_kl(sf, sr, T) if loss == "kl" else topk_hinge(sf, sr, kappa))
        tot = tot / len(samples)
        tot.backward()
        opt.step()
        if log_every and it % log_every == 0:
            print(f"    step {it:>4} loss {tot.item():.4f}", flush=True)
    return RoutingProjection(Wq.detach(), Wk.detach(),
                             {"d": d, "d_r": d_r, "loss": loss, "T": T, "steps": steps, "init": init})


# -- evaluation -------------------------------------------------------------------------------------

def jaccard_topk(s_full, s_route, kappa):
    """Mean Jaccard of the causal top-κ block sets (full vs routing), over query-blocks with ≥κ past."""
    nb = s_full.shape[0]
    causal = _causal(nb, s_full.device)
    sf = s_full.masked_fill(~causal, NEG)
    sr = s_route.masked_fill(~causal, NEG)
    js, top1 = [], []
    for i in range(1, nb):
        kk = min(kappa, i)
        a = set(sf[i, :i].topk(kk).indices.tolist())
        b = set(sr[i, :i].topk(kk).indices.tolist())
        js.append(len(a & b) / len(a | b))
        top1.append(int(sf[i, :i].argmax().item() in b))           # dense-argmax block recovered
    return float(np.mean(js)), float(np.mean(top1))


def eval_projection(proj, samples, kappa=8):
    """Mean Jaccard@κ and top-1-block recall over samples (a (W_q,W_k) or RoutingProjection)."""
    Wq = proj.W_q if isinstance(proj, RoutingProjection) else proj[0]
    Wk = proj.W_k if isinstance(proj, RoutingProjection) else proj[1]
    j, t = [], []
    for (qb, ks, nb, spb) in samples:
        sf = block_scores(qb, ks, nb, spb)
        sr = block_scores(qb, ks, nb, spb, Wq, Wk)
        jj, tt = jaccard_topk(sf, sr, kappa)
        j.append(jj); t.append(tt)
    return float(np.mean(j)), float(np.mean(t))


# -- sample builders --------------------------------------------------------------------------------

def synth_sample(kind, n=8192, d=64, block=128, sub=32, seed=0):
    """One (qb, ks, nb, spb) sample. kind: random | clustered_tight | clustered_diffuse.
    Clustered geometry has BLOCK-level structure (each 128-block belongs to a cluster; a query block's
    top key-blocks are the same-cluster blocks) — the regime where routing is meaningful and a good
    projection should preserve selection. 'random' is full-rank (the P2 adversarial control)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    if kind == "random":
        K = torch.randn(n, d, generator=g, device=DEV)
        Q = torch.randn(n, d, generator=g, device=DEV)
    else:
        spread = 0.10 if kind == "clustered_tight" else 0.45
        nb = n // block
        nc = max(4, nb // 4)
        ctr = torch.randn(nc, d, generator=g, device=DEV)
        blk_c = torch.randint(0, nc, (nb,), generator=g, device=DEV)     # each key block -> a cluster
        K = ctr[blk_c].repeat_interleave(block, 0) + spread * torch.randn(n, d, generator=g, device=DEV)
        q_c = torch.randint(0, nc, (nb,), generator=g, device=DEV)       # each query block -> a cluster
        Q = ctr[q_c].repeat_interleave(block, 0) + 0.30 * torch.randn(n, d, generator=g, device=DEV)
    return block_means_qk(Q, K, block, sub)


def real_samples(bank, layers, block=128, sub=32, max_seq=None):
    """One sample per (layer, sampled q-head paired with its GQA kv-head)."""
    grp, qheads = bank["grp"], list(bank["q_heads"])
    samples = []
    for L in layers:
        K = torch.tensor(bank["K"][L], device=DEV).float()          # (n_kvh, seq, d)
        Q = torch.tensor(bank["Q"][L], device=DEV).float()          # (nqs, seq, d)
        if max_seq:
            K, Q = K[:, :max_seq], Q[:, :max_seq]
        for qi, h in enumerate(qheads):
            kv = h // grp
            samples.append(block_means_qk(Q[qi], K[kv], block, sub))
    return samples


def transfer_matrix(bank, layer_sample, d_r=16, **tk):
    """Train a per-layer projection on each layer, evaluate it on every layer -> Jaccard matrix."""
    projs = {L: train_projection(real_samples(bank, [L]), bank["d"], d_r, **tk) for L in layer_sample}
    M = np.zeros((len(layer_sample), len(layer_sample)))
    for a, La in enumerate(layer_sample):
        for b, Lb in enumerate(layer_sample):
            M[a, b] = eval_projection(projs[La], real_samples(bank, [Lb]))[0]
    return M, projs


def kappa_min_routing(K, Q, proj, B=None, order="cumulant", target=0.9, seed=0):
    """κ_min/n at recall≥target when clustering+ranking happen in the routing space (dense target in full
    space). `proj` None -> full space. Wraps recall_floor.kappa_min via the route_recall K_route hook."""
    from ssa.recall_floor import kappa_min
    Kr = Qr = None
    if proj is not None:
        Kr = (torch.tensor(K, device=DEV).float() @ proj.W_k).cpu().numpy()
        Qr = (torch.tensor(Q, device=DEV).float() @ proj.W_q).cpu().numpy()
    return kappa_min(K, Q, order, target=target, B=B, seed=seed, K_route=Kr, Q_route=Qr)


# -- driver -----------------------------------------------------------------------------------------

def _arms(samples, d, d_r, seed=0):
    """Jaccard@κ for the three arms: untrained random, PCA, trained (KL)."""
    rnd = (random_proj(d, d_r, seed), random_proj(d, d_r, seed + 1))
    pool = torch.cat([s[1] for s in samples], 0)
    W = pca_init(pool, d_r)
    pca = (W, W)
    trained = train_projection(samples, d, d_r, loss="kl", steps=1500, seed=seed)
    return {"random": eval_projection(rnd, samples)[0],
            "pca": eval_projection(pca, samples)[0],
            "trained": eval_projection(trained, samples)[0]}, trained


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--keybank", default="/tmp/keybank_wikitext_8192.npz")
    ap.add_argument("--holdout", default="/tmp/keybank_code_8192.npz")
    ap.add_argument("--d-rs", default="8,16,32")
    ap.add_argument("--out", default="paper/figures/routing_space.json")
    args = ap.parse_args()
    d = 64
    d_rs = [int(x) for x in args.d_rs.split(",")]
    rows = []
    print("=" * 92)
    print("ROUTING SPACE — block-selection Jaccard vs full-d, trained/PCA/untrained (the P2 rebuttal)")
    print("=" * 92)

    print("\n[Rung 1] synthetic geometries  (random reproduces the P2 'bust'; clustered is where it works)")
    print(f"  {'geometry':>18} {'d_r':>4} {'random':>8} {'pca':>8} {'trained':>8}")
    for kind in ("random", "clustered_tight", "clustered_diffuse"):
        samples = [synth_sample(kind, seed=s) for s in range(4)]
        for d_r in d_rs:
            res, _ = _arms(samples, d, d_r)
            rows.append({"rung": 1, "geometry": kind, "d_r": d_r, **res})
            print(f"  {kind:>18} {d_r:>4} {res['random']:>8.3f} {res['pca']:>8.3f} {res['trained']:>8.3f}",
                  flush=True)

    if os.path.exists(args.keybank):
        bank = load_keybank_local(args.keybank)
        hold = load_keybank_local(args.holdout) if os.path.exists(args.holdout) else None
        layers = list(range(0, bank["n_layers"], max(1, bank["n_layers"] // 8)))
        print(f"\n[Rung 2] real Qwen keys (pooled {len(layers)} layers × heads), held-out doc = code")
        print(f"  {'d_r':>4} {'random':>8} {'pca':>8} {'trained':>8} {'trained(holdout)':>17}")
        train_s = real_samples(bank, layers)
        for d_r in d_rs:
            res, trained = _arms(train_s, d, d_r)
            ho = eval_projection(trained, real_samples(hold, layers))[0] if hold else None
            rows.append({"rung": 2, "geometry": "qwen_pooled", "d_r": d_r, **res,
                         "trained_holdout": ho})
            print(f"  {d_r:>4} {res['random']:>8.3f} {res['pca']:>8.3f} {res['trained']:>8.3f} "
                  f"{(f'{ho:.3f}' if ho is not None else '—'):>17}", flush=True)
            if d_r == 16:
                trained.meta.update({"model": "Qwen/Qwen2.5-0.5B", "scope": "shared", "layers": layers,
                                     "rope": "post", "jaccard": res["trained"], "holdout": ho})
                trained.save("runs/routing_space_d16_shared.pt")
        M, _ = transfer_matrix(bank, layers[:6], d_r=16, steps=800)
        offdiag = M[~np.eye(len(M), dtype=bool)]
        print(f"\n  cross-layer transfer (d_r=16): diag {np.diag(M).mean():.3f}  "
              f"off-diag median {np.median(offdiag):.3f}  min {offdiag.min():.3f}")
        rows.append({"rung": 2, "geometry": "transfer", "d_r": 16,
                     "diag_mean": float(np.diag(M).mean()), "offdiag_median": float(np.median(offdiag)),
                     "offdiag_min": float(offdiag.min()), "matrix": M.tolist(), "layers": layers[:6]})
    else:
        print(f"\n[Rung 2] skipped — no keybank at {args.keybank} (run: python -m ssa.qwen_keybank)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"meta": {"d": d, "block": 128, "sub": 32, "kappa": 8, "seed": 0}, "rows": rows},
              open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}")


def load_keybank_local(path):
    from ssa.qwen_keybank import load_keybank
    return load_keybank(path)


if __name__ == "__main__":
    main()
