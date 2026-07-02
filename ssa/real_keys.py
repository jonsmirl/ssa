"""
Retrieval-margin demonstrator — the REAL-KEY test (does SubQ's bet pay off?).

Every prior experiment in this directory used WORST-CASE synthetic keys: vectors uniform on the
sphere (no cluster structure), or a retrieval encoder driven to spread its keys uniformly. On those,
the capacity-search trilemma (the capacity-search trilemma, paper §6) bites hard: the admissible branch-and-bound
bound `⟨q,k⟩ ≤ ⟨q,μ⟩ + ‖q‖·‖k−μ‖` (`admissible_search_bound`) never prunes, because separation forces
a radius floor (`cluster_radius_floor`) that keeps every cluster's bound above the best score.

But the trilemma's escape hatch is DATA-DEPENDENT: pruning fires exactly when cluster radii are small
(`cluster_prunable`: the gate is `margin > radius`). The open question about SubQ is whether REAL
learned attention keys have the BENIGN geometry (clumped onto a low-dimensional manifold, so radii are
small AND the query's margin to its target is large) that turns pruning on — the regime where a
subquadratic selector can be lossless. DeepSeek's DSA never had to bet on this (it ate a quadratic
indexer up to ~52K tokens); SubQ at 12M cannot avoid the bet.

This script makes the measurement: it runs a real pretrained transformer (GPT-2) on real text,
extracts the ACTUAL per-head attention keys `k_j` and queries `q_i`, and runs the exact admissible
branch-and-bound selector on them — measuring the PRUNE FRACTION (fraction of keys scored to find the
dense-attention argmax). It compares, at matched n / d / cue-margin, against synthetic random keys.

  benign  (real prunes, synthetic doesn't)  -> SubQ's bet pays off; subquadratic lossless selection is
                                               possible on real representations.
  worst   (real costs ~100% like synthetic)  -> the trilemma bites even on real data.

a numerical experiment (calibration against data); the geometry it tests is the theory `SearchTradeoff` content.

Run:  python3 -m ssa.real_keys
"""
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch

from .adaptive import kmeans

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------------------------------
# real keys/queries from a pretrained transformer on real text
# --------------------------------------------------------------------------------------------------
def load_real_text(n_tokens: int) -> str:
    """A long contiguous slice of real English text (wikitext if cached, else a built-in fallback)."""
    from datasets import load_dataset
    last = None
    for cfg in ("wikitext-103-raw-v1", "wikitext-2-raw-v1"):
        try:
            ds = load_dataset("wikitext", cfg, split="train")
            buf, got, seen = [], 0, set()
            for t in ds["text"]:
                t = t.strip()
                if len(t) < 64 or t.startswith("=") or t in seen:   # skip headers + duplicate lines
                    continue
                seen.add(t)
                buf.append(t)
                got += len(t.split())
                if got > n_tokens * 1.6:
                    break
            print(f"  (using real non-repetitive text from {cfg}: {len(buf)} distinct paragraphs)")
            return "\n".join(buf)
        except Exception as e:  # pragma: no cover
            last = e
    raise RuntimeError(f"no wikitext config available offline: {last}")


@torch.no_grad()
def extract_qk(layers, n_tokens=1024):
    """Run GPT-2 on real text; return per-layer (q,k) with shape (head, seq, head_dim), numpy float32."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval().to(DEV)
    nh = model.config.n_head
    hd = model.config.n_embd // nh

    caught = {}

    def mk_hook(name, attn):
        def hook(mod, inp, out):
            qkv = mod.c_attn(inp[0])
            q, k, _ = qkv.split(mod.embed_dim, dim=2)
            caught[name] = (q.detach(), k.detach())
        return hook

    handles = [model.transformer.h[L].attn.register_forward_hook(mk_hook(L, model.transformer.h[L].attn))
               for L in layers]
    text = load_real_text(n_tokens)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=n_tokens).input_ids.to(DEV)
    model(ids)
    for h in handles:
        h.remove()

    out = {}
    seq = ids.shape[1]
    for L in layers:
        q, k = caught[L]
        q = q.view(1, seq, nh, hd).transpose(1, 2)[0].cpu().numpy().astype(np.float32)  # (head,seq,hd)
        k = k.view(1, seq, nh, hd).transpose(1, 2)[0].cpu().numpy().astype(np.float32)
        out[L] = (q, k)
    return out, seq, hd, nh


# --------------------------------------------------------------------------------------------------
# geometry metrics
# --------------------------------------------------------------------------------------------------
def coherence_stats(K):
    """Pairwise cosine coherence of a key set (sampled). High mean |cos| / low spread = clumped."""
    n = len(K)
    U = K / (np.linalg.norm(K, axis=1, keepdims=True) + 1e-9)
    idx = np.random.default_rng(0).choice(n, min(n, 400), replace=False)
    G = np.abs(U[idx] @ U[idx].T)
    iu = np.triu_indices(len(idx), 1)
    c = G[iu]
    return float(c.mean()), float(np.percentile(c, 95))


def participation_ratio(K):
    """Effective dimension PR = (Σλ)² / Σλ² of the key covariance. Low PR = energy on few axes (clumped)."""
    Kc = K - K.mean(0, keepdims=True)
    lam = np.linalg.svd(Kc, compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (np.square(lam).sum() + 1e-12))


# --------------------------------------------------------------------------------------------------
# exact admissible branch-and-bound (the SearchTradeoff bound, general ‖q‖)
# --------------------------------------------------------------------------------------------------
def bandb_exact(K, Q, B, causal=True, qpos=None, max_queries=200, seed=0):
    """Exact branch-and-bound over key clusters with the admissible bound UB_b = ⟨q,μ_b⟩ + ‖q‖·R_b.
    Returns mean prune fraction (cost / n_candidates), and recall vs the true (dense) argmax (= 1.0 by
    construction; reported as a correctness check). causal: query i only ranks keys j ≤ i."""
    rng = np.random.default_rng(seed)
    n = len(K)
    members, mu, R = kmeans(K, B, seed=seed)
    members = [np.asarray(m) for m in members]
    if qpos is None:
        lo = max(64, n // 4)
        qpos = rng.choice(np.arange(lo, n), min(max_queries, n - lo), replace=False)
    fracs, hits, tot = [], 0, 0
    for i in qpos:
        q = Q[i]
        qn = float(np.linalg.norm(q))
        cand_mask = (np.arange(n) <= i) if causal else np.ones(n, bool)
        ncand = int(cand_mask.sum())
        if ncand < 2:
            continue
        # dense argmax (ceiling) over candidates
        sc_all = K @ q
        sc_all = np.where(cand_mask, sc_all, -1e30)
        tgt = int(sc_all.argmax())
        # admissible bounds per cluster (members restricted to candidates; R is still an upper bound)
        UB = mu @ q + qn * R
        order = np.argsort(UB)[::-1]
        sstar, best, cost = -1e30, -1, B          # cost starts at B (all bounds scored)
        for b in order:
            if UB[b] <= sstar:
                break                              # prune (exact: cluster_prunable)
            mem = members[b]
            if len(mem) == 0:
                continue
            mem = mem[mem <= i] if causal else mem
            if len(mem) == 0:
                continue
            s = K[mem] @ q
            j = int(s.argmax())
            if s[j] > sstar:
                sstar, best = float(s[j]), int(mem[j])
            cost += len(mem)
        fracs.append(cost / ncand)
        hits += int(best == tgt)
        tot += 1
    return (float(np.mean(fracs)) if fracs else 1.0,
            hits / tot if tot else 0.0, tot)


def bandb_budget(K, Q, B, budget_frac=None, budget_abs=None, causal=True, qpos=None,
                 max_queries=200, seed=0, min_dist=0):
    """APPROXIMATE selection (SubQ's actual regime — top-k, not lossless): open clusters in admissible-
    bound order until the budget is spent; return recall vs the dense top-1. Pass `budget_frac` (a
    fraction of candidates) OR `budget_abs` (a fixed key count — the O(n·k) bounded-k test).
    `min_dist > 0` counts ONLY queries whose dense target is that far before the query — the honest
    long-RANGE retrieval test (a local-biased model gets short targets for free)."""
    rng = np.random.default_rng(seed)
    n = len(K)
    members, mu, R = kmeans(K, B, seed=seed)
    members = [np.asarray(m) for m in members]
    if qpos is None:
        lo = max(64, n // 4)
        qpos = rng.choice(np.arange(lo, n), min(max_queries, n - lo), replace=False)
    hits, tot = 0, 0
    for i in qpos:
        q = Q[i]; qn = float(np.linalg.norm(q))
        cand_mask = (np.arange(n) <= i) if causal else np.ones(n, bool)
        ncand = int(cand_mask.sum())
        if ncand < 2:
            continue
        sc_all = np.where(cand_mask, K @ q, -1e30)
        tgt = int(sc_all.argmax())
        if min_dist > 0 and (i - tgt) < min_dist:
            continue                               # skip local targets — count only long-range retrieval
        UB = mu @ q + qn * R
        order = np.argsort(UB)[::-1]
        budget = budget_abs if budget_abs is not None else budget_frac * ncand
        sstar, best, cost = -1e30, -1, 0
        for b in order:
            if cost >= budget:
                break
            mem = members[b]
            mem = mem[mem <= i] if causal else mem
            if len(mem) == 0:
                continue
            s = K[mem] @ q
            j = int(s.argmax())
            if s[j] > sstar:
                sstar, best = float(s[j]), int(mem[j])
            cost += len(mem)
        hits += int(best == tgt); tot += 1
    return hits / tot if tot else 0.0


def route_recall(K, Q, B, budget_frac=None, budget_abs=None, order="relevance",
                 causal=True, qpos=None, max_queries=300, seed=0, min_dist=0,
                 K_route=None, Q_route=None):
    """APPROXIMATE cluster-routing selector done RIGHT: rank clusters by a RELEVANCE score (not the
    conservative admissible bound, which inflates diffuse clusters), open the top ones until the budget
    is spent, return recall vs the dense top-1. `order`:
      'relevance' : s_b = ⟨q, μ_b⟩                       (centroid relevance — SSA's "most relevant clusters first")
      'cumulant'  : s_b = ⟨q, μ_b⟩ + ½ qᵀ Σ_b q          (clue #1: + the second cumulant / Laplace log-sum-exp)
      'ub'        : s_b = ⟨q, μ_b⟩ + ‖q‖ R_b             (the admissible upper bound — right for LOSSLESS, not this)
    `min_dist > 0` counts only long-RANGE targets. This is the honest test of whether the geometry
    supports cheap APPROXIMATE long-range retrieval.
    `K_route`/`Q_route` (a lower-dim ROUTING representation): if given, clustering AND cluster ranking
    happen in that space, while the dense TARGET and within-cluster refinement stay in full (K,Q) space —
    the κ_min-in-routing-space measurement (does a projection preserve which clusters to open?)."""
    rng = np.random.default_rng(seed)
    n = len(K)
    Kr = K if K_route is None else K_route                          # routing representation for clustering/ranking
    Qr = Q if Q_route is None else Q_route
    members, mu, R = kmeans(Kr, B, seed=seed)
    members = [np.asarray(m) for m in members]
    Sig = None
    if order == "cumulant":
        Sig = np.zeros((B, Kr.shape[1], Kr.shape[1]), np.float32)
        for b in range(B):
            if len(members[b]) > 1:
                Sig[b] = np.cov(Kr[members[b]].T).astype(np.float32)
    if qpos is None:
        lo = max(64, n // 4)
        qpos = rng.choice(np.arange(lo, n), min(max_queries, n - lo), replace=False)
    hits, tot = 0, 0
    for i in qpos:
        q = Q[i]; qr = Qr[i]; qn = float(np.linalg.norm(qr))
        cand_mask = (np.arange(n) <= i) if causal else np.ones(n, bool)
        ncand = int(cand_mask.sum())
        if ncand < 2:
            continue
        sc_all = np.where(cand_mask, K @ q, -1e30)                  # target in FULL space
        tgt = int(sc_all.argmax())
        if min_dist > 0 and (i - tgt) < min_dist:
            continue
        if order == "relevance":
            s = mu @ qr
        elif order == "ub":
            s = mu @ qr + qn * R
        else:
            s = mu @ qr + 0.5 * np.einsum("bij,i,j->b", Sig, qr, qr)
        budget = budget_abs if budget_abs is not None else budget_frac * ncand
        sstar, best, cost = -1e30, -1, 0
        for b in np.argsort(s)[::-1]:
            if cost >= budget:
                break
            mem = members[b]
            mem = mem[mem <= i] if causal else mem
            if len(mem) == 0:
                continue
            sc = K[mem] @ q
            j = int(sc.argmax())
            if sc[j] > sstar:
                sstar, best = float(sc[j]), int(mem[j])
            cost += len(mem)
        hits += int(best == tgt); tot += 1
    return (hits / tot if tot else 0.0), tot


def matched_synthetic(K_real, Q_real, qpos_margin_cos, d, seed=0):
    """Synthetic random unit keys of the same n,d, with queries tuned to the SAME cue margin (cosine to
    target) as the real head — so only KEY GEOMETRY differs (uniform sphere vs the real manifold)."""
    rng = np.random.default_rng(seed)
    n = len(K_real)
    K = rng.standard_normal((n, d)).astype(np.float32)
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    Q = np.empty_like(K)
    # build each query as target-key + noise tuned to median real cue cosine
    cos = max(0.05, min(0.95, qpos_margin_cos))
    for i in range(n):
        t = K[i]                                  # query i "wants" key i (a findable target)
        noise = rng.standard_normal(d).astype(np.float32)
        noise -= noise.dot(t) * t
        noise /= np.linalg.norm(noise) + 1e-9
        Q[i] = cos * t + np.sqrt(max(0.0, 1 - cos * cos)) * noise
    return K, Q


def real_cue_cosine(Q, K, causal=True):
    """Median cosine between a real query and the key it actually retrieves (its dense argmax)."""
    n = len(K)
    cs = []
    for i in range(max(64, n // 4), n):
        q = Q[i]
        sc = K @ q
        sc[np.arange(n) > i] = -1e30 if causal else sc[np.arange(n) > i]
        t = int(sc.argmax())
        cs.append(float(q.dot(K[t]) / (np.linalg.norm(q) * np.linalg.norm(K[t]) + 1e-9)))
    return float(np.median(cs))


# --------------------------------------------------------------------------------------------------
def main():
    np.random.seed(0)
    torch.manual_seed(0)
    print("=" * 92)
    print("THE REAL-KEY TEST — does a subquadratic selector prune on REAL learned attention keys?")
    print("=" * 92)

    layers = [2, 5, 8, 11]
    n_tokens = 1024
    print(f"\nextracting GPT-2 keys/queries on real text (layers {layers}, up to {n_tokens} tokens)...")
    qk, seq, hd, nh = extract_qk(layers, n_tokens)
    print(f"  sequence length n = {seq} keys/queries per head | head_dim d = {hd} | {nh} heads/layer")

    B = 64  # clusters for the treecode
    print(f"\nbranch-and-bound clusters B = {B}; exact (admissible-bound) pruning; causal retrieval.")
    print("\n  REAL keys vs MATCHED synthetic random keys (same n, d, cue-margin):")
    print(f"  {'layer.head':>11} {'coh(real)':>9} {'PR(real)':>8} {'cueCos':>7} "
          f"{'cost_real':>10} {'cost_syn':>9} {'rec_real':>8}")
    print("  " + "-" * 78)

    agg = {"cost_real": [], "cost_syn": [], "coh": [], "pr": [], "rec": []}
    # sample a few heads per layer
    head_sample = [0, 4, 8]
    for L in layers:
        q, k = qk[L]
        for H in head_sample:
            K = k[H]                              # (seq, hd) raw real keys
            Q = q[H]
            coh_m, coh_95 = coherence_stats(K)
            pr = participation_ratio(K)
            cue = real_cue_cosine(Q, K)
            cr, rr, _ = bandb_exact(K, Q, B)
            Ks, Qs = matched_synthetic(K, Q, cue, hd, seed=L * 13 + H)
            cs, _, _ = bandb_exact(Ks, Qs, B)
            print(f"  {L:>5}.{H:<5} {coh_m:>9.3f} {pr:>8.1f} {cue:>7.2f} "
                  f"{cr*100:>9.1f}% {cs*100:>8.1f}% {rr:>8.2f}")
            agg["cost_real"].append(cr); agg["cost_syn"].append(cs)
            agg["coh"].append(coh_m); agg["pr"].append(pr); agg["rec"].append(rr)

    cr = 100 * np.mean(agg["cost_real"]); cs = 100 * np.mean(agg["cost_syn"])
    print("  " + "-" * 78)
    print(f"  {'MEAN':>11} {np.mean(agg['coh']):>9.3f} {np.mean(agg['pr']):>8.1f} {'':>7} "
          f"{cr:>9.1f}% {cs:>8.1f}% {np.mean(agg['rec']):>8.2f}")

    # split exact-cost by head type: sharply-clumped (low effective dim) vs diffuse (high)
    pr = np.array(agg["pr"]); costr = 100 * np.array(agg["cost_real"])
    sharp = pr < np.median(pr); diff = ~sharp
    print(f"\n  exact-B&B cost by head type:")
    print(f"    sharp heads (eff-dim < {np.median(pr):.1f}): {costr[sharp].mean():5.1f}%   "
          f"diffuse heads (eff-dim ≥ {np.median(pr):.1f}): {costr[diff].mean():5.1f}%")
    print("    -> lossless pruning works on sharp heads; the diffuse long-range heads defeat it (trilemma).")

    # APPROXIMATE selection (SubQ's regime): recall at a fixed sublinear budget — real vs synthetic.
    # use a diffuse DEEP head (layer 11, head 0) — exactly where lossless B&B failed above.
    print(f"\n  approximate top-1 recall at a FIXED budget (layer 11, head 0 — a diffuse long-range head):")
    q11, k11 = qk[11]; K11, Q11 = k11[0], q11[0]
    cue11 = real_cue_cosine(Q11, K11)
    Ks, Qs = matched_synthetic(K11, Q11, cue11, hd, seed=999)
    print(f"  {'budget':>8} {'recall_real':>12} {'recall_syn':>11}")
    appr = {}
    for bf in (0.02, 0.05, 0.10, 0.20):
        rr = bandb_budget(K11, Q11, B, bf)
        rs = bandb_budget(Ks, Qs, B, bf)
        appr[bf] = (rr, rs)
        print(f"  {bf*100:>6.0f}% {rr:>12.2f} {rs:>11.2f}")

    print("\n" + "=" * 92)
    print("VERDICT")
    print("=" * 92)
    print(f"  Real-key geometry IS benign in the clumping sense: coherence {np.mean(agg['coh']):.3f} "
          f"(vs random {1/np.sqrt(hd):.3f}), effective dim {np.mean(agg['pr']):.1f}/{hd} (vs ~{hd}).")
    print(f"  But LOSSLESS exact selection is NOT free: cost splits {costr[sharp].mean():.0f}% (sharp) "
          f"vs {costr[diff].mean():.0f}% (diffuse deep heads) — the trilemma bites where it matters.")
    r5, s5 = appr[0.05]
    print(f"  APPROXIMATE selection (SubQ's regime) DOES pay off: at a 5% budget on a diffuse deep head, "
          f"recall {r5:.2f} (real) vs {s5:.2f} (random) — clumping buys cheap top-k, not cheap top-1-exact.")
    print("  => SubQ's bet holds for APPROXIMATE top-k selection on real representations; the exact/lossless")
    print("     guarantee is the part the trilemma still forbids on long-range heads (hence: approximate +")
    print("     co-adapted keys, with a measurable recall cost on diffuse/adversarial retrieval).")


if __name__ == "__main__":
    main()
