"""
Retrieval-margin demonstrator — adaptive Barnes-Hut / branch-and-bound selection.

The flat router (co_train.py) picks a FIXED set of clusters and is approximate (capped well below the
dense ceiling). The FMM/Barnes-Hut clue: refine adaptively. Give each cluster an ADMISSIBLE upper
bound on its best member's score,

    UB_b(q) = q . mu_b + R_b ,   R_b = max_{j in b} || k_j - mu_b ||   (radius),

since q.k_j = q.mu_b + q.(k_j - mu_b) <= q.mu_b + ||k_j - mu_b||. Best-first branch-and-bound: open
clusters in descending UB, track the best score found s*, and STOP once the next cluster's UB <= s*
(no unopened cluster can hold a better key). With an admissible bound this is EXACT (recall = the
dense ceiling) at ADAPTIVE cost; with a budget cap it is an anytime approximation. We measure whether
the cost stays sublinear, and compare to the flat router at matched cost.

Run: python3 -m ssa.adaptive
"""
from __future__ import annotations
import numpy as np
import torch
from .train import Encoder, train_encoder, corrupt, DEV

torch.manual_seed(0)
np.random.seed(0)


def kmeans(K, B, iters=12, seed=0):
    """Plain k-means (tight clusters — what branch-and-bound wants). Returns members, means, radii."""
    rng = np.random.default_rng(seed)
    n = len(K)
    c = K[rng.choice(n, B, replace=False)].copy()
    for _ in range(iters):
        d2 = (c * c).sum(1)[None, :] - 2.0 * (K @ c.T)        # argmin of ||x-c||^2 (||x||^2 const)
        a = d2.argmin(1)
        for b in range(B):
            mb = a == b
            if mb.any():
                c[b] = K[mb].mean(0)
    a = ((c * c).sum(1)[None, :] - 2.0 * (K @ c.T)).argmin(1)
    members = [np.where(a == b)[0] for b in range(B)]
    mu = c.copy()
    R = np.zeros(B, np.float32)
    for b in range(B):
        if len(members[b]):
            R[b] = float(np.max(np.linalg.norm(K[members[b]] - mu[b], axis=1)))
    return members, mu, R


@torch.no_grad()
def probe_bandb(enc, X_test, B=256, beta=20.0, budget=None, trials=600, seed=0,
                mask_frac=0.4, noise=0.1):
    """Branch-and-bound retrieval. budget=None -> exact (prune by admissible bound); else open in UB
    order until `budget` keys scored (anytime approximation)."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    K = enc(X_test).cpu().numpy().astype(np.float32)
    n = len(K)
    members, mu, R = kmeans(K, B, seed=seed)
    hits = exact = 0
    tot_cost = opened_tot = 0
    for _ in range(trials):
        t = int(rng.integers(n))
        q = enc(corrupt(X_test[t:t + 1], mask_frac, noise, gen=gen))[0].cpu().numpy().astype(np.float32)
        exact += int(np.argmax(K @ q) == t)                  # dense ceiling for this cue
        UB = mu @ q + R                                       # admissible upper bound per cluster
        order = np.argsort(UB)[::-1]
        sstar, best, cost, opened = -1e30, -1, B, 0           # cost starts at B (scored all bounds)
        for b in order:
            if budget is None and UB[b] <= sstar:
                break                                         # prune: nothing better can remain (EXACT)
            if budget is not None and cost >= budget:
                break                                         # anytime: budget exhausted
            mem = members[b]
            if len(mem):
                sc = K[mem] @ q
                j = int(sc.argmax())
                if sc[j] > sstar:
                    sstar, best = float(sc[j]), int(mem[j])
                cost += len(mem); opened += 1
        hits += int(best == t)
        tot_cost += cost; opened_tot += opened
    Rbar = float(R[R > 0].mean()) if (R > 0).any() else 0.0
    return dict(recall=hits / trials, ceiling=exact / trials, frac=tot_cost / trials / n,
                opened=opened_tot / trials, B=B, Rbar=Rbar)


def main():
    print("=" * 80)
    print("ADAPTIVE BARNES-HUT / BRANCH-AND-BOUND SELECTION (the FMM clue)")
    print("=" * 80)
    d_raw, d = 64, 128
    X = torch.randn(24000, d_raw, device=DEV)
    X = X / X.norm(dim=-1, keepdim=True)
    X_train, X_test = X[:12000], X[12000:]
    print("\ntraining the retrieval encoder (same as (i))...")
    enc = train_encoder(X_train, d, prox_weight=0.0)
    Xte = X_test[:10000]

    print("\n[exact branch-and-bound] (admissible bound -> recall = dense ceiling), varying #clusters:")
    print(f"  {'B':>6} {'recall':>8} {'ceiling':>8} {'avg cost (frac n)':>18} {'clusters opened':>16}")
    for B in (64, 256, 1024):
        r = probe_bandb(enc, Xte, B=B, budget=None)
        print(f"  {B:>6} {r['recall']:>8.3f} {r['ceiling']:>8.3f} {r['frac']*100:>16.1f}% {r['opened']:>14.1f}/{B}")

    print("\n[anytime branch-and-bound] (UB-ordered, budget-capped) vs the flat router's ~0.48@10%:")
    print(f"  {'budget':>8} {'recall':>8} {'cost (frac n)':>14}")
    Bb = 256
    for budfrac in (0.05, 0.10, 0.20):
        r = probe_bandb(enc, Xte, B=Bb, budget=int(budfrac * 10000))
        print(f"  {budfrac*100:>6.0f}% {r['recall']:>8.3f} {r['frac']*100:>13.1f}%")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    rex = probe_bandb(enc, Xte, B=256, budget=None)
    print(f"  Exact B&B (B=256): recall {rex['recall']:.3f} (= ceiling {rex['ceiling']:.3f}), "
          f"avg cost {rex['frac']*100:.1f}% of n, opening {rex['opened']:.1f}/256 clusters.")
    print("  -> Adaptive refinement is LOSSLESS by construction (admissible bound => exact argmax).")
    print("     Whether it is also CHEAP is set by the cue's margin: a confident cue prunes most")
    print("     clusters (cheap); an ambiguous (corrupted) cue leaves many clusters un-prunable")
    print("     (cost rises). Cost of exact retrieval scales inversely with the detectability margin —")
    print("     the same gap-vs-log-n quantity from recoveryWeight, now as a COMPUTE cost.")


if __name__ == "__main__":
    main()
