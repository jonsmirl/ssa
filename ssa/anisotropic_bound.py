"""
Does the anisotropic (ellipsoidal) admissible bound prune more than the isotropic one? — the test of
the anisotropic bound on real keys.

The flat admissible bound `⟨q,μ⟩ + ‖q‖·R` (isotropic radius) is loose on a cluster spread unevenly — and
that looseness is why lossless hierarchical pruning didn't fire (`hierarchical_routing.py`). The
ellipsoidal bound `⟨q,μ⟩ + R'·√(qᵀΣq)` (`ellipsoidal_search_bound`) is the tightest two-moment upper
bound; its radius term is the SAME `√(qᵀΣq)` the cumulant routing uses (`ellipsoidal_radius_sq`). If real
clusters are anisotropic, it should prune more — making lossless selection cheaper in the benign regime.

We measure exact (lossless) branch-and-bound cost (keys scored) with the isotropic bound, the ellipsoidal
bound, and their min, on the cached Qwen-16K deep-head keys. Both bounds are valid upper bounds, so all
variants return the true argmax (recall = 1); the cheaper one prunes more.

Run:  python3 -m ssa.anisotropic_bound
"""
from __future__ import annotations
import os
import numpy as np

from .adaptive import kmeans

CACHE = "/tmp/qwen_deep_keys.npz"


def cluster_stats(K, members, mu, eps=1e-2):
    """Per cluster: isotropic radius R_iso, and the ellipsoid (Σ+εI)^{1/2}, its Mahalanobis radius R_maha."""
    stats = []
    for b, mem in enumerate(members):
        mem = np.asarray(mem)
        if len(mem) == 0:
            stats.append(None); continue
        X = K[mem] - mu[b]                                   # (m, d) centred
        d = X.shape[1]
        Sig = X.T @ X / len(mem) + eps * np.eye(d, dtype=np.float32)
        lam, V = np.linalg.eigh(Sig)                         # Σ = V diag(λ) Vᵀ
        lam = np.clip(lam, 1e-8, None)
        half = (V * np.sqrt(lam)) @ V.T                      # Σ^{1/2}
        invhalf = (V / np.sqrt(lam)) @ V.T                   # Σ^{-1/2}
        R_iso = float(np.max(np.linalg.norm(X, axis=1)))
        R_maha = float(np.max(np.linalg.norm(X @ invhalf, axis=1)))   # max ‖Σ^{-1/2}(k−μ)‖
        stats.append((mu[b].astype(np.float32), R_iso, half.astype(np.float32), R_maha))
    return stats


def bnb_cost(K, q, members, stats, mode):
    """Lossless best-first B&B with the chosen upper bound. Returns (argmax_key, keys_scored)."""
    qn = float(np.linalg.norm(q))
    ubs = []
    for b, st in enumerate(stats):
        if st is None:
            continue
        mu_b, R_iso, half, R_maha = st
        base = float(mu_b @ q)
        if mode == "iso":
            ub = base + qn * R_iso
        elif mode == "ellip":
            ub = base + R_maha * float(np.linalg.norm(half @ q))
        else:                                                # min of the two valid bounds
            ub = base + min(qn * R_iso, R_maha * float(np.linalg.norm(half @ q)))
        ubs.append((ub, b))
    best, bkey, cost = -1e30, -1, 0
    for ub, b in sorted(ubs, key=lambda t: -t[0]):
        if ub <= best:
            break                                            # prune: nothing better can remain (exact)
        mem = np.asarray(members[b])
        sc = K[mem] @ q
        j = int(sc.argmax())
        if sc[j] > best:
            best, bkey = float(sc[j]), int(mem[j])
        cost += len(mem)
    return bkey, cost


def main():
    if not os.path.exists(CACHE):
        print(f"  cached Qwen keys not found at {CACHE}; run longctx_probe first.")
        return
    d = np.load(CACHE)
    K, Q = d["K_18_0"], d["Q_18_0"]
    n = len(K); B = 64
    print("=" * 82)
    print("ANISOTROPIC vs ISOTROPIC PRUNE BOUND — lossless B&B cost on Qwen-16K deep head")
    print("=" * 82)
    members, mu, _ = kmeans(K, B, seed=0)
    members = [np.asarray(m) for m in members]
    stats = cluster_stats(K, members, mu)
    rng = np.random.default_rng(0)
    qpos = rng.choice(np.arange(max(64, n // 4), n), 120, replace=False)
    cost = {"iso": [], "ellip": [], "min": []}
    rec = {"iso": 0, "ellip": 0, "min": 0}
    tighter = tot = 0
    for i in qpos:
        q = Q[i]
        tgt = int((K @ q).argmax())
        for mode in ("iso", "ellip", "min"):
            bkey, c = bnb_cost(K, q, members, stats, mode)
            cost[mode].append(c); rec[mode] += int(bkey == tgt)
        # how often is the ellipsoidal bound strictly tighter, per cluster?
        qn = float(np.linalg.norm(q))
        for st in stats:
            if st is None:
                continue
            _, R_iso, half, R_maha = st
            tot += 1
            tighter += int(R_maha * float(np.linalg.norm(half @ q)) < qn * R_iso)
    print(f"  n={n}, clusters B={B}; 120 queries; lossless (all return the true argmax).\n")
    print(f"  {'bound':>10} {'avg keys scored':>16} {'frac of n':>10} {'recall':>8}")
    for mode in ("iso", "ellip", "min"):
        m = float(np.mean(cost[mode]))
        print(f"  {mode:>10} {m:>16.0f} {m/n*100:>9.1f}% {rec[mode]/len(qpos):>8.3f}")
    print(f"\n  the ellipsoidal bound is strictly tighter than the isotropic one on "
          f"{tighter/tot*100:.0f}% of (query, cluster) pairs.")
    print("\n" + "=" * 82)
    iso = float(np.mean(cost["iso"])); mn = float(np.mean(cost["min"]))
    if mn < 0.95 * iso:
        print(f"  The anisotropic bound prunes MORE: min-bound lossless cost {mn/n*100:.1f}% vs isotropic "
              f"{iso/n*100:.1f}% — the cluster shape (Σ) is exploitable, so the tightest two-moment bound")
        print("  cuts lossless selection cost on real (anisotropic) keys. Routing and pruning share qᵀΣq.")
    else:
        print(f"  Honest result: on these keys the anisotropic bound does NOT prune much more "
              f"({mn/n*100:.1f}% vs {iso/n*100:.1f}%) — for a generic query the directional radius averages")
        print("  back to the isotropic one; the win needs queries aligned with the clusters' thin axes")
        print("  (which training could induce — the bound-derived regularizer route).")


if __name__ == "__main__":
    main()
