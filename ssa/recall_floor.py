"""
P1 — the recall-vs-κ frontier (the floor map). For each geometry × difficulty × routing-score, sweep the
attention budget κ and find κ_min, the smallest budget that recovers the needle at recall ≥ target. κ_min/n
IS the achievable floor (speedup ≤ n/κ_min); lower is better. The levers that move it down — the cumulant
term and co-training — are read off directly. Doubles as the SubQ test: does benign geometry support a
tiny κ at long range?

Drives the existing `real_keys.route_recall` harness over geometry generators (synthetic clustered/random,
co-trained, and real-model keys when available). Run: python3 -m ssa.recall_floor -> paper/figures/recall_floor.json
"""
from __future__ import annotations
import json
import numpy as np
from ssa.real_keys import route_recall
from ssa.core import clustered_keys


def _norm(x):
    x = np.asarray(x, np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def synth_clustered(n=8192, d=64, spread=0.10, noise=0.30, seed=0):
    K, _ = clustered_keys(n, d, B=max(8, int(n ** 0.45)), spread=spread, seed=seed)
    rng = np.random.default_rng(seed + 1)
    Q = _norm(K) + noise * rng.standard_normal((n, d)).astype(np.float32)
    return _norm(K), _norm(Q)


def synth_random(n=8192, d=64, noise=0.30, seed=0):
    rng = np.random.default_rng(seed)
    K = _norm(rng.standard_normal((n, d)).astype(np.float32))
    Q = K + noise * rng.standard_normal((n, d)).astype(np.float32)
    return K, _norm(Q)


def cotrained(lam, n_clusters=64, cs=128, d=64, noise=0.20, seed=0):
    from ssa.prune_regularizer import train
    K, _ = train(n_clusters, cs, d, lam)
    K = np.asarray(K.detach().cpu().numpy() if hasattr(K, "detach") else K, np.float32)
    rng = np.random.default_rng(seed + 7)
    Q = K + noise * rng.standard_normal(K.shape).astype(np.float32)
    return _norm(K), _norm(Q)


def kappa_min(K, Q, order, target=0.90, min_dist=0, B=None, seed=0):
    """Smallest budget_abs κ with recall ≥ target (returns (κ_min, recall_there, recall_at_max))."""
    n = len(K)
    B = B or max(8, int(round(n ** 0.5)))
    grid = sorted(set(int(n * f) for f in (0.004, 0.008, 0.016, 0.03, 0.06, 0.12, 0.25, 0.5)))
    last = 0.0
    for kap in grid:
        rec, tot = route_recall(K, Q, B, budget_abs=kap, order=order, min_dist=min_dist,
                                max_queries=200, seed=seed)
        last = rec
        if rec >= target and tot > 0:
            return kap, rec, None
    return None, None, last


def main():
    np.random.seed(0)
    target = 0.90
    print("=" * 96)
    print(f"P1 — recall-vs-κ floor map  (κ_min = smallest budget reaching recall ≥ {target}; lower = lower floor)")
    print("=" * 96)
    print(f"  {'geometry':>22} {'difficulty':>11} {'centroid κ_min/n':>18} {'cumulant κ_min/n':>18}")
    results = []

    def row(name, K, Q, min_dist=0, diff="all"):
        kc, rc, lc = kappa_min(K, Q, "relevance", target, min_dist)
        ku, ru, lu = kappa_min(K, Q, "cumulant", target, min_dist)
        n = len(K)
        fc = f"{kc/n*100:.1f}%" if kc else f"miss({lc:.2f})"
        fu = f"{ku/n*100:.1f}%" if ku else f"miss({lu:.2f})"
        print(f"  {name:>22} {diff:>11} {fc:>18} {fu:>18}")
        results.append(dict(geometry=name, difficulty=diff, n=n,
                            centroid_kmin=kc, cumulant_kmin=ku, centroid_floor=(kc/n if kc else None),
                            cumulant_floor=(ku/n if ku else None)))

    # synthetic geometry × difficulty
    Kc, Qc = synth_clustered(spread=0.10)
    row("synth-clustered(tight)", Kc, Qc)
    row("synth-clustered(tight)", Kc, Qc, min_dist=len(Kc) // 4, diff="long-range")
    Kd, Qd = synth_clustered(spread=0.45)
    row("synth-clustered(diffuse)", Kd, Qd)
    Kr, Qr = synth_random()
    row("synth-random(adversarial)", Kr, Qr)
    row("synth-random(adversarial)", Kr, Qr, min_dist=len(Kr) // 4, diff="long-range")

    # co-training lever
    try:
        K0, Q0 = cotrained(lam=0.0)
        row("co-trained(λ=0)", K0, Q0)
        K1, Q1 = cotrained(lam=64.0)
        row("co-trained(λ=64)", K1, Q1)
    except Exception as e:
        print(f"  (co-training skipped: {str(e)[:50]})")

    # real-model keys, if cached (no download)
    import os
    for tag, path in (("qwen-deep(real)", "/tmp/qwen_deep_keys.npz"), ("gemma-deep(real)", "/tmp/gemma_keys.npz")):
        if os.path.exists(path):
            try:
                z = np.load(path)
                kk = [k for k in z.files if "k" in k.lower()][0]
                qq = [k for k in z.files if "q" in k.lower()][0]
                K, Q = _norm(z[kk].reshape(-1, z[kk].shape[-1])), _norm(z[qq].reshape(-1, z[qq].shape[-1]))
                m = min(len(K), len(Q), 16384)
                row(tag, K[:m], Q[:m])
                row(tag, K[:m], Q[:m], min_dist=m // 4, diff="long-range")
            except Exception as e:
                print(f"  ({tag} skipped: {str(e)[:50]})")

    json.dump(dict(target=target, results=results), open("paper/figures/recall_floor.json", "w"), indent=2)
    print("\n  κ_min/n is the floor (speedup ≤ n/κ_min). Read off: cumulant vs centroid (the routing lever),")
    print("  co-trained λ=0 vs 64 (the training lever), tight vs diffuse vs adversarial, and long-range blow-up.")
    print("  wrote paper/figures/recall_floor.json")


if __name__ == "__main__":
    main()
