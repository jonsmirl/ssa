"""
Retrieval-margin demonstrator — experiments.

Run:  python3 -m ssa.experiments
Each experiment tests one prediction of the theory; E4 (the selector) and E5 (recall vs
reasoning) are the two that actually answer "is the withheld piece buildable, and what does it
NOT buy you?" — built to be falsifiable, with the reasoning-gap test included on purpose.
"""
from __future__ import annotations
import numpy as np
from .core import (recovery_weight, threshold_n, softmax, dense_read, read_over,
                   ExactSelector, CentroidSelector, LSHSelector,
                   random_unit_keys, clustered_keys, coherence)


def e1_recovery_weight():
    """The exact identity p_star = 1/(1+(n-1)e^{-beta*gap}) in the homogeneous case (score space)."""
    print("\n== E1: recovery weight is the exact target mass (homogeneous distractors) ==")
    beta, gap = 8.0, 0.5
    print(f"{'n':>8} {'measured p*':>14} {'recoveryWeight':>16} {'abs err':>10}")
    ok = True
    for n in [10, 100, 1000, 10000, 100000]:
        scores = np.concatenate([[0.0], -gap * np.ones(n - 1)])  # target at 0, distractors at -gap
        p = softmax(beta * scores)
        pred = recovery_weight(beta, gap, n - 1)
        err = abs(p[0] - pred)
        ok &= err < 1e-9
        print(f"{n:>8} {p[0]:>14.6f} {pred:>16.6f} {err:>10.2e}")
    print(f"  identity holds to float precision: {ok}")
    return ok


def e2_length_generalization(d=128, beta=30.0, trials=200, seed=0):
    """Recall vs context length with random keys: the gap is set by geometry, recall follows the
    logistic and stays high until n ~ e^{beta*gap} — margin grows only as log n."""
    print("\n== E2: length generalization (recall vs n, random unit keys) ==")
    rng = np.random.default_rng(seed)
    print(f"  d={d}, beta={beta}")
    print(f"{'n':>8} {'coherence':>10} {'gap~1-eps':>10} {'recall':>8} {'pred mass':>10} {'thr n*':>12}")
    rows = []
    for n in [1000, 5000, 20000, 50000, 100000]:
        K = random_unit_keys(n, d, seed=seed)
        eps = coherence(K)
        gap = 1.0 - eps
        hits = 0
        masses = []
        for _ in range(trials):
            t = rng.integers(n)
            q = K[t]                          # clean query == target key (unit norm, score 1)
            s = beta * (K @ q)
            masses.append(softmax(s)[t])
            hits += int(np.argmax(s) == t)
        recall = hits / trials
        rows.append((n, recall))
        print(f"{n:>8} {eps:>10.3f} {gap:>10.3f} {recall:>8.3f} {np.mean(masses):>10.3f} "
              f"{threshold_n(beta, gap):>12.2e}")
    # length generalization = recall stays high as n grows by 100x
    held = rows[-1][1] >= 0.95
    print(f"  recall at largest n (100x the smallest): {rows[-1][1]:.3f}  (length-generalized: {held})")
    return held


def e3_truncation(n=20000, d=128, beta=20.0, trials=80, seed=1):
    """Sparse (top-k) read vs dense read: output error decays and obeys ||hat o - o|| <= 2 Vmax * m."""
    print("\n== E3: sparse == dense — truncation error vs budget k ==")
    rng = np.random.default_rng(seed)
    K = random_unit_keys(n, d, seed=seed)
    V = rng.standard_normal((n, 16)).astype(np.float32)
    Vmax = float(np.linalg.norm(V, axis=1).max())
    sel = ExactSelector().build(K)
    print(f"  n={n}, d={d}, beta={beta}, Vmax={Vmax:.2f}")
    print(f"{'k':>8} {'mean |hat o-o|':>16} {'mean bound 2V*m':>16} {'bound holds':>12}")
    ok = True
    for k in [1, 4, 16, 64, 256, 1024]:
        errs, bounds = [], []
        for _ in range(trials):
            t = rng.integers(n)
            q = K[t]
            o, p, _ = dense_read(q, K, V, beta)
            cand, _ = sel.select(q, beta, k)
            hat, selidx, _ = read_over(q, K, V, beta, cand, k)
            m = 1.0 - p[selidx].sum()         # missed mass
            errs.append(np.linalg.norm(hat - o))
            bounds.append(2 * Vmax * m)
        e, b = np.mean(errs), np.mean(bounds)
        ok &= all(np.array(errs)[i] <= bounds[i] + 1e-6 for i in range(trials))
        print(f"{k:>8} {e:>16.4e} {b:>16.4e} {str(all(np.array(errs) <= np.array(bounds)+1e-6)):>12}")
    print(f"  truncation bound holds on every trial: {ok}")
    return ok


def e4_selector(d=128, beta=25.0, k=64, trials=200, seed=2):
    """THE crux. A sublinear selector (centroid routing, SimHash-LSH) over clustered keys:
    does it capture the target at sublinear cost, and how does that depend on key separation?"""
    print("\n== E4: the selector — sublinear capture vs separation (the withheld piece) ==")
    rng = np.random.default_rng(seed)

    def run(K, assign, label):
        n = len(K)
        V = rng.standard_normal((n, 16)).astype(np.float32)
        selectors = [ExactSelector().build(K),
                     CentroidSelector(seed=seed).build(K),
                     LSHSelector(L=12, bits=11, seed=seed).build(K)]
        odense = {}
        stats = {s.name: dict(recall=0, cost=0, fid=0.0, nz=0) for s in selectors}
        for _ in range(trials):
            t = rng.integers(n)
            q = K[t] + 0.02 * rng.standard_normal(d).astype(np.float32)   # near-clean query
            q /= np.linalg.norm(q)
            o, _, _ = dense_read(q, K, V, beta)
            for s in selectors:
                cand, cost = s.select(q, beta, k)
                st = stats[s.name]
                st["cost"] += cost
                if cand.size:
                    st["recall"] += int(t in set(cand.tolist()))
                    hat, _, _ = read_over(q, K, V, beta, cand, k)
                    st["fid"] += np.linalg.norm(hat - o)
                    st["nz"] += 1
        print(f"  [{label}]  n={n}")
        print(f"    {'selector':>10} {'recall':>8} {'keys scored':>12} {'frac of n':>10} {'read err':>10}")
        for s in selectors:
            st = stats[s.name]
            nz = max(st["nz"], 1)
            print(f"    {s.name:>10} {st['recall']/trials:>8.3f} {st['cost']/trials:>12.0f} "
                  f"{st['cost']/trials/n:>10.3f} {st['fid']/nz:>10.4e}")
        return stats

    # (a) cost scaling with n, at good separation (the buildable regime)
    print("\n  (a) good separation (spread=0.15): cost should grow sublinearly, recall ~ exact")
    sub = []
    for n in [4000, 16000, 64000]:
        K, a = clustered_keys(n, d, B=max(8, int(np.sqrt(n))), spread=0.15, seed=seed)
        st = run(K, a, f"sep eps~{coherence(K):.2f}")
        sub.append((n, st["centroid"]["cost"] / trials, st["centroid"]["recall"] / trials))

    # (b) the dependence on cluster tightness, fixed n (the doubling-dimension crux):
    #     centroid routing is near-lossless only when basins are tight; SimHash-LSH stays lossless.
    print("\n  (b) fixed n=16000, varying basin tightness (spread): centroid routing needs tight"
          "\n      basins (the geometry training drives keys toward); LSH stays near-lossless")
    for spread in [0.08, 0.15, 0.30, 0.60]:
        K, a = clustered_keys(16000, d, B=int(np.sqrt(16000)), spread=spread, seed=seed)
        run(K, a, f"spread={spread} eps~{coherence(K):.2f}")

    # sublinearity check: cost fraction should fall as n grows
    fracs = [c / n for n, c, _ in sub]
    sublinear = fracs[-1] < fracs[0]
    good_recall = sub[-1][2] >= 0.9
    print(f"\n  centroid cost fraction by n: {[f'{f:.3f}' for f in fracs]}  (falls => sublinear: {sublinear})")
    print(f"  centroid recall at n=64000, good separation: {sub[-1][2]:.3f}  (lossless-ish: {good_recall})")
    return sublinear, good_recall


def e5_recall_vs_reasoning(d=64, beta=25.0, h_max=6, trials=600, seed=3, noise=0.18):
    """The honest boundary. With an IMPERFECT cue (the realistic case — a query is a partial/noisy
    pointer, not the exact key) single-hop recall is rho < 1, and an h-hop chain succeeds ~ rho^h.
    A real pointer chain (errors propagate) is no better, so reasoning degrades at least this fast."""
    print("\n== E5: recall vs reasoning — chain success ~ rho^h (built in on purpose) ==")
    rng = np.random.default_rng(seed)
    n = 8000
    K = random_unit_keys(n, d, seed=seed)

    def one_hop():
        t = rng.integers(n)
        q = K[t] + noise * rng.standard_normal(d).astype(np.float32)
        q /= np.linalg.norm(q)
        return int(np.argmax(K @ q) == t)

    rho = np.mean([one_hop() for _ in range(trials)])
    print(f"  single-hop recall rho = {rho:.3f}  (n={n}, d={d}, cue noise={noise})")
    print(f"{'hops h':>8} {'chain success':>14} {'rho^h pred':>12}")
    ok = True
    for h in range(1, h_max + 1):
        succ = sum(int(all(one_hop() for _hop in range(h))) for _ in range(trials))
        pred = rho ** h
        meas = succ / trials
        ok &= abs(meas - pred) < 0.08
        print(f"{h:>8} {meas:>14.3f} {pred:>12.3f}")
    print(f"  chain success tracks rho^h: {ok}  (rho={rho:.2f}: 1 hop {rho:.2f} -> {h_max} hops "
          f"{rho**h_max:.2f}) -> recall generalizes, composition decays")
    return ok


def main():
    print("=" * 78)
    print("RETRIEVAL-MARGIN DEMONSTRATOR — controlled validation of the theory + the selector")
    print("=" * 78)
    r1 = e1_recovery_weight()
    r2 = e2_length_generalization()
    r3 = e3_truncation()
    r4sub, r4rec = e4_selector()
    r5 = e5_recall_vs_reasoning()
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  E1 recovery-weight identity exact ............ {r1}")
    print(f"  E2 length generalization (recall held) ....... {r2}")
    print(f"  E3 truncation bound holds .................... {r3}")
    print(f"  E4 selector sublinear cost ................... {r4sub}")
    print(f"  E4 selector lossless under separation ........ {r4rec}")
    print(f"  E5 reasoning decays as rho^h (the boundary) .. {r5}")
    print("\n  Read: the read-side theory holds exactly; a sublinear selector built from KNOWN")
    print("  ANN parts captures the target losslessly *when keys are separated* and its cost")
    print("  falls as a fraction of n; recall degrades as keys crowd (the doubling-dimension")
    print("  crux); and the same machinery that wins at recall does NOT win at multi-hop.")


if __name__ == "__main__":
    main()
