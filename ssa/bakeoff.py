"""
P3 — sub-linear router bake-off. On benign co-trained keys (where P1 found a real ~0.4% floor), trace
RECALL vs SELECTION COST (keys/nodes scored — hardware-neutral router work) for each candidate router and
report the cost to reach recall ≥ target. Lowest cost wins. Candidates:
  * exact            — ceiling (scans n).
  * centroid (IVF)   — core.CentroidSelector (k-means coarse + scan lists).
  * LSH              — core.LSHSelector (SimHash).
  * faiss-ivf        — faiss IndexIVFFlat (the optimized IVF), nprobe sweep.
  * treecode (ours)  — hierarchical_routing build_tree + hier_approx (recursive-radius beam).

Run: python3 -m ssa.bakeoff  ->  paper/figures/bakeoff.json
"""
from __future__ import annotations
import json
import numpy as np
from ssa.core import CentroidSelector, LSHSelector
from ssa.prune_regularizer import train
from ssa.hierarchical_routing import build_tree, hier_approx
import faiss


def cotrained(lam=64.0, n_clusters=128, cs=128, d=64, noise=0.20, seed=0):
    K, _ = train(n_clusters, cs, d, lam)
    K = (K.detach().cpu().numpy() if hasattr(K, "detach") else np.asarray(K)).astype(np.float32)
    K /= np.linalg.norm(K, axis=1, keepdims=True) + 1e-9
    rng = np.random.default_rng(seed + 3)
    Q = K + noise * rng.standard_normal(K.shape).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9
    return K, Q


def faiss_ivf(K, Q, tgt, qidx, nlist, nprobe, k=8):
    quant = faiss.IndexFlatIP(K.shape[1])
    ix = faiss.IndexIVFFlat(quant, K.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
    ix.train(K); ix.add(K); ix.nprobe = nprobe
    hits = 0
    for j, i in enumerate(qidx):
        _, I = ix.search(Q[i:i + 1], k)
        hits += int(tgt[j] in I[0])
    return hits / len(qidx), nprobe * (len(K) / nlist)


def eval_set(sel_fn, tgt, qidx):
    hits = cost = 0
    for j, i in enumerate(qidx):
        cand, c = sel_fn(i)
        hits += int(tgt[j] in set(np.asarray(cand).tolist())); cost += c
    return hits / len(qidx), cost / len(qidx)


def main():
    np.random.seed(0)
    K, Q = cotrained()
    n, d = K.shape
    rng = np.random.default_rng(1)
    qidx = rng.choice(n, 300, replace=False)
    tgt = [int((K @ Q[i]).argmax()) for i in qidx]
    target = 0.90
    print("=" * 90)
    print(f"P3 — router bake-off on co-trained keys (n={n}, d={d}); cost to reach recall ≥ {target}")
    print("=" * 90)
    frontier = {}

    # centroid (IVF, in-repo): sweep budget k
    cs = CentroidSelector().build(K)
    cen = [(eval_set(lambda i, kk=kk: cs.select(Q[i], 20.0, kk), tgt, qidx), kk) for kk in (8, 32, 128, 512, 2048)]
    frontier["centroid"] = [(r, c) for (r, c), _ in cen]

    # LSH (in-repo): sweep #tables L
    lsh = [(eval_set((lambda i, s=LSHSelector(L=L, bits=10).build(K): s.select(Q[i], 20.0, 8)), tgt, qidx)) for L in (4, 8, 16, 32)]
    frontier["lsh"] = lsh

    # faiss-ivf: sweep nprobe
    nlist = int(n ** 0.5)
    fai = [faiss_ivf(K, Q, tgt, qidx, nlist, npb) for npb in (1, 2, 4, 8, 16)]
    frontier["faiss-ivf"] = fai

    # treecode (ours): build tree once, sweep keep_coarse
    nc = max(2, int(round(n ** (1 / 3)))); fine_per = max(2, int(round(n ** (1 / 3))))
    tree = build_tree(K, nc, fine_per)

    def tc(i, kc):
        best, nodes = hier_approx(K, Q[i], tree, kc, 8)
        return ([best], nodes)
    tcr = [(eval_set(lambda i, kc=kc: tc(i, kc), tgt, qidx)) for kc in (1, 2, 4, 8, 16)]
    frontier["treecode"] = tcr

    def cheapest(pts):
        ok = [(r, c) for (r, c) in pts if r >= target]
        return min(ok, key=lambda x: x[1]) if ok else (max(pts, key=lambda x: x[0]))

    print(f"  {'router':>12} {'recall@target':>14} {'cost (keys/n)':>14} {'reached?':>9}")
    out = {}
    for name, pts in frontier.items():
        r, c = cheapest(pts)
        out[name] = dict(recall=r, cost=c, cost_frac=c / n, reached=bool(r >= target),
                         frontier=[[float(a), float(b)] for a, b in pts])
        print(f"  {name:>12} {r:>14.3f} {c/n*100:>13.1f}% {'YES' if r >= target else 'no':>9}")
    json.dump(dict(n=n, target=target, routers=out), open("paper/figures/bakeoff.json", "w"), indent=2)
    print("\n  Lowest cost-fraction at recall≥target wins the router slot for the fused kernel (P4).")
    print("  wrote paper/figures/bakeoff.json")


if __name__ == "__main__":
    main()
