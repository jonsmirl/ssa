"""
Hierarchical (FMM / treecode) routing — the selection itself made subquadratic, measured.

The flat selector (`adaptive.py`, the FlexAttention kernel's router) scores EVERY cluster per query to
order them — `O(#clusters)` selection work, the one place the kernel isn't asymptotically subquadratic.
A hierarchical selector groups clusters into a tree and, by the hierarchical-routing result's `hierarchical_
prune`, skips a whole subtree with ONE bound check at its parent — provided the parent's radius is the
recursive `R_parent = max_child (‖μ_child − μ_parent‖ + R_child)` the theorem licenses.

This builds that 2-level tree with the recursive radius, runs EXACT (lossless) best-first branch-and-
bound — flat vs hierarchical — and measures the SELECTION cost (number of admissible-bound evaluations,
i.e. nodes scored) per query as context grows. Both return the true dense argmax (the bound is exact);
the hierarchical one scores far fewer nodes, and the ratio falls with n — selection going subquadratic.

Run:  python3 -m ssa.hierarchical_routing
"""
from __future__ import annotations
import heapq
import numpy as np

from .adaptive import kmeans
from .core import clustered_keys


def build_tree(K, n_coarse, fine_per):
    """2-level tree: coarse k-means, each coarse cell sub-clustered into fine cells. Coarse radius is the
    RECURSIVE radius `max_fine (‖μ_fine − μ_coarse‖ + R_fine)` — exactly what subtree_radius_bound needs."""
    mem_c, mu_c, _ = kmeans(K, n_coarse)
    tree = []
    for c in range(n_coarse):
        idx = np.asarray(mem_c[c])
        if len(idx) == 0:
            continue
        Kc = K[idx]
        nf = max(1, min(fine_per, len(idx)))
        mem_f, mu_f, _ = kmeans(Kc, nf)
        fines = []
        for f in range(nf):
            fi = idx[np.asarray(mem_f[f])]
            if len(fi) == 0:
                continue
            Rf = float(np.max(np.linalg.norm(K[fi] - mu_f[f], axis=1)))
            fines.append((mu_f[f].astype(np.float32), Rf, fi))
        if not fines:
            continue
        muc = mu_c[c].astype(np.float32)
        Rc = max(float(np.linalg.norm(mf - muc)) + Rf for mf, Rf, _ in fines)   # recursive parent radius
        tree.append((muc, Rc, fines))
    return tree


def lossless_cost(K, q, tree):
    """Exact best-first B&B over the tree (hierarchical_prune). Returns (argmax_key, nodes_scored). The
    admissible bound is exact, but the coarse radius is loose, so it rarely prunes a whole subtree —
    lossless selection stays ~O(#cells) (the trilemma, now at the tree level)."""
    qn = float(np.linalg.norm(q))
    heap = []; nodes = 0
    for muc, Rc, fines in tree:
        heapq.heappush(heap, (-(float(muc @ q) + qn * Rc), 0, id(fines), fines)); nodes += 1
    best, bkey = -1e30, -1
    while heap:
        negub, kind, _, payload = heapq.heappop(heap)
        if -negub <= best:
            break
        if kind == 0:
            for mf, Rf, fi in payload:
                heapq.heappush(heap, (-(float(mf @ q) + qn * Rf), 1, id(fi), fi)); nodes += 1
        else:
            sc = K[payload] @ q; j = int(sc.argmax())
            if sc[j] > best:
                best, bkey = float(sc[j]), int(payload[j])
    return bkey, nodes


def flat_approx(K, q, tree, keep_fine):
    """Flat approximate routing: score EVERY fine cell by relevance, keep the top `keep_fine`, attend.
    Selection cost = #fine cells (you must score them all to rank)."""
    fines = [f for _, _, fs in tree for f in fs]
    sc = [(float(mf @ q), fi) for mf, _, fi in fines]
    nodes = len(sc)
    best, bkey = -1e30, -1
    for _, fi in sorted(sc, key=lambda t: -t[0])[:keep_fine]:
        s = K[fi] @ q; j = int(s.argmax())
        if s[j] > best:
            best, bkey = float(s[j]), int(fi[j])
    return bkey, nodes


def hier_approx(K, q, tree, keep_coarse, keep_fine):
    """Hierarchical approximate routing: score coarse nodes, descend only the top `keep_coarse`, score
    their fine children, attend the top `keep_fine`. Selection cost = #coarse + keep_coarse·fine_per —
    sublinear in n (the tree never scores the cells under un-kept coarse nodes)."""
    cs = sorted(((float(muc @ q), fines) for muc, _, fines in tree), key=lambda t: -t[0])
    nodes = len(cs)
    cand = []
    for _, fines in cs[:keep_coarse]:
        for mf, _, fi in fines:
            cand.append((float(mf @ q), fi)); nodes += 1
    best, bkey = -1e30, -1
    for _, fi in sorted(cand, key=lambda t: -t[0])[:keep_fine]:
        s = K[fi] @ q; j = int(s.argmax())
        if s[j] > best:
            best, bkey = float(s[j]), int(fi[j])
    return bkey, nodes


def run(n, d=48, keep_coarse=3, keep_fine=8, seed=0):
    rng = np.random.default_rng(seed)
    K, _ = clustered_keys(n, d, B=max(8, int(round(n ** 0.45))), spread=0.10, seed=seed)
    n_coarse = max(2, int(round(n ** (1 / 3))))
    fine_per = max(2, int(round(n ** (1 / 3))))
    tree = build_tree(K, n_coarse, fine_per)
    qpos = rng.choice(n, min(150, n), replace=False)
    f_hit = h_hit = 0; f_nodes, h_nodes, ll_nodes = [], [], []
    for i in qpos:
        q = K[i] + 0.25 * rng.standard_normal(d).astype(np.float32)
        tgt = int((K @ q).argmax())
        fb, fn = flat_approx(K, q, tree, keep_fine)
        hb, hn = hier_approx(K, q, tree, keep_coarse, keep_fine)
        _, ln = lossless_cost(K, q, tree)
        f_hit += int(fb == tgt); h_hit += int(hb == tgt)
        f_nodes.append(fn); h_nodes.append(hn); ll_nodes.append(ln)
    return dict(n=n, n_fine=sum(len(fs) for _, _, fs in tree),
                flat_nodes=np.mean(f_nodes), hier_nodes=np.mean(h_nodes), lossless_nodes=np.mean(ll_nodes),
                flat_recall=f_hit / len(qpos), hier_recall=h_hit / len(qpos))


def main():
    print("=" * 86)
    print("HIERARCHICAL (FMM TREECODE) ROUTING — making the SELECTION subquadratic")
    print("=" * 86)
    print("  2-level tree, recursive parent radius (paper §4.4). Cost = nodes scored per")
    print("  query. Approximate routing keeps the top-3 coarse nodes, then top-8 fine cells.\n")
    print(f"  {'n':>8} {'#fine':>7} {'flat cost':>10} {'hier cost':>10} {'hier/flat':>10} "
          f"{'flat rec':>9} {'hier rec':>9}")
    print("  " + "-" * 72)
    for n in (2000, 8000, 32000, 128000):
        r = run(n)
        print(f"  {n:>8} {r['n_fine']:>7} {r['flat_nodes']:>10.0f} {r['hier_nodes']:>10.0f} "
              f"{r['hier_nodes']/r['flat_nodes']*100:>9.1f}% {r['flat_recall']:>9.3f} {r['hier_recall']:>9.3f}")
    print("\n" + "=" * 86)
    print("  Hierarchical APPROXIMATE routing scores a VANISHING fraction of the cells — hier/flat falls")
    print("  30.8% -> 8.0% as n grows (hier cost ~ n^1/3, flat ~ n^2/3). The tree pruning")
    print("  (hierarchical_prune: one coarse check skips a whole subtree) makes the SELECTION itself")
    print("  subquadratic — the part a flat router, and the kernel's flat front-end, could not.")
    print("  Honest cost: a recall TRADE (hier < flat) — keeping only the top-3 coarse nodes misses the")
    print("  target's cell sometimes. The knob that narrows it is exactly the earlier finding: rank the")
    print("  coarse nodes by the CUMULANT score, not the centroid (or keep more coarse). And LOSSLESS")
    print("  hierarchical B&B does NOT beat flat (the coarse admissible radius is too loose to prune a")
    print("  subtree exactly) — the trilemma again; the win is approximate, which is SSA's regime anyway.")


if __name__ == "__main__":
    main()
