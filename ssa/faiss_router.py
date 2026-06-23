"""
P4 — FAISS-IVF block-router: replace the (n/b)² flat GEMM (and its argsort BlockMask build) with an IVF
search over the nb BLOCK-MEANS. Per query-block it scores ~nlist + nprobe·(nb/nlist) ≈ O(√nb) blocks and
emits the selected block indices directly (no (nb,nb) sel/argsort). This measures the IVF block-router's
block-selection agreement with the flat router + its cost scaling, then projects the 12M kernel
decomposition (and the residual gap to the n·κ floor) using the P0 fit.

Run: python3 -m ssa.faiss_router  ->  paper/figures/faiss_router.json
"""
from __future__ import annotations
import json
import numpy as np
import faiss


def clustered_blocks(nb, d, spread=0.10, seed=0):
    rng = np.random.default_rng(seed)
    nc = max(4, int(nb ** 0.5))
    centers = rng.standard_normal((nc, d)).astype(np.float32)
    mu = centers[rng.integers(0, nc, nb)] + spread * rng.standard_normal((nb, d)).astype(np.float32)
    mu /= np.linalg.norm(mu, axis=1, keepdims=True) + 1e-9
    qb = mu + 0.25 * rng.standard_normal((nb, d)).astype(np.float32)
    qb /= np.linalg.norm(qb, axis=1, keepdims=True) + 1e-9
    return mu, qb


def flat_topc(mu, qb, top_c):
    return np.argpartition(qb @ mu.T, -top_c, axis=1)[:, -top_c:]


def ivf_topc(mu, qb, top_c, nprobe):
    nb, d = mu.shape
    nlist = max(4, int(nb ** 0.5))
    ix = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    ix.train(mu); ix.add(mu); ix.nprobe = nprobe
    _, I = ix.search(qb, top_c)
    blocks_scored = nprobe * (nb / nlist)            # blocks the IVF actually scores per query-block
    return I, blocks_scored


def agree(a, b):
    out = 0.0
    for i in range(len(a)):
        sa, sb = set(a[i].tolist()), set(b[i][b[i] >= 0].tolist())
        out += len(sa & sb) / max(1, len(sa | sb))
    return out / len(a)


def main():
    np.random.seed(0)
    d, top_c = 64, 8
    print("=" * 84)
    print("P4 — FAISS-IVF block-router: agreement with flat top-c + cost scaling (blocks scored/query-block)")
    print("=" * 84)
    print(f"  {'nb':>8} {'flat scored':>12} {'ivf scored':>11} {'cheaper':>8} {'agree(J)':>9}")
    rows = []
    for nb in (1024, 4096, 16384, 65536):
        mu, qb = clustered_blocks(nb, d)
        ft = flat_topc(mu, qb, top_c)
        it, scored = ivf_topc(mu, qb, top_c, nprobe=4)
        a = agree(ft, it)
        rows.append(dict(nb=nb, flat_scored=nb, ivf_scored=float(scored), agree=a))
        print(f"  {nb:>8} {nb:>12} {scored:>11.0f} {nb/scored:>7.0f}x {a:>9.3f}")
    json.dump(rows, open("paper/figures/faiss_router.json", "w"), indent=2)

    # project the 12M kernel with the IVF router, using P0's measured component fits.
    cp = json.load(open("paper/figures/cost_profile.json"))
    fit = cp["fit"]                                    # power-law (p, a) per component from P0
    n12, block = 12e6, 128
    nb12 = n12 / block
    pa = lambda key, x: fit[key][1] * x ** fit[key][0]
    floor = pa("attention", n12)                       # the n·κ floor (linear)
    flat_router = pa("router", n12)
    flat_mask = pa("maskbuild", n12)
    # IVF router scales as nb^1.5 vs flat nb^2 -> ratio (nb^1.5/nb^2)=nb^-0.5; mask becomes linear (direct kv_idx)
    ivf_router = flat_router * (nb12 ** -0.5)
    ivf_mask = flat_mask * (nb12 ** -1.0)              # (nb,nb) argsort -> O(nb·top_c) ~ /nb
    flat_total = floor + flat_router + flat_mask
    ivf_total = floor + ivf_router + ivf_mask
    print(f"\n  Projected @12M (ms):  floor={floor:.0f}  | flat: router={flat_router:.0f} mask={flat_mask:.0f} "
          f"total={flat_total:.0f}")
    print(f"                                          | IVF:  router={ivf_router:.0f} mask={ivf_mask:.0f} "
          f"total={ivf_total:.0f}")
    print(f"  Gap to floor:  flat {flat_total/floor:.0f}x  ->  IVF {ivf_total/floor:.2f}x   "
          f"(kernel {flat_total/ivf_total:.0f}x faster, ~at the floor)")
    json.dump({"rows": rows, "proj_12M": {"floor": floor, "flat_total": flat_total, "ivf_total": ivf_total,
               "flat_gap": flat_total / floor, "ivf_gap": ivf_total / floor}},
              open("paper/figures/faiss_router.json", "w"), indent=2)
    print("  wrote paper/figures/faiss_router.json")


if __name__ == "__main__":
    main()
