"""
What does "98–100% retrieval at 1M–12M tokens" actually show? — characterizing SubQ's headline claim.

Their one 12M claim is single-target needle-in-a-haystack (NIAH) RETRIEVAL ACCURACY. The naive reading is
"selection caps the effective distractor count, so accuracy is flat in n." That is true — but it hides the
load-bearing condition, which this script makes explicit because it IS the thesis of this whole project.

CHEAP selection routes by cluster/block STATISTICS (a mean, a cumulant μ + ½·var) in O(B·d), not by scoring
every key in O(n·d). A single needle is an ISOLATED spike: aggregated into its block's mean, its signal is
divided by the block size and WASHED OUT — and across the growing number of random blocks, chance
fluctuations out-rank it. So cheap moment routing CANNOT reliably find a lone needle in isotropic geometry
(measured below: it collapses with length, and even at margin 0.95 it barely fires). This is the
impossibility wall in miniature — exact cheap selection of an arbitrary spike is not available.

What rescues it is BENIGN geometry: a real answer is not an isolated spike — it sits in a COHERENT span
(its neighbors are also aligned with the query), and a coherent span lifts the whole block's mean, so cheap
routing finds it, length-robustly (measured below: flat ~1.0 while dense sags). THIS is why 98–100%@12M is
achievable by cheap selection — it is the EASY regime: single, HIGH-margin, ACCURACY (not losslessness),
and crucially BENIGN. It does NOT verify the hard claim (lossless subquadratic selection of an adversarial /
low-margin / multi-needle target), which is the side of the wall they report no 12M numbers for (MRCR@128K).

Positional blocks, no k-means — runs in seconds.

Run:  python3 -m ssa.niah_analysis
"""
from __future__ import annotations
import time
import numpy as np


def niah_trial(n, d, margin, mode, rng, block=64, budget=1024, span=24, ctx=0.7):
    """One NIAH trial. mode: dense | isolated | benign.
    isolated = a lone needle; benign = the needle plus a coherent span of aligned neighbors in its block.
    dense scores every key; isolated/benign use cheap block-cumulant selection (attend ~budget keys)."""
    q = rng.standard_normal(d).astype(np.float32); q /= np.linalg.norm(q)
    K = rng.standard_normal((n, d)).astype(np.float32); K /= np.linalg.norm(K, axis=1, keepdims=True)

    def elevate(idx, m):
        p = rng.standard_normal(d).astype(np.float32); p -= p.dot(q) * q; p /= np.linalg.norm(p)
        K[idx] = m * q + np.sqrt(max(0.0, 1 - m * m)) * p

    t = int(rng.integers(n)); elevate(t, margin)                                 # the target needle
    if mode == "benign":                                                         # coherent neighbors in its block
        b0 = (t // block) * block
        for j in [j for j in range(b0, min(b0 + block, n)) if j != t][:span - 1]:
            elevate(j, ctx * margin)
    s = K @ q
    if mode == "dense":
        return int(s.argmax() == t)
    bid = np.arange(n) // block                                                  # fixed positional blocks
    B = int(bid[-1]) + 1
    cnt = np.bincount(bid, minlength=B).astype(np.float64)
    mean = np.bincount(bid, weights=s, minlength=B) / cnt
    var = np.maximum(np.bincount(bid, weights=s * s, minlength=B) / cnt - mean ** 2, 0.0)
    cscore = mean + 0.5 * var                                                    # cheap cumulant routing
    order = np.argsort(cscore)[::-1]
    take = order[:max(1, np.searchsorted(np.cumsum(cnt[order]), budget) + 1)]    # top blocks ~ budget keys
    si = np.where(np.isin(bid, take))[0]
    return int(si[s[si].argmax()] == t)


def accuracy(n, d, margin, mode, trials, seed=0):
    rng = np.random.default_rng(seed)
    return float(np.mean([niah_trial(n, d, margin, mode, rng) for _ in range(trials)]))


def main():
    d, budget = 64, 1024
    t0 = time.time()
    print("=" * 86)
    print("WHAT '98–100% RETRIEVAL AT 1M–12M' ACTUALLY SHOWS (single-needle NIAH)")
    print("=" * 86)

    print("\n[1] NIAH accuracy vs context length — cheap selection needs the needle to be FINDABLE")
    print(f"  (margin 0.55, d={d}, attend ~{budget} keys; isolated = lone spike, benign = coherent span)")
    print(f"  {'n (context)':>12} {'dense':>7} {'SSA isolated':>13} {'SSA benign':>11}")
    for n in (1024, 4096, 16384, 65536, 262144):
        tr = int(np.clip(2_000_000 // n, 12, 60))
        print(f"  {n:>12} {accuracy(n, d, 0.55, 'dense', tr):>7.2f} "
              f"{accuracy(n, d, 0.55, 'isolated', tr):>13.2f} {accuracy(n, d, 0.55, 'benign', tr):>11.2f}",
              flush=True)
    print("  -> dense degrades slowly (more distractors clear the margin). Cheap routing of an ISOLATED")
    print("     needle washes out in the block mean and collapses with length. A BENIGN coherent span lifts")
    print("     its block and stays length-robust — and beats dense at long n (selection caps the distractors).")

    print("\n[2] NIAH accuracy vs needle margin (n=65536) — the claim's HIGH-margin + benign regime")
    print(f"  {'margin':>8} {'dense':>8} {'SSA isolated':>13} {'SSA benign':>11}")
    for mg in (0.45, 0.55, 0.65, 0.80, 0.95):
        print(f"  {mg:>8.2f} {accuracy(65536, d, mg, 'dense', 40):>8.2f} "
              f"{accuracy(65536, d, mg, 'isolated', 40):>13.2f} {accuracy(65536, d, mg, 'benign', 40):>11.2f}",
              flush=True)
    print("  -> a lone needle barely fires at ANY margin (the wash-out); a benign span tracks dense once the")
    print("     margin clears the budget floor. SubQ's 12M NIAH lives in the high-margin + benign regime.")

    print("\n[3] the recoveryWeight extrapolation to 12M (why fixed-budget selection is flat in n)")
    ssa_floor = np.sqrt(2 * np.log(budget) / d)                                  # capped at the fixed budget
    print(f"  detectability threshold βΔ > log(effective distractors). Fixed budget ~{budget} keys:")
    for n in (1e6, 1.2e7):
        dense_floor = np.sqrt(2 * np.log(n) / d)                                  # max distractor for dense
        print(f"    n={n:.0e}: dense needs margin > {dense_floor:.2f}; SSA needs margin > {ssa_floor:.2f} "
              f"(flat in n — the budget, not n, sets the floor — IF the needle is findable)")
    print("\n" + "=" * 86)
    print("  The 98–100%-at-12M claim is REAL and theory-consistent — but it is the EASY regime: a single,")
    print("  HIGH-margin, BENIGN target (its coherent span lifts the block so CHEAP selection finds it),")
    print("  measuring ACCURACY not losslessness. A lone adversarial spike defeats cheap moment routing —")
    print("  the impossibility wall in miniature — which is the low-margin / multi-needle side they report")
    print("  no 12M numbers for (their hard benchmark, MRCR, is at 128K).")
    print(f"  [ran in {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
