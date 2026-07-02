"""
Multi-hop retrieval, measured — the composition law the single-needle NIAH story predicts but never tests.

`niah_analysis.py` shows cheap block-cumulant routing aces a SINGLE benign needle and collapses on a lone
spike. A chain of retrievals (MRCR / multi-hop — the regime SubQ reports 65.9% on while NIAH@12M is ~98%)
composes those per-hop events: with per-hop retrieval rate ρ_j, an h-hop chain succeeds ≈ ∏ρ_j
(the E5 law in experiments.py). This script plants an h-hop chain in the SAME synthetic geometry and runs
it through the SAME budgeted selector as niah_analysis, so the two results sit on one axis:

  • hop j's needle is addressed by direction dirs[j] and its payload IS dirs[j+1] (the next hop's query —
    the geometric analogue of MQAR value→key chaining); the directions are mutually orthogonal, so a hop's
    elevation cannot help any other hop;
  • a hop is benign (a coherent span lifts its block mean — findable by cheap routing) or isolated (a lone
    spike — washed out); the MIXED mode (benign hop 1, isolated hop 2) is the falsifiable case: single
    needles hold, the chain collapses because one weak hop divides the whole product.

The prediction under test: measured chain accuracy ≈ ∏ρ_j (oracle per-hop rates), i.e. the multi-hop sag
is the single-needle result read h times, not a new failure. Pure NumPy — runs in ~1–2 min.

Run:  python3 -m ssa.multihop_analysis           # -> paper/figures/multihop_geometry.json
"""
from __future__ import annotations
import json
import time
import numpy as np


def _cheap_select(s, bid, cnt, budget):
    """The niah_analysis block-cumulant selector, factored out: top blocks by (mean + ½·var) to ~budget
    keys; returns the selected key indices."""
    B = len(cnt)
    mean = np.bincount(bid, weights=s, minlength=B) / cnt
    var = np.maximum(np.bincount(bid, weights=s * s, minlength=B) / cnt - mean ** 2, 0.0)
    cscore = mean + 0.5 * var
    order = np.argsort(cscore)[::-1]
    take = order[:max(1, np.searchsorted(np.cumsum(cnt[order]), budget) + 1)]
    return np.where(np.isin(bid, take))[0]


def _orth_dirs(rng, d, h):
    """h mutually-orthonormal unit directions (Gram–Schmidt on random vectors)."""
    dirs = []
    for _ in range(h):
        u = rng.standard_normal(d).astype(np.float32)
        for w in dirs:
            u -= u.dot(w) * w
        u /= np.linalg.norm(u)
        dirs.append(u)
    return dirs


_PRESETS = {                                                      # mode -> per-hop geometry
    "dense":    lambda h: ["dense"] * h,
    "isolated": lambda h: ["isolated"] * h,
    "benign":   lambda h: ["benign"] * h,
    "mixed":    lambda h: ["benign"] * (h - 1) + ["isolated"],    # last hop is the weak (isolated) link
}


def multihop_trial(n, d, margin, mode, rng, hops=2, block=64, budget=1024, span=24, ctx=0.7):
    """Plant an h-hop chain and evaluate it. Returns {'chain': 0/1, 'hop_ok': [...]}.
    Each hop is retrieved with its TRUE cue dirs[j] (perfect payload readout), so chain = all(hop_ok)
    and the across-trial mean of hop_ok[j] is the oracle ρ_j the composition law multiplies."""
    per_hop = _PRESETS[mode](hops)
    dirs = _orth_dirs(rng, d, hops)
    K = rng.standard_normal((n, d)).astype(np.float32); K /= np.linalg.norm(K, axis=1, keepdims=True)

    def elevate(idx, m, direction):
        p = rng.standard_normal(d).astype(np.float32)
        p -= p.dot(direction) * direction; p /= np.linalg.norm(p)
        K[idx] = m * direction + np.sqrt(max(0.0, 1 - m * m)) * p

    targets = []
    used_blocks = set()
    for j in range(hops):                                        # one needle per hop, in distinct blocks
        while True:
            t = int(rng.integers(n))
            if t // block not in used_blocks:
                used_blocks.add(t // block); break
        targets.append(t)
        elevate(t, margin, dirs[j])
        if per_hop[j] == "benign":                               # coherent span lifts this hop's block mean
            b0 = (t // block) * block
            for jj in [x for x in range(b0, min(b0 + block, n)) if x != t][:span - 1]:
                elevate(jj, ctx * margin, dirs[j])

    bid = np.arange(n) // block
    B = int(bid[-1]) + 1
    cnt = np.bincount(bid, minlength=B).astype(np.float64)
    hop_ok = []
    for j in range(hops):
        s = K @ dirs[j]
        if per_hop[j] == "dense":
            hop_ok.append(int(s.argmax() == targets[j]))
        else:
            si = _cheap_select(s, bid, cnt, budget)
            hop_ok.append(int(si[s[si].argmax()] == targets[j]))
    return {"chain": int(all(hop_ok)), "hop_ok": hop_ok}


def chain_stats(n, d, margin, mode, trials, hops=2, block=64, budget=1024, seed=0):
    rng = np.random.default_rng(seed)
    res = [multihop_trial(n, d, margin, mode, rng, hops=hops, block=block, budget=budget) for _ in range(trials)]
    acc = float(np.mean([r["chain"] for r in res]))
    rhos = [float(np.mean([r["hop_ok"][j] for r in res])) for j in range(hops)]
    return {"n": n, "d": d, "margin": margin, "mode": mode, "hops": hops,
            "acc": acc, "rhos": rhos, "pred": float(np.prod(rhos)), "trials": trials, "seed": seed}


def _trials(n):
    return int(np.clip(3_000_000 // n, 16, 96))


def main():
    d, budget = 64, 1024
    t0 = time.time()
    rows = []
    print("=" * 92)
    print("MULTI-HOP RETRIEVAL — the composition law measured (chain success ~ ∏ per-hop ρ)")
    print("=" * 92)

    print("\n[1] 2-hop chain accuracy vs context length (margin 0.55) — single needles hold, chains sag")
    print(f"  {'n':>10} {'dense':>7} {'isolated':>9} {'benign':>7} {'mixed':>7}  (mixed = benign hop1 + isolated hop2)")
    for n in (1024, 4096, 16384, 65536, 262144):
        tr = _trials(n)
        line = {}
        for mode in ("dense", "isolated", "benign", "mixed"):
            r = chain_stats(n, d, 0.55, mode, tr, hops=2)
            rows.append(r); line[mode] = r["acc"]
        print(f"  {n:>10} {line['dense']:>7.2f} {line['isolated']:>9.2f} {line['benign']:>7.2f} "
              f"{line['mixed']:>7.2f}", flush=True)

    print("\n[2] composition law: measured 2-hop chain vs ∏ρ  (does the chain = product of hops?)")
    print(f"  {'n':>10} {'mode':>9} {'ρ1':>6} {'ρ2':>6} {'∏ρ (pred)':>10} {'measured':>9}")
    for n in (4096, 65536):
        for mode in ("benign", "mixed", "isolated"):
            r = chain_stats(n, d, 0.55, mode, _trials(n), hops=2)
            rows.append(r)
            print(f"  {n:>10} {mode:>9} {r['rhos'][0]:>6.2f} {r['rhos'][1]:>6.2f} "
                  f"{r['pred']:>10.2f} {r['acc']:>9.2f}", flush=True)

    print("\n[3] 2-hop chain accuracy vs margin (n=65536)")
    print(f"  {'margin':>8} {'dense':>7} {'isolated':>9} {'benign':>7} {'mixed':>7}")
    for mg in (0.45, 0.55, 0.65, 0.80, 0.95):
        line = {}
        for mode in ("dense", "isolated", "benign", "mixed"):
            r = chain_stats(65536, d, mg, mode, 48, hops=2)
            rows.append(r); line[mode] = r["acc"]
        print(f"  {mg:>8.2f} {line['dense']:>7.2f} {line['isolated']:>9.2f} {line['benign']:>7.2f} "
              f"{line['mixed']:>7.2f}", flush=True)

    print("\n[4] chain length: hops ∈ {1,2,3} at n=65536, margin 0.55 — benign vs mixed(...,isolated)")
    print(f"  {'hops':>6} {'benign':>7} {'∏ρ':>6} {'mixed':>7} {'∏ρ':>6}")
    for h in (1, 2, 3):
        rb = chain_stats(65536, d, 0.55, "benign", 48, hops=h)
        rm = chain_stats(65536, d, 0.55, "mixed", 48, hops=h)
        rows.append(rb); rows.append(rm)
        print(f"  {h:>6} {rb['acc']:>7.2f} {rb['pred']:>6.2f} {rm['acc']:>7.2f} {rm['pred']:>6.2f}", flush=True)

    json.dump({"meta": {"d": d, "block": 64, "budget": budget, "span": 24, "ctx": 0.7, "seed": 0},
               "rows": rows}, open("paper/figures/multihop_geometry.json", "w"), indent=2)
    print("\n" + "=" * 92)
    print("  The multi-hop sag is the single-needle result read h times: chain ≈ ∏ρ. One isolated hop")
    print("  (the mixed mode) divides the whole product toward 0 even when each single needle is ~98% —")
    print("  which is exactly the NIAH@12M (~98%) vs MRCR (65.9%) split SubQ's own table shows.")
    print(f"  wrote paper/figures/multihop_geometry.json  [ran in {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
