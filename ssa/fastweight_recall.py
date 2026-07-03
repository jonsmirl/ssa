"""
P8-C — the core selection-vs-compression demonstrations.

P3 — WRITE-TIME vs READ-TIME relevance (the load-bearing one). At stream length n ≫ capacity:
  (a) a needle that is SALIENT AT WRITE TIME (an off-distribution key) is kept by a surprise-gated
      fixed-state memory — write-time compression works when relevance is decidable at write time;
  (b) a needle salient ONLY AT READ TIME (one of many identically-distributed facts; the query picks it)
      is LOST by every fixed-state memory once #facts > capacity, while a selection baseline (attention
      over the kept stream) recovers it. This is why compression cannot replace read-time selection —
      the write rule cannot keep what only the future query will make relevant.

P4 — same-key CONFLICT needs a tag (`tag_resolves_conflict`): additive averages, delta keeps the latest,
  an episodic tag recovers BOTH.

P6 — the 2-hop COMPOSITION law is architecture-independent (`chain_le_weakest`): with independently-measured
  per-hop rates, the composition prediction ∏ρ ≤ min hop is the proved bound; the measured joint chain tracks
  it up to inter-hop correlation — the same sag the selection rig measured.

Run:  python3 -m ssa.fastweight_recall           # -> paper/figures/fastweight_recall.json
"""
from __future__ import annotations
import json
import os
import torch
from ssa.fastweight import FastWeightMemory, surprise, _unit


# -- P3 -------------------------------------------------------------------------------------------

def _stream(n, d, regime, g):
    """Return (keys, values, target_idx). regime='write_salient' plants a needle whose KEY is
    off-distribution (high write-time surprise) in a clustered background; regime='read_salient' makes
    every fact i.i.d. (no write-time signal) — only the query at read time reveals the target."""
    if regime == "write_salient":
        centre = _unit(torch.randn(d, generator=g))
        K = _unit(centre + 0.15 * torch.randn(n, d, generator=g))     # a tight background cluster
        t = int(torch.randint(n, (1,), generator=g))
        K[t] = _unit(torch.randn(d, generator=g))                     # the needle: a distinct direction
    else:                                                             # read_salient: all i.i.d., no cluster
        K = _unit(torch.randn(n, d, generator=g))
        t = int(torch.randint(n, (1,), generator=g))
    V = _unit(torch.randn(n, d, generator=g))
    return K, V, t


def _gated_fixed_recall(K, V, t, d, gate_q=0.9, rule="delta"):
    """A surprise-gated fixed-state memory: write a fact only when its key's surprise (residual from the
    key mean) is in the top (1−gate_q). Query the target's key; recover iff argmax-decode hits it."""
    mean = _unit(K.mean(0))
    surprises = (_unit(K) - mean).norm(dim=1)
    thresh = torch.quantile(surprises, gate_q)
    mem = FastWeightMemory(d, rule=rule, beta=1.0, keep_kv=False)
    for i in range(K.shape[0]):
        if surprises[i] >= thresh:
            mem.write(K[i], V[i])
    o = _unit(mem.read_linear(K[t]))
    return int((o @ _unit(V).T).argmax()) == t


def _selection_recall(K, V, t, beta=12.0):
    """The selection baseline: attention over the FULL kept stream (unbounded memory, read-time choice)."""
    o = torch.softmax(beta * (_unit(K) @ _unit(K[t])), dim=0) @ _unit(V)
    return int((_unit(o) @ _unit(V).T).argmax()) == t


def run_p3(d=64, n=512, trials=40):
    rows = []
    print("\n[P3] WRITE-TIME vs READ-TIME relevance (n=512 ≫ d=64) — the selection-vs-compression split")
    print(f"  {'regime':>14} {'gated-fixed memory':>19} {'selection (attention)':>22}")
    for regime in ("write_salient", "read_salient"):
        fixed = sel = 0
        for s in range(trials):
            g = torch.Generator().manual_seed(s)
            K, V, t = _stream(n, d, regime, g)
            fixed += _gated_fixed_recall(K, V, t, d)
            sel += _selection_recall(K, V, t)
        fr, sr = fixed / trials, sel / trials
        rows.append({"test": "P3", "regime": regime, "n": n, "d": d, "gated_fixed": fr, "selection": sr})
        print(f"  {regime:>14} {fr:>19.2f} {sr:>22.2f}", flush=True)
    print("  -> write-time-salient: the gated fixed memory keeps the needle (relevance known at write).")
    print("     read-time-salient: only selection recovers it — the write rule cannot keep what the")
    print("     query has not yet asked for. Compression ≠ selection.")
    return rows


# -- P4 -------------------------------------------------------------------------------------------

def run_p4(d=48, td=8, trials=40):
    rows = []
    print("\n[P4] same-key CONFLICT: additive averages, delta keeps the latest, a TAG recovers both")
    print(f"  {'memory':>16} {'recall v1':>10} {'recall v2':>10}")
    for name in ("additive", "delta", "tagged-delta"):
        r1 = r2 = 0
        for s in range(trials):
            g = torch.Generator().manual_seed(s)
            k = _unit(torch.randn(d, generator=g))
            v1 = _unit(torch.randn(d, generator=g)); v2 = _unit(torch.randn(d, generator=g))
            if name == "tagged-delta":
                # a salient episodic tag (scaled so k⊕t1, k⊕t2 are well-separated despite sharing k)
                t1 = 2.0 * _unit(torch.randn(td, generator=g)); t2 = 2.0 * _unit(torch.randn(td, generator=g))
                mem = FastWeightMemory(d + td, d_v=d, rule="delta", beta=1.0, keep_kv=False)
                mem.write(torch.cat([k, t1]), v1); mem.write(torch.cat([k, t2]), v2)
                o1 = _unit(mem.read_linear(torch.cat([k, t1]))); o2 = _unit(mem.read_linear(torch.cat([k, t2])))
            else:
                mem = FastWeightMemory(d, rule=name, beta=1.0, keep_kv=False)
                mem.write(k, v1); mem.write(k, v2)
                o1 = o2 = _unit(mem.read_linear(k))
            r1 += int((o1 @ v1) > (o1 @ v2)); r2 += int((o2 @ v2) > (o2 @ v1))
        rows.append({"test": "P4", "memory": name, "recall_v1": r1 / trials, "recall_v2": r2 / trials})
        print(f"  {name:>16} {r1 / trials:>10.2f} {r2 / trials:>10.2f}", flush=True)
    print("  -> one key cannot hold two values (same_input_conflict_unservable); the episodic tag")
    print("     (k⊕bucket) makes them distinct keys and recovers BOTH (tag_resolves_conflict).")
    return rows


# -- P6 -------------------------------------------------------------------------------------------

def run_p6(d=64, trials=300, noise=0.55):
    """A 2-hop chain through a fast-weight memory. ρ1 and ρ2 are measured INDEPENDENTLY (each a fresh
    noisy cue on its own hop), so ∏ρ = ρ1·ρ2 is the composition-law 'chain reliability' the theorem
    bounds: chain_le_weakest proves ∏ρ ≤ min hop. `chain` is the measured joint success (hop-1 output
    fed to hop-2) — reported alongside, tracking ∏ρ up to inter-hop correlation."""
    rows = []
    print("\n[P6] 2-hop COMPOSITION through a fast-weight memory — the composition-law bound reproduced")
    hop1 = hop2 = chain = 0
    for s in range(trials):
        g = torch.Generator().manual_seed(s)
        q = _unit(torch.randn(d, generator=g)); mid = _unit(torch.randn(d, generator=g))
        ans = _unit(torch.randn(d, generator=g))
        mem = FastWeightMemory(d, rule="delta", beta=1.0, keep_kv=True)
        for _ in range(20):                                          # distractors so each hop is a real retrieval
            mem.write(_unit(torch.randn(d, generator=g)), _unit(torch.randn(d, generator=g)))
        mem.write(q, mid); mem.write(mid, ans)
        # INDEPENDENT per-hop rates (each hop gets its own fresh noisy cue)
        h1 = int((_unit(mem.read_softmax(_unit(q + noise * torch.randn(d, generator=g)), beta=12.0)) @ mid) > 0.5)
        h2 = int((_unit(mem.read_softmax(_unit(mid + noise * torch.randn(d, generator=g)), beta=12.0)) @ ans) > 0.5)
        # the measured joint chain: feed hop-1's noisy output into hop-2
        r1 = mem.read_softmax(_unit(q + noise * torch.randn(d, generator=g)), beta=12.0)
        c = int((_unit(r1) @ mid > 0.5)
                and (_unit(mem.read_softmax(_unit(r1 + noise * torch.randn(d, generator=g)), beta=12.0)) @ ans) > 0.5)
        hop1 += h1; hop2 += h2; chain += c
    r1r, r2r, cr = hop1 / trials, hop2 / trials, chain / trials
    prod, mh = r1r * r2r, min(r1r, r2r)
    rows.append({"test": "P6", "rho1": round(r1r, 3), "rho2": round(r2r, 3), "prod": round(prod, 3),
                 "chain": round(cr, 3), "min_hop": round(mh, 3), "prod_le_min_hop": bool(prod <= mh + 1e-9)})
    print(f"  ρ1={r1r:.2f}  ρ2={r2r:.2f}  ∏ρ={prod:.2f}  min hop={mh:.2f}  measured chain={cr:.2f}")
    print(f"  ∏ρ ≤ min hop (chain_le_weakest, the theorem): {prod <= mh + 1e-9}   "
          f"| measured chain tracks ∏ρ up to hop correlation")
    return rows


def main():
    out = "paper/figures/fastweight_recall.json"
    print("=" * 92)
    print("P8-C — selection vs compression: write-vs-read-time relevance, conflicts/tags, composition")
    print("=" * 92)
    rows = run_p3() + run_p4() + run_p6()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"meta": {"seed": "0..", "decode": "argmax over stored values",
                        "note": "d≤64 reference memories; the selection baseline is attention over the "
                                "full kept stream (unbounded memory) — the read-time-choice control"},
               "rows": rows}, open(out, "w"), indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
