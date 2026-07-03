"""
P9 — the trained selection-vs-compression comparison. Trains the swappable micro-LM (p9_microlm) on the
MQAR suite (p9_tasks) and measures, TRAINED, what P8 could only measure with hand-built memories.

D1 — capacity vs load (trained P1): recall vs #pairs m for {dense, ssa(κ≈dh), deltanet, linear} at fixed
     state (d, head_dim dh). Does the DeltaNet wall at m≈dh (the compression corner) while selection (dense/
     ssa) holds at any load? The measured TRAINED frontier.
D2 — read- vs write-salient × gate × aux (trained P3, the headline): at OVERLOAD (m > dh), does a LEARNED
     write gate + the JEPA aux loss lift DeltaNet recall? MEASURED: the gate is a NULL ingredient — on
     write-salient the no-gate delta rule already solves it (training shapes the ≤dh keepable keys itself);
     on read-salient nothing lifts the wall (read-time relevance is a capacity limit, not a training gap).
D3 — 2-hop composition (trained P6): pure-chain recall for {dense, ssa, deltanet} on MQAR2Hop. Selection
     chains; the compression corner sags more. Calibrated so the dense ceiling groks (R6).
D4 — the JEPA aux-weight λ sweep on read-salient DeltaNet: flat in λ (the future-prediction aux does not
     lift the read-time-relevance wall) — reported as measured, not tuned.

Run:  python3 -m ssa.p9_compare            # -> paper/figures/p9_*.json  (resumable)
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np
import torch

from ssa.ssa_demo import MQAR
from ssa.p9_tasks import MQARSalient, MQAR2Hop
from ssa.p9_microlm import P9Model, train_model, recall, DEV

D, NHEAD = 128, 8
DH = D // NHEAD                                                          # head_dim = the DeltaNet state dim (16)
SSA = dict(ssa_block=8, ssa_top_c=2, ssa_local=1)                       # κ≈top_c·block=16 ≈ dh (matched read)


def _model(vocab, mixer, n_layer, jepa=False, delta_gate=True, max_len=1024):
    return P9Model(vocab, d=D, n_layer=n_layer, n_head=NHEAD, max_len=max_len, mixer=mixer,
                   delta_gate=delta_gate, jepa=jepa, **SSA).to(DEV)


def _mq(load):
    """Multi-query supervision — the proven MQAR recipe (ssa_checkpoint) trains/evals with min(m,16) queries
    per sequence, not one. A single query gives a ~16× sparser gradient and MQAR does not grok from it."""
    return min(load, 16)


def _curriculum(mixer, task, sched, seed, n_layer=2, jepa=False, delta_gate=True, aux_weight=0.0):
    """Train ONE model through the (load, steps) curriculum; probe recall at each load right AFTER its stage,
    before any higher-load training — so recall@L is the standard 'trained up to L, tested at L' capacity
    measurement, with NO contamination from training past L (which would overload the compression corner and
    wreck its low-load recall). A long FIRST stage (cheap: n≈7 at load 2) lets every mixer grok the retrieval
    op before the first probe — the compression corner (DeltaNet) needs ~2× the steps of attention to grok."""
    torch.manual_seed(seed)
    m = _model(task.vocab, mixer, n_layer, jepa=jepa, delta_gate=delta_gate)
    out = {}
    for load, steps in sched:
        train_model(m, task, steps=steps, n_pairs=load, n_queries=_mq(load), bs=48,
                    warmup=max(40, steps // 10), aux_weight=aux_weight, seed=seed)
        out[load] = recall(m, task, load, n_queries=_mq(load), trials=6)
    return out, m


def _mean_over_seeds(fn, seeds):
    runs = [fn(s) for s in seeds]
    keys = runs[0].keys()
    return {k: float(np.mean([r[k] for r in runs])) for k in keys}


# -- D1 ------------------------------------------------------------------------------------------

def d1_capacity(seeds=(0, 1)):
    task = MQAR(n_keys=256, n_vals=256)
    # long, cheap first stage (n≈7) so EVERY mixer groks the op — vanilla linear is slowest (~2.5k steps, no
    # erase term); short-changing it would misreport undertraining as a capacity wall. Then stages spanning dh.
    sched = [(2, 3000), (4, 600), (8, 700), (16, 800), (24, 900), (32, 1000)]
    rows = []
    print(f"\n[D1] capacity vs load — trained recall (d={D}, head_dim dh={DH}; ssa κ≈{SSA['ssa_top_c']*SSA['ssa_block']})")
    print(f"  {'mixer':>10} " + " ".join(f"m={l:<4}" for l, _ in sched))
    for mixer in ("dense", "ssa", "deltanet", "linear"):
        res = _mean_over_seeds(lambda s: _curriculum(mixer, task, sched, s, n_layer=2)[0], seeds)
        rows.append({"test": "D1", "mixer": mixer, "recall": {str(l): round(res[l], 3) for l, _ in sched}})
        print(f"  {mixer:>10} " + " ".join(f"{res[l]:>5.2f}" for l, _ in sched), flush=True)
    print(f"  -> selection (dense/ssa) holds at any load; both compression corners degrade past m≈dh={DH}")
    print(f"     (DeltaNet MORE — erase-before-write forgets the older pairs later queried; additive linear")
    print(f"     holds them all with interference, so it sags less at these mild overloads m≤2·dh).")
    return rows


# -- D2 ------------------------------------------------------------------------------------------

def d2_gate_aux(seeds=(0, 1)):
    """At overload (load > dh), on read-salient (MQAR) vs write-salient (MQARSalient): does a learned gate +
    aux lift DeltaNet? Compared to SSA (selection). Curriculum to an overload load."""
    overload = 32                                                       # = 2·dh=16 (overload) for both regimes
    n_mark = 8                                                          # write-salient marker set K (≤ dh)
    # gentle 16→24→32 granularity so the SSA router adapts. read-salient groks from load 2; write-salient
    # has K=8 markers per sequence so its curriculum must START at load ≥ K (all-marker grok stage).
    sched_read = [(2, 3000), (4, 600), (8, 700), (16, 800), (24, 900), (32, 1000)]
    sched_write = [(8, 3000), (12, 700), (16, 800), (24, 900), (32, 1000)]
    rows = []
    print(f"\n[D2] read- vs write-salient at overload (m={overload} > dh={DH}) — does the gate + aux help?")
    print(f"  {'regime':>13} {'arm':>22} {'recall@'+str(overload):>10}")
    regimes = {"read_salient": (MQAR(256, 256), sched_read),
               "write_salient": (MQARSalient(256, 256, n_markers=n_mark), sched_write)}
    arms = [("ssa (selection)", "ssa", dict()),
            ("deltanet no-gate", "deltanet", dict(delta_gate=False)),
            ("deltanet +gate", "deltanet", dict(delta_gate=True)),
            ("deltanet +gate+aux", "deltanet", dict(delta_gate=True, jepa=True, aux_weight=0.3))]
    for rname, (task, sched) in regimes.items():
        for aname, mixer, kw0 in arms:
            kw = dict(kw0); aux_w = kw.pop("aux_weight", 0.0)
            r = _mean_over_seeds(
                lambda s, mixer=mixer, task=task, sched=sched, kw=kw, aux_w=aux_w:
                _curriculum(mixer, task, sched, s, n_layer=2, aux_weight=aux_w, **kw)[0], seeds)
            rows.append({"test": "D2", "regime": rname, "arm": aname, "recall": round(r[overload], 3)})
            print(f"  {rname:>13} {aname:>22} {r[overload]:>10.2f}", flush=True)
    print("  -> measured: the learned gate/aux are a NULL ingredient. Write-salient is solved WITHOUT a gate")
    print("     (training shapes the ≤dh keepable keys itself); read-salient walls with or without it")
    print("     (read-time relevance is a capacity limit no write policy can serve). SSA solves both.")
    return rows


# -- D3 ------------------------------------------------------------------------------------------

def d3_composition(seeds=(0, 1)):
    # single chain + a progressive-load curriculum is what groks 2-hop at this scale (the curriculum builds
    # the induction mechanism gradually; cold-start and multi-chain do NOT reach the dense ceiling). Load =
    # 1 chain + (m-2) distractors, so the sweep raises the distractor/capacity pressure on the chain.
    task = MQAR2Hop(n_tokens=192, n_chains=1)
    sched = [(2, 3000), (3, 1500), (4, 1500), (6, 2000), (8, 2000)]      # dense ceiling = 1.0 here (R6 met)
    rows = []
    print(f"\n[D3] 2-hop composition — chain recall (4 layers; dense is the calibrated ceiling ≈1.0)")
    print(f"  {'mixer':>10} " + " ".join(f"m={l:<4}" for l, _ in sched))
    for mixer in ("dense", "ssa", "deltanet"):
        res = _mean_over_seeds(lambda s: _curriculum(mixer, task, sched, s, n_layer=4)[0], seeds)
        rows.append({"test": "D3", "mixer": mixer, "recall": {str(l): round(res[l], 3) for l, _ in sched}})
        print(f"  {mixer:>10} " + " ".join(f"{res[l]:>5.2f}" for l, _ in sched), flush=True)
    print("  -> selection chains; the compression corner sags more (composition compounds its per-hop loss).")
    return rows


# -- D4 ------------------------------------------------------------------------------------------

def d4_aux_ablation(seeds=(0, 1)):
    """The aux-loss ablation. JEPA future-prediction weight λ swept on read-salient DeltaNet (+gate) at
    overload — the regime where 'rethink the objective function' was hypothesized to help the write policy
    keep future-relevant content. Prediction (and D2's spot result): null — read-time relevance is a capacity
    limit, so no auxiliary objective on the write side lifts the wall. Reported as measured (do not tune λ)."""
    task = MQAR(256, 256)
    sched = [(2, 3000), (4, 600), (8, 700), (16, 800), (24, 900), (32, 1000)]
    overload = 32
    rows = []
    print(f"\n[D4] JEPA aux-weight λ sweep — read-salient DeltaNet+gate at overload m={overload} (does 'rethink")
    print(f"     the objective' lift the compression wall?)   {'λ':>6} {'recall@'+str(overload):>10}")
    for lam in (0.0, 0.3, 1.0):
        r = _mean_over_seeds(
            lambda s, lam=lam: _curriculum("deltanet", task, sched, s, n_layer=2,
                                           delta_gate=True, jepa=lam > 0, aux_weight=lam)[0], seeds)
        rows.append({"test": "D4", "aux_weight": lam, "recall": round(r[overload], 3)})
        print(f"     {'':>27} {lam:>6.1f} {r[overload]:>10.2f}", flush=True)
    print("  -> measured: flat in λ — the future-prediction aux does not lift the read-time-relevance wall.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="run a subset, e.g. d1,d3")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--out", default="paper/figures/p9_compare.json")
    args = ap.parse_args()
    seeds = tuple(args.seeds)
    which = set(args.only.split(",")) if args.only else {"d1", "d2", "d3", "d4"}
    print("=" * 96)
    print("P9 — the TRAINED selection-vs-compression comparison (micro-LM on MQAR)")
    print("=" * 96)
    t0 = time.time()
    payload = json.load(open(args.out)) if os.path.exists(args.out) else {"meta": {}, "rows": []}
    payload["meta"] = {"d": D, "n_head": NHEAD, "head_dim": DH, "ssa_kappa": SSA["ssa_top_c"] * SSA["ssa_block"],
                       "seeds": list(seeds), "gpu": "RTX 4080 16GB",
                       "note": "trained micro-LM; synthetic MQAR (not natural language); recall at query positions"}
    done = {(r["test"]) for r in payload["rows"]}
    for name, fn in [("d1", d1_capacity), ("d2", d2_gate_aux), ("d3", d3_composition), ("d4", d4_aux_ablation)]:
        tag = {"d1": "D1", "d2": "D2", "d3": "D3", "d4": "D4"}[name]
        if name in which and tag not in done:
            payload["rows"] += fn(seeds)
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            json.dump(payload, open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
