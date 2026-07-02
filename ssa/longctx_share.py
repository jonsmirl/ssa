"""
Cross-layer routing sharing + the trained routing space, MEASURED on a real model — the first measurement
of what router_variants.py only asserted analytically (the "÷5" cross-layer sharing) and the DSA-comparable
number (routing's share of total prefill).

Reuses longctx_swap's load / measure / prefill_ms; writes a NEW checkpoint (runs/qwen_share.json) so the
P6 baselines stay immutable. For each donor layer we compute the block selection once and reuse it above
the donor (layers below route per-layer — the honest cost accounting), and measure NIAH / two-hop quality
plus prefill wall-clock and the routing's share of it (G.ROUTE_MS / prefill).

Run:  python -m ssa.longctx_share --lengths 8192,32768 --donors 0,4,11,18
      python -m ssa.longctx_share --lengths 8192 --proj runs/routing_space_d16_shared.pt   # trained space
"""
from __future__ import annotations
import argparse
import json
import os

from ssa.longctx_swap import load, prefill_ms, _ids_of_len
from ssa.gemma_ssa_eval import niah_accuracy, two_hop_accuracy


def measure_share(model, tok, n, dev, niah_trials, twohop_trials):
    from ssa import gemma_ssa as G
    ids = _ids_of_len(tok, n, dev)
    G.ROUTE_MS = 0.0
    pms, mem = prefill_ms(model, ids, warmup=2, reps=3)
    route_ms = G.ROUTE_MS / 5.0                                    # ROUTE_MS accumulated over 2 warmup + 3 reps
    return {"niah_acc": round(niah_accuracy(model, tok, n, trials=niah_trials, device=dev), 4),
            "niah2_acc": round(two_hop_accuracy(model, tok, n, trials=twohop_trials, device=dev), 4),
            "prefill_ms": round(pms, 2), "route_ms": round(route_ms, 3),
            "route_share": round(route_ms / pms, 4) if pms else None, "peak_mem_gb": round(mem, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--lengths", default="8192,32768")
    ap.add_argument("--budget", type=float, default=0.12)
    ap.add_argument("--donors", default="0,4,11,18")
    ap.add_argument("--proj", default="")                         # trained routing-space artifact (optional)
    ap.add_argument("--niah-trials", type=int, default=3)
    ap.add_argument("--twohop-trials", type=int, default=3)
    ap.add_argument("--out", default="runs/qwen_share.json")
    args = ap.parse_args()
    import torch
    from ssa import gemma_ssa as G
    from ssa.gemma_ssa import install_ssa
    lengths = [int(x) for x in args.lengths.split(",")]
    donors = [int(x) for x in args.donors.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading {args.model}...", flush=True)
    model, tok = load(args.model, yarn=False)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = {}
    if os.path.exists(args.out):
        for r in json.load(open(args.out)).get("rows", []):
            rows[(r["arm"], r["n"])] = r

    def save():
        json.dump({"meta": {"model": args.model, "budget": args.budget, "block": 128,
                            "note": "route_share = routing wall-clock / prefill wall-clock (the DSA-comparable "
                                    "number); donor=k shares selection to layers >k, layers <k route per-layer"},
                   "rows": sorted(rows.values(), key=lambda r: (r["n"], r["arm"]))},
                  open(args.out, "w"), indent=2)

    print(f"  {'arm':>22} {'n':>7} {'NIAH':>6} {'2hop':>6} {'prefill':>9} {'route':>8} {'route/pref':>10}")

    # 1) per-layer full-d routing (no sharing) — the cost baseline
    arms = [("per_layer_fulld", None, "")]
    for dn in donors:
        arms.append((f"share_from_{dn}_fulld", dn, ""))
    if args.proj:
        arms.append(("per_layer_d16", None, args.proj))
        best = donors[len(donors) // 2]
        arms.append((f"share_from_{best}_d16", best, args.proj))

    for arm, donor, proj in arms:
        install_ssa(model, block=128, budget_frac=args.budget, impl="flex",
                    share_route_from=donor, proj_path=(proj or None))
        for n in lengths:
            if (arm, n) in rows:
                continue
            m = measure_share(model, tok, n, dev, args.niah_trials, args.twohop_trials)
            rows[(arm, n)] = {"arm": arm, "n": n, "donor": donor, "proj": bool(proj), **m}
            save()
            print(f"  {arm:>22} {n:>7} {m['niah_acc']:>6.2f} {m['niah2_acc']:>6.2f} {m['prefill_ms']:>9.1f} "
                  f"{m['route_ms']:>8.2f} {m['route_share']:>10.3f}", flush=True)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
