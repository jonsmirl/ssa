"""
The kappa-sweep driver: frozen-swap SSA quality-vs-budget on a real model. RESUMABLE and ORDERED so a
truncated GPU window still yields the curve. This is the script the GPU hands off to:

    1) smoke gate   — kappa=100% must reproduce the pre-swap dense forward (fail-fast: catches a broken
                      256-dim / K=V / QK-norm handling on Gemma's full layers before wasting the window);
    2) kappa sweep  — for each (context length, budget) measure NIAH accuracy (the load-bearing signal)
                      and LM loss (the weak, local-window-dominated signal), writing a checkpoint after
                      every cell so a re-run skips finished work.

The decisive headline is the NIAH-vs-budget curve, not perplexity (the Qwen dry-run confirmed raw LM
loss barely moves under sparsity — it's dominated by the local window).

Scope note: on a 16 GB GPU + CPU offload the 26B forward is minutes/sequence, so this measures the
quality MECHANISM at MODERATE context (a few k–tens of k), not the literal 12M endpoint — that needs a
cluster (the neocloud/AWS question). Keep lengths/trials modest; resumability covers truncation.

    python -m ssa.gemma_ssa_sweep --model google/gemma-4-26B-A4B --lengths 2048,8192 \
        --budgets 1.0,0.5,0.25,0.12 --niah-trials 4 --out runs/gemma_sweep.json
    # dry-run the orchestration on a small model first:
    python -m ssa.gemma_ssa_sweep --model Qwen/Qwen2.5-0.5B --device-map none \
        --lengths 384,768 --budgets 1.0,0.25 --niah-trials 1 --out runs/dry.json
"""
from __future__ import annotations
import os
import json
import time
import argparse

from ssa.gemma_ssa_eval import niah_accuracy, lm_loss, two_hop_accuracy

# a small held-out real-text set for the LM-loss signal (distinct registers, no incidental needles)
LM_TEXTS = [
    "The history of cartography is the history of how societies chose to see themselves: every map "
    "encodes a theory of what matters, what is central, and what may safely be left blank. ",
    "Photosynthesis converts light into chemical energy through a chain of redox reactions in the "
    "thylakoid membrane, storing the result as the bonds of a sugar the cell can later spend. ",
    "Markets clear not because anyone intends an equilibrium but because mismatched prices create "
    "the very incentives that erode them, a settling no participant can see from the inside. ",
]


def load_model(name, device_map="auto"):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    dm = None if device_map in ("none", "None", "") else device_map
    kw = dict(dtype="auto", low_cpu_mem_usage=True)
    if dm:
        kw["device_map"] = dm
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except Exception as e:
        print(f"  [load] AutoModelForCausalLM failed ({type(e).__name__}); trying the Gemma-4 "
              f"multimodal class (text path)...", flush=True)
        try:
            from transformers import AutoModelForImageTextToText as M
        except Exception:
            from transformers import Gemma4ForConditionalGeneration as M
        model = M.from_pretrained(name, attn_implementation="sdpa", **kw)
    return model.eval(), tok


def pick_device(model, device_map):
    """Where to place inputs. Plain load -> the model's actual device; accelerate-dispatched
    (device_map='auto', the Gemma offload case) -> cuda:0, which its hooks fan out from."""
    import torch
    if device_map in (None, "none", "None", ""):
        return str(next(model.parameters()).device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def smoke_gate(model, tok, block, device, route_full_only=True):
    """kappa=100% (full budget) must reproduce the stock dense forward. Returns (ok, base, gated)."""
    from ssa.gemma_ssa import install_ssa
    txt = ["The quiet harbor filled with morning light as the boats returned with the night's catch. " * 4]
    base = lm_loss(model, tok, txt, max_len=128, device=device)          # stock attention (pre-install)
    install_ssa(model, block=block, budget_frac=1.0, route_full_only=route_full_only)
    gated = lm_loss(model, tok, txt, max_len=128, device=device)          # SSA at full budget
    ok = abs(gated - base) < 5e-2
    print(f"[smoke] dense={base:.4f}  SSA@1.0={gated:.4f}  delta={abs(gated-base):.2e}  "
          f"{'PASS' if ok else 'FAIL — full-layer handling is off; do not trust the sweep'}", flush=True)
    return ok, base, gated


def sweep(model, tok, lengths, budgets, blocks, depths, trials, out, device,
          max_new=12, route_full_only=True, edgeworth=False, beta=2.0, dense_layers=(),
          twohop_trials=3):
    from ssa import gemma_ssa as G
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rows = {}
    if os.path.exists(out):
        for r in json.load(open(out)).get("rows", []):
            rows[(r["n"], r.get("block", 256), r["budget"])] = r
        print(f"  [resume] {len(rows)} cells already done in {out}", flush=True)

    def save():
        json.dump({"rows": sorted(rows.values(), key=lambda r: (r["n"], r.get("block", 256), -r["budget"]))},
                  open(out, "w"), indent=2)

    def set_cfg(blk, b):
        G.CFG = G.SSAConfig(block=blk, budget_frac=b, route_full_only=route_full_only,
                            edgeworth=edgeworth, beta=beta, dense_layers=dense_layers)

    for n in sorted(lengths):                       # cheapest length first -> a full curve lands early
        for blk in blocks:
            for b in budgets:                       # budgets passed 1.0-first (dense ref before sparse)
                if (n, blk, b) in rows:
                    r = rows[(n, blk, b)]
                    if twohop_trials and "niah2_acc" not in r:   # backfill only the missing metric
                        set_cfg(blk, b)
                        r["niah2_acc"] = round(two_hop_accuracy(model, tok, n, trials=twohop_trials,
                                                                device=device), 4)
                        r["niah2_trials"] = twohop_trials
                        save()
                        print(f"  [backfill 2hop] n={n} block={blk} budget={b} niah2={r['niah2_acc']:.3f}",
                              flush=True)
                    else:
                        print(f"  [skip] n={n} block={blk} budget={b}", flush=True)
                    continue
                set_cfg(blk, b)
                t0 = time.time()
                acc = niah_accuracy(model, tok, n, depths=depths, trials=trials,
                                    max_new_tokens=max_new, device=device)
                acc2 = (round(two_hop_accuracy(model, tok, n, trials=twohop_trials, device=device), 4)
                        if twohop_trials else None)
                ll = lm_loss(model, tok, LM_TEXTS, max_len=n, device=device)
                rows[(n, blk, b)] = {"n": n, "block": blk, "budget": b, "niah_acc": round(acc, 4),
                                     "niah2_acc": acc2, "niah2_trials": twohop_trials,
                                     "lm_loss": round(ll, 4), "sec": round(time.time() - t0, 1)}
                save()
                print(f"  [done] n={n:>7} block={blk:>4} budget={b:<5} niah={acc:.3f} "
                      f"niah2={acc2 if acc2 is None else f'{acc2:.3f}'} lm={ll:.4f} "
                      f"({rows[(n, blk, b)]['sec']}s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-26B-A4B")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--lengths", default="2048,8192")
    ap.add_argument("--budgets", default="1.0,0.5,0.25,0.12,0.06")
    ap.add_argument("--block", default="256", help="comma list of block sizes to sweep")
    ap.add_argument("--niah-trials", type=int, default=4)
    ap.add_argument("--niah-depths", default="0.1,0.5,0.9")
    ap.add_argument("--out", default="runs/sweep.json")
    ap.add_argument("--route-all", action="store_true",
                    help="route every attention layer (default: only full/global layers)")
    ap.add_argument("--edgeworth", action="store_true", help="add the 3rd-cumulant (skew) routing term")
    ap.add_argument("--beta", type=float, default=2.0, help="cumulant routing temperature")
    ap.add_argument("--dense-layers", default="", help="comma list of layer_idx to leave dense")
    ap.add_argument("--twohop-trials", type=int, default=3, help="two-hop chain trials/cell (0 disables)")
    ap.add_argument("--no-twohop", action="store_true", help="skip the two-hop metric entirely")
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",")]
    budgets = [float(x) for x in args.budgets.split(",")]
    depths = tuple(float(x) for x in args.niah_depths.split(","))
    route_full_only = not args.route_all
    dense_layers = tuple(int(x) for x in args.dense_layers.split(",") if x.strip())
    blocks = [int(x) for x in args.block.split(",")]

    print(f"loading {args.model} (device_map={args.device_map})...", flush=True)
    model, tok = load_model(args.model, args.device_map)
    device = pick_device(model, args.device_map)
    print(f"  loaded; eval device = {device}", flush=True)

    ok, _, _ = smoke_gate(model, tok, blocks[0], device, route_full_only)
    if not ok:
        print("ABORT: smoke gate failed — the kappa=100% path does not reproduce dense. Fix the "
              "full-layer swap before sweeping (likely the 256-dim/K=V or QK-norm handling).")
        return

    print(f"  routing: edgeworth={args.edgeworth} beta={args.beta} dense_layers={dense_layers}", flush=True)
    twohop_trials = 0 if args.no_twohop else args.twohop_trials
    rows = sweep(model, tok, lengths, budgets, blocks, depths, args.niah_trials,
                 args.out, device, route_full_only=route_full_only,
                 edgeworth=args.edgeworth, beta=args.beta, dense_layers=dense_layers,
                 twohop_trials=twohop_trials)

    print("\n=== kappa-sweep: NIAH (single) + 2-hop (chain) + LM loss vs (block, budget) ===")
    print(f"{'n':>8} {'block':>6} {'budget':>7} {'NIAH':>7} {'2hop':>7} {'LM loss':>9}")
    for r in sorted(rows.values(), key=lambda r: (r["n"], r.get("block", 256), -r["budget"])):
        n2 = r.get("niah2_acc")
        print(f"{r['n']:>8} {r.get('block', 256):>6} {r['budget']:>7.2f} {r['niah_acc']:>7.3f} "
              f"{'—' if n2 is None else f'{n2:>7.3f}'} {r['lm_loss']:>9.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
