"""
The fast kernel inside a real model at long context — the first result that is simultaneously
real-model × long-context × subquadratic-kernel × quality-measured (at 0.5B scale).

The Gemma frozen-swap measured QUALITY with an analytic O(n²) probe; ssa_kernel measured SPEED on
synthetic keys. This wires the fused FlexAttention kernel (gemma_ssa impl="flex") into a real pretrained
model (Qwen2.5-0.5B, which fits in bf16) and measures BOTH at 8K–128K: NIAH + two-hop accuracy vs budget
AND prefill wall-clock vs the unswapped model. Order: dense baseline rows first (stock SDPA), then a smoke
gate (full-budget flex must reproduce the dense LM loss), then the budget grid; every cell checkpointed so
a truncated GPU window still yields a curve.

Scope: 0.5B, single model. n>32768 needs YaRN (Qwen's native window is 32768) — pass --yarn and BOTH the
dense baseline and SSA run under the same YaRN, so ≥65K rows are mechanism + wall-clock evidence, not
absolute-quality claims (rows tagged rope=native|yarn4). The query-BLOCK-granularity flex router differs
from the analytic per-query router; the --impl analytic vs flex A/B at 8K measures that quality cost
directly. Gemma-26B stays on the analytic path (offload makes 32K+ prefill infeasible here).

Run:  python -m ssa.longctx_swap --lengths 8192,16384,32768 --budgets 1.0,0.25,0.12,0.06
      python -m ssa.longctx_swap --lengths 65536,131072 --yarn        # extended window
      python -m ssa.longctx_swap --lengths 8192 --impl analytic       # the granularity A/B leg
"""
from __future__ import annotations
import argparse
import json
import os
import time

from ssa.gemma_ssa_eval import niah_accuracy, two_hop_accuracy, lm_loss
from ssa.gemma_ssa_sweep import LM_TEXTS


def load(model_name, yarn):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(model_name)
    cfg = AutoConfig.from_pretrained(model_name)
    if yarn:                                                       # extend the 32768 native window ×4
        # Qwen2.5 stashes rope_theta inside rope_scaling; preserve the base or YaRN init gets base=None.
        base = getattr(cfg, "rope_theta", None) or (getattr(cfg, "rope_scaling", None) or {}).get("rope_theta") or 1e6
        cfg.rope_theta = base
        cfg.rope_scaling = {"rope_type": "yarn", "factor": 4.0,
                            "original_max_position_embeddings": 32768}
        cfg.max_position_embeddings = 131072
    model = AutoModelForCausalLM.from_pretrained(model_name, config=cfg, dtype=torch.bfloat16).eval()
    model.to("cuda")
    return model, tok


def _ids_of_len(tok, n, dev):
    """A token tensor of length exactly n (a NIAH-style prompt padded with filler then trimmed)."""
    import torch
    from ssa.gemma_ssa_eval import make_niah_text
    text, _ = make_niah_text(50000, 0.5, max(4, n // 8))
    ids = tok(text, return_tensors="pt")["input_ids"]
    if ids.shape[1] < n:                                          # pad by repeating the filler tokenization
        reps = (n // ids.shape[1]) + 1
        ids = ids.repeat(1, reps)
    return ids[:, :n].to(dev)


def prefill_ms(model, ids, warmup=2, reps=3):
    """Median prefill wall-clock (ms) + peak memory (GB). logits_to_keep=1 avoids the full-vocab matmul so
    we time the forward (attention-dominated at long n), and 2 warmups absorb torch.compile per-shape."""
    import torch
    def fwd():
        try:
            return model(ids, logits_to_keep=1)
        except TypeError:
            return model(ids)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            fwd()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s = time.time(); fwd(); torch.cuda.synchronize(); ts.append((time.time() - s) * 1000)
    ts.sort()
    return ts[len(ts) // 2], torch.cuda.max_memory_allocated() / 1e9


def measure(model, tok, n, dev, niah_trials, twohop_trials, lm_cap=4096):
    ids = _ids_of_len(tok, n, dev)
    pms, mem = prefill_ms(model, ids)
    return {
        "niah_acc": round(niah_accuracy(model, tok, n, trials=niah_trials, device=dev), 4),
        "niah2_acc": round(two_hop_accuracy(model, tok, n, trials=twohop_trials, device=dev), 4)
        if twohop_trials else None,
        "lm_loss": round(lm_loss(model, tok, LM_TEXTS, max_len=min(n, lm_cap), device=dev), 4),
        "prefill_ms": round(pms, 2), "peak_mem_gb": round(mem, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--lengths", default="8192,16384,32768")
    ap.add_argument("--budgets", default="1.0,0.25,0.12,0.06")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--impl", default="flex", choices=["flex", "analytic"])
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--niah-trials", type=int, default=3)
    ap.add_argument("--twohop-trials", type=int, default=3)
    ap.add_argument("--yarn", action="store_true", help="YaRN ×4 for n>32768 (native window is 32768)")
    ap.add_argument("--out", default="runs/qwen_longctx.json")
    args = ap.parse_args()

    import torch
    from ssa import gemma_ssa as G
    from ssa.gemma_ssa import install_ssa
    lengths = [int(x) for x in args.lengths.split(",")]
    budgets = [float(x) for x in args.budgets.split(",")]
    rope = "yarn4" if args.yarn else "native"
    if any(n > 32768 for n in lengths) and not args.yarn:
        print("REFUSING n>32768 without --yarn (Qwen native window is 32768; both dense and SSA must "
              "run under the same rope for a fair comparison).")
        return
    dev = "cuda"

    print(f"loading {args.model} (rope={rope}, impl={args.impl})...", flush=True)
    model, tok = load(args.model, args.yarn)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = {}
    if os.path.exists(args.out):
        for r in json.load(open(args.out)).get("rows", []):
            rows[(r["impl"], r["n"], r["budget"], r["rope"])] = r
        print(f"  [resume] {len(rows)} cells in {args.out}", flush=True)

    def save():
        order = {"dense": 0, "analytic": 1, "flex": 2}
        json.dump({"meta": {"model": args.model, "block": args.block, "beta": args.beta,
                            "gpu": "RTX 4080 16GB", "torch": torch.__version__, "seed": 0,
                            "lm_loss_max_len": 4096,
                            "note": "dense baseline = stock SDPA; flex = fused kernel swap; ≥65K under YaRN "
                                    "×4 (rope tag); speedup compares dense vs SSA at equal n and rope only"},
                   "rows": sorted(rows.values(), key=lambda r: (r["n"], order.get(r["impl"], 9), -r["budget"]))},
                  open(args.out, "w"), indent=2)

    # 1) dense baselines (stock attention, pre-install)
    print(f"  {'phase':>9} {'n':>8} {'budget':>7} {'NIAH':>6} {'2hop':>6} {'prefill ms':>11} {'peak GB':>8}")
    for n in sorted(lengths):
        key = ("dense", n, 1.0, rope)
        if key in rows:
            continue
        m = measure(model, tok, n, dev, args.niah_trials, args.twohop_trials)
        rows[key] = {"impl": "dense", "n": n, "budget": 1.0, "rope": rope, **m}
        save()
        print(f"  {'dense':>9} {n:>8} {1.0:>7} {m['niah_acc']:>6.2f} "
              f"{(m['niah2_acc'] if m['niah2_acc'] is not None else 0):>6.2f} {m['prefill_ms']:>11.1f} "
              f"{m['peak_mem_gb']:>8.2f}", flush=True)

    # 2) install SSA + smoke gate (full-budget kernel must reproduce the dense LM loss)
    base = lm_loss(model, tok, LM_TEXTS, max_len=512, device=dev)
    install_ssa(model, block=args.block, budget_frac=1.0, beta=args.beta, impl=args.impl)
    gated = lm_loss(model, tok, LM_TEXTS, max_len=512, device=dev)
    ok = abs(gated - base) < 5e-2
    print(f"  [smoke] dense={base:.4f} SSA@1.0({args.impl})={gated:.4f} delta={abs(gated-base):.2e} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        print("  ABORT: full-budget swap does not reproduce dense.")
        return

    # 3) budget grid
    for n in sorted(lengths):
        for b in budgets:
            key = (args.impl, n, b, rope)
            if key in rows:
                print(f"  [skip] {args.impl} n={n} budget={b}", flush=True)
                continue
            G.CFG = G.SSAConfig(block=args.block, budget_frac=b, beta=args.beta, impl=args.impl)
            m = measure(model, tok, n, dev, args.niah_trials, args.twohop_trials)
            rows[key] = {"impl": args.impl, "n": n, "budget": b, "rope": rope, **m}
            save()
            dense_ms = rows.get(("dense", n, 1.0, rope), {}).get("prefill_ms")
            sp = f"{dense_ms / m['prefill_ms']:.1f}x" if dense_ms and m["prefill_ms"] else "—"
            print(f"  {args.impl:>9} {n:>8} {b:>7} {m['niah_acc']:>6.2f} "
                  f"{(m['niah2_acc'] if m['niah2_acc'] is not None else 0):>6.2f} {m['prefill_ms']:>11.1f} "
                  f"{m['peak_mem_gb']:>8.2f}  speedup {sp}", flush=True)

    save()
    print(f"\n  wrote {args.out}  (rope={rope})")


if __name__ == "__main__":
    main()
