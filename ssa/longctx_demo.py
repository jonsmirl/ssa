"""Complete long-context transformer demo: real model, multi-head CCC routing, quality and speed.

The demo closes the repository's former evidence seam in one process.  It runs an unmodified dense
Qwen2.5-0.5B baseline and then swaps every attention layer to the strictly-causal streaming CCC router
plus FlexAttention.  Qwen contributes real post-RoPE key/query geometry, 14 query heads, two KV heads,
and 24 transformer layers.  A fixed ``top_c`` makes selected attention linear in context length; the
streaming IVF cascade avoids the flat block-by-block routing matrix.

The default run uses YaRN for a fair dense/sparse comparison at 32K and 128K, evaluates single-needle
and two-hop retrieval, measures full-model prefill latency and peak memory, and checkpoints every row.
It is an inference demonstration, not a claim that the frozen sparse swap has been trained optimally.

Run on a CUDA host with cached/downloadable Qwen weights and faiss-gpu::

    python -m ssa.longctx_demo
    python -m ssa.longctx_demo --lengths 8192,32768 --dense-lengths 8192,32768 --quick
"""
from __future__ import annotations

import argparse
import json
import os
import time

from ssa.gemma_ssa_eval import lm_loss, niah_accuracy, two_hop_accuracy
from ssa.gemma_ssa_sweep import LM_TEXTS
from ssa.longctx_swap import _ids_of_len, load


def _csv_ints(value):
    return [int(x) for x in value.split(",") if x]


def _quality(model, tok, n, trials, twohop_trials, device):
    return {
        "niah_acc": round(niah_accuracy(model, tok, n, trials=trials, device=device), 4),
        "twohop_acc": (round(two_hop_accuracy(model, tok, n, trials=twohop_trials,
                                               device=device), 4)
                       if twohop_trials else None),
    }


def _measure_prefill(model, tok, n, device, warmup, reps, routed):
    import torch
    from ssa import gemma_ssa as G
    ids = _ids_of_len(tok, n, device)
    attention_mask = torch.ones_like(ids)

    def fwd():
        try:
            return model(ids, attention_mask=attention_mask, logits_to_keep=1)
        except TypeError:
            return model(ids, attention_mask=attention_mask)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(warmup):
            fwd()
        torch.cuda.synchronize()
        G.ROUTE_MS = 0.0                         # exclude warmup/compile from the router measurement too
        elapsed = []
        for _ in range(reps):
            start = time.time()
            fwd()
            torch.cuda.synchronize()
            elapsed.append((time.time() - start) * 1000)
    elapsed.sort()
    ms = elapsed[len(elapsed) // 2]
    peak = torch.cuda.max_memory_allocated() / 1e9
    route_ms = G.ROUTE_MS / reps if routed else 0.0
    del ids
    return round(ms, 2), round(peak, 3), round(route_ms, 3)


def _selected_fraction(n, block, top_c, local, outlier_cap):
    """Analytic mean selected-block fraction over causal query blocks (an upper bound with outliers)."""
    nb = (n + block - 1) // block
    selected = [min(i + 1, top_c + local + 1 + outlier_cap) for i in range(nb)]
    visible = [i + 1 for i in range(nb)]
    return sum(selected) / sum(visible)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--lengths", default="32768,131072")
    ap.add_argument("--dense-lengths", default="32768,131072")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--top-c", type=int, default=64)
    ap.add_argument("--local", type=int, default=1)
    ap.add_argument("--sub", type=int, default=32)
    ap.add_argument("--chunk-blocks", type=int, default=128)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--search-k", type=int, default=256)
    ap.add_argument("--build-threshold", type=int, default=512)
    ap.add_argument("--outlier-rate", type=float, default=1e-3)
    ap.add_argument("--outlier-cap", type=int, default=4)
    ap.add_argument("--outlier-store-cap", type=int, default=1024,
                    help="fixed top-leverage reservoir; keeps side-channel lookup linear in n")
    ap.add_argument("--share-from", type=int, default=12,
                    help="route here and reuse its mask above; lower layers still route independently")
    ap.add_argument("--trials", type=int, default=1,
                    help="trials at each of three NIAH depths")
    ap.add_argument("--twohop-trials", type=int, default=1,
                    help="trials at each of three two-hop depth pairs")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true",
                    help="skip two-hop and use one timed repetition")
    ap.add_argument("--fresh", action="store_true", help="ignore and replace an existing output artifact")
    ap.add_argument("--rerun-sparse", action="store_true",
                    help="reuse matching dense rows but replace all sparse rows")
    ap.add_argument("--out", default="runs/qwen_complete_longctx_demo.json")
    args = ap.parse_args()

    import torch
    from ssa import gemma_ssa as G
    from ssa.gemma_ssa import install_ssa

    if not torch.cuda.is_available():
        raise RuntimeError("this demo requires CUDA")
    if args.block % args.sub:
        ap.error("--sub must divide --block")
    lengths = sorted(set(_csv_ints(args.lengths)))
    dense_lengths = sorted(set(_csv_ints(args.dense_lengths)) & set(lengths))
    if args.quick:
        args.twohop_trials = 0
        args.reps = 1
    yarn = max(lengths) > 32768
    rope = "yarn4" if yarn else "native"

    print(f"loading {args.model} on {torch.cuda.get_device_name(0)} (rope={rope})...", flush=True)
    model, tok = load(args.model, yarn=yarn)
    text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    hq = int(text_cfg.num_attention_heads)
    hkv = int(text_cfg.num_key_value_heads)
    layers = int(text_cfg.num_hidden_layers)

    rows = {}
    if os.path.exists(args.out) and not args.fresh:
        old = json.load(open(args.out))
        rows = {(r["arm"], r["n"]): r for r in old.get("rows", [])}
        if args.rerun_sparse:
            rows = {key: row for key, row in rows.items() if key[0] != "ccc"}
        print(f"  [resume] loaded {len(rows)} completed rows", flush=True)

    meta = {
        "model": args.model, "transformer_layers": layers,
        "query_heads": hq, "kv_heads": hkv, "head_dim": int(text_cfg.hidden_size // hq),
        "dtype": "bfloat16", "rope": rope, "router": "strictly-causal streaming CCC/FAISS-IVF",
        "attention_kernel": "PyTorch FlexAttention", "block": args.block, "sub": args.sub,
        "top_c": args.top_c, "local": args.local, "nprobe": args.nprobe,
        "search_k": args.search_k, "chunk_blocks": args.chunk_blocks,
        "build_threshold": args.build_threshold, "outlier_rate": args.outlier_rate,
        "outlier_cap": args.outlier_cap, "outlier_store_cap": args.outlier_store_cap,
        "timing_warmup": args.warmup, "timing_reps": args.reps,
        "share_route_from": args.share_from,
        "quality_trials_per_geometry": args.trials,
        "twohop_trials_per_geometry": args.twohop_trials,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "scope": "frozen inference swap; decode uses the analytic sparse fallback after CCC prefill",
    }

    def save(smoke=None):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {"meta": meta, "smoke": smoke,
                   "rows": sorted(rows.values(), key=lambda r: (r["n"], r["arm"]))}
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)

    print(f"  {'arm':>7} {'tokens':>8} {'NIAH':>6} {'2hop':>6} {'prefill ms':>11} "
          f"{'route ms':>10} {'peak GB':>8} {'selected':>9}")

    # Dense rows must run before registering the replacement attention implementation.
    for n in dense_lengths:
        key = ("dense", n)
        if key in rows:
            continue
        ms, peak, route = _measure_prefill(model, tok, n, "cuda", args.warmup, args.reps, False)
        quality = _quality(model, tok, n, args.trials, args.twohop_trials, "cuda")
        row = {"arm": "dense", "n": n, **quality, "prefill_ms": ms,
               "route_ms": route, "peak_mem_gb": peak, "selected_fraction_upper": 1.0}
        rows[key] = row
        save()
        print(f"  {'dense':>7} {n:>8} {row['niah_acc']:>6.2f} "
              f"{(row['twohop_acc'] or 0):>6.2f} {ms:>11.2f} {route:>10.2f} {peak:>8.3f} {1:>9.3f}",
              flush=True)

    # Full-budget fused attention is the wiring gate; it must preserve the pretrained model's loss.
    dense_loss = lm_loss(model, tok, LM_TEXTS, max_len=512, device="cuda")
    install_ssa(model, block=args.block, budget_frac=1.0, top_c=None, impl="ccc")
    fused_loss = lm_loss(model, tok, LM_TEXTS, max_len=512, device="cuda")
    smoke = {"dense_lm_loss": dense_loss, "full_budget_fused_lm_loss": fused_loss,
             "absolute_delta": abs(fused_loss - dense_loss),
             "passed": abs(fused_loss - dense_loss) < 5e-2}
    save(smoke)
    print(f"  smoke: dense loss {dense_loss:.4f}, full-budget fused {fused_loss:.4f}, "
          f"delta {smoke['absolute_delta']:.3g} — {'PASS' if smoke['passed'] else 'FAIL'}", flush=True)
    if not smoke["passed"]:
        raise RuntimeError("full-budget attention swap failed its dense-equivalence smoke gate")

    G.CFG = G.SSAConfig(
        block=args.block, top_c=args.top_c, local_w=args.local, impl="ccc",
        nprobe=args.nprobe, search_k=args.search_k, sub=args.sub,
        chunk_blocks=args.chunk_blocks, build_threshold=args.build_threshold,
        retrain_every=0, outlier_rate=args.outlier_rate, outlier_cap=args.outlier_cap,
        outlier_store_cap=args.outlier_store_cap)
    G.CFG.share_route_from = args.share_from

    for n in lengths:
        key = ("ccc", n)
        if key in rows:
            continue
        ms, peak, route = _measure_prefill(model, tok, n, "cuda", args.warmup, args.reps, True)
        quality = _quality(model, tok, n, args.trials, args.twohop_trials, "cuda")
        frac = _selected_fraction(n, args.block, args.top_c, args.local, args.outlier_cap)
        dense = rows.get(("dense", n))
        row = {"arm": "ccc", "n": n, **quality, "prefill_ms": ms,
               "route_ms": route, "route_share": route / ms if ms else None,
               "peak_mem_gb": peak, "selected_fraction_upper": frac,
               "speedup_vs_dense": dense["prefill_ms"] / ms if dense else None}
        rows[key] = row
        save(smoke)
        print(f"  {'ccc':>7} {n:>8} {row['niah_acc']:>6.2f} "
              f"{(row['twohop_acc'] or 0):>6.2f} {ms:>11.2f} {route:>10.2f} {peak:>8.3f} {frac:>9.3f}",
              flush=True)

    print(f"\nwrote {args.out}")
    print("This run combines one pretrained transformer, real Q/K geometry, all attention heads/layers, "
          "strictly-causal subquadratic routing, a fused sparse kernel, and measured retrieval quality.")


if __name__ == "__main__":
    main()
