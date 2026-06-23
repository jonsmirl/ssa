"""
P0 — cost decomposition of the SSA kernel: where the wall-clock goes, and the gap to the n·κ floor.

Splits one SSA forward into router (block_route: summaries + the (n/b)² score GEMM + topk) /
BlockMask-build / attention (flex_attention over the selected blocks), swept over context length, then
fits a per-component power law and extrapolates the decomposition to 12M. The ATTENTION component is the
theoretical floor (n·κ); ROUTER + MASKBUILD is the gap we can attack. Reuses ssa_kernel internals.

Run: python3 -m ssa.cost_profile  ->  paper/figures/cost_profile.json
"""
from __future__ import annotations
import json
import time
import torch
from torch.nn.attention.flex_attention import BlockMask
from ssa.ssa_kernel import block_route, ssa_flex, dense, _flex, _causal_mod, BLOCK

DEV = "cuda"


def _t(fn, reps=8):
    try:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        s = time.time()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.time() - s) / reps * 1000
    except Exception as e:
        if any(s in str(e).lower() for s in ("out of memory", "cuda error", "device")):
            torch.cuda.empty_cache()
            return None
        raise


def decompose(n, H=8, d=64, block=BLOCK, top_c=8, local=1):
    q = torch.randn(1, H, n, d, device=DEV, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    nb = n // block
    out = {"n": n, "nb": nb}

    # router = block_route (summaries + (n/b)² score GEMM + topk/argsort)
    out["router"] = _t(lambda: block_route(q, k, block, top_c, local))
    kv_num, kv_idx, _ = block_route(q, k, block, top_c, local)
    # mask build
    out["maskbuild"] = _t(lambda: BlockMask.from_kv_blocks(kv_num, kv_idx, BLOCK_SIZE=block, mask_mod=_causal_mod))
    bm = BlockMask.from_kv_blocks(kv_num, kv_idx, BLOCK_SIZE=block, mask_mod=_causal_mod)
    # attention over the selected blocks — this IS the n·κ floor
    out["attention"] = _t(lambda: _flex(q, k, v, block_mask=bm))
    # total SSA forward
    out["total"] = _t(lambda: ssa_flex(q, k, v, block, top_c, local))
    # dense reference (OOMs eventually)
    out["dense"] = _t(lambda: dense(q, k, v))
    del q, k, v
    torch.cuda.empty_cache()
    return out


def _fit(ns, ys):
    import math
    lx = [math.log(x) for x in ns]
    ly = [math.log(y) for y in ys]
    m = len(ns)
    mx, my = sum(lx) / m, sum(ly) / m
    p = sum((lx[i] - mx) * (ly[i] - my) for i in range(m)) / sum((lx[i] - mx) ** 2 for i in range(m))
    a = my - p * mx
    return p, math.exp(a)


def main():
    torch.manual_seed(0)
    ns = (16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152)
    rows = []
    print("=" * 92)
    print("P0 — SSA kernel cost decomposition (H=8, d=64, block=128, top_c=8)")
    print("=" * 92)
    print(f"  {'n':>9} {'router':>9} {'maskbuild':>10} {'attention':>10} {'total':>9} "
          f"{'router %':>9} {'floor(attn)/total':>17}")
    for n in ns:
        try:
            r = decompose(n)
        except Exception as e:
            torch.cuda.empty_cache()
            print(f"  {n:>9}  (memory wall: {str(e)[:40]} — stopping sweep)")
            break
        rows.append(r)
        if r["total"]:
            rf = 100 * (r["router"] + r["maskbuild"]) / r["total"] if r["router"] else float("nan")
            ff = r["attention"] / r["total"] if r["attention"] else float("nan")
            fmt = lambda x: f"{x:.2f}" if x else "OOM"
            print(f"  {n:>9} {fmt(r['router']):>9} {fmt(r['maskbuild']):>10} {fmt(r['attention']):>10} "
                  f"{fmt(r['total']):>9} {rf:>8.0f}% {ff:>16.2f}")
        json.dump(rows, open("paper/figures/cost_profile.json", "w"), indent=2)

    # fit + extrapolate to 12M (use n >= 131072, where scaling is clean)
    fitrows = [r for r in rows if r["n"] >= 131072 and r["router"] and r["attention"]]
    nlist = [r["n"] for r in fitrows]
    comps = {}
    for key in ("router", "maskbuild", "attention"):
        p, a = _fit(nlist, [r[key] for r in fitrows])
        comps[key] = (p, a)
    n12 = 12e6
    pred = {key: comps[key][1] * n12 ** comps[key][0] for key in comps}
    floor = pred["attention"]
    gap = pred["router"] + pred["maskbuild"]
    print("\n  Power-law exponents:  router ~ n^%.2f,  attention ~ n^%.2f,  maskbuild ~ n^%.2f"
          % (comps["router"][0], comps["attention"][0], comps["maskbuild"][0]))
    print(f"  Extrapolated @12M:  attention(floor)={floor:.0f}ms  router={pred['router']:.0f}ms  "
          f"maskbuild={pred['maskbuild']:.0f}ms")
    print(f"  => router+maskbuild is {100*gap/(gap+floor):.0f}% of the SSA forward at 12M; removing it "
          f"would cut time {(gap+floor)/floor:.2f}x toward the floor.")
    json.dump({"rows": rows, "fit": {k: list(v) for k, v in comps.items()},
               "pred_12M": {**pred, "floor": floor, "gap": gap}},
              open("paper/figures/cost_profile.json", "w"), indent=2)
    print("  wrote paper/figures/cost_profile.json")


if __name__ == "__main__":
    main()
