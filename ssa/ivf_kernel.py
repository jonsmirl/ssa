"""
The IVF kernel, end to end — a faiss-GPU IVF block-router wired straight into the FlexAttention
kernel, measured (not projected) to 12M tokens on one 16 GB card.

`router_gpu_compare.py` timed the IVF router in isolation; `faiss_router.py` PROJECTED the 12M kernel
decomposition from the P0 fit. This module closes that gap: `ivf_route` searches an IVF index over the
nb BLOCK-MEANS and emits the compressed `(kv_num_blocks, kv_indices)` contract `BlockMask.from_kv_blocks`
consumes DIRECTLY — no `(nb,nb)` score GEMM, no argsort maskbuild (the two `(n/b)²` terms P0 measured as
the gap to the floor). The BlockMask is built with `compute_q_blocks=False`, which skips the dense
`(nb,nb+1)` transpose that would need 38.7 GB at nb=98,304 — the single change that makes a live 12M
forward fit. `ssa_flex_ivf` = route -> from_kv_blocks -> the same compiled `_flex` as `ssa_kernel`.

Honest scope: SINGLE HEAD (H=8 does not fit at 12M — K alone is 12.3 GB); synthetic random keys, so this
measures SPEED only (selection quality is the P3/P4 story, unchanged). The dense reference is measured to
`--dense-max` (default 4M) and power-law-fit beyond (marked `dense_is_fit`). Inference-only: everything
runs under `torch.no_grad()` (the `compute_q_blocks=False` path rejects a grad-enabled build).

Run:  python3 -m ssa.ivf_kernel                       # full sweep -> paper/figures/ivf_kernel_e2e.json
      python3 -m ssa.ivf_kernel --ns 262144           # one point, for a quick smoke
"""
from __future__ import annotations
import json
import math
import os
import time
import torch
from torch.nn.attention.flex_attention import BlockMask
from ssa.ssa_kernel import BLOCK, _causal_mod, _flex, dense

try:
    import faiss
    import faiss.contrib.torch_utils          # lets faiss GPU indexes consume torch GPU tensors directly
except ImportError as e:                      # faiss is not in requirements.txt — keep ssa_kernel import-clean
    raise ImportError("ssa.ivf_kernel needs faiss-gpu (pip install faiss-gpu-cu12). "
                      "The core kernel in ssa.ssa_kernel has no faiss dependency.") from e

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_RES = None


def _gpu_res(temp_mb=512):
    """Lazy faiss GPU-resources singleton with a capped scratch arena so it coexists with torch."""
    global _RES
    if _RES is None:
        _RES = faiss.StandardGpuResources()
        _RES.setTempMemory(temp_mb * 1024 * 1024)
    return _RES


def block_means(x, block=BLOCK, chunk=1 << 20):
    """(n, d) fp16 CUDA -> (nb, d) fp32 block means, fp32-accumulated and chunked over blocks so no full
    fp32 copy of x is ever materialized (that copy is exactly what block_route's `.float()` pays at 12M)."""
    n, d = x.shape
    nb = n // block
    out = torch.empty(nb, d, device=x.device, dtype=torch.float32)
    cb = max(1, chunk // block)                                    # blocks per chunk
    for s in range(0, nb, cb):
        e = min(nb, s + cb)
        out[s:e] = x[s * block:e * block].view(e - s, block, d).mean(1, dtype=torch.float32)
    return out


def build_ivf(mu, nlist=None, nprobe=4, res=None):
    """IVF-flat index over the block means (inner product = the routing score)."""
    nb, d = mu.shape
    nlist = nlist or min(nb, max(4, int(nb ** 0.5)))
    ix = faiss.GpuIndexIVFFlat(res or _gpu_res(), d, nlist, faiss.METRIC_INNER_PRODUCT)
    ix.train(mu); ix.add(mu); ix.nprobe = min(nprobe, nlist)
    return ix


def _route_head(mu, qb, top_c, local, nprobe, search_k, res):
    """One head: IVF search over block means -> (kv_num (nb,), kv_idx (nb, W)) int32, W = top_c+local+1.

    Post-processing is pure GPU torch, O(nb·W log W): drop -1 pads and future blocks, keep the first
    `top_c` causal hits in faiss rank order (budget parity with the flat top-c router), OR in the own
    block + `local` predecessors, dedupe by a two-sort with sentinel=nb, count -> kv_num."""
    nb, d = mu.shape
    search_k = min(search_k, nb)
    ix = build_ivf(mu, nprobe=nprobe, res=res)
    _, I = ix.search(qb, search_k)                                # (nb, search_k) int64, distance-sorted
    qi = torch.arange(nb, device=mu.device)
    valid = (I >= 0) & (I <= qi[:, None])                         # -1 pads and future blocks out
    keep = valid & (valid.cumsum(1) <= top_c)                    # first top_c causal hits, faiss order
    SENT = nb
    cand = torch.where(keep, I, torch.full_like(I, SENT))
    loc = qi[:, None] - torch.arange(local + 1, device=mu.device)[None, :]   # own block + `local` before
    loc = torch.where(loc >= 0, loc, torch.full_like(loc, SENT)).to(I.dtype)
    cand = torch.cat([cand, loc], dim=1)                         # (nb, search_k + local + 1)
    cand, _ = cand.sort(1)                                        # sentinels to the right
    dup = torch.zeros_like(cand, dtype=torch.bool)
    dup[:, 1:] = cand[:, 1:] == cand[:, :-1]
    cand = torch.where(dup, torch.full_like(cand, SENT), cand)
    cand, _ = cand.sort(1)                                        # re-pack uniques first
    W = min(top_c + local + 1, nb)                               # never more distinct blocks than exist
    cand = cand[:, :W]
    kv_num = (cand < SENT).sum(1).to(torch.int32)
    kv_idx = torch.where(cand < SENT, cand, torch.zeros_like(cand)).to(torch.int32)  # pads -> block 0 (causal)
    return kv_num, kv_idx


def ivf_route(q, k, block=BLOCK, top_c=8, local=1, nprobe=4, search_k=None, res=None):
    """(B,H,n,d) q,k -> (kv_num (B,H,nb) int32, kv_idx (B,H,nb,W) int32) for BlockMask.from_kv_blocks.
    One IVF index is built per (b,h) over that head's block means (benchmarks use B=1, small H)."""
    B, H, n, d = q.shape
    nb = n // block
    search_k = search_k or min(2 * top_c, nb)
    W = min(top_c + local + 1, nb)                                 # must match _route_head's cap
    kv_num = torch.empty(B, H, nb, device=q.device, dtype=torch.int32)
    kv_idx = torch.empty(B, H, nb, W, device=q.device, dtype=torch.int32)
    for b in range(B):
        for h in range(H):
            mu = block_means(k[b, h], block)
            qb = block_means(q[b, h], block)
            kn, ki = _route_head(mu, qb, top_c, local, nprobe, search_k, res)
            kv_num[b, h] = kn
            kv_idx[b, h] = ki
            del mu, qb
    return kv_num, kv_idx


def _build_mask(kv_num, kv_idx, n, block):
    """BlockMask from the compressed contract. seq_lengths=(n,n) is REQUIRED: the kv_indices width W < nb,
    so without it from_kv_blocks infers kv_len = W·BLOCK instead of the real n. compute_q_blocks=False
    skips the dense (nb,nb+1) transpose (38.7 GB at nb=98,304) — forward-only, under no_grad."""
    return BlockMask.from_kv_blocks(kv_num, kv_idx, BLOCK_SIZE=block, mask_mod=_causal_mod,
                                    seq_lengths=(n, n), compute_q_blocks=False)


def ssa_flex_ivf(q, k, v, block=BLOCK, top_c=8, local=1, nprobe=4, search_k=None):
    """Full IVF-routed SSA inference: IVF route -> sparse BlockMask (no dense transpose) -> fused kernel."""
    n = q.shape[2]
    kv_num, kv_idx = ivf_route(q, k, block, top_c, local, nprobe, search_k)
    return _flex(q, k, v, block_mask=_build_mask(kv_num, kv_idx, n, block))


# ---------------------------------------------------------------------------------------------------
# end-to-end benchmark
# ---------------------------------------------------------------------------------------------------

def _t(fn, warmup, reps):
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        s = time.time()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.time() - s) / reps * 1000
    except Exception as e:
        if any(s in str(e).lower() for s in ("out of memory", "cuda error", "device", "assert", "handles")):
            torch.cuda.empty_cache()
            return None
        raise


def _fill(n, d, g):
    """Preallocate q,k,v (1,1,n,d) fp16 and fill per-slice (no fp32 intermediate, no host transfer)."""
    q = torch.empty(1, 1, n, d, device=DEV, dtype=torch.float16)
    k = torch.empty_like(q); v = torch.empty_like(q)
    for t in (q, k, v):
        t.view(-1).normal_(generator=g)
    return q, k, v


@torch.no_grad()
def decompose_ivf(n, d=64, block=BLOCK, top_c=8, local=1, nprobe=4,
                  warmup=3, reps=6, dense_max=1 << 22, g=None):
    """Single-head IVF-kernel cost decomposition at context length n, mirroring cost_profile.decompose."""
    nb = n // block
    q, k, v = _fill(n, d, g)
    torch.cuda.reset_peak_memory_stats()
    out = {"n": n, "nb": nb, "reps": reps}

    # router: full IVF route (index build + search + post-process), with the build/search split reported
    out["router_ms"] = _t(lambda: ivf_route(q, k, block, top_c, local, nprobe), warmup, reps)
    mu = block_means(k[0, 0], block); qb = block_means(q[0, 0], block)
    search_k = min(2 * top_c, nb)
    out["router_build_ms"] = _t(lambda: build_ivf(mu, nprobe=nprobe), warmup, reps)
    _ixb = build_ivf(mu, nprobe=nprobe)
    out["router_search_ms"] = _t(lambda: _ixb.search(qb, search_k), warmup, reps)
    del mu, qb, _ixb

    kv_num, kv_idx = ivf_route(q, k, block, top_c, local, nprobe)
    out["mean_blocks_per_row"] = kv_num.float().mean().item()
    out["kv_frac"] = out["mean_blocks_per_row"] / ((nb + 1) / 2)
    out["maskbuild_ms"] = _t(lambda: _build_mask(kv_num, kv_idx, n, block), warmup, reps)
    bm = _build_mask(kv_num, kv_idx, n, block)
    out["attention_ms"] = _t(lambda: _flex(q, k, v, block_mask=bm), warmup, reps)   # the measured n·κ floor
    out["total_ms"] = _t(lambda: ssa_flex_ivf(q, k, v, block, top_c, local, nprobe), warmup, reps)

    # dense reference — measured while it is feasible, fit beyond
    if n <= dense_max:
        dreps = 1 if n >= (1 << 22) else max(2, reps // 2)
        out["dense_ms"] = _t(lambda: dense(q, k, v), 1, dreps)
        out["dense_is_fit"] = False
    else:
        out["dense_ms"] = None
        out["dense_is_fit"] = True

    out["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
    del q, k, v, kv_num, kv_idx, bm
    torch.cuda.empty_cache()
    return out


def _fit(ns, ys):
    lx = [math.log(x) for x in ns]; ly = [math.log(y) for y in ys]
    m = len(ns); mx = sum(lx) / m; my = sum(ly) / m
    p = sum((lx[i] - mx) * (ly[i] - my) for i in range(m)) / sum((lx[i] - mx) ** 2 for i in range(m))
    return p, math.exp(my - p * mx)


def _free_gb():
    free, _ = torch.cuda.mem_get_info()
    return free / 1e9


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[262144, 524288, 1048576, 2097152, 4194304, 8388608, 12582912])
    ap.add_argument("--top-c", type=int, default=8)
    ap.add_argument("--local", type=int, default=1)
    ap.add_argument("--nprobe", type=int, default=4)
    ap.add_argument("--dense-max", type=int, default=1 << 22)      # 4M: ~52 s/rep dense, feasible
    ap.add_argument("--out", default="paper/figures/ivf_kernel_e2e.json")
    args = ap.parse_args()

    torch._dynamo.config.cache_size_limit = 64                    # 7 shapes would blow the default 8
    g = torch.Generator(device=DEV).manual_seed(0)
    free = _free_gb()
    print("=" * 96)
    print("THE IVF KERNEL END TO END — measured single-head wall-clock to 12M (faiss-GPU IVF + FlexAttention)")
    print("=" * 96)
    print(f"  free VRAM at start: {free:.1f} GB" + ("  [WARN <10 GB: free the GPU before the >=8M points]"
                                                     if free < 10 else ""))
    print(f"  {'n':>10} {'nb':>7} {'router':>8} {'maskbld':>8} {'attn(floor)':>11} {'total':>9} "
          f"{'dense':>10} {'speedup':>8} {'blk/row':>8} {'peak GB':>8}")
    rows = []
    for n in args.ns:
        r = decompose_ivf(n, top_c=args.top_c, local=args.local, nprobe=args.nprobe,
                          warmup=3 if n <= (1 << 21) else 2, reps=6 if n <= (1 << 21) else 3,
                          dense_max=args.dense_max, g=g)
        rows.append(r)
        fmt = lambda x: f"{x:.2f}" if x else ("OOM" if x is None else "0")
        dstr = f"{r['dense_ms']:.1f}" if r["dense_ms"] else "fit"
        sp = (r["dense_ms"] / r["total_ms"]) if (r["dense_ms"] and r["total_ms"]) else None
        r["speedup_vs_dense"] = sp
        print(f"  {n:>10} {r['nb']:>7} {fmt(r['router_ms']):>8} {fmt(r['maskbuild_ms']):>8} "
              f"{fmt(r['attention_ms']):>11} {fmt(r['total_ms']):>9} {dstr:>10} "
              f"{(f'{sp:.0f}x' if sp else '—'):>8} {r['mean_blocks_per_row']:>8.1f} {r['peak_mem_gb']:>8.2f}")
        _write(args, rows, g)                                     # incremental, crash-safe

    # fit dense beyond dense_max from our own measured single-head points, and backfill fitted dense_ms
    meas = [(r["n"], r["dense_ms"]) for r in rows if r["dense_ms"]]
    fit = None
    if len(meas) >= 2:
        p, a = _fit([m[0] for m in meas], [m[1] for m in meas])
        fit = {"p": p, "a": a, "fit_points_n": [m[0] for m in meas]}
        for r in rows:
            if r["dense_ms"] is None:
                r["dense_ms"] = a * r["n"] ** p
                r["speedup_vs_dense"] = r["dense_ms"] / r["total_ms"] if r["total_ms"] else None
        print(f"\n  dense single-head fit: dense_ms ~ {a:.3e}·n^{p:.2f}  (from n<= {meas[-1][0]})")
    _write(args, rows, g, fit)
    big = rows[-1]
    if big["total_ms"]:
        print(f"  @{big['n']}:  total={big['total_ms']:.0f} ms  vs dense(fit)={big['dense_ms']:.0f} ms  "
              f"=> {big['speedup_vs_dense']:.0f}x;  total/attention = {big['total_ms']/big['attention_ms']:.1f}x "
              f"(the gap to the floor, MEASURED)")
    print(f"  wrote {args.out}")


def _write(args, rows, g, fit=None):
    meta = {"B": 1, "H": 1, "d": 64, "block": BLOCK, "top_c": args.top_c, "local": args.local,
            "nprobe": args.nprobe, "dtype": "float16", "seed": 0, "gpu": "RTX 4080 16GB",
            "dense_max_measured": args.dense_max,
            "note": "single-head; synthetic random keys (SPEED ONLY — selection quality is P3/P4); "
                    "dense measured to dense_max then power-law fit (dense_is_fit)."}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"meta": meta, "rows": rows, "dense_fit": fit}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
