"""
The decode path — per-step economics of IVF-routed SSA vs dense, measured to 12M.

Prefill got the headline (ivf_kernel.py); autoregressive DECODE is where the serving cost actually lives,
and the SUBQ_ASSESSMENT "serving paradox" asserts an O(n)-per-step dense path vs O(κ) for selection but
never measures it. This does. Per generated token: maintain block means incrementally (a running sum for
the partial tail block; when a block completes, add its mean to the IVF index — build once at prefill,
add-only, NO quantizer retrain, so the index holds only COMPLETED PAST blocks and selection is
automatically causal), IVF-search the single query, gather ≈κ keys, one κ-length softmax row
(`decode_attend`). The dense reference is a FAIR one: an fp16 flash-decode row (sdpa, q_len=1) over the
full (pos+1)-length prefix with no fp32 K/V copy — measurable even at 12M (a ~3.2 GB fp16 K/V read),
so BOTH sides are measured, no cost model. The previous fp32-upcasting reference (which copied the whole
prefix to fp32 every step, overstating the dense cost ~5×) is kept as `dense_decode_naive` and reported
alongside — the headline speedup uses the fair baseline.

Honest scope: single head; synthetic keys (speed only); add-only index (a real serving loop retrains the
quantizer every R blocks — centroids drift as n grows materially; noted in the JSON meta).

Run:  python3 -m ssa.ivf_decode                  # -> paper/figures/ivf_decode.json
      python3 -m ssa.ivf_decode --ns 1048576     # one point
"""
from __future__ import annotations
import json
import math
import os
import time
import torch
import torch.nn.functional as F
from ssa.ssa_kernel import BLOCK

try:
    import faiss
    import faiss.contrib.torch_utils
except ImportError as e:
    raise ImportError("ssa.ivf_decode needs faiss-gpu (pip install faiss-gpu-cu12).") from e

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_RES = None


def _gpu_res(temp_mb=512):
    global _RES
    if _RES is None:
        _RES = faiss.StandardGpuResources()
        _RES.setTempMemory(temp_mb * 1024 * 1024)
    return _RES


@torch.no_grad()
def decode_attend(qvec, k, v, blocks, pos, block=BLOCK):
    """One decode attention row over the SELECTED blocks (+ causal clip at `pos`), fp32 softmax.
    Shared primitive: with the blocks a query-block's prefill selected, this reproduces that block's last
    prefill row — the equivalence test in test_ivf_decode.py pins it. scale = 1/√d matches sdpa / flex."""
    d = qvec.shape[-1]
    idx = torch.cat([torch.arange(b * block, min((b + 1) * block, pos + 1), device=k.device)
                     for b in sorted(set(int(x) for x in blocks))])
    idx = idx[idx <= pos]
    Ksel = k.index_select(0, idx).float()
    Vsel = v.index_select(0, idx).float()
    s = (qvec.float() @ Ksel.T) / math.sqrt(d)
    w = torch.softmax(s, dim=-1)
    return (w @ Vsel).to(qvec.dtype)


@torch.no_grad()
def dense_decode(qvec, k, v, pos):
    """FAIR reference: one fp16 flash-decode row (sdpa, q_len=1) over the whole causal prefix [0..pos].
    No fp32 copy of K/V — reads ~2·(pos+1)·d·2 bytes, the strongest dense step available here.
    Bandwidth-bound, measured at every n. Pinned against `dense_decode_naive` in test_ivf_decode.py."""
    d = qvec.shape[-1]
    out = F.scaled_dot_product_attention(qvec.view(1, 1, 1, d),
                                         k[:pos + 1].view(1, 1, pos + 1, d),
                                         v[:pos + 1].view(1, 1, pos + 1, d))
    return out.view(d)


@torch.no_grad()
def dense_decode_naive(qvec, k, v, pos):
    """The PREVIOUS reference (kept for the side-by-side): upcasts the WHOLE prefix K/V to fp32 every
    step — two full-prefix fp32 allocations+copies on top of the read, ~5× slower than `dense_decode`
    at large n. Reported as dense_naive_step_ms_mean; never used for the headline speedup."""
    d = qvec.shape[-1]
    Kp = k[:pos + 1].float(); Vp = v[:pos + 1].float()
    w = torch.softmax((qvec.float() @ Kp.T) / math.sqrt(d), dim=-1)
    return (w @ Vp).to(qvec.dtype)


def _build_index(mu, nprobe=4):
    nb, d = mu.shape
    nlist = min(nb, max(4, int(nb ** 0.5)))
    ix = faiss.GpuIndexIVFFlat(_gpu_res(), d, nlist, faiss.METRIC_INNER_PRODUCT)
    ix.train(mu); ix.add(mu); ix.nprobe = min(nprobe, nlist)
    return ix


@torch.no_grad()
def _prefix(n, d, extra, g, block=BLOCK, chunk=1 << 20):
    """Prealloc k,v of length n+extra, fill [0..n) in chunks, return k, v and the completed-block means."""
    k = torch.empty(n + extra, d, device=DEV, dtype=torch.float16)
    v = torch.empty_like(k)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        k[s:e].normal_(generator=g); v[s:e].normal_(generator=g)
    nb = n // block
    mu = torch.empty(nb, d, device=DEV, dtype=torch.float32)
    for s in range(0, nb, max(1, chunk // block)):
        e = min(nb, s + max(1, chunk // block))
        mu[s:e] = k[s * block:e * block].view(e - s, block, d).mean(1, dtype=torch.float32)
    return k, v, mu


@torch.no_grad()
def bench_decode(n, d=64, block=BLOCK, top_c=8, local=1, nprobe=4, steps=128, g=None):
    """Measure per-step decode latency: SSA (IVF search + κ-gather + row) vs dense (full-prefix row)."""
    k, v, mu = _prefix(n, d, steps, g, block)
    ix = _build_index(mu, nprobe)
    nb = n // block
    qbuf = torch.randn(steps, d, generator=g, device=DEV, dtype=torch.float16)

    def sel_blocks(qvec, pos):
        _, I = ix.search(qvec.view(1, d).float(), min(2 * top_c, ix.ntotal))
        cand = [int(b) for b in I[0].tolist() if 0 <= b][:top_c]
        last = pos // block
        cand += [last - j for j in range(local + 1) if last - j >= 0]     # own + local tail blocks
        return set(cand)

    # append the decode tokens and grow the index as blocks complete (add-only)
    def do_step(t, timed_dense=False):
        pos = n + t
        k[pos] = qbuf[t]; v[pos] = qbuf[t]                                # trivial synthetic k/v for the new token
        blocks = sel_blocks(qbuf[t], pos)
        o = decode_attend(qbuf[t], k, v, blocks, pos, block)
        if (pos + 1) % block == 0:                                       # a block just completed -> index it
            b = pos // block
            ix.add(k[b * block:(b + 1) * block].mean(0, keepdim=True, dtype=torch.float32))
        return dense_decode(qbuf[t], k, v, pos) if timed_dense else o

    # Pass A — throughput (unsynced loop / steps), SSA then dense (fair fp16 sdpa) then naive (fp32-upcast)
    for _ in range(3):
        do_step(0)
    torch.cuda.synchronize(); s = time.time()
    for t in range(steps):
        do_step(t)
    torch.cuda.synchronize(); ssa_ms = (time.time() - s) / steps * 1000
    for _ in range(3):
        dense_decode(qbuf[0], k, v, n)
    torch.cuda.synchronize(); s = time.time()
    for t in range(steps):
        dense_decode(qbuf[t], k, v, n + t)
    torch.cuda.synchronize(); dense_ms = (time.time() - s) / steps * 1000
    for _ in range(3):
        dense_decode_naive(qbuf[0], k, v, n)
    torch.cuda.synchronize(); s = time.time()
    for t in range(steps):
        dense_decode_naive(qbuf[t], k, v, n + t)
    torch.cuda.synchronize(); naive_ms = (time.time() - s) / steps * 1000

    # Pass B — component medians over a short run
    def _med(fn, reps=32):
        xs = []
        for t in range(reps):
            torch.cuda.synchronize(); a = time.time(); fn(t); torch.cuda.synchronize()
            xs.append((time.time() - a) * 1000)
        xs.sort(); return xs[len(xs) // 2]
    search_ms = _med(lambda t: ix.search(qbuf[t].view(1, d).float(), min(2 * top_c, ix.ntotal)))
    attend_ms = _med(lambda t: decode_attend(qbuf[t], k, v, sel_blocks(qbuf[t], n + t), n + t, block))

    kappa = (top_c + local + 1) * block
    bw_gb = 2 * (n * d * 2) / 1e9                                        # fair dense reads k+v fp16 of the prefix
    del k, v, mu, ix, qbuf
    torch.cuda.empty_cache()
    return {"n": n, "nb": nb, "ssa_step_ms_mean": ssa_ms, "dense_step_ms_mean": dense_ms,
            "dense_naive_step_ms_mean": naive_ms, "dense_measured": True,
            "speedup": dense_ms / ssa_ms if ssa_ms else None,
            "speedup_vs_naive": naive_ms / ssa_ms if ssa_ms else None,
            "search_ms_p50": search_ms, "gather_attend_ms_p50": attend_ms,
            "kappa_keys": kappa, "dense_read_gb": bw_gb, "steps": steps}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[1048576, 2097152, 4194304, 8388608, 12582912])
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--out", default="paper/figures/ivf_decode.json")
    args = ap.parse_args()
    g = torch.Generator(device=DEV).manual_seed(0)
    print("=" * 92)
    print("IVF DECODE — per-step latency, IVF-routed SSA vs dense full-prefix (both measured), single head")
    print("=" * 92)
    print(f"  {'n':>10} {'nb':>7} {'SSA step':>9} {'dense step':>11} {'naive':>9} {'speedup':>8} "
          f"{'vs naive':>9} {'search':>8} {'attend':>8} {'κ keys':>8}")
    rows = []
    for n in args.ns:
        r = bench_decode(n, steps=args.steps, g=g)
        rows.append(r)
        print(f"  {n:>10} {r['nb']:>7} {r['ssa_step_ms_mean']:>8.3f}m {r['dense_step_ms_mean']:>10.3f}m "
              f"{r['dense_naive_step_ms_mean']:>8.3f}m "
              f"{(f'{r['speedup']:.1f}x' if r['speedup'] else '—'):>8} "
              f"{(f'{r['speedup_vs_naive']:.0f}x' if r['speedup_vs_naive'] else '—'):>9} "
              f"{r['search_ms_p50']:>7.3f}m "
              f"{r['gather_attend_ms_p50']:>7.3f}m {r['kappa_keys']:>8}", flush=True)
        _write(args, rows)
    print(f"\n  SSA decode is ~flat in n (κ fixed); dense grows with the prefix. wrote {args.out}")


def _write(args, rows):
    meta = {"H": 1, "d": 64, "block": BLOCK, "top_c": 8, "local": 1, "nprobe": 4, "steps": args.steps,
            "seed": 0, "dtype": "float16", "gpu": "RTX 4080 16GB",
            "index_policy": "prefill-build + add-on-block-complete; no retrain (steps<<n; a serving loop "
                            "would retrain the quantizer every R blocks as centroids drift)",
            "dense_reference": "FAIR: measured fp16 flash-decode row (sdpa q_len=1) over the full prefix, "
                               "no fp32 K/V copy (reads ~2·n·d·2 bytes, in dense_read_gb). The previous "
                               "fp32-upcasting reference (copies whole-prefix K and V to fp32 every step, "
                               "~5x slower) is reported as dense_naive_step_ms_mean; the headline speedup "
                               "uses the fair baseline"}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"meta": meta, "rows": rows}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
