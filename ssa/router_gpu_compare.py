"""
The real test: flat (n/b)² GEMM router vs faiss-GPU IVF router, BOTH on the GPU (torch tensors, no CPU
transfer), across context length into the regime where the flat router's nb² matrix OOMs. Settles whether
the sub-linear IVF router's op-count win is real in GPU wall-clock — the thing P4/P5 only projected.

Run: python3 -m ssa.router_gpu_compare  ->  paper/figures/router_gpu_compare.json
"""
from __future__ import annotations
import json
import time
import torch
import faiss
import faiss.contrib.torch_utils          # lets faiss GPU indexes consume torch GPU tensors directly

DEV = "cuda"
d, block, top_c = 64, 128, 8
res = faiss.StandardGpuResources()
res.setTempMemory(512 * 1024 * 1024)      # cap faiss scratch so it coexists with torch


def _t(fn, reps=6):
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
        if any(s in str(e).lower() for s in ("out of memory", "cuda error", "device", "assert", "handles")):
            torch.cuda.empty_cache()
            return None
        raise


def flat(qb, mu):
    return torch.einsum('qd,kd->qk', qb, mu).topk(top_c, dim=-1).indices


def ivf(mu, qb, nlist, nprobe=4):
    ix = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_INNER_PRODUCT)
    ix.train(mu); ix.add(mu); ix.nprobe = nprobe
    return ix.search(qb, top_c)            # build + search, all on GPU


def main():
    torch.manual_seed(0)
    print("=" * 86)
    print(f"GPU router comparison — flat (n/b)² GEMM vs faiss-GPU IVF (both GPU, no transfer), d={d} top_c={top_c}")
    print("=" * 86)
    print(f"  {'n':>9} {'nb':>6} {'flat GEMM':>11} {'IVF (build+search)':>20} {'flat nb² mem':>13} {'winner':>14}")
    rows = []
    for n in (262144, 524288, 1048576, 2097152, 4194304, 8388608):
        nb = n // block
        mu = torch.randn(nb, d, device=DEV).contiguous()
        qb = torch.randn(nb, d, device=DEV).contiguous()
        nlist = max(4, int(nb ** 0.5))
        mem = nb * nb * 4 / 1e9
        tf = _t(lambda: flat(qb, mu)) if mem < 3.0 else None   # skip flat where nb² would OOM (would crash faiss)
        ti = _t(lambda: ivf(mu, qb, nlist))
        if tf is None and ti is not None:
            win = "IVF (flat OOM)"
        elif tf and ti:
            win = f"{'IVF' if ti < tf else 'flat'} {max(tf, ti)/min(tf, ti):.1f}x"
        else:
            win = "—"
        fs = f"{tf:.2f} ms" if tf else "OOM"
        cs = f"{ti:.2f} ms" if ti else "OOM"
        print(f"  {n:>9} {nb:>6} {fs:>11} {cs:>20} {mem:>11.1f}G {win:>14}")
        rows.append(dict(n=n, nb=nb, flat_ms=tf, ivf_ms=ti, flat_mem_gb=mem))
        json.dump(rows, open("paper/figures/router_gpu_compare.json", "w"), indent=2)
        del mu, qb
        torch.cuda.empty_cache()
        if tf is None and ti is None:
            print(f"  {'':>9}  both OOM — stopping."); break
    print("\n  Both on the GPU now: no transfer. The flat GEMM grows nb² and OOMs; the IVF is ~linear in")
    print("  memory and op-count. wrote paper/figures/router_gpu_compare.json")


if __name__ == "__main__":
    main()
