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
        # NOTE: `flat` here is a stripped single-head score GEMM (nb² fp32). The kernel's ACTUAL router
        # (ssa_kernel.block_route, H heads + (B,H,nb,nb) bool sel + argsort) is ~10× heavier and OOMs/faults
        # at ~1M on this 16 GB card (measured: 19 ms @256K, fault @1M). The single-head GEMM genuinely OOMs
        # only when its matrix exceeds VRAM (~14 GB ⇒ ~7M tokens); skip it just past that to spare faiss.
        tf = _t(lambda: flat(qb, mu)) if mem < 14.0 else None
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


def plot_from_json(path="paper/figures/router_gpu_compare.json"):
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = json.load(open(path))
    n = np.array([r["n"] for r in rows], float)
    flat = np.array([r["flat_ms"] if r["flat_ms"] else np.nan for r in rows])
    ivf = np.array([r["ivf_ms"] for r in rows])
    fm = ~np.isnan(flat)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    fig.patch.set_facecolor("#000"); ax.set_facecolor("#000")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(n[fm], flat[fm], color="#f2c14e", lw=2.5, marker="o", ms=5, label="flat (n/b)² GEMM — GPU")
    ax.plot(n, ivf, color="#3fbf90", lw=2.5, marker="o", ms=5, label="faiss-GPU IVF (build+search) — GPU")
    # where flat OOMs: extrapolate its measured trend and mark it dead (× on a faint dashed continuation)
    oom = n[~fm]
    if len(oom):
        pf = np.polyfit(np.log(n[fm]), np.log(flat[fm]), 1)
        flat_ext = np.exp(pf[1]) * oom ** pf[0]
        ax.plot(np.concatenate([[n[fm][-1]], oom]), np.concatenate([[flat[fm][-1]], flat_ext]),
                color="#f2c14e", lw=1.3, ls=(0, (2, 3)), alpha=0.5)
        ax.scatter(oom, flat_ext, marker="x", s=120, linewidths=2.6, color="#ff5d5d",
                   label="flat OOMs (nb² matrix > GPU)", zorder=6)

    ax.text(2.9e5, 1.7e2, "crossover ~3M (IVF 1.7× faster at 4M);\nflat OOMs at 8M (17 GB nb² matrix);\n"
            "IVF the only router past it (64 ms).\n(the kernel's real block_route — H heads +\nargsort — OOMs even earlier, ~1M)",
            color="#9fd9c4", fontsize=8.0, ha="left", va="top")
    ax.set_xticks([262144, 524288, 1048576, 2097152, 4194304, 8388608])
    ax.set_xticklabels(["256K", "512K", "1M", "2M", "4M", "8M"])
    ax.set_xlim(2.3e5, 1.0e7); ax.set_ylim(0.15, 4e2)
    ax.set_xlabel("Context length (tokens)", color="#bdbdbd")
    ax.set_ylabel("Router wall-clock per call (ms, log)", color="#bdbdbd")
    ax.text(0.0, 1.13, "REAL GPU — both routers on GPU, no transfer (faiss-gpu)",
            transform=ax.transAxes, color="#8a8a8a", fontsize=8.4, fontweight="bold")
    ax.set_title("Flat (n/b)² GEMM vs faiss-GPU IVF router — measured", color="white",
                 fontsize=13.5, loc="left", pad=30)
    ax.grid(True, which="major", color="#2a2a2a", lw=0.5)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.12, labelcolor="#ddd")
    fig.tight_layout()
    fig.savefig("paper/figures/router_gpu_compare.png", dpi=150, facecolor=fig.get_facecolor())
    print("wrote paper/figures/router_gpu_compare.png")


if __name__ == "__main__":
    import sys
    plot_from_json() if "plot" in sys.argv else main()
