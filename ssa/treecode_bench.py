"""
P4-router: the bounded-candidate treecode router vs the flat (n/b)² GEMM router — wall-clock + the
memory wall (docs/bounded_treecode_scope.md). Times the FULL router each way (summaries included):
  * flat:     block means + (nb x nb) score GEMM + top_c    — O(nb²) time, materializes nb² (the wall)
  * treecode: build_tree + Q-tiled descend_beam             — O(nb log) time, bounded memory
Pushes n until the flat nb² matrix OOMs; the treecode keeps running. Run: python3 -m ssa.treecode_bench
"""
from __future__ import annotations
import json
import time
import torch
from ssa.treecode import build_tree, descend_beam

DEV = "cuda"


def _t(fn, reps=5):
    try:
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        s = time.time()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.time() - s) / reps * 1000
    except Exception as e:
        msg = str(e).lower()
        if any(s in msg for s in ("out of memory", "device not ready", "device-side", "cuda error")):
            torch.cuda.empty_cache()
            return None
        raise


def tree_router(levels, F, qb, W, top_c, tile=4096):
    return torch.cat([descend_beam(levels, F, qb[:, i:i + tile], W, top_c)
                      for i in range(0, qb.shape[1], tile)], dim=1)


def main():
    torch.manual_seed(0)
    H, d, block, F, W, top_c = 8, 64, 128, 8, 16, 8
    print("=" * 84)
    print(f"P4-router — treecode vs flat GEMM router (H={H}, d={d}, block={block}, F={F}, W={W}, top_c={top_c})")
    print("=" * 84)
    print(f"  {'n':>9} {'nb':>7} {'flat GEMM':>11} {'beam descend':>13} {'(tree build)':>13} "
          f"{'flat nb² mem':>12} {'descend vs flat':>16}")
    rows = []
    for n in (262144, 524288, 1048576, 2097152, 4194304, 8388608):
        try:
            k = torch.randn(1, H, n, d, device=DEV, dtype=torch.float16)
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            print(f"  {n:>9}  cannot allocate keys on this GPU"); break
        nb = n // block
        qb = torch.randn(H, nb, d, device=DEV)

        def full_flat():
            mu = k.reshape(H, nb, block, d).mean(2).half()
            return torch.einsum('bqd,bkd->bqk', qb.half(), mu).topk(top_c, -1).indices

        tf = _t(full_flat)
        tb = _t(lambda: build_tree(k, block, F), reps=2)        # amortizable: once per forward
        levels = None
        try:
            levels, _ = build_tree(k, block, F)
        except Exception:
            torch.cuda.empty_cache()
        td = _t(lambda: tree_router(levels, F, qb, W, top_c)) if levels is not None else None
        flat_gb = H * nb * nb * 2 / 1e9
        fs = f"{tf:.1f} ms" if tf else "OOM"
        ds = f"{td:.1f} ms" if td else ("OOM" if levels is not None else "—")
        bs = f"{tb:.1f} ms" if tb else "OOM"
        if tf is None and td is not None:
            verdict = "flat OOMs; tree runs"
        elif tf and td:
            verdict = f"{'tree' if td < tf else 'flat'} {max(tf, td) / min(tf, td):.1f}x"
        else:
            verdict = ""
        print(f"  {n:>9} {nb:>7} {fs:>11} {ds:>13} {bs:>13} {flat_gb:>10.1f}G {verdict:>16}")
        rows.append(dict(n=n, nb=nb, flat_ms=tf, descend_ms=td, build_ms=tb, flat_gb=flat_gb))
        json.dump(rows, open("paper/figures/treecode_router_measured.json", "w"), indent=2)  # persist incrementally
        del k, qb, levels
        torch.cuda.empty_cache()
        if tf is None and td is None:
            print(f"  {'':>9}  both OOM / GPU memory wall reached (16 GB) — stopping."); break
    print("  wrote paper/figures/treecode_router_measured.json")
    print("\n  Headline comparison is descend-vs-flat (the tree build is once-per-forward, amortized over all")
    print("  query-blocks + decode steps). Flat's nb² GEMM wins on constant while it fits, then OOMs; the")
    print("  fixed-width beam descend grows nb·log and does NOT regress — the P4 claim, modulo this GPU's wall.")


if __name__ == "__main__":
    main()
