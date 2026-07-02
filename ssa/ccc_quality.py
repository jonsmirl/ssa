"""
The trilemma-completing table: which CCC component rescues which failure regime. For each retrieval
regime we plant a needle a probe query-block should select, and measure needle-BLOCK selection recall for
each selector variant — showing sub-block granularity rescues isolated needles and the outlier channel
rescues high-norm spikes, where the plain block-IVF router misses them.

Regimes:  benign (coherent span) | isolated (lone unit-norm key) | spike (k=c·q, high norm)
Variants: ivf (block-level centroid) | +subblock (32) | +outlier (32 + side-channel)

`--fire-rate` instead measures the certificate gate-rate per geometry (soundness is pinned in tests).

Run:  python3 -m ssa.ccc_quality                     # -> paper/figures/ccc_quality.json
"""
from __future__ import annotations
import json
import os
import numpy as np
import torch
from ssa.ssa_kernel import BLOCK

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SUB = 32


def _variant(sub, outlier_cap):
    from ssa.cascade_router import CausalCascade
    return dict(block=BLOCK, sub=sub, top_c=2, local=0, outlier_cap=outlier_cap, search_k=256)


VARIANTS = {"ivf": (BLOCK, 0), "+subblock": (SUB, 0), "+outlier": (SUB, 4)}


@torch.no_grad()
def _trial(regime, n, target_blk, c, g):
    """Plant one needle in `target_blk` addressed by a coherent probe query in the last block; return
    (q,k) and the target. The probe query-block points at a random unit u; the needle aligns with u."""
    from ssa.cascade_router import CausalCascade
    d = 64
    q = torch.nn.functional.normalize(torch.randn(n, d, generator=g, device=DEV), dim=1).half()
    k = torch.nn.functional.normalize(torch.randn(n, d, generator=g, device=DEV), dim=1).half()
    u = torch.nn.functional.normalize(torch.randn(d, generator=g, device=DEV), dim=0).half()
    q[n - BLOCK:] = u                                              # coherent probe query block
    lo = target_blk * BLOCK
    if regime == "benign":
        k[lo:lo + 24] = u                                         # a coherent span (24 aligned keys)
    elif regime == "isolated":
        k[lo + 7] = u                                            # one lone unit-norm key
    elif regime == "spike":
        k[lo + 7] = c * u                                        # one high-norm key (the impossibility case)
    hit = {}
    for name, (sub, cap) in VARIANTS.items():
        cc = CausalCascade(d, **_variant(sub, cap))
        cc.append(k)
        kn, ki, _, _ = cc.route(q, qpos=0, search_k=256)
        last = n // BLOCK - 1
        hit[name] = int(target_blk in ki[last, :int(kn[last])].tolist())
    return hit


def quality_matrix(ns=(65536,), trials=40, cs=(2, 4, 8)):
    rng = np.random.default_rng(0)
    rows = []
    print(f"  {'n':>8} {'regime':>10} {'c':>3} {'ivf':>6} {'+subblock':>10} {'+outlier':>9} {'trials':>7}")
    for n in ns:
        nb = n // BLOCK
        for regime in ("benign", "isolated", "spike"):
            for c in (cs if regime == "spike" else (0,)):
                g = torch.Generator(device=DEV).manual_seed(0)
                agg = {k: 0 for k in VARIANTS}
                for _ in range(trials):
                    tb = int(rng.integers(1, nb - 1))
                    h = _trial(regime, n, tb, c, g)
                    for kk in VARIANTS:
                        agg[kk] += h[kk]
                row = {"n": n, "regime": regime, "c": c, "trials": trials,
                       **{kk: agg[kk] / trials for kk in VARIANTS}}
                rows.append(row)
                print(f"  {n:>8} {regime:>10} {c:>3} {row['ivf']:>6.2f} {row['+subblock']:>10.2f} "
                      f"{row['+outlier']:>9.2f} {trials:>7}", flush=True)
    return rows


@torch.no_grad()
def fire_rate(ns=(1 << 20,), trials=1):
    """Certificate gate-rate per geometry (soundness itself is pinned in test_ccc_certificates.py)."""
    from ssa.cascade_router import CausalCascade
    rows = []
    print(f"  {'n':>10} {'geometry':>10} {'cert_rate':>10} {'mean_rounds':>12}")
    for n in ns:
        for geom in ("clustered", "random"):
            g = torch.Generator(device=DEV).manual_seed(0)
            if geom == "random":
                q = torch.randn(1, 1, n, 64, generator=g, device=DEV, dtype=torch.float16)
                k = torch.randn(1, 1, n, 64, generator=g, device=DEV, dtype=torch.float16)
            else:
                nb4 = n // SUB; nc = max(8, int(nb4 ** 0.5))
                ctr = torch.randn(nc, 64, generator=g, device=DEV)
                a = torch.randint(0, nc, (nb4,), generator=g, device=DEV)
                k = (ctr[a].repeat_interleave(SUB, 0) + 0.1 * torch.randn(n, 64, generator=g, device=DEV)).half().view(1, 1, n, 64)
                nbq = n // BLOCK
                qc = ctr[torch.randint(0, nc, (nbq,), generator=g, device=DEV)]
                q = (qc.repeat_interleave(BLOCK, 0) + 0.2 * torch.randn(n, 64, generator=g, device=DEV)).half().view(1, 1, n, 64)
            cc = CausalCascade(64, n_hint=n, chunk_blocks=1024)
            cb = 1024 * BLOCK; certs, rounds = [], []
            for t in range(0, n, cb):
                e = min(n, t + cb)
                cc.append(k[0, 0, t:e])
                _, _, cert, st = cc.route(q[0, 0, t:e], qpos=t, certify=True)
                certs.append(cert.float().mean().item()); rounds.append(st["rounds"].float().mean().item())
            rows.append({"n": n, "geometry": geom, "cert_rate": float(np.mean(certs)),
                         "mean_rounds": float(np.mean(rounds))})
            print(f"  {n:>10} {geom:>10} {np.mean(certs):>10.3f} {np.mean(rounds):>12.3f}", flush=True)
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fire-rate", action="store_true")
    ap.add_argument("--ns", type=int, nargs="+", default=[65536, 1048576])
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--out", default="paper/figures/ccc_quality.json")
    args = ap.parse_args()
    print("=" * 92)
    print("CCC QUALITY — which component rescues which regime" if not args.fire_rate
          else "CCC — certificate fire-rate per geometry")
    print("=" * 92)
    if args.fire_rate:
        rows = fire_rate(tuple(args.ns))
        kind = "fire_rate"
    else:
        rows = quality_matrix(tuple(args.ns), args.trials)
        kind = "quality"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    payload = {"meta": {"block": BLOCK, "sub": SUB, "kind": kind, "seed": 0}, "rows": rows}
    prev = json.load(open(args.out)) if os.path.exists(args.out) else {"meta": {}, "rows": []}
    if kind == "quality":
        json.dump(payload, open(args.out, "w"), indent=2)
    else:                                                          # keep quality rows, add fire-rate
        prev.setdefault("fire_rate", rows)
        json.dump(prev, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
