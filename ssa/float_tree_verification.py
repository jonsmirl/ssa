"""Stress-test outward-rounded tree radii and score caps against float64 descendant oracles.

This is numerical verification, not a formal proof about CUDA arithmetic.  It exercises the same
``CausalTree`` construction used by the long-context runner and compares both its guarded values and the
former unguarded float32 formulas on scale-separated and cancellation-heavy geometries.

Run on CUDA with::

    python -m ssa.float_tree_verification --out runs/float_tree_verification.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch

from .cascade_router import CausalTree, _outward_nonnegative, _outward_score_cap


def _raw_levels(leaves, fanout):
    levels, radii = [leaves], [torch.zeros(len(leaves), device=leaves.device)]
    while len(levels[-1]) > 1:
        children = levels[-1].view(-1, fanout, leaves.shape[1])
        child_r = radii[-1].view(-1, fanout)
        center = children.mean(1)
        radius = ((children - center[:, None]).norm(dim=-1) + child_r).amax(1)
        levels.append(center)
        radii.append(radius)
    return levels, radii


def _guarded_formula_levels(leaves, fanout):
    levels, radii = [leaves], [torch.zeros(len(leaves), device=leaves.device)]
    while len(levels[-1]) > 1:
        children = levels[-1].view(-1, fanout, leaves.shape[1])
        child_r = radii[-1].view(-1, fanout)
        center = children.mean(1)
        candidate = (children - center[:, None]).norm(dim=-1) + child_r
        radius = _outward_nonnegative(candidate, leaves.shape[1]).amax(1)
        radius = torch.nextafter(radius, torch.full_like(radius, float("inf")))
        levels.append(center)
        radii.append(radius)
    return levels, radii


def _guarded_tree(leaves, fanout):
    tree = CausalTree(
        leaves.shape[1], block=1, sub=1, query_sub=1, n_hint=len(leaves),
        tree_fanout=fanout, tree_beam=1, search_k=1, top_c=1, local=0,
        outlier_rate=0, outlier_cap=0,
    )
    tree.stage = leaves
    tree.n_sub = len(leaves)
    tree._commit()
    depth = 1
    while depth < len(tree.levels) and tree.counts[depth]:
        depth += 1
    return tree.levels[:depth], tree.radii[:depth]


def _cases(n, d, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    normal = torch.randn(n, d, generator=g, device=device, dtype=torch.float32)
    row_scale = torch.pow(2.0, torch.linspace(-40, 40, n, device=device)).unsqueeze(1)
    wide = torch.randn(n, d, generator=g, device=device, dtype=torch.float32) * row_scale
    offset = torch.full((n, d), 2.0**16, device=device)
    offset += 2.0**-5 * torch.randn(n, d, generator=g, device=device)
    signs = torch.where(torch.arange(n, device=device)[:, None] % 2 == 0, 1.0, -1.0)
    cancellation = signs * (2.0**18) + torch.randn(n, d, generator=g, device=device)
    axis = torch.zeros(n, d, device=device)
    axis[torch.arange(n, device=device), torch.arange(n, device=device) % d] = row_scale[:, 0]
    return {
        "normal": normal,
        "wide_scale": wide,
        "large_offset": offset,
        "cancellation": cancellation,
        "axis_scale": axis,
    }


def _timed_build(leaves, fanout, guarded, repeats=5):
    build = _guarded_formula_levels if guarded else _raw_levels
    for _ in range(2):
        build(leaves, fanout)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        build(leaves, fanout)
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - started) / repeats


def verify_case(leaves, fanout, queries=12, seed=0):
    raw_levels, raw_radii = _raw_levels(leaves, fanout)
    safe_levels, safe_radii = _guarded_tree(leaves, fanout)
    if len(raw_levels) != len(safe_levels):
        raise AssertionError("raw and guarded trees have different depths")
    g = torch.Generator(device=leaves.device).manual_seed(seed + 10_000)
    Q = torch.randn(queries, leaves.shape[1], generator=g, device=leaves.device)

    raw_radius_violations = safe_radius_violations = 0
    raw_cap_violations = safe_cap_violations = 0
    checked_nodes = checked_caps = 0
    max_raw_radius_deficit = max_safe_radius_deficit = 0.0
    max_raw_cap_deficit = max_safe_cap_deficit = 0.0
    max_relative_inflation = 0.0

    span = 1
    leaves64 = leaves.double()
    q64 = Q.double()
    for level, (raw_c, raw_r, safe_c, safe_r) in enumerate(
            zip(raw_levels, raw_radii, safe_levels, safe_radii)):
        if not torch.equal(raw_c, safe_c):
            raise AssertionError(f"guard changed centers at level {level}")
        members = leaves64.view(len(raw_c), span, leaves.shape[1])
        true_radius = (members - raw_c.double()[:, None]).norm(dim=2).amax(1)
        raw_deficit = true_radius - raw_r.double()
        safe_deficit = true_radius - safe_r.double()
        raw_radius_violations += int((raw_deficit > 0).sum())
        safe_radius_violations += int((safe_deficit > 0).sum())
        max_raw_radius_deficit = max(max_raw_radius_deficit, float(raw_deficit.max()))
        max_safe_radius_deficit = max(max_safe_radius_deficit, float(safe_deficit.max()))
        checked_nodes += len(raw_c)

        raw_cap = ((Q[:, None] * raw_c).sum(2)
                   + Q.norm(dim=1)[:, None] * raw_r[None]).double()
        safe_cap = _outward_score_cap(Q, safe_c, safe_r).double()
        true_best = torch.einsum("qd,nsd->qns", q64, members).amax(2)
        raw_gap = true_best - raw_cap
        safe_gap = true_best - safe_cap
        raw_cap_violations += int((raw_gap > 0).sum())
        safe_cap_violations += int((safe_gap > 0).sum())
        max_raw_cap_deficit = max(max_raw_cap_deficit, float(raw_gap.max()))
        max_safe_cap_deficit = max(max_safe_cap_deficit, float(safe_gap.max()))
        checked_caps += queries * len(raw_c)

        denom = raw_r.clamp_min(torch.finfo(torch.float32).tiny)
        relative = (safe_r - raw_r) / denom
        max_relative_inflation = max(max_relative_inflation, float(relative.max()))
        span *= fanout

    return {
        "leaves": len(leaves),
        "dimension": leaves.shape[1],
        "fanout": fanout,
        "depth": len(raw_levels),
        "nodes_checked": checked_nodes,
        "caps_checked": checked_caps,
        "raw_radius_violations": raw_radius_violations,
        "guarded_radius_violations": safe_radius_violations,
        "raw_cap_violations": raw_cap_violations,
        "guarded_cap_violations": safe_cap_violations,
        "max_raw_radius_deficit": max_raw_radius_deficit,
        "max_guarded_radius_deficit": max_safe_radius_deficit,
        "max_raw_cap_deficit": max_raw_cap_deficit,
        "max_guarded_cap_deficit": max_safe_cap_deficit,
        "max_relative_radius_inflation": max_relative_inflation,
    }


def run(out=None, n=4096, d=64, fanouts=(2, 4, 16), seed=0, bench_n=65536):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU floating-point verification")
    rows = []
    with torch.no_grad():
        for fanout in fanouts:
            usable = fanout ** int(math.floor(math.log(n, fanout)))
            for name, leaves in _cases(usable, d, "cuda", seed + fanout).items():
                row = verify_case(leaves, fanout, seed=seed + fanout)
                row["geometry"] = name
                rows.append(row)
        g = torch.Generator(device="cuda").manual_seed(seed + 99)
        bench_leaves = torch.randn(bench_n, d, generator=g, device="cuda",
                                   dtype=torch.float32)
        timing = {
            "leaves": bench_n,
            "dimension": d,
            "fanout": 16,
            "raw_ms": _timed_build(bench_leaves, 16, False),
            "guarded_ms": _timed_build(bench_leaves, 16, True),
        }
        timing["overhead_ratio"] = timing["guarded_ms"] / timing["raw_ms"]

    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": "float32",
        "guard": "8*(dimension+4)*eps plus nextafter(+inf)",
        "rows": rows,
        "totals": {
            key: sum(row[key] for row in rows)
            for key in ("nodes_checked", "caps_checked", "raw_radius_violations",
                        "guarded_radius_violations", "raw_cap_violations",
                        "guarded_cap_violations")
        },
        "timing": timing,
        "scope": "empirical CUDA verification against float64 oracles; not an IEEE proof",
    }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/float_tree_verification.json")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--fanouts", default="2,4,16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bench-n", type=int, default=65536)
    args = parser.parse_args()
    result = run(args.out, args.n, args.d, tuple(map(int, args.fanouts.split(","))),
                 args.seed, args.bench_n)
    print(json.dumps({"device": result["device"], "totals": result["totals"],
                      "timing": result["timing"]}, indent=2))


if __name__ == "__main__":
    main()
