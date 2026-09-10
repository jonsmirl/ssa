"""Score-tail certificate experiment on synthetic and cached Qwen geometry.

The hard comparison uses identical queries and causal prefixes for every mode.
CCC-style block-mean routing supplies only initial blocks.  The new attention
certificate independently evaluates admissible block score caps, compresses
unopened key counts into score bands, and sends the resulting mass cap through
the restricted-read TV/KL/output algebra.

Run locally with the cached fixture::

    python -m ssa.score_tail_experiment \
      --cache /tmp/ssa_bennett_qwen_8192.npz \
      --out runs/score_tail_certificate.json

The cache contains Q/K geometry but not values, so the Qwen mass measurements
are real post-RoPE geometry while its output-error measurement uses a declared
deterministic proxy value map.  This is float64 verification, not an IEEE proof.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from .bennett_mass_experiment import _actual_omitted_mass
from .hierarchical_certified_attention import CertifiedTreeAttention
from .score_tail_certificate import ScoreTailCertifiedAttention


TREE_MODES = {
    "radius": ("radius", 0),
    "bennett_trace": ("bennett", 0),
    "bennett_covariance": ("bennett_covariance", 0),
    "best_existing_peeled": ("bennett_peel_covariance", 4),
}


def _actual_output_error(K, V, q, beta, prefix, result):
    logits = beta * (K[:prefix] @ q)
    top = float(logits.max())
    weights = np.exp(logits - top)
    dense = (weights / weights.sum()) @ V[:prefix]
    return float(np.linalg.norm(dense - result.output))


def _summary(rows, elapsed, *, tail_levels, storage_scalars, query_work):
    a = np.asarray(rows, dtype=np.float64)
    mass_bad = a[:, 5] > 2e-11
    output_bad = a[:, 6] > 2e-11
    margins = a[np.isfinite(a[:, 8]), 8] if a.shape[1] > 8 else np.empty(0)
    return {
        "queries": len(rows),
        "soundness_violations": int(np.sum(mass_bad | output_bad)),
        "mass_soundness_violations": int(np.sum(mass_bad)),
        "output_soundness_violations": int(np.sum(output_bad)),
        "maximum_mass_deficit": float(np.maximum(a[:, 5], 0).max(initial=0)),
        "maximum_output_deficit": float(np.maximum(a[:, 6], 0).max(initial=0)),
        "mean_certified_omitted_mass": float(a[:, 0].mean()),
        "median_certified_omitted_mass": float(np.median(a[:, 0])),
        "mean_actual_omitted_mass": float(a[:, 1].mean()),
        "median_actual_omitted_mass": float(np.median(a[:, 1])),
        "mean_keys_scored": float(a[:, 2].mean()),
        "mean_blocks_opened": float(a[:, 3].mean()),
        "mean_node_or_bound_evaluations": float(a[:, 4].mean()),
        "certified_fraction": float(a[:, 7].mean()),
        "mean_finite_certificate_margin": (float(margins.mean()) if len(margins) else None),
        "tail_levels": tail_levels,
        "storage_scalars": int(storage_scalars),
        "mean_query_work_units": float(np.mean(query_work)),
        "elapsed_seconds": elapsed,
    }


def _evaluate_hard_modes(K, V, Q, positions, beta, eta, block, route_fraction, tail_levels,
                         max_block_fraction=None):
    tail_index = ScoreTailCertifiedAttention(K, V, block)
    routes = []
    for q, pos in zip(Q, positions):
        visible = math.ceil((int(pos) + 1) / block)
        seeds = max(1, math.ceil(route_fraction * visible))
        routes.append(tail_index.route_by_block_mean(q, seeds, prefix=int(pos) + 1))

    result = {}
    for label, (mode, peel_count) in TREE_MODES.items():
        started = time.perf_counter()
        index = CertifiedTreeAttention(K, V, block, mass_bound=mode, peel_count=peel_count)
        rows, work = [], []
        for q, pos, route in zip(Q, positions, routes):
            prefix = int(pos) + 1
            visible = math.ceil(prefix / block)
            max_blocks = (None if max_block_fraction is None else
                          max(len(route.block_indices) + bool(prefix % block),
                              math.ceil(max_block_fraction * visible)))
            read = index.read(q, beta=beta, prefix=prefix, mass_tol=eta,
                              initial_blocks=route.block_indices, max_blocks=max_blocks)
            actual = _actual_omitted_mass(K, q, beta, prefix, read.indices)
            actual_error = _actual_output_error(K, V, q, beta, prefix, read)
            rows.append((read.mass_upper, actual, read.keys_scored, read.blocks_opened,
                         read.bounds_evaluated, actual - read.mass_upper,
                         actual_error - read.output_error_upper, float(read.certified), np.nan))
            bound_charge = (K.shape[1] if mode in ("radius", "bennett") else
                            K.shape[1] ** 2 if mode == "bennett_covariance" else
                            K.shape[1] ** 2 + peel_count * K.shape[1])
            work.append(read.bounds_evaluated * bound_charge
                        + read.keys_scored * K.shape[1])
        nodes = len(index._nodes)
        base = nodes * (2 * K.shape[1] + 2)  # key/value means and radii
        extra = (0 if mode == "radius" else nodes if mode == "bennett" else
                 nodes * K.shape[1] ** 2 if mode == "bennett_covariance" else
                 nodes * (4 * (K.shape[1] + 1) + K.shape[1] + 2 + K.shape[1] ** 2))
        result[label] = _summary(rows, time.perf_counter() - started, tail_levels=0,
                                 storage_scalars=base + extra, query_work=work)

    for label, levels in (("one_threshold_block_cap", 1),
                          ("multi_threshold_tail_profile", tail_levels)):
        started = time.perf_counter()
        rows, work = [], []
        for q, pos, route in zip(Q, positions, routes):
            prefix = int(pos) + 1
            visible = math.ceil(prefix / block)
            max_blocks = (None if max_block_fraction is None else
                          max(len(route.block_indices) + bool(prefix % block),
                              math.ceil(max_block_fraction * visible)))
            read = tail_index.read(q, beta=beta, prefix=prefix, routing=route,
                                   tail_levels=levels, mass_tol=eta, max_blocks=max_blocks)
            actual = _actual_omitted_mass(K, q, beta, prefix, read.indices)
            actual_error = _actual_output_error(K, V, q, beta, prefix, read)
            rows.append((read.mass_upper, actual, read.keys_scored, read.blocks_opened,
                         read.bounds_evaluated, actual - read.mass_upper,
                         actual_error - read.output_error_upper, float(read.certified),
                         read.certificate_margin))
            work.append(read.tail_query_work + read.keys_scored * K.shape[1])
        nb = math.ceil(len(K) / block)
        storage = nb * (2 * K.shape[1] + 2) + 2 * levels
        result[label] = _summary(rows, time.perf_counter() - started, tail_levels=levels,
                                 storage_scalars=storage, query_work=work)
    return result


def _oracle_floor(K, Q, positions, beta, eta, block):
    exact_rows, max_rows = [], []
    for q, pos in zip(Q, positions):
        prefix = int(pos) + 1
        logits = beta * (K[:prefix] @ q)
        order = np.lexsort((-np.arange(prefix), -logits))
        sorted_logits = logits[order]
        scaled = np.exp(sorted_logits - sorted_logits[0])
        cumulative = np.cumsum(scaled)
        total = float(cumulative[-1])
        exact_k = int(np.searchsorted(cumulative, (1 - eta) * total, side="left") + 1)
        one_k = prefix
        for k in range(1, prefix):
            residual_cap = (prefix - k) * scaled[k]
            if residual_cap / (cumulative[k - 1] + residual_cap) <= eta:
                one_k = k
                break
        exact_rows.append((prefix, exact_k, len(np.unique(order[:exact_k] // block))))
        max_rows.append((prefix, one_k, len(np.unique(order[:one_k] // block))))

    def pack(rows, label, levels):
        a = np.asarray(rows, dtype=np.float64)
        return {
            "oracle_only": True,
            "meaning": label,
            "median_keys_retained_fraction": float(np.median(a[:, 1] / a[:, 0])),
            "mean_keys_retained": float(a[:, 1].mean()),
            "mean_blocks_touched": float(a[:, 2].mean()),
            "mean_keys_scored_to_construct_oracle": float(a[:, 0].mean()),
            "tail_levels": levels,
            "soundness_violations": 0,
            "certified_fraction": 1.0,
        }
    return {
        "exact_residual_mass_floor": pack(
            exact_rows, "exact histogram/true residual mass after exact top-key retention", "exact"),
        "exact_top_keys_plus_one_residual_max": pack(
            max_rows, "N_remaining * exp(exact largest residual score)", 1),
    }


def _fixed_budget_finding(K, V, Q, positions, beta, block, fraction=0.10):
    index = CertifiedTreeAttention(K, V, block, mass_bound="radius")
    rows = []
    for q, pos in zip(Q, positions):
        prefix = int(pos) + 1
        visible = math.ceil(prefix / block)
        cap = max(1, math.ceil(fraction * visible))
        read = index.read(q, beta=beta, prefix=prefix, max_blocks=cap, mass_tol=0)
        rows.append((read.blocks_opened / visible,
                     _actual_omitted_mass(K, q, beta, prefix, read.indices),
                     read.mass_upper))
    a = np.asarray(rows)
    return {
        "target_block_fraction": fraction,
        "mean_opened_block_fraction": float(a[:, 0].mean()),
        "median_actual_omitted_mass": float(np.median(a[:, 1])),
        "mean_actual_omitted_mass": float(a[:, 1].mean()),
        "median_certified_omitted_mass": float(np.median(a[:, 2])),
    }


def _synthetic(kind, n, d, queries, seed):
    rng = np.random.default_rng(seed)
    if kind == "concentrated":
        K = 0.04 * rng.normal(size=(n, d))
        K[:64, 0] += 55
        Q = 0.05 * rng.normal(size=(queries, d)); Q[:, 0] += 1
    elif kind == "random":
        K, Q = rng.normal(size=(n, d)), rng.normal(size=(queries, d))
    elif kind == "adversarial":
        K = 0.02 * rng.normal(size=(n, d))
        for start in range(0, n, 64):
            K[start, 0] += 14
            K[start + 1:start + 8, 0] -= 2
        Q = 0.02 * rng.normal(size=(queries, d)); Q[:, 0] += 1
    else:
        raise ValueError(kind)
    V = np.tanh(K[:, :8])
    positions = np.linspace(n // 2, n - 1, queries, dtype=int)
    return K, V, Q, positions


def run(out, cache="/tmp/ssa_bennett_qwen_8192.npz", queries=32, block=64,
        tail_levels=16, route_fraction=0.10, seed=0):
    result = {
        "method": "geometry-routed deterministic attention score-tail certificate",
        "command": ("python -m ssa.score_tail_experiment "
                    f"--cache {cache} --out {out}"),
        "scope": "CPU float64 reference; direct block attention caps; no FlexAttention integration",
        "parameters": {"queries": queries, "block": block, "tail_levels": tail_levels,
                       "route_fraction": route_fraction, "seed": seed},
        "synthetic": {},
    }
    for kind in ("concentrated", "random", "adversarial"):
        K, V, Q, positions = _synthetic(kind, 2048, 64, queries, seed)
        beta = 1 / math.sqrt(K.shape[1])
        result["synthetic"][kind] = {
            "eta_0.10": _evaluate_hard_modes(
                K, V, Q, positions, beta, 0.10, block, route_fraction, tail_levels),
            "eta_0.01": _evaluate_hard_modes(
                K, V, Q, positions, beta, 0.01, block, route_fraction, tail_levels),
            "oracle": _oracle_floor(K, Q, positions, beta, 0.10, block),
        }

    fixture = Path(cache)
    if not fixture.exists():
        raise FileNotFoundError(f"Qwen geometry cache not found: {fixture}")
    z = np.load(fixture)
    K, all_Q = z["K"].astype(np.float64), z["Q"].astype(np.float64)
    rng = np.random.default_rng(seed)
    eligible = np.arange(max(64, len(K) // 2), len(K))
    positions = np.sort(rng.choice(eligible, min(queries, len(eligible)), replace=False))
    Q = all_Q[positions]
    V = np.tanh(K[:, :8])
    beta = 1 / math.sqrt(K.shape[1])
    result["qwen"] = {
        "model": "Qwen/Qwen2.5-0.5B", "layer": 18, "kv_head": 0, "query_head": 0,
        "sequence": len(K), "dimension": K.shape[1], "cache": str(fixture),
        "value_scope": "deterministic tanh(K[:,:8]) proxy; mass geometry is real Q/K",
        "fixed_10pct_blocks": _fixed_budget_finding(K, V, Q, positions, beta, block),
        "eta_0.10": _evaluate_hard_modes(
            K, V, Q, positions, beta, 0.10, block, route_fraction, tail_levels),
        "eta_0.01": _evaluate_hard_modes(
            K, V, Q, positions, beta, 0.01, block, route_fraction, tail_levels),
        "fixed_10pct_block_comparison": _evaluate_hard_modes(
            K, V, Q, positions, beta, 0.10, block, route_fraction, tail_levels,
            max_block_fraction=0.10),
        "oracle_eta_0.10": _oracle_floor(K, Q, positions, beta, 0.10, block),
        "honest_fences": [
            "routing certification is not attention-logit certification",
            "attention soundness requires supplied admissible block caps and exact band counts",
            "no favorable real-model tail profile is assumed",
            "conditional subquadraticity also requires subquadratic evaluated bounds, opened keys, and levels",
            "oracle rows score every key and are not implementations",
        ],
    }
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/score_tail_certificate.json")
    parser.add_argument("--cache", default="/tmp/ssa_bennett_qwen_8192.npz")
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--tail-levels", type=int, default=16)
    parser.add_argument("--route-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run(args.out, args.cache, args.queries, args.block,
                 args.tail_levels, args.route_fraction, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
