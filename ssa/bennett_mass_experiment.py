"""Variance-sensitive mass-tree experiment on synthetic and real Qwen geometry.

The experiment compares the existing center/radius exponential-mass cap with the
trace, covariance, and outlier-peeled Bennett caps formalized in Substrate commits
6b3da713a, da13ebeba, and 9c6b1ad35. All are evaluated inside the same certified
binary-tree traversal. Each candidate takes the minimum with the radius cap.

Run on the local CUDA host with cached Qwen weights::

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      python -m ssa.bennett_mass_experiment --out runs/bennett_mass_tree.json

The Qwen forward uses CUDA; tree construction, exact log-mass oracles, and the
reference traversal use float64 NumPy.  This is floating-point verification, not
an interval-arithmetic proof.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .hierarchical_certified_attention import CertifiedTreeAttention

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _bound_configs(d):
    configs = {
        "radius": ("radius", 0),
        "bennett": ("bennett", 0),
        "bennett_covariance": ("bennett_covariance", 0),
    }
    configs.update({f"peel_{t}": ("bennett_peel", t) for t in (1, 2, 4, 8)})
    configs["peel_covariance_4"] = ("bennett_peel_covariance", 4)
    return configs


def _logsumexp(x):
    top = float(np.max(x))
    return top + float(np.log(np.exp(x - top).sum()))


def _actual_omitted_mass(K, q, beta, prefix, selected):
    logits = beta * (K[:prefix] @ q)
    top = float(logits.max())
    weights = np.exp(logits - top)
    kept = float(weights[np.asarray(selected, dtype=int)].sum())
    return max(0.0, min(1.0, 1.0 - kept / float(weights.sum())))


def audit_node_caps(K, Q, beta, block=64):
    """Compare every tree-node cap to its exact descendant log partition."""
    values = np.zeros((len(K), 1))
    configs = _bound_configs(K.shape[1])
    indexes = {
        label: CertifiedTreeAttention(
            K, values, block, mass_bound=mode, peel_count=peel_count)
        for label, (mode, peel_count) in configs.items()
    }
    reductions = {mode: [] for mode in indexes if mode != "radius"}
    slack = {mode: [] for mode in indexes}
    violations = {mode: 0 for mode in indexes}
    max_deficit = {mode: 0.0 for mode in indexes}
    started = time.perf_counter()
    for q in Q:
        qnorm = float(np.linalg.norm(q))
        for node_ids in zip(*(index._nodes for index in indexes.values())):
            nodes = dict(zip(indexes, node_ids))
            lo, hi = nodes["radius"].start * block, nodes["radius"].end * block
            exact = _logsumexp(beta * (K[lo:hi] @ q))
            caps = {
                mode: index._node_log_upper(nodes[mode], q, beta, qnorm)
                for mode, index in indexes.items()
            }
            for mode, cap in caps.items():
                deficit = exact - cap
                violations[mode] += int(deficit > 2e-11)
                max_deficit[mode] = max(max_deficit[mode], deficit)
                slack[mode].append(cap - exact)
            for mode in reductions:
                reductions[mode].append(caps["radius"] - caps[mode])
    result = {
        "query_node_pairs": int(len(slack["radius"])),
        "per_node_extra_scalars": {
            label: (0 if label == "radius" else
                    1 if label == "bennett" else
                    int(K.shape[1] ** 2) if label == "bennett_covariance" else
                    int(peel_count * K.shape[1] + K.shape[1] + 2
                        + (K.shape[1] ** 2 if "covariance" in label else 0)))
            for label, (_, peel_count) in configs.items()
        },
        "seconds": time.perf_counter() - started,
    }
    for mode in indexes:
        values = np.asarray(slack[mode])
        result[f"{mode}_violations"] = violations[mode]
        result[f"max_{mode}_deficit_log"] = max(0.0, float(max_deficit[mode]))
        result[f"{mode}_excess_log_median"] = float(np.median(values))
    for mode in reductions:
        values = np.asarray(reductions[mode])
        result[f"{mode}_strictly_tighter_fraction"] = float(np.mean(values > 1e-12))
        result[f"{mode}_cap_reduction_log_median"] = float(np.median(values))
        result[f"{mode}_cap_reduction_log_p90"] = float(np.quantile(values, 0.9))
        result[f"{mode}_cap_reduction_log_max"] = float(values.max())
    return result


def compare_traversal(K, Q, positions, beta, block=64,
                      budget_fractions=(0.01, 0.05, 0.10), tolerances=(0.1, 0.01)):
    """Measure fixed-budget quality and work needed for certified mass tolerances."""
    values = np.zeros((len(K), 1))
    indexes, build_seconds = {}, {}
    for mode, (implementation, peel_count) in _bound_configs(K.shape[1]).items():
        started = time.perf_counter()
        indexes[mode] = CertifiedTreeAttention(
            K, values, block, mass_bound=implementation, peel_count=peel_count)
        build_seconds[mode] = time.perf_counter() - started

    fixed, stopping, force_kept_stopping = {}, {}, {}
    for frac in budget_fractions:
        by_mode = {}
        for mode, index in indexes.items():
            rows = []
            for q, pos in zip(Q, positions):
                prefix = int(pos) + 1
                visible_blocks = math.ceil(prefix / block)
                cap = max(1, math.ceil(frac * visible_blocks))
                result = index.read(q, beta=beta, prefix=prefix, max_blocks=cap, mass_tol=0)
                actual = _actual_omitted_mass(K, q, beta, prefix, result.indices)
                rows.append((result.blocks_opened, result.bounds_evaluated,
                             result.mass_upper, actual))
            a = np.asarray(rows)
            by_mode[mode] = {
                "mean_blocks_opened": float(a[:, 0].mean()),
                "mean_bounds_evaluated": float(a[:, 1].mean()),
                "mean_mass_upper": float(a[:, 2].mean()),
                "mean_actual_omitted_mass": float(a[:, 3].mean()),
                "median_mass_upper": float(np.median(a[:, 2])),
                "max_soundness_deficit": float(np.maximum(a[:, 3] - a[:, 2], 0).max()),
            }
        fixed[f"{frac:.3f}"] = by_mode

    for tol in tolerances:
        by_mode = {}
        for mode, index in indexes.items():
            rows = []
            for q, pos in zip(Q, positions):
                prefix = int(pos) + 1
                result = index.read(q, beta=beta, prefix=prefix, mass_tol=tol)
                actual = _actual_omitted_mass(K, q, beta, prefix, result.indices)
                rows.append((result.blocks_opened, result.keys_scored,
                             result.bounds_evaluated, result.mass_upper, actual,
                             float(result.certified)))
            a = np.asarray(rows)
            by_mode[mode] = {
                "mean_blocks_opened": float(a[:, 0].mean()),
                "mean_keys_scored": float(a[:, 1].mean()),
                "mean_bounds_evaluated": float(a[:, 2].mean()),
                "mean_mass_upper": float(a[:, 3].mean()),
                "mean_actual_omitted_mass": float(a[:, 4].mean()),
                "certified_fraction": float(a[:, 5].mean()),
                "max_soundness_deficit": float(np.maximum(a[:, 4] - a[:, 3], 0).max()),
            }
        stopping[f"{tol:.3g}"] = by_mode

    def visible_cover(index, node_id, full):
        node = index._nodes[node_id]
        if node.start >= full:
            return []
        if node.end <= full:
            return [node_id]
        return sum((visible_cover(index, child, full) for child in node.children), [])

    # Instantiate the theorem's force-kept arm: every exposed key in the
    # current causal frontier promotes its containing leaf block into S.
    for tol in tolerances:
        by_mode = {}
        for mode, index in indexes.items():
            if not mode.startswith("peel_"):
                continue
            rows = []
            for q, pos in zip(Q, positions):
                prefix = int(pos) + 1
                full = prefix // block
                cover = visible_cover(index, index._root, full) if full else []
                seeds = sorted({
                    int(key_id) // block for node_id in cover
                    for key_id in index._nodes[node_id].peel.exposed_indices
                })
                result = index.read(
                    q, beta=beta, prefix=prefix, mass_tol=tol, initial_blocks=seeds)
                actual = _actual_omitted_mass(K, q, beta, prefix, result.indices)
                rows.append((len(seeds), result.blocks_opened, result.keys_scored,
                             result.bounds_evaluated, result.mass_upper, actual))
            a = np.asarray(rows)
            by_mode[mode] = {
                "mean_forced_seed_blocks": float(a[:, 0].mean()),
                "mean_blocks_opened": float(a[:, 1].mean()),
                "mean_keys_scored": float(a[:, 2].mean()),
                "mean_bounds_evaluated": float(a[:, 3].mean()),
                "mean_mass_upper": float(a[:, 4].mean()),
                "mean_actual_omitted_mass": float(a[:, 5].mean()),
                "max_soundness_deficit": float(np.maximum(a[:, 5] - a[:, 4], 0).max()),
            }
        force_kept_stopping[f"{tol:.3g}"] = by_mode
    return {"build_seconds": build_seconds, "fixed_budget": fixed, "stopping": stopping,
            "force_kept_stopping": force_kept_stopping}


def _synthetic(n, d, queries, seed):
    rng = np.random.default_rng(seed)
    K = 0.04 * rng.normal(size=(n, d))
    # A single high-mass block plus rare, directionally diverse extremes.  The
    # extremes enlarge max radius much more than average squared spread.
    K[:64, 0] += 55
    direction = rng.normal(size=(n // 64, d))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    K[np.arange(n // 64) * 64 + 1] += 6 * direction
    Q = 0.05 * rng.normal(size=(queries, d))
    Q[:, 0] += 1
    positions = np.linspace(n // 2, n - 1, queries, dtype=int)
    return K, Q, positions


def _qwen_geometry(n_tokens, layer, queries, cache, seed):
    cache = Path(cache)
    if cache.exists():
        import torch
        z = np.load(cache)
        K, Q = z["K"].astype(np.float64), z["Q"].astype(np.float64)
        meta = {
            "cache": str(cache), "cache_hit": True,
            "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "torch": torch.__version__,
        }
    else:
        import torch
        from .longctx_keys import extract_qk_hf

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to extract real Qwen geometry")
        captured, seq, d, n_q, n_kv = extract_qk_hf(
            "Qwen/Qwen2.5-0.5B", [layer], n_tokens)
        q, k = captured[layer]
        group = n_q // n_kv
        K, Q = k[0].astype(np.float64), q[0 * group].astype(np.float64)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, K=K.astype(np.float32), Q=Q.astype(np.float32))
        meta = {
            "cache": str(cache), "cache_hit": False, "sequence": seq,
            "dimension": d, "query_heads": n_q, "kv_heads": n_kv,
            "cuda_device": torch.cuda.get_device_name(),
        }
    rng = np.random.default_rng(seed)
    eligible = np.arange(max(64, len(K) // 2), len(K))
    positions = np.sort(rng.choice(eligible, min(queries, len(eligible)), replace=False))
    return K, Q[positions], positions, meta


def run(out=None, n_tokens=8192, layer=18, queries=32, block=64, seed=0,
        cache="/tmp/ssa_bennett_qwen_8192.npz", synthetic_only=False):
    synthetic = _synthetic(n_tokens, 64, queries, seed)
    result = {
        "method": "radius vs min-radius trace/covariance/outlier-peeled Bennett caps",
        "beta": 1 / math.sqrt(64),
        "block": block,
        "scope": "float64 oracle verification; Qwen extraction on CUDA; not an IEEE proof",
        "synthetic": {
            "node_caps": audit_node_caps(synthetic[0], synthetic[1], 1 / math.sqrt(64), block),
            "traversal": compare_traversal(*synthetic, 1 / math.sqrt(64), block),
        },
    }
    if not synthetic_only:
        K, Q, positions, meta = _qwen_geometry(n_tokens, layer, queries, cache, seed)
        beta = 1 / math.sqrt(K.shape[1])
        result["qwen"] = {
            "model": "Qwen/Qwen2.5-0.5B", "layer": layer, "kv_head": 0,
            "query_head": 0, "sequence": len(K), "dimension": K.shape[1],
            "queries": len(Q), "meta": meta,
            "node_caps": audit_node_caps(K, Q, beta, block),
            "traversal": compare_traversal(K, Q, positions, beta, block),
        }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/bennett_mass_tree.json")
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--queries", type=int, default=32)
    parser.add_argument("--block", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", default="/tmp/ssa_bennett_qwen_8192.npz")
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    result = run(args.out, args.n, args.layer, args.queries, args.block,
                 args.seed, args.cache, args.synthetic_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
