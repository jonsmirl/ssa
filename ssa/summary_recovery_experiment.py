"""Cell-summary minimax and persistent all-prefix ridge CPU diagnostics.

Dense weights/feature reads are charged oracles. Worst-case scalar-cube radii
are NOT a Qwen accuracy floor; ridge reader gains need a supplied mismatch bound
before they imply an error certificate. This is not an efficient attention kernel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from .cell_summary_minimax import cell_summary_minimax, decode_cell_summary
from .persistent_ridge import PersistentRidge
from .span_memory import fit_cells, joint_update, sequential_update
from .span_memory_experiment import (
    _array, _assign, _digest, _value_metrics, route_blocks,
    run_case as old_run_case,
)


def feature_spec(K, budget, dv, seed, kind):
    """Budget includes basis, normalization, Gram, cross, W, count and lambda.

    Both families use a fixed random projection and intercept. The linear
    projection is not the earlier full raw-key/intercept model.
    """
    if kind not in ("linear_projection", "tanh"):
        raise ValueError("unknown feature family")
    d = K.shape[1]
    r = 2
    def size(r):
        return 2*d + (d+1)*(r-1) + r*r + 2*r*dv + 2
    if size(r) > budget:
        raise ValueError("budget too small for ridge feature and statistics state")
    while size(r+1) <= budget:
        r += 1
    rng = np.random.default_rng(seed)
    return {"kind": kind, "mean": K.mean(0),
            "scale": np.maximum(K.std(0), 1e-6),
            "projection": rng.normal(size=(d, r-1))/np.sqrt(d),
            "bias": rng.uniform(-1, 1, size=r-1)}, r, size(r)


def features(K, spec):
    z = ((K-spec["mean"])/spec["scale"]) @ spec["projection"] + spec["bias"]
    if spec["kind"] == "tanh":
        z = np.tanh(z)
    return np.column_stack((np.ones(len(K)), z))


def select_penalty(X, V, fit_stop, train_stop, penalties):
    """Frozen fit on [0,fit_stop), selection on [fit_stop,train_stop) only.

    Uses value MSE, not suffix attention errors. Candidates are absolute positive
    penalties on SUM loss; the chosen lambda never grows with prefix length.
    Equal validation loss prefers the earlier supplied candidate.
    """
    candidates = np.asarray(penalties, dtype=float)
    if (candidates.ndim != 1 or not len(candidates) or
            not np.isfinite(candidates).all() or np.any(candidates <= 0) or
            not 0 < fit_stop < train_stop <= len(X)):
        raise ValueError("positive penalties and nonempty fit/validation required")
    scores = []
    for penalty in candidates:
        model = PersistentRidge(X.shape[1], V.shape[1], float(penalty))
        model.append(X[:fit_stop], V[:fit_stop])
        error = X[fit_stop:train_stop] @ model.solve()-V[fit_stop:train_stop]
        scores.append(float(np.mean(error*error)))
    best = int(np.argmin(scores))
    return float(candidates[best]), {"candidates": candidates.tolist(),
        "validation_mse": scores, "selected_index": best,
        "fit_rows": fit_stop, "validation_rows": train_stop-fit_stop,
        "fit_value_rows_across_candidates": fit_stop*len(candidates),
        "validation_value_rows_across_candidates": (train_stop-fit_stop)*len(candidates),
        "selection_metric": "frozen validation value MSE; chronological prefix only"}


def _weights(q, K):
    logits = K @ q / np.sqrt(K.shape[1])
    weights = np.exp(logits-logits.max())
    return weights/weights.sum(), logits


def _row(output, dense, position, selected):
    error = output-dense
    return {"position": int(position), "selected_count": len(selected),
            "output": output.tolist(), "signed_error": error.tolist(),
            "l2_error": float(np.linalg.norm(error))}


def _mean(rows, field):
    return float(np.mean([r[field] for r in rows]))


def _charge_control(work, name, diag):
    row = work.setdefault(name, {"rows_processed": 0, "fit_ops_estimate": 0,
        "diagnostic_ops_estimate": 0, "svd_work_proxy": 0, "max_temporary_bytes_estimate": 0})
    for key, source in (("rows_processed", "rows_read"),
            ("fit_ops_estimate", "fit_arithmetic_ops_estimate"),
            ("diagnostic_ops_estimate", "diagnostic_dense_ops_estimate"),
            ("svd_work_proxy", "svd_work_proxy")):
        row[key] += diag[source]
    row["max_temporary_bytes_estimate"] = max(row["max_temporary_bytes_estimate"], diag["temporary_bytes_estimate"])
    row["svd_is_fit_work"] = name == "joint"


def run_case(Q, K, V, *, budget=8192, seed=0, calibration=512,
             fit_stop=3072, train_stop=4096, batch_size=64,
             penalties=(.01, 1., 100., 10000.), query_positions=None,
             block=64, top_blocks=2, include_existing=True):
    Q, K, V = map(_array, (Q, K, V))
    n, d = K.shape
    dv = V.shape[1]
    positions = np.asarray(query_positions if query_positions is not None else
                           np.linspace(train_stop, n-1, 16, dtype=int))
    if (Q.shape != K.shape or len(V) != n or
            not 1 <= calibration <= fit_stop < train_stop < n or batch_size < 1 or
            positions.ndim != 1 or not len(positions) or positions.dtype.kind not in "iu" or
            np.any(positions < train_stop) or np.any(positions >= n) or
            np.any(np.diff(positions) <= 0)):
        raise ValueError("invalid shape, causal split or query schedule")
    started = time.perf_counter()
    models, runtime = {}, {}
    for kind in ("linear_projection", "tanh"):
        spec, r, scalars = feature_spec(K[:calibration], budget, dv, seed, kind)
        X = features(K, spec)  # Full reference workspace, not persistent state.
        penalty, validation = select_penalty(X, V, fit_stop, train_stop, penalties)
        ridge = PersistentRidge(r, dv, penalty)
        joint = np.zeros((r, dv))
        sequential = joint.copy()
        anchors, controls_work = [], {}
        for first in range(0, train_stop, batch_size):
            last = min(first+batch_size, train_stop)
            old = X[:min(first, 256)] @ ridge.solve()-V[:min(first, 256)]
            ridge.append(X[first:last], V[first:last])
            W = ridge.solve()
            new = X[:min(first, 256)] @ W-V[:min(first, 256)]
            if first:
                anchors.append({"end": last, "old_l2_before": float(np.linalg.norm(old, axis=1).mean()),
                    "old_l2_after": float(np.linalg.norm(new, axis=1).mean()),
                    "fraction_old_worsened": float(np.mean(np.linalg.norm(new, axis=1) >
                                                           np.linalg.norm(old, axis=1)+1e-12))})
            joint, diag = joint_update(joint, X[first:last], V[first:last], rate=.5)
            _charge_control(controls_work, "joint", diag)
            sequential, diag = sequential_update(sequential, X[first:last], V[first:last], rate=.5)
            _charge_control(controls_work, "sequential", diag)
        optimal = np.linalg.lstsq(X[:train_stop], V[:train_stop], rcond=1e-10)[0]
        arrays = [v for v in spec.values() if isinstance(v, np.ndarray)]
        base_state = sum(a.size for a in arrays)
        models[kind] = {"feature_dimension": r, "state_scalars": scalars,
            "state_bytes_float64": 8*scalars, "feature_state_scalars": base_state,
            "statistics_accounting": ridge.state_accounting(), "penalty": penalty,
            "validation": validation, "frozen_fit_sha256": _digest(ridge.solve(), *arrays),
            "frozen_heldout_value": _value_metrics(X[train_stop:] @ ridge.solve(), V[train_stop:]),
            "training_value": _value_metrics(X[:train_stop] @ ridge.solve(), V[:train_stop]),
            "oracle_training_irreducible_fro": float(np.linalg.norm(V[:train_stop]-X[:train_stop]@optimal)),
            "prefix_old_anchor_interference": anchors,
            "matched_feature_control_state_scalars": base_state+r*dv,
            "matched_feature_control_work": controls_work,
            "frozen_attention": [], "online_attention": [],
            "matched_joint_attention": [], "matched_sequential_attention": [],
            "online_rows_processed_including_query_copies": 0,
            "statistic_append_calls": (train_stop+batch_size-1)//batch_size,
            "prefix_fit_solves": (train_stop+batch_size-1)//batch_size,
            "online_solves": 0}
        runtime[kind] = {"X": X, "frozen": ridge.copy(), "online": ridge,
                         "joint": joint, "sequential": sequential}

    cells = min(calibration, budget//(d+dv+1))
    if cells < 1:
        raise ValueError("budget too small for cell state")
    centers, _, _, _ = fit_cells(K[:calibration], np.zeros((calibration, dv)), cells, seed=seed)
    assignment = _assign(K, centers)
    sums = np.zeros((cells, dv))
    np.add.at(sums, assignment[:train_stop], V[:train_stop])
    cell_rows, sparse_rows, routes = [], [], []
    oracle_selected_rows, global_median_rows = [], []
    global_sum = V[:train_stop].sum(0, keepdims=True)
    cell_cursor = fit_cursor = train_stop
    for pos in positions.tolist():
        prefix = pos+1
        while fit_cursor+batch_size <= prefix:
            last = fit_cursor+batch_size
            for kind, state in runtime.items():
                x, y = state["X"][fit_cursor:last], V[fit_cursor:last]
                state["online"].append(x, y)
                state["joint"], diag = joint_update(state["joint"], x, y, rate=.5)
                _charge_control(models[kind]["matched_feature_control_work"], "joint", diag)
                state["sequential"], diag = sequential_update(state["sequential"], x, y, rate=.5)
                _charge_control(models[kind]["matched_feature_control_work"], "sequential", diag)
                models[kind]["online_rows_processed_including_query_copies"] += len(x)
                models[kind]["statistic_append_calls"] += 1
            fit_cursor = last
        np.add.at(sums, assignment[cell_cursor:prefix], V[cell_cursor:prefix])
        # Preserve row order across measurement schedules, as with cell sums.
        np.add.at(global_sum, np.zeros(prefix-cell_cursor, dtype=np.int64), V[cell_cursor:prefix])
        cell_cursor = prefix
        ids = route_blocks(K, Q[pos], prefix, block, top_blocks)
        p, logits = _weights(Q[pos], K[:prefix])
        dense = p @ V[:prefix]
        selected_p = np.exp(logits[ids]-logits[ids].max())
        sparse_rows.append(_row((selected_p/selected_p.sum()) @ V[ids], dense, pos, ids))
        unread = np.ones(prefix, dtype=bool)
        unread[ids] = False
        oracle_selected_rows.append(_row(p[ids] @ V[ids], dense, pos, ids))
        global_profile = cell_summary_minimax(p, np.zeros(prefix, dtype=np.int64), ids, cells=1)
        global_row = _row(decode_cell_summary(global_profile, global_sum, V[ids]), dense, pos, ids)
        global_row["radius"] = global_profile["radius"]
        global_median_rows.append(global_row)
        routes.append({"position": pos, "indices": ids.tolist(),
            "omitted_mass": float(p[unread].sum()), "oracle_key_logits": prefix,
            "routing_key_rows_scanned": ((prefix-1)//block)*block})
        profile = cell_summary_minimax(p, assignment[:prefix], ids, cells=cells)
        median_out = decode_cell_summary(profile, sums, V[ids])
        mean_out = decode_cell_summary(profile, sums, V[ids], coefficients=profile["mean_coefficients"])
        h = profile["dual_witness"]
        # Integer sums are exactly zero; target pairing remains a float diagnostic.
        kernel = np.zeros(cells, dtype=np.int64)
        np.add.at(kernel, assignment[:prefix], h.astype(np.int64))
        radius = profile["radius"]
        residual = profile["residual"]
        vector_bound = radius*float(np.linalg.norm(V[:prefix], axis=1).max())
        cell_rows.append({"position": pos, "selected_count": len(ids),
            "radius": radius, "mean_coefficient_radius": profile["mean_radius"],
            "omitted_mass_without_summary": float(p[unread].sum()),
            "dual_target": float(p@h), "primal_dual_gap": float(abs(radius-p@h)),
            "kernel_max_abs_integer": int(np.abs(kernel).max(initial=0)),
            "selected_witness_max_abs_integer": int(np.abs(h[ids]).max(initial=0)),
            "witness_sha256": hashlib.sha256(h.tobytes()).hexdigest(),
            "median_output": median_out.tolist(), "mean_output": mean_out.tolist(),
            "median_l2_error": float(np.linalg.norm(median_out-dense)),
            "mean_l2_error": float(np.linalg.norm(mean_out-dense)),
            "row_norm_scaled_output_bound": vector_bound,
            "output_bound_deficit": max(0., float(np.linalg.norm(median_out-dense))-vector_bound),
            "signed_identity_deficit": float(np.linalg.norm(median_out-dense+residual@V[:prefix])),
            "coefficient_sum_including_exact_selected": float(p[ids].sum()+
                (profile["coefficients"][assignment[:prefix]][unread]).sum())})
        for kind, state in runtime.items():
            online = state["online"].copy()
            joint, sequential = state["joint"], state["sequential"]
            if fit_cursor < prefix:
                x, y = state["X"][fit_cursor:prefix], V[fit_cursor:prefix]
                online.append(x, y)
                joint, diag = joint_update(joint, x, y, rate=.5)
                _charge_control(models[kind]["matched_feature_control_work"], "joint", diag)
                sequential, diag = sequential_update(sequential, x, y, rate=.5)
                _charge_control(models[kind]["matched_feature_control_work"], "sequential", diag)
                models[kind]["online_rows_processed_including_query_copies"] += len(x)
                models[kind]["statistic_append_calls"] += 1
            a = p[unread] @ state["X"][:prefix][unread]
            exact = p[ids] @ V[ids]
            for protocol, model in (("frozen", state["frozen"]), ("online", online)):
                W = model.solve()
                out = exact+a@W
                row = _row(out, dense, pos, ids)
                signed = p[unread] @ (state["X"][:prefix][unread]@W-V[:prefix][unread])
                row.update(reader_gain=model.reader_gain(a),
                    universal_reader_gain_bound=float(np.linalg.norm(a)/(2*np.sqrt(models[kind]["penalty"]))),
                    signed_identity_deficit=float(np.linalg.norm(signed-(out-dense))))
                models[kind][protocol+"_attention"].append(row)
            models[kind]["online_solves"] += 1
            models[kind]["matched_joint_attention"].append(_row(exact+a@joint, dense, pos, ids))
            models[kind]["matched_sequential_attention"].append(_row(exact+a@sequential, dense, pos, ids))
            models[kind]["last_online_state_sha256"] = _digest(online.gram, online.cross, online.solve())

    total_prefix = sum(pos+1 for pos in positions.tolist())
    for kind, model in models.items():
        r = model["feature_dimension"]
        model["summary"] = {key: _mean(model[key], "l2_error") for key in
            ("frozen_attention", "online_attention", "matched_joint_attention", "matched_sequential_attention")}
        processed = train_stop+model["online_rows_processed_including_query_copies"]
        model["work"] = {
            "fit_value_rows": processed, "distinct_admitted_value_rows": int(positions[-1])+1,
            "fit_statistic_multiply_add_ops_estimate": 2*processed*(r*r+r*dv),
            "statistic_accumulation_add_ops_estimate": model["statistic_append_calls"]*(r*r+r*dv),
            "solve_work_proxy_r3_plus_r2dv": (model["prefix_fit_solves"]+model["online_solves"])*(r**3+r*r*dv),
            "reader_gain_solves": 2*len(positions), "reader_gain_solve_work_proxy": 2*len(positions)*r**3,
            "reader_gain_quadratic_work_proxy": 6*len(positions)*r*r,
            "oracle_unread_feature_sum_ops_estimate": 2*total_prefix*r,
            "oracle_signed_check_ops_estimate_two_protocols": 4*total_prefix*r*dv,
            "reference_materialized_feature_scalars": n*r,
            "temporary_query_state_copy_scalars": r*r+2*r*dv+2,
            "basis_generation_ops_estimate": 2*n*d*(r-1),
            "normalization_and_bias_scalar_ops_estimate": 2*n*d+n*(r-1),
            "tanh_scalar_evaluations": n*(r-1) if kind == "tanh" else 0,
            "validation_candidate_stat_ops_estimate": len(penalties)*fit_stop*2*(r*r+r*dv),
            "validation_candidate_solve_work_proxy": len(penalties)*(r**3+r*r*dv),
            "validation_prediction_ops_estimate": len(penalties)*2*(train_stop-fit_stop)*r*dv,
            "oracle_prefix_lstsq_input_scalars": train_stop*(r+dv),
            "oracle_prefix_lstsq_work_proxy": train_stop*r*r+r*r*dv,
            "oracle_training_and_frozen_heldout_prediction_ops_estimate": 2*n*r*dv,
            "oracle_irreducible_prediction_ops_estimate": 2*train_stop*r*dv,
            "old_anchor_value_rows_two_checks": 2*sum(min(first,256) for first in range(0,train_stop,batch_size)),
            "old_anchor_prediction_ops_estimate": 4*r*dv*sum(min(first,256) for first in range(0,train_stop,batch_size)),
            "reference_replica_arrays_scalars_estimate": 3*(r*r+2*r*dv)+4*r*dv,
            "maximum_supplied_batch_scalars": batch_size*(r+dv),
            "note": "Estimates, not allocator peaks. LAPACK internals and Python overhead excluded. Full archive, feature matrices, frozen/online/query replicas and matched controls coexist in this reference but are not deployed state. Control SVD work is itemized separately. Floating reductions and solver diagnostics are not interval certificates."}
    result = {"budget_scalars": budget, "seed": seed, "calibration": calibration,
        "fit_stop": fit_stop, "train_stop": train_stop, "batch_size": batch_size,
        "models": models, "routes": routes, "sparse_attention": sparse_rows,
        "sparse_mean_l2": _mean(sparse_rows, "l2_error"),
        "oracle_selected_only": {"rows": oracle_selected_rows, "mean_l2": _mean(oracle_selected_rows, "l2_error"),
            "scope": "True globally normalized selected weights, zero unread values; no tail recovery. Dense denominator is an oracle."},
        "global_sum_median": {"rows": global_median_rows, "mean_l2": _mean(global_median_rows, "l2_error"),
            "mean_radius": _mean(global_median_rows, "radius"), "state_scalars": dv,
            "work": {"summary_value_rows_admitted": int(positions[-1])+1,
                "oracle_coefficient_rows_and_membership_tests": total_prefix,
                "sort_work_upper_proxy_n_log2n": float(sum((p+1)*np.log2(p+1) for p in positions))},
            "scope": "One global value sum, no geometry cells; oracle unread-weight median, not cheap evaluation."},
        "cells": {"count": cells, "state_scalars": cells*(d+dv+1), "rows": cell_rows,
            "mean_radius": _mean(cell_rows, "radius"),
            "median_radius": float(np.median([x["radius"] for x in cell_rows])),
            "mean_mean_coefficient_radius": _mean(cell_rows, "mean_coefficient_radius"),
            "mean_omitted_mass": _mean(cell_rows, "omitted_mass_without_summary"),
            "median_decoder_mean_l2": _mean(cell_rows, "median_l2_error"),
            "mean_decoder_mean_l2": _mean(cell_rows, "mean_l2_error"),
            "radius_fraction_at_most": {str(e): float(np.mean([x["radius"] <= e for x in cell_rows])) for e in (.01, .1)},
            "max_primal_dual_gap": max(x["primal_dual_gap"] for x in cell_rows),
            "max_integer_kernel_deficit": max(x["kernel_max_abs_integer"] for x in cell_rows),
            "max_output_bound_deficit": max(x["output_bound_deficit"] for x in cell_rows),
            "work": {"calibration_distance_evaluations": 11*calibration*cells,
                "reference_assignment_distance_evaluations": n*cells,
                "oracle_coefficient_rows": total_prefix,
                "profile_cell_membership_tests": total_prefix*cells,
                "profile_workspace_scalars_order_proxy": int(positions[-1])+1+cells,
                "sort_work_upper_proxy_n_log2n": float(sum((p+1)*np.log2(p+1) for p in positions)),
                "summary_value_rows_admitted": int(positions[-1])+1}},
        "shared_work": {"exact_selected_value_reads": sum(len(r["indices"]) for r in routes),
            "oracle_key_logits_and_dense_truth_value_rows": total_prefix,
            "reference_routing_key_rows_scanned": sum(r["routing_key_rows_scanned"] for r in routes),
            "full_fixture_archive_scalars": Q.size+K.size+V.size}}
    if include_existing:
        old = old_run_case(Q, K, V, budget=budget, seed=seed, train_stop=train_stop,
            calibration=calibration, batch_size=batch_size, query_positions=positions,
            block=block, top_blocks=top_blocks)
        assert [r["indices"] for r in old["routes"]] == [r["indices"] for r in routes]
        result["existing_budget_controls"] = {name: {"state_scalars": v["state_scalars"],
            "summary": v["attention_summary"], "heldout_value": v["heldout_value"],
            "work": {key: value for key, value in v.items() if
                any(part in key for part in ("work", "ops_estimate", "rows", "query_copy"))}}
            for name, v in old["models"].items()}
        result["existing_control_work"] = old["work"]
    result["seconds_including_controls_and_oracles"] = time.perf_counter()-started
    return result


def run_synthetic():
    """Predefined small controls; not selected for a favorable Qwen result."""
    rng = np.random.default_rng(19)
    K, Q = rng.normal(size=(2, 256, 8))
    linear = K @ rng.normal(size=(8, 4))
    independent = rng.normal(size=(256, 4))
    cases = {}
    for name, q, v in (("realizable_linear", Q, linear),
                        ("independent_values", Q, independent),
                        ("concentrated_attention", 8*Q, independent)):
        cases[name] = run_case(q, K, v, budget=512, seed=19, calibration=32,
            fit_stop=128, train_stop=192, batch_size=16, penalties=(.01,1.,100.,10000.),
            query_positions=np.linspace(192,255,8,dtype=int), block=16, top_blocks=2,
            include_existing=False)
    return cases


def audit_cases(cases, tolerance=1e-10):
    """Numerical regression audit, not interval or model-quality certification."""
    cells = [row for c in cases for row in c["cells"]["rows"]]
    ridge = [row for c in cases for m in c["models"].values()
             for mode in ("frozen_attention", "online_attention") for row in m[mode]]
    maximum_gap = max(row["primal_dual_gap"] for row in cells)
    signed = max([row["signed_identity_deficit"] for row in cells+ridge])
    output = max(row["output_bound_deficit"] for row in cells)
    gain = max(0., max(row["reader_gain"]-row["universal_reader_gain_bound"] for row in ridge))
    integer_violations = sum(bool(row["kernel_max_abs_integer"] or row["selected_witness_max_abs_integer"]) for row in cells)
    state_violations = sum(m["state_scalars"] > c["budget_scalars"] for c in cases for m in c["models"].values())
    state_violations += sum(c["cells"]["state_scalars"] > c["budget_scalars"] for c in cases)
    return {"tolerance": tolerance, "cell_query_count": len(cells),
        "ridge_protocol_query_count": len(ridge), "integer_witness_violations": integer_violations,
        "state_cap_violations": state_violations, "maximum_primal_dual_gap": maximum_gap,
        "maximum_signed_identity_deficit": signed, "maximum_output_bound_deficit": output,
        "maximum_reader_gain_bound_deficit": gain,
        "passed": not integer_violations and not state_violations and max(maximum_gap,signed,output,gain) <= tolerance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="/tmp/ssa_qwen_qkv_8192.npz")
    parser.add_argument("--out", default="runs/summary_recovery_comparison.json")
    parser.add_argument("--budgets", default="8192,16384")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--calibration", type=int, default=512)
    parser.add_argument("--fit-stop", type=int, default=3072)
    parser.add_argument("--train-stop", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--penalties", default="0.01,1,100,10000")
    args = parser.parse_args()
    if not 1 <= args.queries:
        parser.error("positive query count required")
    fixture = Path(args.cache)
    with np.load(fixture) as data:
        Q, K, V = [data[k] for k in ("Q", "K", "V")]
    source_names = ("summary_recovery_experiment.py", "cell_summary_minimax.py",
                    "persistent_ridge.py", "span_memory.py", "span_memory_experiment.py")
    report = {"config": vars(args), "base_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in source_names},
        "math_source_commit": "940cee72ffb2d8ccd3c78bb70a1c2b5ce06382fb",
        "scope": "CPU float64, previously inspected Qwen layer18 KV0 document. No new model forward, CE, retrieval or generalization claim. Cube minimax and actual Qwen head errors are distinct. Dense oracle weights and feature sums do not implement the open efficient streaming construction.",
        "penalty_protocol": "Chronological fit [0,3072), validation [3072,4096) by default; refit all prefix with selected fixed absolute penalty, including intercept. No suffix-based selection.",
        "environment": {"numpy": np.__version__, "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                        "omp_threads": os.environ.get("OMP_NUM_THREADS")}, "cases": []}
    positions = np.linspace(args.train_stop, len(K)-1, args.queries, dtype=int)
    for budget in map(int, args.budgets.split(",")):
        for seed in map(int, args.seeds.split(",")):
            case = run_case(Q, K, V, budget=budget, seed=seed, calibration=args.calibration,
                fit_stop=args.fit_stop, train_stop=args.train_stop, batch_size=args.batch_size,
                penalties=tuple(map(float, args.penalties.split(","))), query_positions=positions)
            report["cases"].append(case)
            print(json.dumps({"budget": budget, "seed": seed, "sparse_l2": case["sparse_mean_l2"],
                "cell_radius": case["cells"]["mean_radius"],
                "cell_median_l2": case["cells"]["median_decoder_mean_l2"],
                "ridge": {k: v["summary"] for k, v in case["models"].items()}}), flush=True)
    report["synthetic_cases"] = run_synthetic()
    report["numerical_audit"] = audit_cases(report["cases"]+list(report["synthetic_cases"].values()))
    report["status"] = "complete" if report["numerical_audit"]["passed"] else "failed_numerical_audit"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if report["status"] != "complete":
        raise RuntimeError("numerical audit failed; see output artifact")


if __name__ == "__main__":
    main()
