"""Capacity/interference tests for span fitting; not a deployable attention kernel.

Frozen suffix prediction and causal online reconstruction are separate protocols.
Dense-weight attention is an explicitly charged oracle diagnosing value recovery.
Only the online cell-centroid arm approximates both weights and values. No Qwen
CE, retrieval, IEEE certificate, or sufficient-statistic minimax claim follows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from .span_memory import joint_update, sequential_update, fit_cells, predict_cells, linear_obstruction


def _array(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError("nonempty finite matrices required")
    return a


def route_blocks(K, q, prefix, block=64, top_blocks=2):
    """Flat centroid route: past blocks plus the entire visible current block.

    This reference rebuilds past block means from visible keys; charge that scan.
    Stable sorting prefers lower block ids on equal scores. No value inputs.
    """
    K = _array(K)
    if not 1 <= prefix <= len(K) or block < 1 or top_blocks < 0:
        raise ValueError("invalid causal prefix or route budget")
    current = (prefix - 1) // block
    local = np.arange(current * block, prefix, dtype=np.int64)
    if current == 0 or top_blocks == 0:
        return local
    means = K[:current * block].reshape(current, block, -1).mean(axis=1)
    selected = np.argsort(-(means @ q), kind="stable")[:top_blocks]
    return np.sort(np.concatenate((
        (selected[:, None] * block + np.arange(block)).ravel(), local)))


def _indices(indices, n):
    raw = np.asarray(indices)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
        raise ValueError("one-dimensional integer selected indices required")
    ids = raw.astype(np.int64)
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= n):
        raise ValueError("selected indices must be unique and visible")
    return ids


def reconstruct_attention(q, K, V, indices, predicted_values):
    """Dense-weight ORACLE; only selected values replace predicted values.

    All visible key logits and all predicted values are computed. V outside the
    selected set is never consumed by this estimator (only by external scoring).
    """
    K, predicted = _array(K), _array(predicted_values)
    V = np.asarray(V,dtype=np.float64)
    if V.ndim != 2 or len(K) != len(V) or V.shape != predicted.shape:
        raise ValueError("incompatible reconstruction matrices")
    ids = _indices(indices, len(K))
    if not np.isfinite(V[ids]).all():
        raise ValueError("selected values must be finite")
    logits = K @ q / np.sqrt(K.shape[1])
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return _read_with_weights(weights, V, ids, predicted)


def _read_with_weights(weights, V, ids, predicted):
    mixed = predicted.copy()
    mixed[ids] = V[ids]
    return weights @ mixed


def centroid_attention(q, K, V, indices, centers, counts, sums):
    """Existing uncalibrated cell-tail formula with causal selected subtraction.

    Counts/sums must cover exactly this visible prefix and these assignments.
    This is an approximation, not an admissible mass or output certificate.
    """
    K, centers, sums = map(_array, (K, centers, sums))
    V = np.asarray(V,dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    ids = _indices(indices, len(K))
    if (V.ndim != 2 or len(V) != len(K) or not np.isfinite(V[ids]).all()
            or counts.shape != (len(centers),)
            or sums.shape != (len(centers), V.shape[1])
            or not np.isfinite(counts).all() or np.any(counts < 0)
            or counts.sum() != len(K)):
        raise ValueError("cell summaries must describe the visible prefix")
    removed = _assign(K[ids], centers) if len(ids) else np.empty(0, dtype=int)
    tail_counts = counts - np.bincount(removed, minlength=len(centers))
    tail_sums = sums.copy()
    np.add.at(tail_sums, removed, -V[ids])
    if np.any(tail_counts < 0):
        raise ValueError("selected assignments disagree with counts")
    active = tail_counts > 0
    selected_logits = K[ids] @ q / np.sqrt(K.shape[1])
    tail_logits = centers[active] @ q / np.sqrt(K.shape[1]) + np.log(tail_counts[active])
    logits = np.concatenate((selected_logits, tail_logits))
    values = np.concatenate((V[ids], tail_sums[active] / tail_counts[active, None]))
    weights = np.exp(logits - logits.max())
    return (weights @ values) / weights.sum()


def _assign(K, centers):
    # Matrix-form squared distance avoids an n*c*d temporary.
    distances = (K * K).sum(axis=1)[:, None] + (centers * centers).sum(axis=1)[None] - 2 * K @ centers.T
    return distances.argmin(axis=1)


def _features(K, spec):
    z = (K - spec["mean"]) / spec["scale"]
    if spec["kind"] == "tanh":
        z = np.tanh(z @ spec["projection"] + spec["bias"])
    return np.column_stack((np.ones(len(K)), z))


def _spec(K, kind, budget, dv, seed):
    d = K.shape[1]
    spec = {"kind": kind, "mean": K.mean(axis=0),
            "scale": np.maximum(K.std(axis=0), 1e-6)}
    if kind == "tanh":
        width = (budget - 2*d - dv) // (d + 1 + dv)
        if width < 1:
            raise ValueError("budget too small for nonlinear feature and value state")
        rng = np.random.default_rng(seed)
        spec.update(projection=rng.normal(size=(d, width)) / np.sqrt(d),
                    bias=rng.uniform(-1, 1, size=width))
    r = d + 1 if kind == "linear" else width + 1
    scalars = sum(a.size for a in spec.values() if isinstance(a, np.ndarray)) + r * dv
    if scalars > budget:
        raise ValueError("budget too small for linear feature and value state")
    return spec, r, scalars


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a, dtype=np.float64)
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _value_metrics(prediction, truth):
    residual = prediction - truth
    energy = float(np.sum(truth ** 2))
    return {"relative_frobenius_error": float(np.linalg.norm(residual) / np.sqrt(energy)) if energy else None,
            "mse": float(np.mean(residual ** 2)),
            "mean_l2": float(np.linalg.norm(residual, axis=1).mean())}


def _summary(rows):
    errors = np.asarray([r["l2_error"] for r in rows])
    return {"queries": len(rows), "mean_l2_error": float(errors.mean()),
            "p95_l2_error": float(np.quantile(errors, .95)),
            "mean_selected_count": float(np.mean([r["selected_count"] for r in rows]))}


def run_case(Q, K, V, *, budget=8192, seed=0, train_stop=4096, calibration=512,
             batch_size=64, rate=.5, query_positions=None, block=64, top_blocks=2):
    """Matched persistent-state cap, identical observed values and selected keys.

    All fitting sees the same prefix once. Joint batches and sequential delta
    have identical features/state but different update work. Frozen evaluation
    never fits suffix V. Online evaluation explicitly admits suffix V before its
    query; this is not held-out value prediction. SVD workspace is charged apart
    from persistent inference state. No Gram matrix is retained across batches.
    """
    Q, K, V = map(_array, (Q, K, V))
    n, d = K.shape; dv = V.shape[1]
    if (Q.shape != K.shape or len(V) != n or not 1 <= calibration <= train_stop < n
            or batch_size < 1 or not 0 <= rate <= 1):
        raise ValueError("invalid fixture geometry or split/update settings")
    positions = np.asarray(query_positions if query_positions is not None else
                           np.linspace(train_stop, n-1, 16, dtype=int))
    if (positions.ndim != 1 or not len(positions) or positions.dtype.kind not in "iu"
            or np.any(positions < train_stop) or np.any(positions >= n)
            or np.any(np.diff(positions) <= 0)):
        raise ValueError("query positions must increase within held-out suffix")
    models, runtime = {}, {}
    for kind in ("linear", "tanh"):
        spec, r, scalars = _spec(K[:calibration], kind, budget, dv, seed)
        X = _features(K, spec)
        for update_name, updater in (("joint", joint_update), ("sequential", sequential_update)):
            name = kind + "_" + update_name
            W = np.zeros((r, dv)); interference = []; start_time = time.perf_counter()
            fit_ops = diagnostic_ops = svd_proxy = diagnostic_workspace = 0
            for first in range(0, train_stop, batch_size):
                last = min(first + batch_size, train_stop)
                anchor_ids = np.arange(min(first, 256))
                before = X[anchor_ids] @ W - V[anchor_ids]
                W, diag = updater(W, X[first:last], V[first:last], rate=rate)
                fit_ops += diag["fit_arithmetic_ops_estimate"]
                diagnostic_ops += diag["diagnostic_dense_ops_estimate"]
                svd_proxy += diag["svd_work_proxy"]
                diagnostic_workspace = max(diagnostic_workspace,diag["temporary_bytes_estimate"]//8)
                after = X[anchor_ids] @ W - V[anchor_ids]
                if len(anchor_ids):
                    bnorm, anorm = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
                    interference.append({"fit_start": first, "anchor_count": len(anchor_ids),
                        "old_mean_l2_before": float(bnorm.mean()), "old_mean_l2_after": float(anorm.mean()),
                        "new_batch_residual_fro_before": diag["residual_before_fro"],
                        "new_batch_residual_fro_after": diag["residual_after_fro"],
                        "new_batch_irreducible_residual_fro": diag["joint_irreducible_residual_fro"],
                        "new_batch_numerical_rank": diag["numerical_rank"],
                        "new_batch_retained_condition": diag["retained_condition"],
                        "fraction_old_associations_worsened": float(np.mean(anorm > bnorm + 1e-12))})
            elapsed = time.perf_counter() - start_time
            # These oracle fits diagnose realizability, not an extra deployable mode.
            optimal, _, rank, singular = np.linalg.lstsq(X[:train_stop], V[:train_stop], rcond=1e-10)
            fit_hash = _digest(W, *[v for v in spec.values() if isinstance(v, np.ndarray)])
            # Workspace counts are conservative scalar estimates, excluding LAPACK internals.
            m = min(batch_size, train_stop)
            workspace = (m*r + m*dv + m*min(m,r) + min(m,r)*r + r*m + r*dv
                         if update_name == "joint" else r + 2*dv)
            models[name] = {"state_scalars": scalars, "state_bytes_float64": scalars*8,
                "unused_state_budget": budget-scalars, "feature_dimension": r,
                "frozen_fit_sha256": fit_hash, "training_value_rows": train_stop,
                "fit_seconds_including_interference_diagnostics": elapsed,
                "prefix_fit_work": {"arithmetic_ops_estimate": fit_ops,
                    "diagnostic_dense_ops_estimate": diagnostic_ops,
                    "svd_work_proxy": svd_proxy, "svd_is_fit_work": update_name == "joint",
                    "note": "Sequential SVD is diagnostic only; both modes execute it. Old-anchor diagnostics and full-prefix oracle solve are additional work."},
                "algorithm_update_workspace_scalars_estimate_excluding_library_internals": workspace,
                "reference_diagnostic_workspace_scalars_estimate_excluding_library_internals": diagnostic_workspace,
                "temporary_query_copy_scalars": r*dv,
                "heldout_value": _value_metrics(X[train_stop:] @ W, V[train_stop:]),
                "training_value": _value_metrics(X[:train_stop] @ W, V[:train_stop]),
                "oracle_prefix_lstsq": {"label": "full-prefix solver diagnostic, not deployed state",
                    "numerical_rank": int(rank),
                    "training": _value_metrics(X[:train_stop] @ optimal, V[:train_stop]),
                    "heldout": _value_metrics(X[train_stop:] @ optimal, V[train_stop:]),
                    "condition": float(singular[0]/singular[rank-1]) if rank else None,
                    "input_workspace_scalars": train_stop*(r+dv)},
                "interference": interference, "frozen_attention": [], "online_attention": [],
                "online_value_rows_admitted": 0, "online_update_rows_processed": 0,
                "online_fit_work": {"arithmetic_ops_estimate": 0, "svd_work_proxy": 0,
                                    "diagnostic_dense_ops_estimate": 0}}
            runtime[name] = {"X": X, "frozen": W.copy(), "online": W.copy(), "updater": updater}

    cells = min(calibration, budget // (d+dv+1))
    if cells < 1:
        raise ValueError("budget must fit at least one cell")
    # Only keys calibrate centers; do not make a second target-reading fit pass.
    centers, _, _, _ = fit_cells(K[:calibration], np.zeros((calibration,dv)), cells, seed=seed)
    assignment = _assign(K, centers)
    counts = np.bincount(assignment[:train_stop], minlength=cells).astype(float)
    sums = np.zeros((cells,dv)); np.add.at(sums, assignment[:train_stop], V[:train_stop])
    frozen_values = predict_cells(K, centers, counts, sums)
    models["cells"] = {"state_scalars": cells*(d+dv+1), "state_bytes_float64": cells*(d+dv+1)*8,
        "unused_state_budget": budget-cells*(d+dv+1), "feature_dimension": cells,
        "frozen_fit_sha256": _digest(centers,counts,sums), "training_value_rows": train_stop,
        "heldout_value": _value_metrics(frozen_values[train_stop:],V[train_stop:]),
        "training_value": _value_metrics(frozen_values[:train_stop],V[:train_stop]),
        "frozen_attention": [], "online_attention": [], "online_centroid_attention": [],
        "online_value_rows_admitted": 0}

    cursor = train_stop
    fit_cursor = train_stop
    sparse_rows = []; routes = []; started = time.perf_counter()
    for pos in positions.tolist():
        prefix = pos+1
        # Persistent updates use a fixed grid, independent of query sampling.
        # A partial current batch is fitted to a temporary copy for this query.
        # Its reads/work are charged again if another query evaluates it later.
        while fit_cursor + batch_size <= prefix:
            first, last = fit_cursor, fit_cursor + batch_size
            for name, state in runtime.items():
                state["online"], diag = state["updater"](state["online"], state["X"][first:last], V[first:last], rate=rate)
                models[name]["online_update_rows_processed"] += last-first
                for field in models[name]["online_fit_work"]:
                    source = "fit_arithmetic_ops_estimate" if field == "arithmetic_ops_estimate" else field
                    models[name]["online_fit_work"][field] += diag[source]
            fit_cursor=last
        for name,state in runtime.items():
            state["evaluation_online"]=state["online"]
            if fit_cursor < prefix:
                state["evaluation_online"],diag=state["updater"](state["online"],state["X"][fit_cursor:prefix],V[fit_cursor:prefix],rate=rate)
                models[name]["online_update_rows_processed"] += prefix-fit_cursor
                for field in models[name]["online_fit_work"]:
                    source = "fit_arithmetic_ops_estimate" if field == "arithmetic_ops_estimate" else field
                    models[name]["online_fit_work"][field] += diag[source]
            models[name]["online_value_rows_admitted"]=prefix-train_stop
        counts += np.bincount(assignment[cursor:prefix], minlength=cells)
        np.add.at(sums,assignment[cursor:prefix],V[cursor:prefix])
        models["cells"]["online_value_rows_admitted"] += prefix-cursor
        cursor=prefix
        ids=route_blocks(K,Q[pos],prefix,block,top_blocks)
        kp,vp=K[:prefix],V[:prefix]
        logits=kp@Q[pos]/np.sqrt(d); weights=np.exp(logits-logits.max()); weights/=weights.sum()
        dense=weights@vp
        selected_weights=np.exp(logits[ids]-logits[ids].max())
        selected_weights/=selected_weights.sum()
        sparse=selected_weights@vp[ids]
        def row(out):
            return {"position":pos,"output":out.tolist(),"l2_error":float(np.linalg.norm(out-dense)),
                    "selected_count":len(ids)}
        sparse_rows.append(row(sparse))
        routes.append({"position":pos,"indices":ids.tolist(),"actual_selected_mass":float(weights[ids].sum()),
                       "routing_block_scores":(prefix-1)//block,
                       "routing_key_rows_scanned":((prefix-1)//block)*block,
                       "oracle_key_logits":prefix,"oracle_truth_value_rows":prefix})
        for name,state in runtime.items():
            for protocol in ("frozen","online"):
                W = state["frozen"] if protocol == "frozen" else state["evaluation_online"]
                predicted=state["X"][:prefix]@W
                models[name][protocol+"_attention"].append(row(_read_with_weights(weights,vp,ids,predicted)))
        online_values=predict_cells(kp,centers,counts,sums)
        models["cells"]["frozen_attention"].append(row(_read_with_weights(weights,vp,ids,frozen_values[:prefix])))
        models["cells"]["online_attention"].append(row(_read_with_weights(weights,vp,ids,online_values)))
        models["cells"]["online_centroid_attention"].append(row(centroid_attention(Q[pos],kp,vp,ids,centers,counts,sums)))
    for model in models.values():
        model["attention_summary"]={key:_summary(model[key]) for key in
            ("frozen_attention","online_attention","online_centroid_attention") if key in model}
        model["oracle_predicted_value_rows_per_protocol"] = sum(r["oracle_key_logits"] for r in routes)
        if "interference" in model:
            model["oracle_reconstruction_matmul_ops_estimate_two_protocols"] = (
                4*sum(r["oracle_key_logits"] for r in routes)*model["feature_dimension"]*dv)
    return {"budget_scalars":budget,"seed":seed,"train_stop":train_stop,"calibration":calibration,
        "batch_size":batch_size,"rate":rate,"models":models,"routes":routes,
        "sparse_attention":sparse_rows,"sparse_summary":_summary(sparse_rows),
        "evaluation_seconds_including_updates_and_oracles":time.perf_counter()-started,
        "protocol":"Fixed prefix features; same prefix value observations; frozen suffix prediction separated from online admission. Joint correction fits each supplied batch, not all historical held keys. Persistent batches anchored at train_stop; query partial batches use discarded temporary copies. Equal representation-state caps, not equal FLOPs, total resident memory, solver workspace or repeated diagnostic accesses. All modes share a full diagnostic archive.",
        "work":{"shared_query_exact_selected_values":sum(len(r["indices"]) for r in routes),
                "shared_oracle_visible_key_logits":sum(r["oracle_key_logits"] for r in routes),
                "shared_oracle_truth_value_rows":sum(r["oracle_truth_value_rows"] for r in routes),
                "reference_routing_key_rows_scanned":sum(r["routing_key_rows_scanned"] for r in routes),
                "feature_calibration_key_rows":calibration,
                "offline_feature_rows_materialized_per_family":n,
                "cell_calibration_distance_evaluations":11*calibration*cells,
                "cell_prefix_assignment_distance_evaluations":n*cells,
                "cell_frozen_prediction_distance_evaluations":n*cells,
                "cell_online_prediction_distance_evaluations":sum(r["oracle_key_logits"] for r in routes)*cells,
                "cell_selected_replacement_distance_evaluations":sum(len(r["indices"]) for r in routes)*cells,
                "online_prefix_value_rows_per_mode":cursor,
                "full_cache_arrays_and_dense_predictions_are_diagnostic_workspace":True}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",default="/tmp/ssa_qwen_qkv_8192.npz")
    parser.add_argument("--out",default="runs/span_memory_comparison.json")
    parser.add_argument("--budgets",default="8192,16384")
    parser.add_argument("--seeds",default="0,1,2")
    parser.add_argument("--train-stop",type=int,default=4096)
    parser.add_argument("--calibration",type=int,default=512)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--queries",type=int,default=16)
    args=parser.parse_args()
    if args.queries<1: parser.error("positive query count required")
    fixture=Path(args.cache); data=np.load(fixture)
    Q,K,V=[data[k] for k in ("Q","K","V")]
    positions=np.linspace(args.train_stop,len(K)-1,args.queries,dtype=int)
    report={"status":"running","config":vars(args),
        "scope":"Cached Qwen layer18 KV0, previously inspected document; CPU float64 diagnostic, no model CE/retrieval or efficient kernel claim. Random tanh features are fixed, not learned.",
        "environment":{"numpy":np.__version__,"processor":platform.machine(),"dtype":"float64",
                       "openblas_threads":os.environ.get("OPENBLAS_NUM_THREADS"),
                       "omp_threads":os.environ.get("OMP_NUM_THREADS")},
        "exact_linear_realizability_check":linear_obstruction(K,V),
        "base_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "fixture_sha256":hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "source_sha256":{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ("span_memory.py","span_memory_experiment.py")},"cases":[]}
    for budget in map(int,args.budgets.split(",")):
        for seed in map(int,args.seeds.split(",")):
            result=run_case(Q,K,V,budget=budget,seed=seed,train_stop=args.train_stop,
                calibration=args.calibration,batch_size=args.batch_size,query_positions=positions)
            report["cases"].append(result)
            print(json.dumps({"budget":budget,"seed":seed,
                "sparse_l2":result["sparse_summary"]["mean_l2_error"],
                "online_oracle_l2":{k:v["attention_summary"]["online_attention"]["mean_l2_error"]
                                    for k,v in result["models"].items()}}),flush=True)
    report["summary"]={}
    for budget in sorted({c["budget_scalars"] for c in report["cases"]}):
        cases=[c for c in report["cases"] if c["budget_scalars"]==budget]
        modes={}
        for name in cases[0]["models"]:
            rows=[c["models"][name] for c in cases]
            modes[name]={"state_scalars":rows[0]["state_scalars"],
                "feature_dimension":rows[0]["feature_dimension"],
                "heldout_relative_frobenius_error_mean":float(np.mean([r["heldout_value"]["relative_frobenius_error"] for r in rows])),
                "attention_mean_l2":{protocol:float(np.mean([r["attention_summary"][protocol]["mean_l2_error"] for r in rows]))
                    for protocol in rows[0]["attention_summary"]}}
            if "interference" in rows[0]:
                anchors=[a for r in rows for a in r["interference"]]
                modes[name]["mean_fraction_old_associations_worsened_per_batch"]=float(np.mean([
                    a["fraction_old_associations_worsened"] for a in anchors])) if anchors else None
        report["summary"][str(budget)]={"seeds":len(cases),
            "same_query_positions_for_every_mode_and_seed":True,
            "linear_models_identical_across_seeds_and_budgets":True,
            "sparse_mean_l2":cases[0]["sparse_summary"]["mean_l2_error"],"models":modes}
    report["status"]="complete"
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
