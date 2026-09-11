"""Partial-key-coordinate reads feeding SSA's existing score-tail certificate.

No learned predictor or new covariance cap. This measures logical coordinate
and value reads, separately from the dense work of this NumPy reference.
Exact top-set certification is NOT attention-mass certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from .partial_coordinate_attention import CoordinateBounds
from .score_tail_certificate import (
    ScoreTailCertifiedAttention, _logsumexp, certificate_margin,
    mass_share_upper_from_logs, tail_profile_from_item_caps,
)


def _order(scores, ids=None):
    ids = np.arange(len(scores)) if ids is None else np.asarray(ids, dtype=np.int64)
    return ids[np.lexsort((ids, -np.asarray(scores)[ids]))]


def _finite(value):
    return float(value) if np.isfinite(value) else None


def run_read(K, V, q, *, prefix=None, coordinates=8, mode="block_interval",
             eta=.1, max_values=None, seed_values=128, batch=128, block=64,
             levels=16, coordinate_schedule=()):
    """Fixed-r sparse fetch or progressive coordinate-then-value refinement.

    A partial boundary block is force-kept, inside the stated value budget.
    Other seeds are ranked by partial scores (index resolves ties). Additional
    values are fetched by upper score. No dense oracle influences these choices.
    Each schedule entry increases globally known coordinates BEFORE extra values
    are fetched; this conservative policy is not claimed to be work-optimal.
    """
    K, V, q = [np.asarray(a, dtype=np.float64) for a in (K,V,q)]
    prefix = len(K) if prefix is None else prefix
    if (K.ndim != 2 or V.ndim != 2 or len(K) != len(V) or V.shape[1] < 1 or
            not isinstance(prefix, (int,np.integer)) or not 1 <= prefix <= len(K) or
            not 0 < eta < 1 or block < 1 or batch < 1):
        raise ValueError("invalid shapes, prefix or settings")
    K, V = K[:prefix], V[:prefix]
    n, d = K.shape
    if not np.isfinite(V).all():
        raise ValueError("visible values must be finite for value-bound construction")
    if not isinstance(coordinates, (int,np.integer)) or not 0 <= coordinates <= d:
        raise ValueError("invalid coordinate count")
    schedule = [coordinates]+list(coordinate_schedule)
    if any(not isinstance(r, (int,np.integer)) or not 0 <= r <= d for r in schedule) or any(
            a >= b for a,b in zip(schedule,schedule[1:])):
        raise ValueError("coordinate schedule must strictly increase within dimension")
    max_values = n if max_values is None else max_values
    if not isinstance(max_values, (int,np.integer)) or not 1 <= max_values <= n or seed_values < 1:
        raise ValueError("invalid value budget")
    boundary = np.arange((n//block)*block, n, dtype=np.int64)
    if len(boundary) > max_values:
        raise ValueError("value budget must fit partial causal boundary")
    start = time.perf_counter()
    index = CoordinateBounds(K, block=block, mode=mode)
    state = index.start(q, beta=1/math.sqrt(d))
    state.refine_coordinates(state.coordinate_order[:coordinates])
    seed_count = min(n, max_values, max(seed_values,len(boundary)))
    candidates = np.setdiff1d(np.arange(n), boundary, assume_unique=True)
    seed = np.sort(np.r_[boundary, _order(state.partial,candidates)[:seed_count-len(boundary)]])
    state.open_keys(seed)
    selected = np.zeros(n,dtype=bool)
    selected[seed] = True
    trace = []
    best_cap = np.inf
    profile_items = sorts = 0
    stage = 0
    value_bound = float(np.linalg.norm(V,axis=1).max())  # append/build-time metadata
    while True:
        ids, unread = np.flatnonzero(selected), np.flatnonzero(~selected)
        log_z = _logsumexp(state.partial[ids])
        profile = tail_profile_from_item_caps(state.upper[unread], levels=levels)
        profile_items += len(unread)
        best_cap = min(best_cap, profile.log_mass_upper)
        margin = certificate_margin(best_cap,log_z,eta)
        bound = mass_share_upper_from_logs(best_cap,log_z)
        trace.append({"coordinates": schedule[stage], "value_rows": len(ids),
            "key_coordinate_reads": state.coordinate_scalars_read,
            "log_tail_cap": _finite(best_cap), "tail_empty": len(unread)==0,
            "omitted_mass_upper": bound, "margin": _finite(margin),
            "certified": margin <= 0})
        if margin <= 0 or not len(unread):
            break
        if stage+1 < len(schedule):
            stage += 1
            state.refine_coordinates(state.coordinate_order[:schedule[stage]])
            continue
        if len(ids) >= max_values:
            break
        take = min(batch,max_values-len(ids),len(unread))
        fetched = _order(state.upper,unread)[:take]
        sorts += len(unread)
        state.open_keys(fetched)
        selected[fetched] = True
    output = np.exp(state.partial[ids]-log_z) @ V[ids]
    one_cap = tail_profile_from_item_caps(state.upper[unread],levels=1).log_mass_upper
    unquantized_cap = _logsumexp(state.upper[unread])
    work = state.work()
    work.update(value_rows_read=len(ids), value_scalars_read=len(ids)*V.shape[1],
        logical_query_read_bytes_float64=8*(state.coordinate_scalars_read+len(ids)*V.shape[1]),
        dense_KV_read_bytes_float64=8*n*(d+V.shape[1]),
        logical_key_fraction=state.coordinate_scalars_read/(n*d),
        logical_value_fraction=len(ids)/n,
        profile_item_evaluations=profile_items, diagnostic_cap_items_two_modes=2*len(unread),
        tail_levels=levels, tail_profile_storage_scalars=2*levels,
        seed_sort_items=len(candidates), subsequent_sort_items=sorts,
        index_value_bound_rows=n,
        note="Logical key ledger excludes dense index construction and NumPy interval refreshes, all itemized separately. No sparse kernel, hardware latency, or subquadratic total-work claim.")

    # Dense reference computation begins only after the read policy has stopped.
    scores = K @ q / math.sqrt(d)
    log_total = _logsumexp(scores)
    weights = np.exp(scores-log_total)
    dense = weights@V
    omitted = float(weights[unread].sum())
    actual_kl = max(0.,log_total-_logsumexp(scores[ids]))
    kl_upper = float(np.logaddexp(0.,best_cap-log_z))
    error = float(np.linalg.norm(output-dense))
    output_upper = 2*value_bound*bound
    strict = {}
    for k in (1,16):
        target = _order(scores)[:min(k,n)]
        nominated = _order(state.partial,ids)[:min(k,len(ids))]
        check = state.topk_certificate(nominated)
        strict[str(k)] = {"recall": float(np.isin(target,ids).mean()),
            "certified": bool(len(nominated)==len(target) and check["certified"]),
            "margin": check["strict_margin"],
            "nominated_set_matches_oracle": bool(np.array_equal(np.sort(nominated),np.sort(target))),
            "selected_top_ids": nominated.tolist(), "oracle_top_ids": target.tolist()}
    return {"prefix": n, "coordinates_requested": coordinates, "final_coordinates": schedule[stage],
        "coordinate_schedule": schedule, "interval_mode": mode, "eta": eta,
        "max_values": max_values, "indices": ids.tolist(), "output": output.tolist(),
        "certified": margin <= 0, "stop_reason": "certificate" if margin <= 0 else "value_budget",
        "omitted_mass_upper": bound, "actual_omitted_mass": omitted,
        "certificate_margin": _finite(margin), "kl_upper": kl_upper, "actual_supported_kl": actual_kl,
        "output_error_upper": output_upper, "actual_output_l2": error,
        "one_threshold_mass_upper": mass_share_upper_from_logs(one_cap,log_z),
        "unquantized_item_cap_mass_upper": mass_share_upper_from_logs(unquantized_cap,log_z),
        "top_set": strict, "trace": trace, "work": work,
        "audit": {"interval_deficit": float(max(0.,(state.lower-scores).max(),(scores-state.upper).max())),
            "mass_deficit": max(0.,omitted-bound), "output_deficit": max(0.,error-output_upper),
            "kl_deficit": max(0.,actual_kl-kl_upper),
            "false_top_certificates": sum(v["certified"] and not v["nominated_set_matches_oracle"] for v in strict.values())},
        "oracle_work": {"key_logits": n,"truth_value_rows": n,"top_sort_items": n},
        "seconds_including_build_and_oracle": time.perf_counter()-start}


def _summarize(rows):
    fields = ("omitted_mass_upper","actual_omitted_mass","actual_output_l2","kl_upper")
    result = {"queries": len(rows), **{f"mean_{k}":float(np.mean([r[k] for r in rows])) for k in fields},
        "certified_fraction":float(np.mean([r["certified"] for r in rows])),
        "mean_key_coordinate_fraction":float(np.mean([r["work"]["logical_key_fraction"] for r in rows])),
        "mean_value_fraction":float(np.mean([r["work"]["logical_value_fraction"] for r in rows])),
        "mean_logical_KV_bytes_fraction":float(np.mean([r["work"]["logical_query_read_bytes_float64"]/r["work"]["dense_KV_read_bytes_float64"] for r in rows])),
        "mean_bound_evaluations":float(np.mean([r["work"]["bound_evaluations"] for r in rows])),
        "mean_profile_item_evaluations":float(np.mean([r["work"]["profile_item_evaluations"] for r in rows])),
        "mean_fully_scored_keys":float(np.mean([r["work"]["fully_scored_keys"] for r in rows])),
        "mean_values_read":float(np.mean([r["work"]["value_rows_read"] for r in rows])),
        "top1_recall":float(np.mean([r["top_set"]["1"]["recall"] for r in rows])),
        "top16_recall":float(np.mean([r["top_set"]["16"]["recall"] for r in rows])),
        "top1_certified_fraction":float(np.mean([r["top_set"]["1"]["certified"] for r in rows])),
        "top16_certified_fraction":float(np.mean([r["top_set"]["16"]["certified"] for r in rows]))}
    return result


def evaluate_case(K,V,Q,positions,*,coordinate_counts=(0,8,16,32,64),block=64,levels=16):
    d = K.shape[1]
    counts = sorted({min(int(r),d) for r in coordinate_counts})
    fixed, stopping, progressive, radius, oracle = {}, {}, {}, [], []
    for mode in ("global_abs","block_interval"):
        for r in counts:
            rows = []
            for q,pos in zip(Q,positions):
                n = int(pos)+1
                budget = min(n,128+n%block)
                rows.append(run_read(K,V,q,prefix=n,coordinates=r,mode=mode,eta=.1,
                    max_values=budget,seed_values=budget,block=block,levels=levels))
            fixed[f"{mode}/r{r}"] = {"summary":_summarize(rows),"rows":rows}
    for eta in (.1,.01):
        for r in counts:
            rows = [run_read(K,V,q,prefix=int(pos)+1,coordinates=r,eta=eta,
                seed_values=min(int(pos)+1,128+(int(pos)+1)%block),block=block,levels=levels)
                for q,pos in zip(Q,positions)]
            stopping[f"eta{eta}/r{r}"] = {"summary":_summarize(rows),"rows":rows}
        rows = []
        for q,pos in zip(Q,positions):
            n = int(pos)+1
            cap = min(n,max(128+n%block,math.ceil(.1*n)))
            rows.append(run_read(K,V,q,prefix=n,coordinates=counts[0],coordinate_schedule=counts[1:],
                eta=eta,max_values=cap,seed_values=min(n,128+n%block),block=block,levels=levels))
        progressive[f"eta{eta}"] = {"summary":_summarize(rows),"rows":rows}
    # Existing mean/radius mass reader is a baseline, with different block-level
    # fetching. Queries/prefixes/eta match, but routes and granularity do not.
    for q,pos in zip(Q,positions):
        n = int(pos)+1
        ix = ScoreTailCertifiedAttention(K[:n],V[:n],block)
        for eta in (.1,.01):
            old = ix.read(q,beta=1/math.sqrt(d),mass_tol=eta,tail_levels=levels)
            radius.append({"prefix":n,"eta":eta,"mass_upper":old.mass_upper,
                "keys_scored":old.keys_scored,"value_fraction":len(old.indices)/n,
                "bounds_evaluated":old.bounds_evaluated,"certified":old.certified})
        logits = K[:n]@q/math.sqrt(d)
        p = np.exp(logits-_logsumexp(logits))
        order = _order(logits)
        for eta in (.1,.01):
            k = min(n,int(np.searchsorted(np.cumsum(p[order]),1-eta))+1)
            oracle.append({"prefix":n,"eta":eta,"minimum_values_for_mass":k,
                "fraction":k/n,"key_logits_to_construct":n})
    groups = list(fixed.values())+list(stopping.values())+list(progressive.values())
    rows = [row for group in groups for row in group["rows"]]
    audits = [row["audit"] for row in rows]
    audit = {k:max(a[k] for a in audits) for k in audits[0]}
    audit["queries_across_modes"] = len(rows)
    audit["violations_above_1e_9"] = sum(any(a[k]>1e-9 for k in a) for a in audits)
    return {"fixed_budget":fixed,"stopping":stopping,"progressive_10percent_value_cap":progressive,
        "existing_radius_reader":radius,"oracle_minimum_values":oracle,"audit":audit}


def synthetic(kind,n=1024,d=64,queries=8):
    rng = np.random.default_rng(71)
    K = rng.normal(size=(n,d))
    Q = rng.normal(size=(queries,d))
    if kind == "concentrated":
        K *= .02
        K[:16,0] += 70
        Q *= .02
        Q[:,0] += 1
    elif kind == "adversarial":
        K *= .02
        for b in range(0,n,64):
            K[b,-1] += 80
            K[b+1:b+64,-1] -= 80/63
        Q *= .02
        Q[:,-1] = .6
        Q[:,0] = 1.  # larger query coordinate is uninformative
    elif kind != "random":
        raise ValueError(kind)
    V = rng.normal(size=(n,8))
    positions = np.linspace(n//2,n-1,queries,dtype=int)
    return K,V,Q,positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",default="/tmp/ssa_qwen_qkv_8192.npz")
    parser.add_argument("--out",default="runs/partial_coordinate_attention.json")
    parser.add_argument("--queries",type=int,default=16)
    parser.add_argument("--coordinates",default="0,8,16,32,64")
    args = parser.parse_args()
    if args.queries < 1:
        parser.error("positive query count required")
    counts = tuple(map(int,args.coordinates.split(",")))
    if not counts or min(counts)<0:
        parser.error("nonnegative coordinate counts required")
    path = Path(args.cache)
    with np.load(path) as z:
        Q,K,V = [z[k].astype(np.float64) for k in ("Q","K","V")]
    source = ("partial_coordinate_attention.py","partial_coordinate_experiment.py","score_tail_certificate.py")
    report = {"config":vars(args),"base_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "fixture_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256":{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source},
        "scope":"Previously inspected Qwen layer18 KV0 document, real QKV; CPU float64. No new controller training, decoding, semantic retrieval or speedup. Top-set and omitted-mass certificates separate. Ordinary guarded arithmetic is not a formal IEEE proof.",
        "math_commits":["495dcb05f","7395e8a8a","f0ce0f345"],"cases":{}}
    cases = {name:synthetic(name,queries=min(args.queries,8)) for name in ("concentrated","random","adversarial")}
    positions = np.linspace(len(K)//2,len(K)-1,args.queries,dtype=int)
    cases["qwen"] = (K,V,Q[positions],positions)
    for name,data in cases.items():
        started = time.perf_counter()
        report["cases"][name] = evaluate_case(*data,coordinate_counts=counts)
        report["cases"][name]["seconds"] = time.perf_counter()-started
        print(name,json.dumps(report["cases"][name]["audit"]),flush=True)
        print(name,{k:v["summary"]["mean_logical_KV_bytes_fraction"] for k,v in report["cases"][name]["stopping"].items()},flush=True)
    report["status"] = "complete" if all(c["audit"]["violations_above_1e_9"]==0 for c in report["cases"].values()) else "failed_audit"
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    if report["status"] != "complete":
        raise RuntimeError("numerical audit failed; see saved artifact")


if __name__ == "__main__":
    main()
