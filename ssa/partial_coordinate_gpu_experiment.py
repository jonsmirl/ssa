"""Charged GPU latency and dense-oracle audit on fresh multi-head Qwen fixtures.

No timing includes the dense oracle. Query timings DO include routing, sorting,
adaptive stopping, synchronization, and output. Index construction is separate.
This is sampled decode-shaped attention, not complete-model serving or prefill.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F


def measure(fn, repeats=5, warmup=2):
    """Synchronized wall latency, including all host control and GPU work."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    durations = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.cuda.synchronize()
        durations.append(1000*(time.perf_counter()-start))
    return result, {"median_ms":float(np.median(durations)),
        "minimum_ms":float(np.min(durations)), "p90_ms":float(np.quantile(durations,.9)),
        "samples_ms":durations, "warmup_calls":warmup,
        "scope":"synchronized wall time; includes host control, launches and device work"}


def dense_read(q, K, V):
    h = len(q)
    return F.scaled_dot_product_attention(q[None,:,None,:],
        K[None,None].expand(1,h,-1,-1), V[None,None].expand(1,h,-1,-1),
        dropout_p=0.,is_causal=False)[0,:,0]


def audit_read(q, K, V, result, eta):
    """Independent float64 dense oracle, executed only after sparse policy stops."""
    q, K, V = (x.detach().double() for x in (q,K,V))
    s = q@K.T / K.shape[1]**.5
    p = s.softmax(-1)
    truth = p@V
    keep = result["selected_mask"]
    delta = (p*~keep).sum(-1)
    restricted = s.masked_fill(~keep,-torch.inf).softmax(-1)
    restricted_out = restricted@V
    actual_kl = -torch.log1p(-delta)
    head_rows = []
    for h in range(len(q)):
        bound = float(result["mass_upper"][h])
        kl_bound = float(result["kl_upper"][h]) if "kl_upper" in result else float(-np.log1p(-bound))
        out_bound = float(result["output_error_upper"][h])
        error = float(torch.linalg.vector_norm(result["output"][h].double()-truth[h]))
        numerical_output = float(torch.linalg.vector_norm(result["output"][h].double()-restricted_out[h]))
        lower,upper = result["lower"][h].double(),result["upper"][h].double()
        interval_deficit = float(torch.maximum((lower-s[h]).max(),(s[h]-upper).max()).clamp_min(0))
        order = torch.argsort(s[h],descending=True,stable=True)
        one,top16 = order[:1],order[:min(16,len(K))]
        claimed = bool(result["certified"][h])
        head_rows.append({"mass_upper":bound,"actual_omitted_mass":float(delta[h]),
            "certified":claimed,"output_error_upper":out_bound,"actual_output_l2":error,
            "restricted_output_numerical_l2":numerical_output,
            "actual_supported_kl":float(actual_kl[h]),
            "kl_upper":kl_bound,
            "kl_upper_from_mass":float(-np.log1p(-bound)) if bound<1 else None,
            "top1_id":int(one[0]),"top1_coverage":float(keep[h,one].double().mean()),
            "top16_coverage":float(keep[h,top16].double().mean()),
            "values_read":int(keep[h].sum()),"value_fraction":float(keep[h].double().mean()),
            "audit":{"interval_deficit":interval_deficit,
                "mass_deficit":max(0.,float(delta[h])-bound),
                "kl_deficit":max(0.,float(actual_kl[h])-kl_bound),
                "output_deficit":max(0.,error-out_bound),
                "false_stop":claimed and (bound>eta or float(delta[h])>eta+1e-6)}})
    return head_rows,truth


def _jsonable(x):
    if isinstance(x,torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x,np.generic):
        return x.item()
    if isinstance(x,dict):
        return {str(k):_jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):
        return [_jsonable(v) for v in x]
    return x


def summarize(rows):
    groups = {}
    for row in rows:
        key = f"r{row['coordinates']}/eta{row['eta']}"
        groups.setdefault(key,[]).append(row)
    result = {}
    for key,group in groups.items():
        heads = [h for r in group for h in r["heads"]]
        result[key] = {"head_queries":len(heads),"group_queries":len(group),
            "certified_fraction":float(np.mean([h["certified"] for h in heads])),
            "mean_value_fraction":float(np.mean([h["value_fraction"] for h in heads])),
            "mean_mass_upper":float(np.mean([h["mass_upper"] for h in heads])),
            "mean_actual_omitted_mass":float(np.mean([h["actual_omitted_mass"] for h in heads])),
            "mean_output_l2":float(np.mean([h["actual_output_l2"] for h in heads])),
            "mean_output_error_upper":float(np.mean([h["output_error_upper"] for h in heads])),
            "full_value_read_fraction":float(np.mean([h["value_fraction"]==1 for h in heads])),
            "top1_index0_fraction":float(np.mean([h["top1_id"]==0 for h in heads])),
            "median_sparse_ms":float(np.median([r["query_timing"]["median_ms"] for r in group])),
            "median_dense_native_ms":float(np.median([r["dense_native_timing"]["median_ms"] for r in group])),
            "median_dense_fp32_ms":float(np.median([r["dense_fp32_timing"]["median_ms"] for r in group])),
            "median_paired_slowdown_vs_native":float(np.median([r["query_timing"]["median_ms"]/r["dense_native_timing"]["median_ms"] for r in group])),
            "sparse_faster_fraction":float(np.mean([r["query_timing"]["median_ms"]<r["dense_native_timing"]["median_ms"] for r in group]))}
        if all("work" in r for r in group):
            work = [h for r in group for h in r["work"]]
            result[key]["mean_requested_KV_byte_fraction"] = float(np.mean([
                (h["key_read_bytes"]+h["value_read_bytes"])/h["dense_KV_read_bytes"] for h in work]))
            result[key]["median_query_plus_build_ms"] = float(np.median([r["query_plus_build_ms"] for r in group]))
            result[key]["mean_adaptive_stages"] = float(np.mean([h["stages"] for h in work]))
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures",default="/tmp/ssa_partial_fresh")
    parser.add_argument("--out",default="runs/partial_coordinate_gpu.json")
    parser.add_argument("--coordinates",default="32,64")
    parser.add_argument("--repeats",type=int,default=5)
    parser.add_argument("--limit-groups",type=int,default=0,help="smoke limit; zero means all")
    parser.add_argument("--uncached-tail",action="store_true",help="ablate cached upper-score order/suffix masses")
    args = parser.parse_args()
    if args.repeats<1:
        parser.error("positive repeats required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use approved GPU access, do not silently benchmark CPU")
    from .partial_coordinate_gpu import CoordinateGPUIndex
    root = Path(args.fixtures)
    manifest = json.loads((root/"manifest.json").read_text())
    files = sorted(root.glob("article_*.npz"))
    if len(files)!=2:
        raise ValueError("expected exactly two fresh article fixtures")
    sha = lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
    expected = {Path(a["path"]).name:a["sha256"] for a in manifest["articles"]}
    if any(expected.get(f.name)!=sha(f) for f in files):
        raise ValueError("fresh fixture does not match its extraction manifest")
    report = {"config":vars(args),"base_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "device":torch.cuda.get_device_name(),"torch":torch.__version__,
        "fixture_manifest":manifest,"fixture_sha256":{f.name:sha(f) for f in files},
        "source_sha256":{f:sha(Path(__file__).with_name(f)) for f in
            ("partial_coordinate_gpu.py","partial_coordinate_gpu_experiment.py","partial_coordinate_fixture.py")},
        "scope":"Decode-shaped sampled query groups, fresh articles, not complete model inference speed. Warm synchronized wall timings; index build separate, oracle outside timing. Guards empirically checked, not formal IEEE proof.",
        "rows":[]}
    groups = 0
    for path in files:
        with np.load(path,allow_pickle=False) as z:
            positions,heads = z["positions"].tolist(),z["query_heads"].tolist()
            for layer in (6,12,18):
                K_all = torch.tensor(z[f"K_{layer}"],device="cuda",dtype=torch.bfloat16)
                V_all = torch.tensor(z[f"V_{layer}"],device="cuda",dtype=torch.bfloat16)
                Q_all = torch.tensor(z[f"Q_{layer}"],device="cuda",dtype=torch.bfloat16)
                for kv in (0,1):
                    slots = [i for i,h in enumerate(heads) if h//7==kv]
                    for pi,pos in enumerate(positions):
                        if args.limit_groups and groups>=args.limit_groups:
                            break
                        n = pos+1
                        K,V,q = K_all[kv,:n].contiguous(),V_all[kv,:n].contiguous(),Q_all[slots,pi].contiguous()
                        ix,build_time = measure(lambda:CoordinateGPUIndex(K,V,block=64),repeats=3,warmup=1)
                        native,native_time = measure(lambda:dense_read(q,K,V),args.repeats)
                        q32,k32,v32 = q.float(),K.float(),V.float()
                        fp32,fp32_time = measure(lambda:dense_read(q32,k32,v32),args.repeats)
                        for r in map(int,args.coordinates.split(",")):
                            for eta in (.1,.01):
                                output,timing = measure(lambda:ix.read(q,coordinates=r,eta=eta,seed=128+n%64,
                                    reuse_tail_order=not args.uncached_tail),args.repeats)
                                diagnostics,truth = audit_read(q,K,V,output,eta)
                                for slot,h in enumerate(diagnostics):
                                    h["query_head"] = heads[slots[slot]]
                                    h["dense_native_output_l2"] = float(torch.linalg.vector_norm(native[slot].double()-truth[slot]))
                                    h["dense_fp32_output_l2"] = float(torch.linalg.vector_norm(fp32[slot].double()-truth[slot]))
                                report["rows"].append({"article":path.name,"layer":layer,"kv_head":kv,
                                    "prefix":n,"coordinates":r,"eta":eta,"heads":diagnostics,
                                    "query_timing":timing,"index_build_timing":build_time,
                                    "dense_native_timing":native_time,"dense_fp32_timing":fp32_time,
                                    "index_storage":ix.storage(),"seed_values":128+n%64,
                                    "tail_mode":output["tail_mode"],
                                    "floating_point_status":output["floating_point_status"],
                                    "work":_jsonable(output["counts"]),
                                    "query_plus_build_ms":timing["median_ms"]+build_time["median_ms"]})
                        groups += 1
                        print(f"{path.name} layer={layer} kv={kv} prefix={n} group={groups}",flush=True)
    report["summary"] = summarize(report["rows"])
    audits = [h["audit"] for r in report["rows"] for h in r["heads"]]
    report["audit"] = {k:max(a[k] for a in audits) for k in audits[0]}
    report["audit"]["violations_above_1e_6"] = sum(any(float(v)>1e-6 for v in a.values()) for a in audits)
    report["status"] = "passed" if report["audit"]["violations_above_1e_6"]==0 else "failed_audit"
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"summary":report["summary"],"audit":report["audit"]},indent=2),flush=True)
    if report["status"]!="passed":
        raise RuntimeError("GPU numerical audit failed; results saved without claiming success")


if __name__ == "__main__":
    main()
