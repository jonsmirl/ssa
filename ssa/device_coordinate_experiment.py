"""Fixed-stage device decision versus matched dense CUDA Graph replay.

No host acceptance branch. Index/plan/capture costs are recorded separately.
All proposal, certificate, conditional fallback and output work is timed.
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

from .partial_coordinate_gpu_experiment import dense_read, _jsonable


def capture_measure(fn, repetitions=7, launches=32):
    """Identical harness for sparse and dense, graph setup outside replay timing."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    started = time.perf_counter()
    with torch.cuda.graph(graph):
        result = fn()
    torch.cuda.synchronize()
    capture_ms = 1000*(time.perf_counter()-started)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    wall,device = [],[]
    for _ in range(repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        wall.append(1000*(time.perf_counter()-started))
        begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(launches):
            graph.replay()
        end.record()
        end.synchronize()
        device.append(begin.elapsed_time(end)/launches)
    return result, {"wall_median_ms":float(np.median(wall)),"wall_samples_ms":wall,
        "event_median_ms":float(np.median(device)),"event_samples_ms":device,
        "event_launches_per_sample":launches,"capture_ms":capture_ms,
        "scope":"warm CUDA Graph replay; wall includes sync, events include queued replay interval"}


def audit(q,K,V,result,eta):
    s = q.double()@K.double().T/8
    p = s.softmax(-1)
    dense = p@V.double()
    proposal = result["proposed_selected_mask"]
    final = result["selected_mask"]
    proposed_delta = (p*~proposal).sum(-1)
    final_delta = (p*~final).sum(-1)
    rows = []
    for h in range(len(q)):
        mass = float(result["proposed_mass_upper"][h])
        error = float(torch.linalg.vector_norm(result["output"][h].double()-dense[h]))
        error_bound = float(result["output_error_upper"][h])
        accepted = bool(result["accepted"][h])
        valid = bool(result.get("numerically_valid",torch.ones(len(q),dtype=torch.bool))[h])
        certified = bool(result.get("certified",torch.ones(len(q),dtype=torch.bool))[h])
        interval_deficit = max(0.,float((result["lower"][h].double()-s[h]).max()),
            float((s[h]-result["upper"][h].double()).max()))
        correct_selection = torch.equal(final[h],proposal[h] if accepted else torch.ones_like(final[h]))
        rows.append({"accepted":accepted,"proposal_mass_upper":mass,
            "proposal_actual_omitted_mass":float(proposed_delta[h]),
            "actual_omitted_mass":float(final_delta[h]),"final_mass_upper":float(result["mass_upper"][h]),
            "actual_output_l2":error,"output_error_upper":error_bound,
            "selected_values":int(final[h].sum()),"proposed_keys":int(proposal[h].sum()),
            "audit":{"interval_deficit":interval_deficit,
                "proposal_mass_deficit":max(0.,float(proposed_delta[h])-mass),
                "final_mass_deficit":max(0.,float(final_delta[h])-float(result["mass_upper"][h])),
                "output_deficit":max(0.,error-error_bound),
                "invalid_numerics":not valid,
                "uncertified_final":not certified,
                "false_accept":accepted and (mass>eta or float(proposed_delta[h])>eta+1e-6),
                "bad_final_selection":not correct_selection}})
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        key = f"n{row['prefix']}/r{row['coordinates']}/f{row['fraction']}/eta{row['eta']}"
        groups.setdefault(key,[]).append(row)
    result = {}
    for key,group in groups.items():
        heads = [h for r in group for h in r["heads"]]
        result[key] = {"head_queries":len(heads),"fallback_fraction":float(np.mean([not h["accepted"] for h in heads])),
            "mean_actual_omitted_mass":float(np.mean([h["actual_omitted_mass"] for h in heads])),
            "mean_output_l2":float(np.mean([h["actual_output_l2"] for h in heads])),
            "sparse_wall_ms":float(np.median([r["sparse_timing"]["wall_median_ms"] for r in group])),
            "dense_wall_ms":float(np.median([r["dense_timing"]["wall_median_ms"] for r in group])),
            "sparse_event_ms":float(np.median([r["sparse_timing"]["event_median_ms"] for r in group])),
            "dense_event_ms":float(np.median([r["dense_timing"]["event_median_ms"] for r in group])),
            "median_paired_event_ratio":float(np.median([r["sparse_timing"]["event_median_ms"]/r["dense_timing"]["event_median_ms"] for r in group])),
            "event_win_fraction":float(np.mean([r["sparse_timing"]["event_median_ms"]<r["dense_timing"]["event_median_ms"] for r in group]))}
    return result


def fixture_paths(roots):
    answer = []
    for root in map(Path,roots):
        manifest = json.loads((root/"manifest.json").read_text())
        # Both extraction protocols record their individual data file digests.
        for item in manifest.get("articles",[manifest]):
            path = root/Path(item["path"]).name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=item["sha256"]:
                raise ValueError(f"fixture digest mismatch: {path}")
            answer.append((path,manifest))
    return answer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures",nargs="+",default=["/tmp/ssa_partial_fresh","/tmp/ssa_device_growth"])
    parser.add_argument("--out",default="runs/device_coordinate_attention.json")
    parser.add_argument("--limit-groups",type=int,default=0)
    parser.add_argument("--repetitions",type=int,default=7)
    parser.add_argument("--launches",type=int,default=32)
    args = parser.parse_args()
    if args.repetitions<1 or args.launches<1:
        parser.error("positive timing counts required")
    if not torch.cuda.is_available():
        raise RuntimeError("approved CUDA access required")
    from .device_coordinate_attention import DeviceCoordinateIndex
    sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    files = fixture_paths(args.fixtures)
    report = {"config":vars(args),"device":torch.cuda.get_device_name(),"torch":torch.__version__,
        "base_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "source_sha256":{p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name("device_coordinate_attention.py"),
            Path(__file__).with_name("partial_coordinate_gpu.py"),Path(__file__).with_name("partial_coordinate_gpu_experiment.py"))},
        "fixtures":[{"path":str(p),"sha256":sha(p),"manifest":m} for p,m in files],
        "scope":"Fixed-stage, device-side decision and conditional dense fallback; warm graph replay, frozen dense geometry, not model serving or physical DRAM counters.","rows":[]}
    count = 0
    for path,manifest in files:
        with np.load(path,allow_pickle=False) as z:
            positions,heads = z["positions"].tolist(),z["query_heads"].tolist()
            for layer in (6,12,18):
                Kall,Vall,Qall = (torch.tensor(z[f"{kind}_{layer}"],device="cuda",dtype=torch.bfloat16) for kind in ("K","V","Q"))
                for kv in (0,1):
                    slots = [i for i,h in enumerate(heads) if h//7==kv]
                    for pi,pos in enumerate(positions):
                        if args.limit_groups and count>=args.limit_groups:
                            break
                        n = pos+1
                        K,V,q = Kall[kv,:n].contiguous(),Vall[kv,:n].contiguous(),Qall[slots,pi].contiguous()
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        index = DeviceCoordinateIndex(K,V)
                        torch.cuda.synchronize()
                        build_ms = 1000*(time.perf_counter()-started)
                        # Alternating order is deterministic, not performance-selected.
                        for r in (32,64):
                            for fraction in (.25,.5):
                                for eta in (.1,.01):
                                    started = time.perf_counter()
                                    plan = index.prepare(q,coordinates=r,fraction=fraction,eta=eta)
                                    torch.cuda.synchronize()
                                    prepare_ms = 1000*(time.perf_counter()-started)
                                    if len(report["rows"])%2:
                                        result,st = capture_measure(plan.run,args.repetitions,args.launches)
                                        native,dt = capture_measure(lambda:dense_read(q,K,V),args.repetitions,args.launches)
                                    else:
                                        native,dt = capture_measure(lambda:dense_read(q,K,V),args.repetitions,args.launches)
                                        result,st = capture_measure(plan.run,args.repetitions,args.launches)
                                    diagnostics = audit(q,K,V,result,eta)
                                    for i,h in enumerate(diagnostics):
                                        h["query_head"] = heads[slots[i]]
                                    report["rows"].append({"fixture":str(path),"layer":layer,"kv_head":kv,"prefix":n,
                                        "coordinates":r,"fraction":fraction,"eta":eta,"heads":diagnostics,
                                        "sparse_timing":st,"dense_timing":dt,"index_build_ms":build_ms,
                                        "prepare_ms":prepare_ms,"work":_jsonable(result["counts"]),
                                        "storage":index.storage()})
                        count += 1
                        print(f"{path.name} layer={layer} kv={kv} n={n} group={count}",flush=True)
    report["summary"] = summarize(report["rows"])
    audits = [h["audit"] for r in report["rows"] for h in r["heads"]]
    report["audit"] = {k:max(a[k] for a in audits) for k in audits[0]}
    report["audit"]["violations_above_1e_6"] = sum(any(float(v)>1e-6 for v in a.values()) for a in audits)
    report["status"] = "passed" if report["audit"]["violations_above_1e_6"]==0 else "failed"
    out = Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"audit":report["audit"],"summary":report["summary"]},indent=2),flush=True)
    if report["status"]!="passed":
        raise RuntimeError("numerical audit failed; artifact saved")


if __name__ == "__main__":
    main()
