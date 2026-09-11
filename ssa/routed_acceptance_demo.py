"""Qwen endpoint acceptance: two charged forwards, no claimed Taylor certificate.

Full-vocabulary token predictions and four-candidate NIAH are separate scopes.
Targets are used only to score losses after the label-free acceptance decision.
This prefill/teacher-forced experiment does not implement dual-cache serving.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F
from .endpoint_acceptance import accept_logits
from .qwen_tail_demo import replace_attention


@torch.no_grad()
def paired_hidden(model, ids, gains, config):
    hidden, times = {}, {}
    for mode in ("sparse", "tail"):
        torch.cuda.synchronize(); start = time.perf_counter()
        with replace_attention(mode, gains, **config) as calls:
            hidden[mode] = model.model(ids, use_cache=False).last_hidden_state
        if calls[0] != model.config.num_hidden_layers:
            raise AssertionError("not every layer used the requested attention")
        torch.cuda.synchronize()
        times[mode] = time.perf_counter() - start
    return hidden, times


@torch.no_grad()
def paired_window(model, ids, gains, config, projection_chunk=256):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    hidden, forward_times = paired_hidden(model, ids, gains, config)
    modes = {name: {"nll_sum": 0., "correct": 0} for name in ("reference", "candidate", "accepted")}
    counts = {key: 0 for key in ("targets", "accepted_rows", "reference_ties", "candidate_ties",
                                "changed_top1_proposals", "preservation_violations", "projection_calls")}
    for first in range(0, ids.shape[1] - 1, projection_chunk):
        last = min(first + projection_chunk, ids.shape[1] - 1)
        r = model.lm_head(hidden["sparse"][:, first:last]).float()
        c = model.lm_head(hidden["tail"][:, first:last]).float()
        output, decision = accept_logits(r, c)  # no target label supplied
        target = ids[:, first + 1:last + 1]
        for name, logits in (("reference", r), ("candidate", c), ("accepted", output)):
            modes[name]["nll_sum"] += float(F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum"))
            modes[name]["correct"] += int((logits.argmax(-1) == target).sum())
        counts["targets"] += target.numel()
        counts["accepted_rows"] += int(decision["accepted"].sum())
        counts["reference_ties"] += int((~decision["reference_strict"]).sum())
        counts["candidate_ties"] += int((~decision["candidate_strict"]).sum())
        counts["changed_top1_proposals"] += int((decision["reference_choice"] != decision["candidate_choice"]).sum())
        counts["preservation_violations"] += int((output.argmax(-1) != r.argmax(-1)).sum())
        counts["projection_calls"] += 2
    torch.cuda.synchronize()
    return {"modes": modes, **counts, "tokens": ids.numel(), "full_model_forwards": 2,
            "forward_seconds": forward_times, "total_seconds": time.perf_counter() - start,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}


def aggregate(rows):
    targets = sum(r["targets"] for r in rows)
    result = {key: sum(r[key] for r in rows) for key in
              ("targets", "accepted_rows", "reference_ties", "candidate_ties", "changed_top1_proposals",
               "preservation_violations", "full_model_forwards", "projection_calls", "total_seconds")}
    result["accepted_fraction"] = result["accepted_rows"] / targets
    result["modes"] = {}
    for name in ("reference", "candidate", "accepted"):
        ce = sum(r["modes"][name]["nll_sum"] for r in rows) / targets
        result["modes"][name] = {"ce_nats": ce, "perplexity": math.exp(ce),
                                 "token_accuracy": sum(r["modes"][name]["correct"] for r in rows) / targets}
    result["windows"] = rows
    return result


def main():
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .streaming_qwen import make_choice_niah_ids, score_choice
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="512,8192,32768")
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--probe-contexts", default="8192,32768")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--gains", default="runs/qwen_tail_final/results.json")
    parser.add_argument("--out", default="runs/routed_acceptance_qwen.json")
    args = parser.parse_args()
    lengths = [int(x) for x in args.contexts.split(",") if x]
    probes = [int(x) for x in args.probe_contexts.split(",") if x]
    if (args.windows < 1 or not lengths or min(lengths) < 2 or len(set(lengths)) != len(lengths)
            or any(n < 128 or n % 64 for n in probes)):
        parser.error("positive windows, distinct contexts >=2, and block-aligned probes >=128 required")
    torch.set_num_threads(4); torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
        attn_implementation="sdpa", dtype=torch.bfloat16).cuda().eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    arrow = next((Path.home() / ".cache/huggingface/datasets/wikitext").glob("wikitext-2-raw-v1/*/*/wikitext-validation.arrow"))
    ds = Dataset.from_file(str(arrow))
    content = "\n".join(s for s in ds["text"] if s.strip())
    tokens = np.asarray(tokenizer(content, truncation=False)["input_ids"], dtype=np.int64)
    if max(lengths) * args.windows > len(tokens):
        parser.error("not enough validation tokens for complete requested windows")
    gains = torch.tensor(json.loads(Path(args.gains).read_text())["trained_log_gains"], device="cuda")
    if gains.shape != (model.config.num_hidden_layers, model.config.num_attention_heads) or not torch.isfinite(gains).all():
        raise ValueError("gain geometry or finiteness mismatch")
    config = dict(block=64, top_blocks=2, cells=16, router="batched_tree", query_chunk=256,
                  tail_mode="prototype", max_tail_share=.25)
    report = {"status": "running", "config": vars(args), "attention": config,
              "scope": "Exact comparison of computed endpoints, NOT the nonlinear path certificate. Development validation previously inspected; no fitting; independent paired prefills, not dual-cache generation.",
              "hardware": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "model_config": model.config.to_dict(), "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
              "gain_sha256": hashlib.sha256(Path(args.gains).read_bytes()).hexdigest(),
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("routed_acceptance_demo.py", "endpoint_acceptance.py", "qwen_tail_demo.py",
                                             "hybrid_tail_attention.py", "tail_correction.py", "tail_tree_router.py", "streaming_qwen.py")},
              "corpus": {}, "probes": []}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = out.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(out)
    save()
    try:
        for n in lengths:
            rows = []
            for window in range(args.windows):
                ids = torch.tensor(tokens[window*n:(window+1)*n], device="cuda")[None]
                row = paired_window(model, ids, gains, config)
                row["start"] = window * n; rows.append(row)
                report["corpus"][str(n)] = aggregate(rows); save()
            print(n, {k: v for k, v in report["corpus"][str(n)].items() if k != "windows"}, flush=True)
        for n in probes:
            for depth in (.1, .5, .9):
                ids, labels, gold = make_choice_niah_ids(tokenizer, n, depth=depth, block=64, device="cuda")
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                with torch.no_grad():
                    hidden, times = paired_hidden(model, ids, gains, config)
                    r = model.lm_head(hidden["sparse"][:, -1:]).float()[:, 0]
                    c = model.lm_head(hidden["tail"][:, -1:]).float()[:, 0]
                    accepted, decision = accept_logits(r, c, labels)
                modes = {}
                for name, logits in (("reference", r), ("candidate", c), ("accepted", accepted)):
                    entry = score_choice(logits, labels, gold)
                    scores = logits[0, labels]
                    entry["strict_gold_win"] = bool(scores[gold] > torch.cat((scores[:gold], scores[gold+1:])).max())
                    modes[name] = entry
                torch.cuda.synchronize()
                row = {"context": n, "depth": depth, "scope": "four supplied candidates, not full vocabulary",
                       "accepted_candidate": bool(decision["accepted"].item()), "modes": modes,
                       "full_model_forwards": 2, "forward_seconds": times,
                       "seconds": time.perf_counter() - started, "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}
                report["probes"].append(row); save(); print("probe", row, flush=True)
                del hidden, r, c, accepted
        report["status"] = "complete"; save()
    except Exception as error:
        report.update(status="failed", error=repr(error)); save(); raise


if __name__ == "__main__":
    main()
