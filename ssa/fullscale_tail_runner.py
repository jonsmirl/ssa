"""Frozen-gain, whole-corpus and long-context evaluation; offline Kaggle driver.

No validation/test fitting. Whole-corpus CE is target-count weighted and includes
the final partial window. Logits are projected in chunks to avoid n*vocab memory.
Atomic telemetry is saved after every window/probe, including failures.
"""
from __future__ import annotations
import argparse
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from .qwen_tail_demo import replace_attention


def validate_protocol(contexts, probe_contexts, window_limit, max_tail_share):
    """Reject malformed runs before allocating a model or writing result arms."""
    def lengths(value, *, allow_empty=False):
        if not value and allow_empty:
            return []
        try:
            result = [int(part) for part in value.split(",")]
        except ValueError as error:
            raise ValueError("contexts must be comma-separated integers") from error
        if any(length < 2 for length in result):
            raise ValueError("contexts must contain at least two tokens")
        if len(set(result)) != len(result):
            raise ValueError("duplicate contexts would overwrite result arms")
        return result

    lengths(contexts)
    lengths(probe_contexts, allow_empty=True)
    if window_limit < 0:
        raise ValueError("window-limit must be nonnegative")
    if max_tail_share is not None and not (math.isfinite(max_tail_share) and 0 <= max_tail_share <= 1):
        raise ValueError("max-tail-share must be finite and in [0,1]")


@torch.no_grad()
def score_window(model, ids, mode, gains, config, *, projection_chunk=256, last_only=False):
    if projection_chunk < 1:
        raise ValueError("projection_chunk must be positive")
    if ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 1:
        raise ValueError("ids must be a nonempty [batch, tokens] tensor")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with replace_attention(mode, gains, **config) as calls:
        hidden = model.model(ids, use_cache=False).last_hidden_state
    if calls[0] != model.config.num_hidden_layers:
        raise AssertionError("not every transformer layer used the requested attention")
    last = model.lm_head(hidden[:, -1:]).float()[:, 0]
    nll = correct = count = 0
    if not last_only:
        for start in range(0, ids.shape[1] - 1, projection_chunk):
            stop = min(start + projection_chunk, ids.shape[1] - 1)
            logits = model.lm_head(hidden[:, start:stop]).float()
            target = ids[:, start + 1:stop + 1]
            nll += float(F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction="sum"))
            correct += int((logits.argmax(-1) == target).sum())
            count += target.numel()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {"tokens": ids.numel(), "targets": count, "nll_sum": nll, "top1_correct": correct,
            "ce_nats": nll / count if count else None, "seconds": elapsed,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9}, last


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .streaming_qwen import make_choice_niah_ids, score_choice
    from .kaggle_10m_runner import machine_info, find_model
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="/kaggle/working/tail_bundle")
    parser.add_argument("--model")
    parser.add_argument("--out", default="/kaggle/working/ssa_tail_fullscale.json")
    parser.add_argument("--contexts", default="512,4096,8192,32768")
    parser.add_argument("--probe-contexts", default="8192,32768,131072")
    parser.add_argument("--window-limit", type=int, default=0, help="debug only; zero uses every test window")
    parser.add_argument("--allow-other-gpu", action="store_true", help="local preflight only")
    parser.add_argument("--tail-mode", choices=("prototype", "jensen"), default="prototype")
    parser.add_argument("--gain-mode", choices=("saved", "zero"), default="saved")
    parser.add_argument("--max-tail-share", type=float)
    args = parser.parse_args()
    try:
        validate_protocol(args.contexts, args.probe_contexts, args.window_limit, args.max_tail_share)
    except ValueError as error:
        parser.error(str(error))
    torch.set_num_threads(4)
    torch.manual_seed(0)
    root = Path(args.bundle)
    report = {"status": "starting", "protocol": vars(args), "manifest": json.loads((root / "manifest.json").read_text()),
              "started": time.time(), "corpus": {}, "probes": []}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temp = out.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(out)

    def log(message):
        print(message, flush=True)
        report["last_progress"] = str(message); save()

    try:
        report["machine"] = machine_info(torch)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA allocation missing")
        if not args.allow_other_gpu and "6000" not in report["machine"]["gpu"]:
            raise RuntimeError("requested RTX 6000 was not allocated")
        path = args.model or find_model()
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, attn_implementation="sdpa", dtype=torch.bfloat16).cuda().eval()
        model.requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        gains = torch.tensor(json.loads((root / "gains.json").read_text())["trained_log_gains"], device="cuda")
        if (gains.shape != (model.config.num_hidden_layers, model.config.num_attention_heads)
                or not torch.isfinite(gains).all()):
            raise ValueError("saved gain/model geometry or finiteness mismatch")
        if args.gain_mode == "zero":
            gains.zero_()
        config = {"block": 64, "top_blocks": 2, "cells": 16, "router": "batched_tree", "query_chunk": 256,
                  "tail_mode": args.tail_mode, "max_tail_share": args.max_tail_share}
        report["attention"] = config
        report["model_config"] = model.config.to_dict()
        tokens = np.load(root / "tokens.npz")["test"]
        report["test_tokens"] = len(tokens)
        report["test_token_sha256"] = hashlib.sha256(tokens.tobytes()).hexdigest()
        report["scope"] = "fixed algorithm and gain mode specified in protocol; full official WikiText-2 test; unscaled positions; no test fitting; approximate tail, not a certificate"
        log({"machine": report["machine"], "test_tokens": len(tokens)})

        # Check streaming projection loss, dense fallback and batched tree wiring
        # before any expensive corpus/long-context arm.
        smoke = torch.tensor(tokens[:512].astype(np.int64), device="cuda")[None]
        dense_row, dense_last = score_window(model, smoke, "dense", gains, config)
        full_config = dict(config, top_blocks=8, router="flat")
        full_row, full_last = score_window(model, smoke, "tail", gains, full_config)
        with torch.no_grad():
            hf_ce = float(model(smoke, labels=smoke, use_cache=False).loss)
        gate = {"max_abs_last_logit_delta": float((dense_last - full_last).abs().max()),
                "dense_fallback_ce_delta": abs(dense_row["ce_nats"] - full_row["ce_nats"]),
                "chunked_projection_ce_delta": abs(dense_row["ce_nats"] - hf_ce)}
        gate["passed"] = gate["max_abs_last_logit_delta"] < 0.5 and gate["dense_fallback_ce_delta"] < 0.02 and gate["chunked_projection_ce_delta"] < 1e-5
        report["gate"] = gate; log({"gate": gate})
        if not gate["passed"]:
            raise RuntimeError("preflight model equivalence gate failed")
        del smoke, dense_last, full_last
        report["status"] = "running"
        for context in map(int, args.contexts.split(",")):
            rows = report["corpus"][str(context)] = {}
            for mode in ("dense", "sparse", "tail"):
                record = rows[mode] = {"windows": [], "status": "running"}
                try:
                    for index, start in enumerate(range(0, len(tokens) - 1, context)):
                        if args.window_limit and index >= args.window_limit:
                            break
                        ids = torch.tensor(tokens[start:start + context].astype(np.int64), device="cuda")[None]
                        row, _ = score_window(model, ids, mode, gains, config)
                        row["start"] = start
                        record["windows"].append(row)
                        if index % 10 == 0:
                            log({"context": context, "mode": mode, "window": index, "ce": row["ce_nats"]})
                        else:
                            save()
                    count = sum(x["targets"] for x in record["windows"])
                    nll = sum(x["nll_sum"] for x in record["windows"])
                    record.update(targets=count, ce_nats=nll / count, perplexity=math.exp(nll / count),
                                  token_top1_accuracy=sum(x["top1_correct"] for x in record["windows"]) / count,
                                  seconds=sum(x["seconds"] for x in record["windows"]),
                                  peak_allocated_gb=max(x["peak_allocated_gb"] for x in record["windows"]), status="complete")
                    log({"context": context, "mode": mode, "summary": {k: v for k, v in record.items() if k != "windows"}})
                except Exception:
                    record.update(status="error", error=traceback.format_exc()); log(record["error"])
                    torch.cuda.empty_cache()
        for context in filter(None, args.probe_contexts.split(",")):
            for depth in (0.1, 0.5, 0.9):
                ids, labels, gold = make_choice_niah_ids(tokenizer, int(context), depth=depth, block=64, device="cuda")
                trial = {"context": int(context), "depth": depth, "modes": {}}
                report["probes"].append(trial)
                for mode in ("dense", "sparse", "tail"):
                    try:
                        row, last = score_window(model, ids, mode, gains, config, last_only=True)
                        row["retrieval"] = score_choice(last, labels, gold)
                        trial["modes"][mode] = row
                        log({"probe": context, "depth": depth, "mode": mode, "result": row})
                    except Exception:
                        trial["modes"][mode] = {"error": traceback.format_exc()}; log(trial["modes"][mode])
                        torch.cuda.empty_cache()
        errors = any(row.get("status") == "error" for modes in report["corpus"].values() for row in modes.values())
        errors |= any("error" in row for trial in report["probes"] for row in trial["modes"].values())
        report["status"] = "complete_with_errors" if errors else "complete"
    except Exception:
        report.update(status="failed", error=traceback.format_exc()); print(report["error"], flush=True)
    finally:
        report["finished"] = time.time(); save()
    if report["status"] == "failed":
        raise RuntimeError("full-scale run failed; see saved telemetry")


if __name__ == "__main__":
    main()
