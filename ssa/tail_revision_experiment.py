"""Validation-only algorithm comparison; test evaluation is a separate invocation.

Candidate definitions are fixed before observing validation results. Saved gains
remain unchanged. No test result is used to choose a cell rule or influence cap.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from .fullscale_tail_runner import score_window


CANDIDATES = {
    "dense": ("dense", "prototype", None, "saved"),
    "sparse": ("sparse", "prototype", None, "saved"),
    "old_tail": ("tail", "prototype", None, "saved"),
    "jensen": ("tail", "jensen", None, "zero"),
    "jensen_saved": ("tail", "jensen", None, "saved"),
    "prototype_capped": ("tail", "prototype", .25, "saved"),
    "jensen_capped": ("tail", "jensen", .25, "zero"),
}


def main():
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/tail_revision_validation.json")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--contexts", default="512,8192,32768")
    p.add_argument("--windows", type=int, default=2)
    p.add_argument("--candidates", default=",".join(CANDIDATES))
    args = p.parse_args()
    torch.set_num_threads(4)
    model_name = "Qwen/Qwen2.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, attn_implementation="sdpa", dtype=torch.bfloat16).cuda().eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    arrow = next((Path.home() / ".cache/huggingface/datasets/wikitext").glob(f"wikitext-2-raw-v1/*/*/wikitext-{args.split}.arrow"))
    dataset = Dataset.from_file(str(arrow))
    content = "\n".join(s for s in dataset["text"] if len(s.strip()) > 0)
    tokens = np.asarray(tokenizer(content, truncation=False)["input_ids"], dtype=np.int64)
    gains_path = Path("runs/qwen_tail_final/results.json")
    saved = torch.tensor(json.loads(gains_path.read_text())["trained_log_gains"], device="cuda")
    report = {"config": vars(args), "candidates": CANDIDATES, "hardware": torch.cuda.get_device_name(0),
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("tail_correction.py", "hybrid_tail_attention.py", "tail_revision_experiment.py")},
              "token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
              "gain_sha256": hashlib.sha256(gains_path.read_bytes()).hexdigest(), "results": {}}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    for length in map(int, args.contexts.split(",")):
        report["results"][str(length)] = {}
        for candidate in args.candidates.split(","):
            mode, tail_mode, cap, gain_kind = CANDIDATES[candidate]
            gains = torch.zeros_like(saved) if gain_kind == "zero" else saved
            cfg = {"block": 64, "top_blocks": 2, "cells": 16, "router": "batched_tree", "query_chunk": 256,
                   "tail_mode": tail_mode, "max_tail_share": cap}
            rows = []
            for index in range(args.windows):
                ids = torch.tensor(tokens[index * length:(index + 1) * length], device="cuda")[None]
                row, _ = score_window(model, ids, mode, gains, cfg)
                rows.append(row)
            targets = sum(row["targets"] for row in rows)
            ce = sum(row["nll_sum"] for row in rows) / targets
            result = {"ce_nats": ce, "perplexity": math.exp(ce), "windows": rows,
                      "seconds": sum(row["seconds"] for row in rows)}
            report["results"][str(length)][candidate] = result
            out.write_text(json.dumps(report, indent=2) + "\n")
            print(length, candidate, ce, result["perplexity"], flush=True)


if __name__ == "__main__":
    main()
