"""Complete frozen Qwen comparison: dense, sparse, fixed-state tail correction.

Gain selection uses validation tokens only. Evaluation text is disjoint from
both validation and optional CE training. A scoped SDPA hook restores the
original function even on exceptions. No download is required with cached data.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .hybrid_tail_attention import hybrid_tail_attention

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@contextmanager
def replace_attention(mode, gains=0.0, *, block=64, top_blocks=2, cells=16, router="flat", query_chunk=32,
                      tail_mode="prototype", max_tail_share=None):
    original = F.scaled_dot_product_attention
    calls = [0]

    def replacement(q, k, v, *args, **kw):
        layer = calls[0]; calls[0] += 1
        if mode == "dense":
            return original(q, k, v, *args, **kw)
        mask = args[0] if args else kw.get("attn_mask")
        if mask is not None:
            raise ValueError("demo expects unpadded causal self-attention without an explicit mask")
        if q.shape[2] <= block:
            return original(q, k, v, *args, **kw)
        gain = gains[layer] if isinstance(gains, torch.Tensor) and gains.ndim == 2 else gains
        routes = None
        if router == "tree":
            from .tail_tree_router import tail_tree_routes
            routes = tail_tree_routes(q, k, block=block, top_blocks=top_blocks)
        elif router == "batched_tree":
            from .tail_tree_router import batched_tail_tree_routes
            routes = batched_tail_tree_routes(q, k, block=block, top_blocks=top_blocks, query_chunk=query_chunk)
        if torch.is_grad_enabled() and isinstance(gain, torch.Tensor) and gain.requires_grad:
            def compute(aq, ak, av, ag):
                return hybrid_tail_attention(aq, ak, av, block=block, top_blocks=top_blocks,
                                             cells=cells, log_gain=ag, use_tail=mode == "tail", routing_blocks=routes, query_chunk=query_chunk,
                                             tail_mode=tail_mode, max_tail_share=max_tail_share)
            return checkpoint(compute, q, k, v, gain, use_reentrant=False)
        return hybrid_tail_attention(q, k, v, block=block, top_blocks=top_blocks, cells=cells,
                                     log_gain=gain, use_tail=mode == "tail", routing_blocks=routes, query_chunk=query_chunk,
                                     tail_mode=tail_mode, max_tail_share=max_tail_share)
    F.scaled_dot_product_attention = replacement
    try:
        yield calls
    finally:
        F.scaled_dot_product_attention = original


def loss(model, ids, mode, gain, config):
    with replace_attention(mode, gain, **config) as calls:
        output = model(ids, labels=ids, use_cache=False)
    if calls[0] != model.config.num_hidden_layers:
        raise AssertionError("not every decoder layer used the attention hook")
    return output.loss


@torch.no_grad()
def evaluate(model, examples, mode, gains, config):
    values = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for ids in examples:
        values.append(float(loss(model, ids[None], mode, gains, config)))
    torch.cuda.synchronize()
    mean = sum(values) / len(values)
    return {"ce_nats": mean, "perplexity": math.exp(mean), "per_example_ce": values,
            "seconds": time.perf_counter() - start}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/qwen_tail_demo")
    p.add_argument("--context", type=int, default=1024)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--test-examples", type=int, default=6)
    p.add_argument("--official-splits", action="store_true")
    p.add_argument("--eval-contexts", default="")
    p.add_argument("--load-gains", help="results.json containing trained_log_gains; evaluate without retraining")
    p.add_argument("--router", choices=("flat", "tree", "batched_tree"), default="flat")
    args = p.parse_args()
    if args.context <= 64 or args.test_examples < 1 or args.steps < 0:
        p.error("context must exceed 64; test-examples positive; steps nonnegative")
    if args.load_gains and args.steps:
        p.error("--load-gains is evaluation only")
    if args.eval_contexts and not args.official_splits:
        p.error("context extension requires --official-splits")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    name = "Qwen/Qwen2.5-0.5B"
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True,
               attn_implementation="sdpa", torch_dtype=torch.bfloat16).cuda().eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n".join(s for s in dataset["text"] if len(s.strip()) > 0)
    tokens = tokenizer(text, return_tensors="pt", truncation=False).input_ids[0]
    ctx = args.context
    # Fixed non-overlapping windows. The final test is never used for selection.
    train = [tokens[i * ctx:(i + 1) * ctx].cuda() for i in range(16)]
    valid = [tokens[i * ctx:(i + 1) * ctx].cuda() for i in range(16, 18)]
    test = [tokens[i * ctx:(i + 1) * ctx].cuda() for i in range(20, 20 + args.test_examples)]
    if args.official_splits:
        def split_examples(split, count, length=None):
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            tx = "\n".join(s for s in ds["text"] if len(s.strip()) > 0)
            ids = tokenizer(tx, return_tensors="pt", truncation=False).input_ids[0]
            length = ctx if length is None else length
            return [ids[i * length:(i + 1) * length].cuda() for i in range(count)]
        train = [tokens[i * ctx:(i + 1) * ctx].cuda() for i in range(64)]
        valid = split_examples("validation", 4)
        test = split_examples("test", args.test_examples)
    config = {"block": 64, "top_blocks": 2, "cells": 16, "router": args.router}
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    report = {"model": name, "config": vars(args), "attention": config,
              "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_sha256": {s: hashlib.sha256(Path(__file__).with_name(s).read_bytes()).hexdigest()
                                for s in ("hybrid_tail_attention.py", "qwen_tail_demo.py", "tail_tree_router.py")},
              "test_token_sha256": hashlib.sha256(torch.stack(test).cpu().numpy().tobytes()).hexdigest(),
              "hardware": torch.cuda.get_device_name(0), "validation": {}, "test": {},
              "scope": "complete 24-layer 14-query-head frozen model; quality experiment; router specified in attention config"}
    if args.load_gains:
        best_gain = json.loads(Path(args.load_gains).read_text())["selected_gain"]
    else:
        for mode in ("dense", "sparse"):
            report["validation"][mode] = evaluate(model, valid, mode, 0., config)
            print("validation", mode, report["validation"][mode], flush=True)
        for gain in (-2., 0., 2., 4., 6.):
            report["validation"][str(gain)] = evaluate(model, valid, "tail", gain, config)
            print("validation tail", gain, report["validation"][str(gain)], flush=True)
        best_gain = min((-2., 0., 2., 4., 6.), key=lambda g: report["validation"][str(g)]["ce_nats"])
    report["selected_gain"] = best_gain
    gains = nn.Parameter(torch.full((model.config.num_hidden_layers, model.config.num_attention_heads), best_gain, device="cuda"))
    if args.load_gains:
        loaded = torch.tensor(json.loads(Path(args.load_gains).read_text())["trained_log_gains"], device="cuda")
        if loaded.shape != gains.shape or not torch.isfinite(loaded).all():
            raise ValueError("checkpoint gain shape or finiteness mismatch")
        gains = loaded
        report["loaded_gain_sha256"] = hashlib.sha256(Path(args.load_gains).read_bytes()).hexdigest()
    if args.steps:
        opt = torch.optim.Adam([gains], lr=0.05)
        history = []
        best = report["validation"][str(best_gain)]["ce_nats"]
        best_gains = gains.detach().clone()
        for step in range(args.steps):
            objective = loss(model, train[step % len(train)][None], "tail", gains, config)
            opt.zero_grad(); objective.backward(); opt.step()
            if (step + 1) % 10 == 0 or step == args.steps - 1:
                score = evaluate(model, valid, "tail", gains, config)
                if score["ce_nats"] < best:
                    best = score["ce_nats"]; best_gains = gains.detach().clone()
                history.append({"step": step + 1, "train_ce": float(objective.detach()), "validation_ce": score["ce_nats"]})
                print(history[-1], flush=True)
        gains = best_gains
        report["training"] = history
    for mode, gain in (("dense", 0.), ("sparse", 0.), ("tail", best_gain)):
        report["test"][mode] = evaluate(model, test, mode, gain, config)
        print("test", mode, report["test"][mode], flush=True)
    if args.steps or args.load_gains:
        report["test"]["ce_trained_tail"] = evaluate(model, test, "tail", gains, config)
        print("test ce_trained_tail", report["test"]["ce_trained_tail"], flush=True)
    if args.eval_contexts:
        if not args.official_splits:
            raise ValueError("context extension uses the official held-out test split")
        report["extended_test"] = {}
        for length in map(int, args.eval_contexts.split(",")):
            examples = split_examples("test", args.test_examples, length)
            report["extended_test"][str(length)] = {}
            for mode, gain in (("dense", 0.), ("sparse", 0.), ("tail", gains.detach())):
                result = evaluate(model, examples, mode, gain, config)
                report["extended_test"][str(length)][mode] = result
                print("extended test", length, mode, result, flush=True)
    torch.save({"gains": gains.detach(), "config": config}, root / "gains.pt")
    report["trained_log_gains"] = gains.detach().cpu().tolist()
    report["persistent_tail_scalars"] = model.config.num_hidden_layers * model.config.num_key_value_heads * config["cells"] * (model.config.hidden_size // model.config.num_attention_heads + 1)
    report["trainable_parameters"] = gains.numel()
    report["peak_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("saved", root / "results.json", flush=True)


if __name__ == "__main__":
    main()
