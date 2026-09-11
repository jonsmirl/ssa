"""Training-only oracle for bounded correction at one frozen dense-model layer.

The exact fixed linear reader is W_O, not the nonlinear remainder of the model.
Oracle gates use dense outputs and are NOT deployable inference gates.
"""
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F


def scalar_oracle_gate(error, direction):
    """Minimize ||error-theta*direction||^2 for one theta per row in [0,1]."""
    denominator = direction.square().sum(-1)
    numerator = (error * direction).sum(-1)
    safe = denominator.masked_fill(denominator == 0, 1)
    return (numerator / safe).clamp(0, 1).masked_fill(denominator == 0, 0)


def head_diagonal_error(error, projection, heads):
    """Sum separate head read errors; omits cross-head terms of the true loss."""
    if error.shape[-1] % heads:
        raise ValueError("head dimensions must divide the concatenated output")
    width = error.shape[-1] // heads
    result = error.new_zeros(error.shape[:-1])
    for head in range(heads):
        part = slice(head * width, (head + 1) * width)
        result += F.linear(error[..., part], projection[:, part]).square().sum(-1)
    return result


def joint_head_gate(error, directions, initial, iterations=100):
    """Feasible box-QP iterate for [query,head,read_dim] directions.

    Cyclic exact coordinate minimization includes the full cross-head Gram.
    Returns a feasible solution, not an asserted exact optimizer. The reported
    first-order gap is a convex lower-bound gap in exact arithmetic, and is
    evaluated numerically here without an outward-rounding certificate.
    """
    gram = directions @ directions.transpose(-1, -2)
    rhs = (directions * error[:, None]).sum(-1)
    diagonal = gram.diagonal(dim1=-2, dim2=-1)
    denominator = diagonal.masked_fill(diagonal <= 0, 1)
    gate = initial.clone().clamp(0, 1)
    for iteration in range(iterations):
        gradient = (gram @ gate[..., None])[..., 0] - rhs
        for head in range(gate.shape[-1]):
            new = (gate[:, head] - gradient[:, head] / denominator[:, head]).clamp(0, 1)
            new = new.masked_fill(diagonal[:, head] <= 0, 0)
            delta = new - gate[:, head]
            gate[:, head] = new
            gradient += gram[:, :, head] * delta[:, None]
        gradient = (gram @ gate[..., None])[..., 0] - rhs
        projected_gradient = gate - (gate - gradient / denominator).clamp(0, 1)
        if projected_gradient.abs().max().item() < 1e-5:
            break
    # Gradient is for half the squared norm. Minimize its affine model over
    # the box to obtain a first-order suboptimality bound (exact arithmetic).
    corner = (gradient < 0).to(gate.dtype)
    gap = (2 * gradient * (gate - corner)).sum(-1).clamp_min(0)
    return gate, {"iterations": iteration + 1,
                  "projected_kkt_max": projected_gradient.abs().max().item(),
                  "projected_kkt_mean": projected_gradient.abs().amax(-1).mean().item(),
                  "first_order_gap_mean": gap.mean().item(),
                  "first_order_gap_max": gap.max().item()}


@contextmanager
def capture_dense_attention(layer):
    """Call original SDPA unchanged, retaining one layer's actual Q/K/V/output."""
    original = F.scaled_dot_product_attention
    captured = {"calls": 0}
    def wrapper(q, k, v, *args, **kwargs):
        index = captured["calls"]
        captured["calls"] += 1
        output = original(q, k, v, *args, **kwargs)
        if index == layer:
            if (args[0] if args else kwargs.get("attn_mask")) is not None:
                raise ValueError("diagnostic requires unpadded causal attention")
            captured.update(q=q.detach().clone(), k=k.detach().clone(),
                            v=v.detach().clone(), dense=output.detach().clone())
        return output
    F.scaled_dot_product_attention = wrapper
    try:
        yield captured
    finally:
        F.scaled_dot_product_attention = original


def tensor_hash(tensor):
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@torch.no_grad()
def main():
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .hybrid_tail_attention import hybrid_tail_attention
    from .tail_tree_router import batched_tail_tree_routes
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--layer", type=int, default=18, help="zero-based decoder layer")
    parser.add_argument("--out", default="runs/reader_weighted_tail_diagnostic.json")
    args = parser.parse_args()
    if args.context <= 192:
        parser.error("context must exceed the exact 192-key budget")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    started = time.perf_counter()
    name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    arrow = next((Path.home() / ".cache/huggingface/datasets/wikitext").glob(
        "wikitext-2-raw-v1/*/*/wikitext-validation.arrow"))
    data = Dataset.from_file(str(arrow))
    content = "\n".join(s for s in data["text"] if s.strip())
    tokens = np.asarray(tokenizer(content, truncation=False)["input_ids"][:args.context], dtype=np.int64)
    if len(tokens) != args.context:
        raise ValueError("not enough cached validation tokens")
    model = AutoModelForCausalLM.from_pretrained(name, local_files_only=True,
        attn_implementation="sdpa", dtype=torch.bfloat16).cuda().eval().requires_grad_(False)
    if not 0 <= args.layer < model.config.num_hidden_layers:
        parser.error("layer outside model")
    with capture_dense_attention(args.layer) as captured:
        model.model(torch.tensor(tokens, device="cuda")[None], use_cache=False)
    if captured["calls"] != model.config.num_hidden_layers or "q" not in captured:
        raise AssertionError("not every decoder layer used SDPA")
    q, k, v, dense = (captured[key] for key in ("q", "k", "v", "dense"))
    projection = model.model.layers[args.layer].self_attn.o_proj.weight.detach().float()
    gain_path = Path("runs/qwen_tail_final/results.json")
    gains = torch.tensor(json.loads(gain_path.read_text())["trained_log_gains"], device="cuda")[args.layer]
    config = dict(block=64, top_blocks=2, cells=16, query_chunk=256)
    routes = batched_tail_tree_routes(q, k, block=64, top_blocks=2, query_chunk=256)
    sparse = hybrid_tail_attention(q, k, v, routing_blocks=routes, use_tail=False, **config)
    capped = hybrid_tail_attention(q, k, v, routing_blocks=routes, log_gain=gains,
                                  max_tail_share=.25, tail_mode="prototype", **config)
    flatten = lambda x: x[0].transpose(0, 1).reshape(args.context, -1).float()
    # Initial <=192-key prefixes are completely selected; focus on nontrivial reads.
    dense, sparse, capped = (flatten(x)[192:] for x in (dense, sparse, capped))
    error, direction = dense - sparse, capped - sparse
    read_error, read_direction = F.linear(error, projection), F.linear(direction, projection)
    weighted_gate = scalar_oracle_gate(read_error, read_direction)
    raw_gate = scalar_oracle_gate(error, direction)
    width, heads = q.shape[-1], q.shape[1]
    per_head_read_direction = torch.stack([
        F.linear(direction[:, head * width:(head + 1) * width],
                 projection[:, head * width:(head + 1) * width])
        for head in range(heads)], dim=1)
    joint_gate, joint_solver = joint_head_gate(read_error, per_head_read_direction,
                                              weighted_gate[:, None].expand(-1, heads))
    joint_residual = error - (direction.reshape(-1, heads, width) * joint_gate[..., None]).flatten(1)
    residuals = {"sparse": error, "capped": error - direction,
                 "raw_oracle": error - raw_gate[:, None] * direction,
                 "reader_oracle": error - weighted_gate[:, None] * direction,
                 "joint_head_feasible_oracle": joint_residual}
    rows = {}
    for mode, residual in residuals.items():
        raw = residual.square().sum(-1)
        projected = F.linear(residual, projection).square().sum(-1)
        diagonal = head_diagonal_error(residual, projection, q.shape[1])
        rows[mode] = {"raw_mean_squared_l2": raw.mean().item(),
                      "projected_mean_squared_l2": projected.mean().item(),
                      "head_diagonal_projected_mean_squared_l2": diagonal.mean().item(),
                      "cross_head_term_mean": (projected - diagonal).mean().item(),
                      "cross_head_term_mean_absolute": (projected - diagonal).abs().mean().item()}
    losses = {mode: F.linear(residual, projection).square().sum(-1) for mode, residual in residuals.items()}
    oracle_excess = (losses["reader_oracle"] - torch.minimum(losses["sparse"], losses["capped"])).max().item()
    def gate_summary(gate):
        return {"mean": gate.mean().item(), "zero_fraction": (gate == 0).float().mean().item(),
                "one_fraction": (gate == 1).float().mean().item(),
                "quantiles_0_25_50_75_100": gate.quantile(torch.linspace(0, 1, 5, device=gate.device)).tolist()}
    report = {"config": vars(args), "model": name, "split": "official validation, first window; development diagnostic",
        "hardware": torch.cuda.get_device_name(0), "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {f: hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()
            for f in ("reader_weighted_tail_diagnostic.py", "hybrid_tail_attention.py", "tail_correction.py", "tail_tree_router.py")},
        "token_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        "gain_sha256": hashlib.sha256(gain_path.read_bytes()).hexdigest(),
        "tensor_sha256": {key: tensor_hash(captured[key]) for key in ("q", "k", "v")},
        "projection_sha256": tensor_hash(projection), "q_shape": list(q.shape), "k_shape": list(k.shape),
        "queries_measured": len(error), "query_positions_zero_based": [192, args.context - 1],
        "attention": {**config, "router": "batched_tree", "max_tail_share": .25, "tail_mode": "prototype", "gain_mode": "saved"},
        "rows": rows, "weighted_gate": gate_summary(weighted_gate), "raw_gate": gate_summary(raw_gate),
        "joint_head_solver": {**joint_solver, "variables_per_query": heads,
            "algorithm": "cyclic coordinate descent, full cross-head Gram, box [0,1], initialized at shared oracle",
            "scope": "Training-only feasible numerical solution; first-order gap is not an outward-rounded certificate or an exact optimum claim",
            "gate": gate_summary(joint_gate.flatten()),
            "max_loss_excess_over_shared": (losses["joint_head_feasible_oracle"] - losses["reader_oracle"]).max().item()},
        "gate_mean_absolute_difference": (weighted_gate - raw_gate).abs().mean().item(),
        "capped_projected_better_fraction": (losses["capped"] < losses["sparse"]).float().mean().item(),
        "oracle_max_excess_over_best_endpoint": oracle_excess,
        "scope": "One layer at identical dense hidden inputs. Theta oracle uses unavailable dense inference target; no trained gate, no end-to-end CE or retrieval claim. W_O is the exact fixed linear reader; no theorem for later nonlinear layers. Attention outputs are bf16, diagnostic projections float32.",
        "seconds": time.perf_counter() - started}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": rows, "weighted_gate": report["weighted_gate"],
                      "gate_mean_absolute_difference": report["gate_mean_absolute_difference"],
                      "oracle_max_excess": oracle_excess, "out": str(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
