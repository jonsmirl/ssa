"""Local scaling diagnostic for the streamed Qwen long-context retrieval failure."""
from __future__ import annotations

import argparse
import json
import os
import time


def _ints(value):
    return [int(item) for item in value.split(",") if item]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--lengths", default="131072,524288")
    parser.add_argument("--query-subs", default="128,32")
    parser.add_argument("--rope-factor", type=float, default=306.0)
    parser.add_argument("--force-needle", action="store_true")
    parser.add_argument("--route-geometry", choices=("pre_rope", "post_rope"), default="pre_rope")
    parser.add_argument("--share-route-from", type=int, default=12)
    parser.add_argument("--out", default="runs/query_subblock_diagnostic.json")
    args = parser.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from ssa.streaming_qwen import (
        StreamingQwenConfig, StreamingQwenPrefill, make_choice_niah_ids, score_choice,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    native = int(config.max_position_embeddings)
    config.rope_theta = (getattr(config, "rope_theta", None)
                         or (getattr(config, "rope_scaling", None) or {}).get("rope_theta")
                         or 1e6)
    config.rope_scaling = {
        "rope_type": "yarn", "factor": args.rope_factor,
        "original_max_position_embeddings": native,
    }
    config.max_position_embeddings = int(native * args.rope_factor)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16, local_files_only=True,
    ).eval().cuda()
    model.config.use_cache = False

    payload = {
        "gpu": torch.cuda.get_device_name(0), "model": args.model,
        "rope": {"type": "static_yarn", "factor": args.rope_factor, "native": native},
        "rows": [],
    }

    def save():
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)

    for n in _ints(args.lengths):
        ids, labels, gold, metadata = make_choice_niah_ids(
            tokenizer, n, depth=0.5, device="cuda", return_metadata=True)
        for query_sub in _ints(args.query_subs):
            cfg = StreamingQwenConfig(
                query_sub=query_sub, route_geometry=args.route_geometry,
                share_route_from=args.share_route_from,
            )
            print(f"starting n={n:,} query_sub={query_sub}", flush=True)
            streamer = StreamingQwenPrefill(model, cfg)
            started = time.time()
            logits, stats = streamer(
                ids, probe_block=metadata["needle_block"],
                force_block=(metadata["needle_block"] if args.force_needle else None),
            )
            row = {
                "tokens": n, "query_sub": query_sub, "forced_needle": args.force_needle,
                "route_geometry": args.route_geometry,
                "share_route_from": args.share_route_from,
                "wall_s": round(time.time() - started, 3),
                "quality": score_choice(logits, labels, gold), "stats": stats,
            }
            payload["rows"].append(row)
            save()
            print(json.dumps({
                "tokens": n, "query_sub": query_sub, "wall_s": row["wall_s"],
                "quality": row["quality"],
                "probe": [{
                    "layer": item["layer"], "selected": item["heads_selected"],
                    "exact_top_c": item["heads_exact_top_c"],
                } for item in stats["route_probe"]],
            }), flush=True)
            del logits, streamer
            torch.cuda.empty_cache()
        del ids
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
