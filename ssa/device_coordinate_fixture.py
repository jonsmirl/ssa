"""Offline natural 32K mixed-article geometry for context-growth measurements.

No article or geometry is repeated. The two previous 8K articles are excluded.
This is a deterministic test-split stream, not a claim of untouched evaluation
data or semantic quality: WikiText has already been inspected in this project.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from .partial_coordinate_fixture import ARTICLES, LAYERS, MODEL, QUERY_HEADS, REVISION, capture_geometry


def article_stream(rows, excluded=ARTICLES):
    """Yield distinct top-level articles in source order, excluding named ones."""
    headings = []
    for i, row in enumerate(rows):
        match = re.fullmatch(r"\s*= ([^=]+) =\s*", row)
        if match:
            headings.append((i, match.group(1).strip()))
    seen = set()
    for index, (start, title) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(rows)
        if title in excluded:
            continue
        if title in seen:
            raise ValueError(f"duplicate article title: {title}")
        seen.add(title)
        yield {"title": title, "row_start": start, "row_end_exclusive": end,
               "text": "\n".join(row for row in rows[start:end] if row.strip())}


def build_stream(rows, tokenizer, n):
    """Join consecutive eligible articles until their natural text supplies n tokens."""
    if n < 1:
        raise ValueError("positive token count required")
    pieces, metadata = [], []
    text = ""
    for article in article_stream(rows):
        offset = len(text) + (2 if pieces else 0)
        pieces.append(article["text"])
        text = "\n\n".join(pieces)
        metadata.append({k: v for k, v in article.items() if k != "text"} | {
            "character_start": offset, "character_end_exclusive": len(text),
            "text_sha256": hashlib.sha256(article["text"].encode()).hexdigest()})
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(encoded["input_ids"]) >= n:
            ids = np.asarray(encoded["input_ids"][:n], dtype=np.int64)
            used_end = int(encoded["offset_mapping"][n - 1][1])
            for item in metadata:
                item["used_character_end_exclusive"] = min(item["character_end_exclusive"], used_end)
                item["truncated_by_token_limit"] = item["character_end_exclusive"] > used_end
            return ids, {"articles": metadata, "separator": "\n\n",
                         "excluded_articles": list(ARTICLES),
                         "candidate_stream_tokens": len(encoded["input_ids"]),
                         "candidate_stream_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                         "used_text_character_end_exclusive": used_end,
                         "used_text_prefix_sha256": hashlib.sha256(text[:used_end].encode()).hexdigest()}
    raise ValueError("eligible natural text does not supply the requested token count")


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="/tmp/ssa_device_growth")
    parser.add_argument("--n", type=int, default=32768)
    args = parser.parse_args()
    if args.n < 4:
        parser.error("--n must be at least 4")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA access required for frozen dense geometry extraction")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    arrow = (Path.home() / ".cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/"
             "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow")
    rows = list(Dataset.from_file(str(arrow))["text"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    ids, provenance = build_stream(rows, tokenizer, args.n)
    positions = np.asarray([args.n // 4 - 1, args.n // 2 - 1, args.n - 1], dtype=np.int64)
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"model": MODEL, "model_revision": REVISION, "dataset": "wikitext/wikitext-2-raw-v1",
                "split": "test", "arrow_path": str(arrow),
                "arrow_sha256": hashlib.sha256(arrow.read_bytes()).hexdigest(),
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    (Path(__file__), Path(__file__).with_name("partial_coordinate_fixture.py"))},
                "token_ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                "positions_zero_based": positions.tolist(), "n": args.n,
                "query_heads": list(QUERY_HEADS), "kv_heads_for_queries": [h // 7 for h in QUERY_HEADS],
                "layers_zero_based": list(LAYERS), "torch": torch.__version__,
                "hardware": torch.cuda.get_device_name(0), "stream": provenance,
                "scope": "Natural mixed-article stream, no article or geometry repetition; excludes the two prior8K articles. Context-growth timing fixture, not semantic quality or untouched held-out evidence. Frozen BF16 full24-layer dense geometry, saved as FP32 casts."}
    print(json.dumps({"articles": [a["title"] for a in provenance["articles"]],
                      "positions": positions.tolist(), "tokens": args.n}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
        local_files_only=True, attn_implementation="sdpa", dtype=torch.bfloat16).cuda().eval()
    model.requires_grad_(False)
    if (model.config.num_hidden_layers, model.config.num_attention_heads,
            model.config.num_key_value_heads) != (24, 14, 2):
        raise ValueError("unexpected Qwen model geometry")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.inference_mode(), capture_geometry(args.n, positions.tolist()) as capture:
        output = model.model(torch.from_numpy(ids)[None].cuda(), use_cache=False, return_dict=True)
        del output
    torch.cuda.synchronize()
    if capture["calls"] != 24 or len(capture["arrays"]) != 9:
        raise AssertionError("incomplete 24-layer dense capture")
    if not all(np.isfinite(x).all() for x in capture["arrays"].values()):
        raise ArithmeticError("nonfinite captured geometry")
    manifest.update(sdpa_calls=capture["calls"], sdpa_shapes=capture["shapes"],
                    prefill_capture_seconds=time.perf_counter() - start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
    path = root / "mixed_test_32768.npz" if args.n == 32768 else root / f"mixed_test_{args.n}.npz"
    np.savez_compressed(path, **capture["arrays"], positions=positions, query_heads=np.asarray(QUERY_HEADS),
                        token_ids=ids, metadata_json=np.asarray(json.dumps(manifest, sort_keys=True)))
    manifest.update(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"path": str(path), "sha256": manifest["sha256"],
                      "seconds": manifest["prefill_capture_seconds"],
                      "peak_GB": manifest["peak_allocated_bytes"] / 1e9}), flush=True)


if __name__ == "__main__":
    main()
