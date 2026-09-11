"""Fresh, named-document post-RoPE Qwen geometry; strictly offline extraction.

This is a frozen dense-model fixture, not an end-to-end sparse-model evaluation.
The four query heads are stored separately; heads 0/3 share KV head 0 and heads
7/10 share KV head 1. Query positions are zero-based and include their own key.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
LAYERS = (6, 12, 18)
QUERY_HEADS = (0, 3, 7, 10)
ARTICLES = ("Ise @-@ class battleship", "Second Battle of Naktong Bulge")


def named_articles(rows, names=ARTICLES):
    """Extract entire articles by exact top-level heading, never row guesses."""
    headings = []
    for i, row in enumerate(rows):
        match = re.fullmatch(r"\s*= ([^=]+) =\s*", row)
        if match:
            headings.append((i, match.group(1).strip()))
    found = {}
    for j, (start, title) in enumerate(headings):
        if title in names:
            if title in found:
                raise ValueError(f"duplicate article: {title}")
            end = headings[j + 1][0] if j + 1 < len(headings) else len(rows)
            found[title] = {"title": title, "row_start": start, "row_end_exclusive": end,
                            "text": "\n".join(x for x in rows[start:end] if x.strip())}
    if set(found) != set(names):
        raise ValueError(f"missing articles: {set(names) - set(found)}")
    return [found[name] for name in names]


def compact_kv(x):
    """Undo an explicitly repeated 14-head KV tensor, verifying all repeats."""
    import torch
    if x.shape[1] == 2:
        return x
    if x.shape[1] != 14:
        raise ValueError("expected 2 native or 14 repeated KV heads")
    for group in range(2):
        reference = x[:, group * 7:group * 7 + 1]
        if not torch.equal(x[:, group * 7:(group + 1) * 7], reference.expand(-1, 7, -1, -1)):
            raise ValueError("KV tensor does not match Qwen's 7-to-1 GQA grouping")
    return x[:, [0, 7]]


@contextmanager
def capture_geometry(n, positions, layers=LAYERS):
    """Scoped SDPA hook; restores global function even if the model raises."""
    import torch
    import torch.nn.functional as F
    original = F.scaled_dot_product_attention
    result = {"calls": 0, "arrays": {}, "shapes": []}

    def hook(q, k, v, *args, **kwargs):
        layer = result["calls"]
        result["calls"] += 1
        if tuple(q.shape) != (1, 14, n, 64):
            raise ValueError(f"unexpected query shape at layer {layer}: {tuple(q.shape)}")
        if k.shape != v.shape or k.shape[0] != 1 or tuple(k.shape[2:]) != (n, 64):
            raise ValueError("unexpected key/value shape")
        if k.shape[1] not in (2, 14):
            raise ValueError("unexpected KV head count")
        result["shapes"].append({"layer": layer, "q": list(q.shape), "k": list(k.shape)})
        if layer in layers:
            kh, vh = compact_kv(k), compact_kv(v)
            qs = q[0, list(QUERY_HEADS)][:, positions]
            for name, value in ((f"Q_{layer}", qs), (f"K_{layer}", kh[0]), (f"V_{layer}", vh[0])):
                result["arrays"][name] = value.detach().to(dtype=torch.float32, device="cpu").numpy()
        return original(q, k, v, *args, **kwargs)

    F.scaled_dot_product_attention = hook
    try:
        yield result
    finally:
        F.scaled_dot_product_attention = original


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from datasets import Dataset
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="/tmp/ssa_partial_fresh")
    parser.add_argument("--n", type=int, default=8192)
    args = parser.parse_args()
    if args.n < 8:
        parser.error("--n must be at least 8")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA access is required for dense frozen-model extraction")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    arrow = (Path.home() / ".cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/0.0.0/"
             "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow")
    dataset = Dataset.from_file(str(arrow))
    articles = named_articles(list(dataset["text"]))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True,
        attn_implementation="sdpa", dtype=torch.bfloat16).cuda().eval()
    model.requires_grad_(False)
    if (model.config.num_hidden_layers, model.config.num_attention_heads,
            model.config.num_key_value_heads) != (24, 14, 2):
        raise ValueError("unexpected model geometry")
    positions = np.linspace(args.n // 2, args.n - 1, 4, dtype=np.int64)
    manifest = {"model": MODEL, "model_revision": REVISION, "split": "test",
                "dataset": "wikitext/wikitext-2-raw-v1", "arrow_path": str(arrow),
                "arrow_sha256": hashlib.sha256(arrow.read_bytes()).hexdigest(),
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "torch": torch.__version__, "hardware": torch.cuda.get_device_name(0),
                "layers_zero_based": list(LAYERS), "query_heads": list(QUERY_HEADS),
                "kv_heads_for_queries": [h // 7 for h in QUERY_HEADS],
                "positions_zero_based": positions.tolist(), "n": args.n, "articles": [],
                "scope": "Frozen BF16 dense prefill, full 24 layers; saved FP32 casts of post-RoPE QKV. No sparse-model rollout, training, or document tuning."}
    for index, article in enumerate(articles):
        ids = tokenizer(article["text"], add_special_tokens=False, return_tensors="pt").input_ids
        if ids.shape[1] < args.n:
            raise ValueError(f"article too short: {article['title']}: {ids.shape[1]}")
        tokens = ids[:, :args.n].contiguous()
        metadata = {k: v for k, v in manifest.items() if k != "articles"}
        metadata.update({k: v for k, v in article.items() if k != "text"})
        metadata.update(article_text_sha256=hashlib.sha256(article["text"].encode()).hexdigest(),
                        token_ids_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                        article_total_tokens=ids.shape[1])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), capture_geometry(args.n, positions.tolist()) as capture:
            result = model.model(tokens.cuda(), use_cache=False, return_dict=True)
            del result
        torch.cuda.synchronize()
        if capture["calls"] != 24 or len(capture["arrays"]) != 3 * len(LAYERS):
            raise AssertionError("incomplete 24-layer dense capture")
        metadata.update(sdpa_calls=capture["calls"], sdpa_shapes=capture["shapes"],
                        prefill_capture_seconds=time.perf_counter() - start,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated())
        output = root / f"article_{index}.npz"
        np.savez_compressed(output, **capture["arrays"], positions=positions,
                            query_heads=np.asarray(QUERY_HEADS), token_ids=tokens.numpy()[0],
                            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))
        metadata.update(path=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest())
        manifest["articles"].append(metadata)
        print(json.dumps({"path": str(output), "title": article["title"],
                          "seconds": metadata["prefill_capture_seconds"],
                          "peak_GB": metadata["peak_allocated_bytes"] / 1e9}), flush=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(root / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
