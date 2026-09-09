"""Kaggle RTX Pro 6000 driver for the kill-surviving >10M complete-transformer experiment."""
from __future__ import annotations

import glob
import json
import os
import platform
import time
import traceback


OUT = os.environ.get("SSA_10M_OUT", "/kaggle/working/ssa_10m_result.json")
N_MAIN = int(os.environ.get("SSA_10M_TOKENS", "10000128"))  # 78,126 exact 128-token blocks
N_REF = int(os.environ.get("SSA_10M_REF_TOKENS", "131072"))
RESULT = {"status": "starting", "target_tokens": N_MAIN, "started": time.time()}


def dump():
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(RESULT, f, indent=2)
    os.replace(tmp, OUT)


def log(message):
    print(message, flush=True)
    RESULT.setdefault("log", []).append(str(message))
    RESULT["log"] = RESULT["log"][-100:]
    dump()


def find_model():
    patterns = [
        "/kaggle/input/**/qwen2.5/transformers/0.5b/1/config.json",
        "/kaggle/input/**/qwen2-5/transformers/0.5b/1/config.json",
        "/kaggle/input/**/0.5b/1/config.json",
    ]
    hits = []
    for pattern in patterns:
        hits.extend(glob.glob(pattern, recursive=True))
    hits = sorted(set(hits), key=len)
    if not hits:
        raise FileNotFoundError("Qwen2.5-0.5B model attachment not found under /kaggle/input")
    return os.path.dirname(hits[0])


def machine_info(torch):
    import subprocess
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:
        smi = repr(exc)
    try:
        import psutil
        ram_gb = round(psutil.virtual_memory().total / 2**30, 2)
    except Exception:
        ram_gb = None
    return {
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_gib": (round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
                          if torch.cuda.is_available() else None),
        "system_ram_gib": ram_gb, "nvidia_smi": smi,
    }


def run():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from ssa.streaming_qwen import (
        StreamingQwenConfig, StreamingQwenPrefill, make_choice_niah_ids, score_choice,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle did not allocate a CUDA GPU")
    RESULT["machine"] = machine_info(torch)
    dump()
    log(f"machine: {RESULT['machine']}")
    if "6000" not in RESULT["machine"]["gpu"]:
        raise RuntimeError(f"expected RTX Pro 6000 allocation, got {RESULT['machine']['gpu']}")
    if RESULT["machine"]["gpu_total_gib"] < 80:
        raise RuntimeError("the >10M full-model run requires the 96 GiB RTX Pro 6000 tier")

    model_path = find_model()
    log(f"model attachment: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    native = int(getattr(config, "max_position_embeddings", 32768))
    rope_factor = max(1.0, N_MAIN / native)
    # Round upward so every tested token is inside the configured static-YaRN range.
    rope_factor = float(int(rope_factor + 0.999999))
    base = (getattr(config, "rope_theta", None)
            or (getattr(config, "rope_scaling", None) or {}).get("rope_theta") or 1e6)
    config.rope_theta = base
    config.rope_scaling = {
        "rope_type": "yarn", "factor": rope_factor,
        "original_max_position_embeddings": native,
    }
    config.max_position_embeddings = int(native * rope_factor)
    RESULT["rope"] = {"type": "static_yarn", "factor": rope_factor,
                      "native_tokens": native, "configured_tokens": config.max_position_embeddings,
                      "caveat": "far beyond training range; capacity/mechanism evidence, not absolute LM quality"}
    dump()

    load_t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, dtype=torch.bfloat16, local_files_only=True,
    ).eval().cuda()
    model.config.use_cache = False
    text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    RESULT["model"] = {
        "name": "Qwen2.5-0.5B", "layers": int(text_cfg.num_hidden_layers),
        "hidden_size": int(text_cfg.hidden_size),
        "query_heads": int(text_cfg.num_attention_heads),
        "kv_heads": int(text_cfg.num_key_value_heads),
        "head_dim": int(getattr(text_cfg, "head_dim",
                                text_cfg.hidden_size // text_cfg.num_attention_heads)),
        "dtype": "bfloat16", "load_s": round(time.time() - load_t0, 3),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    dump()
    log(f"loaded model: {RESULT['model']}")

    # 4K full-budget equivalence is the exact wiring gate for layer streaming and native GQA.
    smoke_n = 4096
    smoke_ids, labels, gold = make_choice_niah_ids(tokenizer, smoke_n, device="cuda")
    with torch.no_grad():
        dense = model(smoke_ids, use_cache=False, logits_to_keep=1).logits[:, 0].float()
    smoke_cfg = StreamingQwenConfig(
        chunk_blocks=8, top_c=smoke_n // 128, local=smoke_n // 128,
        search_k=(smoke_n // 32), build_threshold=1 << 20,
        outlier_rate=0.0, outlier_cap=0, share_route_from=0,
    )
    streamer = StreamingQwenPrefill(model, smoke_cfg)
    streamed, smoke_stats = streamer(smoke_ids)
    max_abs = float((dense - streamed).abs().max())
    RESULT["smoke"] = {
        "tokens": smoke_n, "max_abs_logit_delta": max_abs,
        "dense_quality": score_choice(dense, labels, gold),
        "streamed_quality": score_choice(streamed, labels, gold),
        "stats": smoke_stats, "passed": max_abs < 0.5,
    }
    dump()
    log(f"4K dense-equivalence smoke: {RESULT['smoke']}")
    if not RESULT["smoke"]["passed"]:
        raise RuntimeError("streaming equivalence smoke failed")
    del dense, streamed, smoke_ids
    torch.cuda.empty_cache()

    production_cfg = StreamingQwenConfig()
    streamer.cfg = production_cfg
    RESULT["router"] = {
        **production_cfg.__dict__,
        "complexity": "fixed selected blocks/token; streaming IVF summaries; fixed outlier reservoir",
        "strict_causal": True, "native_gqa": True,
    }
    dump()

    # The 128K reference is small enough for stock dense SDPA on this GPU.  It anchors the direct-word
    # NIAH ranking before the same streamed implementation is scaled 76x farther.
    ref_ids, ref_labels, ref_gold = make_choice_niah_ids(tokenizer, N_REF, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        ref_dense_logits = model(ref_ids, use_cache=False, logits_to_keep=1).logits[:, 0].float()
    torch.cuda.synchronize()
    RESULT["reference_dense"] = {
        "tokens": N_REF, "elapsed_s": round(time.time() - t0, 3),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "quality": score_choice(ref_dense_logits, ref_labels, ref_gold),
    }
    dump()
    log(f"128K dense reference: {RESULT['reference_dense']}")

    ref_stream_logits, ref_stream_stats = streamer(ref_ids)
    RESULT["reference_streamed"] = {
        "tokens": N_REF, "stats": ref_stream_stats,
        "quality": score_choice(ref_stream_logits, ref_labels, ref_gold),
    }
    dump()
    log(f"128K streamed reference: {RESULT['reference_streamed']}")
    del ref_dense_logits, ref_stream_logits, ref_ids
    torch.cuda.empty_cache()

    main_ids, main_labels, main_gold = make_choice_niah_ids(
        tokenizer, N_MAIN, depth=0.5, device="cuda")
    RESULT["status"] = "running_10m"
    RESULT["progress"] = {"layer": 0, "layers": int(text_cfg.num_hidden_layers)}
    dump()
    log(f"starting complete streamed prefill at {N_MAIN:,} tokens")

    def progress(record):
        RESULT["progress"] = record
        dump()
        log(f"layer {record['layer']:02d}/{record['layers']} elapsed={record['elapsed_s']:.1f}s "
            f"peak={record['peak_allocated_gb']:.1f}GB")

    main_logits, main_stats = streamer(main_ids, progress=progress)
    RESULT["main"] = {
        "tokens": N_MAIN, "over_10m": N_MAIN > 10_000_000,
        "complete_transformer": True, "all_layers": main_stats["layers"] == int(text_cfg.num_hidden_layers),
        "stats": main_stats, "quality": score_choice(main_logits, main_labels, main_gold),
        "scope": "one final-token semantic-candidate NIAH ranking; no full decode cache retained",
    }
    RESULT["status"] = "complete"
    RESULT["finished"] = time.time()
    RESULT["total_s"] = round(RESULT["finished"] - RESULT["started"], 3)
    dump()
    log(f"COMPLETE: {RESULT['main']}")


def main():
    dump()
    try:
        run()
    except Exception as exc:
        RESULT["status"] = "error"
        RESULT["error"] = repr(exc)
        RESULT["traceback"] = traceback.format_exc()
        RESULT["finished"] = time.time()
        RESULT["total_s"] = round(RESULT["finished"] - RESULT["started"], 3)
        dump()
        print(RESULT["traceback"], flush=True)
        raise


if __name__ == "__main__":
    main()
