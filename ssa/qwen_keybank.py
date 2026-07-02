"""
Extract a bank of POST-RoPE per-head q/k from a real model, across ALL layers, for training and
evaluating the low-dim routing space (routing_space.py). Reuses the SDPA-intercept trick from
longctx_keys.extract_qk_hf but captures every layer and lets the caller pass the text (so we can hold out
a code document from the wikitext training documents — the overfit control).

Cached to disk (default /tmp; configurable — /tmp evicts) as fp16 npz per document.

Run:  python3 -m ssa.qwen_keybank --n 8192 --out-dir /tmp
"""
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import glob
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def code_text(n_tokens):
    """A code document: the repo's own ssa/*.py concatenated (offline, no download) — a register held out
    from the wikitext training docs."""
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.py")))
    buf = []
    for f in files:
        buf.append(open(f).read())
        if sum(len(b.split()) for b in buf) > n_tokens * 1.6:
            break
    return "\n".join(buf)


def _doc_text(doc, n_tokens):
    from ssa.real_keys import load_real_text
    if doc == "code":
        return code_text(n_tokens)
    return load_real_text(n_tokens)                                 # wikitext (deterministic slice)


@torch.no_grad()
def extract_all_layers(model_name, text, n_tokens):
    """Capture post-RoPE q,k for EVERY attention layer. Returns {layer: (q (n_qh,seq,hd), k (n_kvh,seq,hd))}."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa",
                                                 torch_dtype=torch.float32).eval().to(DEV)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=n_tokens).input_ids.to(DEV)
    cap, call = {}, {"i": 0}
    orig = F.scaled_dot_product_attention

    def patched(q, k, v, *a, **kw):
        i = call["i"]; call["i"] += 1
        cap[i] = (q.detach()[0].cpu().half().numpy(), k.detach()[0].cpu().half().numpy())
        return orig(q, k, v, *a, **kw)

    F.scaled_dot_product_attention = patched
    try:
        model(ids)
    finally:
        F.scaled_dot_product_attention = orig
    seq = ids.shape[1]
    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return cap, seq


def build_keybank(model_name="Qwen/Qwen2.5-0.5B", n_tokens=8192,
                  docs=("wikitext", "code"), out_dir="/tmp", q_heads=(0, 3, 7, 10)):
    """Extract + cache one npz per doc: K_{L} (n_kvh, seq, hd) fp16, Q_{L} (len(q_heads), seq, hd), meta."""
    paths = []
    for doc in docs:
        text = _doc_text(doc, n_tokens)
        cap, seq = extract_all_layers(model_name, text, n_tokens)
        arrs = {}
        for L, (q, k) in cap.items():
            arrs[f"K_{L}"] = k                                     # (n_kvh, seq, hd)
            arrs[f"Q_{L}"] = q[list(q_heads)]                     # sampled q-heads
        arrs["meta"] = np.array([len(cap), seq, cap[0][1].shape[-1],
                                 cap[0][1].shape[0], cap[0][0].shape[0]], dtype=np.int64)
        path = os.path.join(out_dir, f"keybank_{doc}_{seq}.npz")
        np.savez(path, **arrs, q_heads=np.array(q_heads))
        paths.append(path)
        print(f"  wrote {path}  ({len(cap)} layers, seq={seq}, "
              f"{cap[0][0].shape[0]} q-heads / {cap[0][1].shape[0]} kv-heads, d={cap[0][1].shape[-1]})",
              flush=True)
    return paths


def load_keybank(path):
    """-> dict: {'n_layers','seq','d','n_kvh','n_qh','q_heads', 'K'[L]->(n_kvh,seq,d), 'Q'[L]->(nqs,seq,d)}."""
    z = np.load(path)
    nl, seq, d, n_kvh, n_qh = [int(x) for x in z["meta"]]
    return {"n_layers": nl, "seq": seq, "d": d, "n_kvh": n_kvh, "n_qh": n_qh,
            "q_heads": z["q_heads"], "grp": n_qh // n_kvh,
            "K": {L: z[f"K_{L}"] for L in range(nl)},
            "Q": {L: z[f"Q_{L}"] for L in range(nl)}}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--docs", default="wikitext,code")
    ap.add_argument("--out-dir", default="/tmp")
    args = ap.parse_args()
    build_keybank(args.model, args.n, tuple(args.docs.split(",")), args.out_dir)


if __name__ == "__main__":
    main()
