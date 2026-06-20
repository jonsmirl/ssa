"""
The frontier-scale check — do the routing findings hold on Gemma-4-26B's real keys?

Everything tested so far (GPT-2, TinyLlama, Qwen) used head_dim 64. Gemma-4-26B-A4B is a different point:
26B params (MoE, 4B active), RoPE + GQA, and — the genuinely new variable — **head_dim 256**. This
extracts its real post-RoPE attention keys (text path, via the official AutoProcessor) and re-runs the
core results: centroid vs cumulant routing, and the tempered-temperature sweep, on a deep head.

Loads the 47GB model with device_map=auto (GPU + CPU offload; needs the ~78GB RAM). Caches the extracted
deep-head keys to /tmp/gemma_keys.npz so the routing analysis can be re-run without reloading the model.

Run:  python3 -m ssa.gemma_keys
"""
from __future__ import annotations
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import glob
import numpy as np
import torch
import torch.nn.functional as F

from .real_keys import load_real_text, coherence_stats, participation_ratio, real_cue_cosine
from .tempered_routing import route_recall_tempered

# Load from the local snapshot directory directly (the pre-existing cache's offline hub-resolution is
# broken under transformers 5.12.1; a direct path bypasses it). Pick the COMPLETE snapshot (the offline
# attempts left an empty second one).
_snaps = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--google--gemma-4-26B-A4B/snapshots/*/"))
MODEL = next((s for s in _snaps if os.path.exists(s + "config.json")
              and os.path.exists(s + "model.safetensors.index.json")),
             _snaps[0] if _snaps else "google/gemma-4-26B-A4B")
CACHE = "/tmp/gemma_keys.npz"
LAYERS = [8, 16, 22, 29]


@torch.no_grad()
def extract():
    from transformers import AutoTokenizer
    try:
        from transformers import AutoModelForImageTextToText as _AutoModel
    except Exception:
        from transformers import Gemma4ForConditionalGeneration as _AutoModel
    print("  loading tokenizer + model (Gemma4ForConditionalGeneration, 47GB, device_map=auto)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = _AutoModel.from_pretrained(MODEL, dtype="auto", device_map="auto",
                                       attn_implementation="sdpa", low_cpu_mem_usage=True).eval()
    text = load_real_text(2048)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=2048)["input_ids"].to("cuda")
    print(f"  forwarding {ids.shape[1]} tokens...", flush=True)

    want = set(LAYERS)
    cap, call = {}, {"i": 0}
    orig = F.scaled_dot_product_attention

    def patched(q, k, v, *a, **kw):
        i = call["i"]; call["i"] += 1
        if i in want:
            cap[i] = (q.detach()[0].float().cpu().numpy(), k.detach()[0].float().cpu().numpy())
        return orig(q, k, v, *a, **kw)

    F.scaled_dot_product_attention = patched
    try:
        model(ids)
    finally:
        F.scaled_dot_product_attention = orig

    out = {}
    n_q = cap[LAYERS[0]][0].shape[0]; n_kv = cap[LAYERS[0]][1].shape[0]; hd = cap[LAYERS[0]][1].shape[-1]
    grp = max(1, n_q // n_kv)
    for L in LAYERS:
        q, k = cap[L]
        for kv in range(n_kv):
            out[f"K_{L}_{kv}"] = k[kv]
            out[f"Q_{L}_{kv}"] = q[min(kv * grp, n_q - 1)]
    out["meta"] = np.array([ids.shape[1], hd, n_q, n_kv])
    np.savez(CACHE, **out)
    print(f"  extracted: n={ids.shape[1]}, head_dim={hd}, {n_q} q-heads / {n_kv} kv-heads -> cached", flush=True)


def main():
    if not os.path.exists(CACHE):
        extract()
    d = np.load(CACHE)
    seq, hd, n_q, n_kv = (int(x) for x in d["meta"])
    print("=" * 84)
    print(f"GEMMA-4-26B routing check — n={seq}, head_dim={hd}, {n_q} q-heads / {n_kv} kv-heads")
    print("=" * 84)
    md = seq // 4

    rr = lambda K, Q, beta, mode: route_recall_tempered(K, Q, 64, 0.05, beta, mode, md, max_queries=200)
    print(f"\n  centroid vs cumulant routing, per head of the deep layers (5% budget, long-range):")
    print(f"  {'head':>8} {'coh':>7} {'effdim':>8} {'cue':>5} {'centroid':>9} {'cumulant':>9}")
    print("  " + "-" * 56)
    best = None
    for L in (22, 29):
        for kv in range(min(n_kv, 4)):
            K, Q = d[f"K_{L}_{kv}"], d[f"Q_{L}_{kv}"]
            coh, _ = coherence_stats(K); pr = participation_ratio(K); cue = real_cue_cosine(Q, K)
            rc = rr(K, Q, 0.0, "centroid")
            rk = rr(K, Q, 1.0, "second")
            print(f"  {L}.{kv:<5} {coh:>7.3f} {pr:>8.1f} {cue:>5.2f} {rc:>9.3f} {rk:>9.3f}")
            if best is None or rk - rc > best[0]:
                best = (rk - rc, L, kv)

    _, Lb, kvb = best
    K, Q = d[f"K_{Lb}_{kvb}"], d[f"Q_{Lb}_{kvb}"]
    print(f"\n  temperature sweep on head {Lb}.{kvb} (5% budget, long-range, cheap 2nd-order routing):")
    print(f"  {'β':>6} {'recall':>8}")
    cen = rr(K, Q, 0.0, "centroid")
    orc = rr(K, Q, 0.0, "oracle")
    for beta in (1.0, 2.0, 4.0, 8.0):
        rs = rr(K, Q, beta, "second")
        tag = "  <- cumulant (β=1)" if beta == 1.0 else ""
        print(f"  {beta:>6.1f} {rs:>8.3f}{tag}")
    print(f"  centroid (β→0): {cen:.3f}    oracle (β→∞): {orc:.3f}")

    print("\n" + "=" * 84)
    print(f"  Gemma-4-26B (head_dim {hd}, 26B MoE) — does the cumulant gain + β≈2 optimum survive at")
    print("  frontier scale and 4x the head dimension of every earlier model? Compare to Qwen (hd 64):")
    print("  cumulant 0.90 / centroid ~0 / β≈2 peak 0.94. The geometry of attention routing, tested up.")


if __name__ == "__main__":
    main()
