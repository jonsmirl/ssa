"""Memory-bounded, complete Qwen2 prefill with CCC + FlexAttention.

The ordinary Hugging Face forward keeps sequence-wide Q/K/MLP temporaries alive.  At ten million
tokens those intermediates exceed even a 96 GiB GPU.  This executor preserves the exact decoder-layer
dataflow while walking each layer left-to-right in aligned token chunks:

* each token passes through every pretrained attention, projection, norm, MLP, and residual;
* K/V stay in their native GQA shape and FlexAttention runs with ``enable_gqa=True``;
* CCC sees the same post-RoPE real-model Q/K geometry, but emits one chunk's BlockMask at a time;
* K/V are discarded after their layer, MLP intermediates after their chunk, and residuals update in place.

This is inference-only and Qwen2-family-specific.  It returns final-token logits, which is enough for
the one-forward multiple-choice NIAH probe used by the >10M Kaggle experiment.  It deliberately does
not return a decode KV cache: Qwen2.5-0.5B's 10M-token bf16 cache alone is about 123 GB.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from ssa.cascade_router import StreamingGQARouter


@dataclass
class StreamingQwenConfig:
    block: int = 128
    chunk_blocks: int = 128
    top_c: int = 64
    local: int = 1
    sub: int = 32
    nprobe: int = 16
    search_k: int = 256
    build_threshold: int = 512
    outlier_rate: float = 1e-3
    outlier_cap: int = 4
    outlier_store_cap: int = 1024
    share_route_from: int | None = 12

    def router_kwargs(self):
        return {
            "top_c": self.top_c, "local": self.local, "sub": self.sub,
            "chunk_blocks": self.chunk_blocks, "nprobe": self.nprobe,
            "search_k": self.search_k, "build_threshold": self.build_threshold,
            "retrain_every": 0, "outlier_rate": self.outlier_rate,
            "outlier_cap": self.outlier_cap, "outlier_store_cap": self.outlier_store_cap,
        }


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


def _stream_mask(kv_num, kv_idx, q_offset, q_len, kv_len, block):
    # A device scalar avoids specializing the compiled kernel on every Python chunk offset.
    offset = torch.tensor(q_offset, device=kv_num.device, dtype=torch.int64)

    def causal(bb, hh, qi, ki):
        return ki <= qi + offset

    return BlockMask.from_kv_blocks(
        kv_num, kv_idx, BLOCK_SIZE=block, mask_mod=causal,
        seq_lengths=(q_len, kv_len), compute_q_blocks=False,
    )


class StreamingQwenPrefill:
    """Execute all layers of a pretrained Qwen2-family CausalLM with bounded activations."""

    def __init__(self, model, cfg: StreamingQwenConfig | None = None):
        self.model = model
        self.cfg = cfg or StreamingQwenConfig()
        self.flex = torch.compile(flex_attention, dynamic=True)

    @torch.no_grad()
    def __call__(self, input_ids, progress: Callable[[dict], None] | None = None):
        model, cfg = self.model, self.cfg
        core = model.model
        device = next(model.parameters()).device
        if device.type != "cuda":
            raise RuntimeError("StreamingQwenPrefill requires CUDA")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        batch, n = input_ids.shape
        if n % cfg.block:
            raise ValueError(f"sequence length {n} must be divisible by block={cfg.block}")
        chunk = cfg.chunk_blocks * cfg.block
        if chunk <= 0 or chunk % cfg.block:
            raise ValueError("chunk size must be a positive multiple of block")

        text_cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
        hq = int(text_cfg.num_attention_heads)
        hkv = int(text_cfg.num_key_value_heads)
        head_dim = int(getattr(text_cfg, "head_dim", text_cfg.hidden_size // hq))
        hidden_size = int(text_cfg.hidden_size)
        dtype = next(model.parameters()).dtype
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        hidden = core.embed_tokens(input_ids)
        del input_ids

        donor_num = donor_idx = None
        route_s = attention_s = mlp_s = 0.0
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)

        for layer_idx, layer in enumerate(core.layers[: text_cfg.num_hidden_layers]):
            layer_started = time.time()
            k_cache = torch.empty(batch, hkv, n, head_dim, device=device, dtype=dtype)
            v_cache = torch.empty_like(k_cache)
            reuse = (cfg.share_route_from is not None and layer_idx > cfg.share_route_from)
            router = None if reuse else StreamingGQARouter(
                batch, hq, hkv, n, head_dim, block=cfg.block, **cfg.router_kwargs())
            if cfg.share_route_from is not None and layer_idx == cfg.share_route_from:
                width = router.width
                donor_num = torch.empty(batch, hq, n // cfg.block, device=device, dtype=torch.int32)
                donor_idx = torch.zeros(batch, hq, n // cfg.block, width,
                                        device=device, dtype=torch.int32)
            if reuse and (donor_num is None or donor_idx is None):
                raise RuntimeError("route donor plan was not produced")

            for start in range(0, n, chunk):
                stop = min(n, start + chunk)
                hs = hidden[:, start:stop]
                x = layer.input_layernorm(hs)
                shape = (batch, stop - start, -1, head_dim)
                attn = layer.self_attn
                q = attn.q_proj(x).view(shape).transpose(1, 2)
                k = attn.k_proj(x).view(shape).transpose(1, 2)
                v = attn.v_proj(x).view(shape).transpose(1, 2)
                pos = torch.arange(start, stop, device=device, dtype=torch.long).unsqueeze(0)
                cos, sin = core.rotary_emb(x, pos)
                q, k = _apply_rope(q, k, cos, sin)
                k_cache[:, :, start:stop].copy_(k)
                v_cache[:, :, start:stop].copy_(v)
                del x, pos, cos, sin, k, v

                b0, b1 = start // cfg.block, stop // cfg.block
                torch.cuda.synchronize(device)
                tick = time.time()
                if reuse:
                    kv_num = donor_num[:, :, b0:b1]
                    kv_idx = donor_idx[:, :, b0:b1]
                else:
                    # route_chunk needs the post-RoPE unrepeated K; read its just-written cache view.
                    kv_num, kv_idx, _ = router.route_chunk(
                        q, k_cache[:, :, start:stop], start)
                    if cfg.share_route_from is not None and layer_idx == cfg.share_route_from:
                        donor_num[:, :, b0:b1].copy_(kv_num)
                        donor_idx[:, :, b0:b1].copy_(kv_idx)
                torch.cuda.synchronize(device)
                route_s += time.time() - tick

                bm = _stream_mask(kv_num, kv_idx, start, stop - start, stop, cfg.block)
                torch.cuda.synchronize(device)
                tick = time.time()
                y = self.flex(q, k_cache[:, :, :stop], v_cache[:, :, :stop],
                              block_mask=bm, scale=attn.scaling, enable_gqa=True)
                y = y.transpose(1, 2).reshape(batch, stop - start, hidden_size)
                y = attn.o_proj(y)
                hs.add_(y)
                torch.cuda.synchronize(device)
                attention_s += time.time() - tick
                del q, y, bm, kv_num, kv_idx

                tick = time.time()
                z = layer.post_attention_layernorm(hs)
                mlp = layer.mlp
                z = mlp.down_proj(mlp.act_fn(mlp.gate_proj(z)) * mlp.up_proj(z))
                hs.add_(z)
                torch.cuda.synchronize(device)
                mlp_s += time.time() - tick
                del z, hs

            del k_cache, v_cache, router
            torch.cuda.empty_cache()
            record = {
                "layer": layer_idx + 1,
                "layers": int(text_cfg.num_hidden_layers),
                "elapsed_s": round(time.time() - started, 3),
                "layer_s": round(time.time() - layer_started, 3),
                "route_s": round(route_s, 3),
                "attention_s": round(attention_s, 3),
                "mlp_s": round(mlp_s, 3),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
            }
            if progress is not None:
                progress(record)

        last = core.norm(hidden[:, -1:])
        logits = model.lm_head(last).float().squeeze(1)
        elapsed = time.time() - started
        stats = {
            "tokens": n, "layers": int(text_cfg.num_hidden_layers),
            "elapsed_s": round(elapsed, 3), "tokens_per_s": round(n / elapsed, 3),
            "route_s": round(route_s, 3), "attention_s": round(attention_s, 3),
            "mlp_s": round(mlp_s, 3),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated(device) / 1e9, 3),
            "selected_fraction_upper": sum(
                min(i + 1, cfg.top_c + cfg.local + 1 + cfg.outlier_cap)
                for i in range(n // cfg.block)
            ) / sum(range(1, n // cfg.block + 1)),
        }
        del hidden, last
        return logits, stats


def make_choice_niah_ids(tokenizer, n, depth=0.5, block=128, device="cpu"):
    """Build an exact, block-aligned context and a one-next-token retrieval ranking.

    All four candidate words are single Qwen tokens.  Ranking their next-token logits measures retrieval
    without retaining a 123 GB decode cache; unlike letter choices, it has no fixed-position label bias.
    """
    if n % block:
        raise ValueError("NIAH context length must be block aligned")
    filler = tokenizer(
        " The garden path wound past the old stone wall in the quiet afternoon light.",
        add_special_tokens=False,
    )["input_ids"]
    needle = tokenizer(
        " Remember this fact exactly: the secret access word is walnut.",
        add_special_tokens=False,
    )["input_ids"]
    question = tokenizer(
        " Question: what is the secret access word? Answer: the secret access word is",
        add_special_tokens=False,
    )["input_ids"]
    available = n - len(needle) - len(question)
    if available < len(filler):
        raise ValueError("context is too short for the NIAH prompt")
    repeated = (filler * ((available + len(filler) - 1) // len(filler)))[:available]
    cut = int(round(available * min(max(depth, 0.0), 1.0)))
    ids = repeated[:cut] + needle + repeated[cut:] + question
    labels = []
    for word in (" walnut", " crimson", " lantern", " marble"):
        encoded = tokenizer(word, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(f"expected one token for candidate {word!r}, got {encoded}")
        labels.append(encoded[0])
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0), labels, 0


def score_choice(logits, labels, gold):
    scores = logits[0, labels].detach().float().cpu()
    predicted = int(scores.argmax())
    names = ("walnut", "crimson", "lantern", "marble")
    return {
        "gold": names[gold], "predicted": names[predicted], "correct": predicted == gold,
        "candidate_logits": {name: round(float(score), 5)
                             for name, score in zip(names, scores)},
    }
