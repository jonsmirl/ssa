"""Memory-bounded, complete Qwen2 prefill with CCC + FlexAttention.

The ordinary Hugging Face forward keeps sequence-wide Q/K/MLP temporaries alive.  At ten million
tokens those intermediates exceed even a 96 GiB GPU.  This executor preserves the exact decoder-layer
dataflow while walking each layer left-to-right in aligned token chunks:

* each token passes through every pretrained attention, projection, norm, MLP, and residual;
* K/V stay in their native GQA shape and FlexAttention runs with ``enable_gqa=True``;
* CCC routes on real-model pre-RoPE content Q/K while attention uses the unchanged post-RoPE Q/K;
* bounded layer-1 head consensus preserves high-vote long-range evidence through later sparse plans;
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

from ssa.cascade_router import CausalTree, StreamingGQARouter


@dataclass
class StreamingQwenConfig:
    backend: str = "tree"
    block: int = 128
    chunk_blocks: int = 128
    top_c: int = 64
    local: int = 1
    sub: int = 32
    query_sub: int = 128
    route_geometry: str = "pre_rope"
    nprobe: int = 16
    search_k: int = 256
    build_threshold: int = 512
    outlier_rate: float = 1e-3
    outlier_cap: int = 4
    outlier_store_cap: int = 1024
    # With 14 heads × at most 70 base selections, cap=128 necessarily retains every block with at
    # least nine head votes (there can be at most floor(980/9)=108 such blocks).
    persistent_consensus_cap: int = 128
    persistent_consensus_min_heads: int = 2
    share_route_from: int | None = 12

    def router_kwargs(self):
        kwargs = {
            "top_c": self.top_c, "local": self.local, "sub": self.sub,
            "chunk_blocks": self.chunk_blocks, "nprobe": self.nprobe,
            "search_k": self.search_k, "build_threshold": self.build_threshold,
            "retrain_every": 0, "outlier_rate": self.outlier_rate,
            "outlier_cap": self.outlier_cap, "outlier_store_cap": self.outlier_store_cap,
        }
        if self.backend == "tree":
            kwargs["query_sub"] = self.query_sub
        return kwargs


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
    def __call__(self, input_ids, progress: Callable[[dict], None] | None = None,
                 probe_block: int | None = None, force_block: int | None = None):
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
        if cfg.route_geometry not in {"pre_rope", "post_rope"}:
            raise ValueError("route_geometry must be 'pre_rope' or 'post_rope'")
        if input_ids.device != device:
            input_ids = input_ids.to(device)
        hidden = core.embed_tokens(input_ids)
        del input_ids

        donor_num = donor_idx = None
        persistent_num = persistent_idx = None
        if cfg.persistent_consensus_cap > 0:
            persistent_num = torch.zeros(batch, n // cfg.block, device=device, dtype=torch.int32)
            persistent_idx = torch.zeros(
                batch, n // cfg.block, cfg.persistent_consensus_cap,
                device=device, dtype=torch.int32,
            )
        route_probe = []
        route_s = attention_s = mlp_s = 0.0
        started = time.time()
        torch.cuda.reset_peak_memory_stats(device)

        for layer_idx, layer in enumerate(core.layers[: text_cfg.num_hidden_layers]):
            layer_started = time.time()
            k_cache = torch.empty(batch, hkv, n, head_dim, device=device, dtype=dtype)
            v_cache = torch.empty_like(k_cache)
            reuse = (cfg.share_route_from is not None and layer_idx > cfg.share_route_from)
            router = None if reuse else StreamingGQARouter(
                batch, hq, hkv, n, head_dim, block=cfg.block, backend=cfg.backend,
                **cfg.router_kwargs())
            if cfg.share_route_from is not None and layer_idx == cfg.share_route_from:
                width = router.width + cfg.persistent_consensus_cap
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
                raw_q, raw_k = q, k
                pos = torch.arange(start, stop, device=device, dtype=torch.long).unsqueeze(0)
                cos, sin = core.rotary_emb(x, pos)
                q, k = _apply_rope(q, k, cos, sin)
                k_cache[:, :, start:stop].copy_(k)
                v_cache[:, :, start:stop].copy_(v)
                route_q = raw_q if cfg.route_geometry == "pre_rope" else q
                route_k = raw_k if cfg.route_geometry == "pre_rope" else k
                del x, pos, cos, sin, v

                b0, b1 = start // cfg.block, stop // cfg.block
                torch.cuda.synchronize(device)
                tick = time.time()
                if reuse:
                    kv_num = donor_num[:, :, b0:b1]
                    kv_idx = donor_idx[:, :, b0:b1]
                else:
                    # Routing may use content-only pre-RoPE geometry; attention always uses the
                    # pretrained post-RoPE tensors stored in the layer cache.
                    kv_num, kv_idx, _ = router.route_chunk(
                        route_q, route_k, start)
                    if persistent_idx is not None:
                        if layer_idx == 0:
                            extra_num, extra_idx = self._head_consensus(
                                kv_num, kv_idx, n // cfg.block,
                                cfg.persistent_consensus_cap,
                                cfg.persistent_consensus_min_heads,
                            )
                            persistent_num[:, b0:b1].copy_(extra_num)
                            persistent_idx[:, b0:b1].copy_(extra_idx)
                        else:
                            extra_num = persistent_num[:, b0:b1]
                            extra_idx = persistent_idx[:, b0:b1]
                        kv_num, kv_idx = self._augment_plan(
                            kv_num, kv_idx, extra_num, extra_idx, n // cfg.block)
                    if force_block is not None and stop == n:
                        self._force_final_query_block(kv_num, kv_idx, force_block)
                    if cfg.share_route_from is not None and layer_idx == cfg.share_route_from:
                        donor_num[:, :, b0:b1].copy_(kv_num)
                        donor_idx[:, :, b0:b1].copy_(kv_idx)
                    if probe_block is not None and stop == n:
                        route_probe.append(self._probe_route(
                            router, route_q, kv_num, kv_idx, probe_block, layer_idx))
                del raw_q, raw_k, route_q, route_k, k
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
                min(i + 1, cfg.top_c + cfg.local + 1 + cfg.outlier_cap
                    + cfg.persistent_consensus_cap)
                for i in range(n // cfg.block)
            ) / sum(range(1, n // cfg.block + 1)),
            "route_probe": route_probe if probe_block is not None else None,
        }
        del hidden, last
        return logits, stats

    @staticmethod
    def _head_consensus(kv_num, kv_idx, total_blocks, cap, min_heads):
        """Find a fixed number of blocks independently selected by the most heads per query row."""
        batch, heads, queries, width = kv_idx.shape
        ids = kv_idx.permute(0, 2, 1, 3).reshape(batch * queries, heads * width).long()
        counts = kv_num.permute(0, 2, 1)
        valid = (torch.arange(width, device=kv_idx.device)[None, None, None, :]
                 < counts[..., None])
        valid = valid.reshape(batch * queries, heads * width)
        row = torch.arange(batch * queries, device=kv_idx.device)[:, None].expand_as(ids)
        encoded = row[valid] * total_blocks + ids[valid]
        unique, frequency = torch.unique(encoded, sorted=True, return_counts=True)
        unique_row, unique_id = unique // total_blocks, unique % total_blocks

        # Compact the sorted variable-length groups to a small dense [row, heads*width] workspace.
        position = torch.arange(unique.numel(), device=kv_idx.device)
        group_start = torch.zeros_like(position)
        if unique.numel():
            starts = torch.ones_like(position, dtype=torch.bool)
            starts[1:] = unique_row[1:] != unique_row[:-1]
            group_start = torch.where(starts, position, torch.zeros_like(position)).cummax(0).values
        rank = position - group_start
        dense_frequency = torch.zeros(
            batch * queries, heads * width, device=kv_idx.device, dtype=frequency.dtype)
        dense_ids = torch.full(
            (batch * queries, heads * width), total_blocks,
            device=kv_idx.device, dtype=torch.long)
        dense_frequency[unique_row, rank] = frequency
        dense_ids[unique_row, rank] = unique_id
        take = min(cap, heads * width)
        values, pick = dense_frequency.topk(take, dim=1)
        selected = dense_ids.gather(1, pick)
        selected = torch.where(values >= min_heads, selected, torch.full_like(selected, total_blocks))
        if take < cap:
            selected = torch.cat((selected, torch.full(
                (batch * queries, cap - take), total_blocks,
                device=kv_idx.device, dtype=selected.dtype)), dim=1)
        number = (selected < total_blocks).sum(1).to(torch.int32)
        selected = torch.where(selected < total_blocks, selected, torch.zeros_like(selected)).to(torch.int32)
        return number.view(batch, queries), selected.view(batch, queries, cap)

    @staticmethod
    def _augment_plan(kv_num, kv_idx, extra_num, extra_idx, SENT):
        """Union per-query persistent candidates into every head's current sparse plan."""
        batch, heads, queries, width = kv_idx.shape
        cap = extra_idx.shape[-1]
        base_valid = (torch.arange(width, device=kv_idx.device)[None, None, None, :]
                      < kv_num[..., None])
        extra_valid = (torch.arange(cap, device=kv_idx.device)[None, None, None, :]
                       < extra_num[:, None, :, None])
        base = torch.where(base_valid, kv_idx.long(), torch.full_like(kv_idx.long(), SENT))
        extra = extra_idx[:, None].expand(-1, heads, -1, -1)
        extra = torch.where(extra_valid, extra.long(), torch.full_like(extra.long(), SENT))
        candidates = torch.cat((base, extra), dim=-1).sort(dim=-1).values
        duplicate = torch.zeros_like(candidates, dtype=torch.bool)
        duplicate[..., 1:] = candidates[..., 1:] == candidates[..., :-1]
        candidates = torch.where(duplicate, torch.full_like(candidates, SENT), candidates)
        candidates = candidates.sort(dim=-1).values
        number = (candidates < SENT).sum(-1).to(torch.int32)
        indices = torch.where(candidates < SENT, candidates, torch.zeros_like(candidates)).to(torch.int32)
        return number, indices

    @staticmethod
    def _force_final_query_block(kv_num, kv_idx, key_block):
        """Oracle diagnostic: ensure the final query row can see one specified key block."""
        width = kv_idx.shape[-1]
        for batch in range(kv_num.shape[0]):
            for head in range(kv_num.shape[1]):
                count = int(kv_num[batch, head, -1])
                if bool((kv_idx[batch, head, -1, :count] == key_block).any()):
                    continue
                slot = count if count < width else width - 1
                kv_idx[batch, head, -1, slot] = key_block
                if count < width:
                    kv_num[batch, head, -1] = count + 1

    def _probe_route(self, router, q_chunk, kv_num, kv_idx, target_block, layer_idx):
        """Diagnose the final query block against the exact sub-block routing metric.

        ``exact_rank <= top_c`` but ``selected=False`` isolates tree-beam pruning. A larger exact rank
        means the query-block/sub-block metric itself does not rank the needle inside the fixed budget.
        """
        cfg = self.cfg
        if not 0 <= target_block < router.n // cfg.block:
            raise ValueError("probe_block is outside the context")
        rows = []
        qtokens = q_chunk[0].view(router.hq, -1, cfg.block, router.d).float()[:, -1]
        qmean = qtokens.mean(1)
        final_qblock = router.n // cfg.block - 1
        for hq in range(router.hq):
            hk = hq // router.groups
            cascade = router.cascades[0][hk]
            if not isinstance(cascade, CausalTree):
                continue
            means = cascade.levels[0][:cascade.committed]
            if cascade.stage is not None:
                means = torch.cat((means, cascade.stage), dim=0)
            qreps = qtokens[hq].view(cascade.qspb, cascade.query_sub, router.d).mean(1)
            scores = qreps @ means.T
            parent_scores = scores.view(cascade.qspb, -1, cascade.spb).amax((0, 2))[:final_qblock]
            mean_parent_scores = (qmean[hq] @ means.T).view(-1, cascade.spb).amax(1)[:final_qblock]
            target_score = parent_scores[target_block]
            rank = 1 + int((parent_scores > target_score).sum())
            mean_target_score = mean_parent_scores[target_block]
            mean_rank = 1 + int((mean_parent_scores > mean_target_score).sum())
            count = int(kv_num[0, hq, -1])
            chosen = kv_idx[0, hq, -1, :count]
            rows.append({
                "head": hq, "kv_head": hk, "selected": bool((chosen == target_block).any()),
                "exact_rank": rank, "within_top_c": rank <= cfg.top_c,
                "target_score": round(float(target_score), 6),
                "whole_query_mean_rank": mean_rank,
                "whole_query_mean_score": round(float(mean_target_score), 6),
            })
        return {
            "layer": layer_idx + 1, "target_block": target_block,
            "heads_selected": sum(row["selected"] for row in rows),
            "heads_exact_top_c": sum(row["within_top_c"] for row in rows),
            "heads": rows,
        }


def make_choice_niah_ids(tokenizer, n, depth=0.5, block=128, device="cpu",
                         return_metadata=False):
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
    result = (torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0), labels, 0)
    if return_metadata:
        walnut_offset = needle.index(labels[0])
        walnut_pos = cut + walnut_offset
        return (*result, {"needle_token_start": cut, "needle_block": walnut_pos // block,
                          "answer_token_position": walnut_pos,
                          "needle_tokens": len(needle), "depth": depth})
    return result


def score_choice(logits, labels, gold):
    scores = logits[0, labels].detach().float().cpu()
    predicted = int(scores.argmax())
    names = ("walnut", "crimson", "lantern", "marble")
    return {
        "gold": names[gold], "predicted": names[predicted], "correct": predicted == gold,
        "candidate_logits": {name: round(float(score), 5)
                             for name, score in zip(names, scores)},
    }
