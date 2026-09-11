"""Fused CUDA reference for partial-coordinate attention certificates.

The real-arithmetic certificate uses block coordinate extrema, NOT a CCC
routing-score certificate. FP32 guards are conservative engineering allowances,
not verified IEEE interval arithmetic. Sorting, adaptive control and log-mass
reductions remain PyTorch operations; this is not an end-to-end fused decoder.
Inputs are immutable visible-prefix snapshots. No unseen logits or values are
used to choose reads. Index construction reads all visible keys and values.
"""
from __future__ import annotations

import math
import operator

import torch
import triton
import triton.language as tl


@triton.jit
def _intervals(K, Q, ORDER, MIN, MAX, SCORE, ABS, LOW, HIGH,
               N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr,
               R: tl.constexpr, BETA: tl.constexpr, GUARD: tl.constexpr):
    h, b = tl.program_id(0), tl.program_id(1)
    rows = b * BLOCK + tl.arange(0, BLOCK)
    a = tl.arange(0, D)
    coord = tl.load(ORDER + h * D + a)
    q = tl.load(Q + h * D + coord).to(tl.float32) * BETA
    k = tl.load(K + rows[:, None] * D + coord[None, :],
                (rows[:, None] < N) & (a[None, :] < R), 0).to(tl.float32)
    terms = k * q[None, :]
    partial = tl.sum(terms, 1)
    absolute = tl.sum(tl.abs(terms), 1)
    mn = tl.load(MIN + b * D + coord)
    mx = tl.load(MAX + b * D + coord)
    low_terms = tl.minimum(q * mn, q * mx)
    high_terms = tl.maximum(q * mn, q * mx)
    missing_low = tl.sum(tl.where(a >= R, low_terms, 0), 0)
    missing_high = tl.sum(tl.where(a >= R, high_terms, 0), 0)
    missing_abs = tl.sum(tl.where(a >= R, tl.maximum(tl.abs(low_terms), tl.abs(high_terms)), 0), 0)
    guard = GUARD * (absolute + missing_abs) + 1.0e-30
    tl.store(SCORE + h * N + rows, partial, rows < N)
    tl.store(ABS + h * N + rows, absolute, rows < N)
    tl.store(LOW + h * N + rows, partial + missing_low - guard, rows < N)
    tl.store(HIGH + h * N + rows, partial + missing_high + guard, rows < N)


@triton.jit
def _complete(K, Q, ORDER, IDS, SCORE, ABS, LOW, HIGH,
              N: tl.constexpr, D: tl.constexpr, COUNT: tl.constexpr,
              HEAD: tl.constexpr, R: tl.constexpr, BETA: tl.constexpr,
              GUARD: tl.constexpr, TILE: tl.constexpr):
    j = tl.program_id(0) * TILE + tl.arange(0, TILE)
    rows = tl.load(IDS + j, j < COUNT, 0)
    a = tl.arange(0, D)
    coord = tl.load(ORDER + HEAD * D + a)
    q = tl.load(Q + HEAD * D + coord).to(tl.float32) * BETA
    k = tl.load(K + rows[:, None] * D + coord[None, :],
                (j[:, None] < COUNT) & (a[None, :] >= R), 0).to(tl.float32)
    terms = k * q[None, :]
    initial = tl.load(SCORE + HEAD * N + rows, j < COUNT, 0)
    score = initial + tl.sum(terms, 1)
    magnitude = tl.load(ABS + HEAD * N + rows, j < COUNT, 0) + tl.sum(tl.abs(terms), 1)
    guard = GUARD * magnitude + 1.0e-30
    tl.store(SCORE + HEAD * N + rows, score, j < COUNT)
    tl.store(LOW + HEAD * N + rows, score - guard, j < COUNT)
    tl.store(HIGH + HEAD * N + rows, score + guard, j < COUNT)


@triton.jit
def _value_reduce(V, IDS, SCORE, CENTER, LOGZ_CENTERED, OUT, COUNT: tl.constexpr,
                  DV: tl.constexpr, VD: tl.constexpr, TILE: tl.constexpr):
    b = tl.program_id(0)
    j = b * TILE + tl.arange(0, TILE)
    rows = tl.load(IDS + j, j < COUNT, 0)
    scores = tl.load(SCORE + rows, j < COUNT, -float("inf"))
    center = tl.load(CENTER).to(tl.float32)
    z = tl.load(LOGZ_CENTERED).to(tl.float32)
    w = tl.exp((scores - center) - z)
    d = tl.arange(0, VD)
    values = tl.load(V + rows[:, None] * DV + d[None, :],
                     (j[:, None] < COUNT) & (d[None, :] < DV), 0).to(tl.float32)
    out = tl.sum(w[:, None] * values, 0)
    tl.store(OUT + b * DV + d, out, d < DV)


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


class CoordinateGPUIndex:
    """CUDA snapshot; keys/values preserve their supplied floating dtype.

    Each query head routes independently against one shared KV head. Different
    causal prefixes require different snapshots. ``read`` includes host/device
    synchronization, stable sorts, O(n) tail preprocessing, repeated O(n)
    retained-mass reductions and O(n*r) coordinate reads. There is no
    subquadratic full-prefill guarantee.
    """

    def __init__(self, K, V, block=64):
        block = _integer(block, "block")
        if (K.ndim != 2 or V.ndim != 2 or len(K) != len(V) or len(K) < 1
                or K.shape[1] != 64 or V.shape[1] < 1 or block != 64):
            raise ValueError("requires nonempty K[n,64], V[n,dv] and block=64")
        if not K.is_cuda or not V.is_cuda or K.device != V.device:
            raise ValueError("K and V must share a CUDA device")
        if K.dtype not in (torch.float16, torch.bfloat16, torch.float32) or V.dtype not in (
                torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("supported archive dtypes are float16, bfloat16, float32")
        if not bool(torch.isfinite(K).all() & torch.isfinite(V).all()):
            raise ValueError("visible archive must be finite")
        self.K, self.V = K.detach().contiguous().clone(), V.detach().contiguous().clone()
        self.n, self.d, self.dv = len(K), K.shape[1], V.shape[1]
        self.block, self.device = block, K.device
        self.blocks = triton.cdiv(self.n, block)
        padded = torch.full((self.blocks * block, self.d), float("nan"), device=K.device)
        padded[:self.n] = self.K.float()
        grouped = padded.view(self.blocks, block, self.d)
        self.minimum = torch.nan_to_num(grouped, nan=float("inf")).amin(1)
        self.maximum = torch.nan_to_num(grouped, nan=-float("inf")).amax(1)
        # FP64 build-time metadata, conservatively rounded by an engineering guard.
        self.value_bound = self.V.double().norm(dim=1).max() * (1 + 64 * torch.finfo(torch.float64).eps)

    def storage(self):
        return {"archive_bytes": self.K.numel() * self.K.element_size() + self.V.numel() * self.V.element_size(),
                "summary_bytes": self.minimum.numel() * 8 + 8,
                "index_build_key_scalars": self.n * self.d,
                "index_build_value_scalars": self.n * self.dv}

    @torch.no_grad()
    def read(self, q, coordinates=32, eta=.1, seed=128, max_values=None,
             reuse_tail_order=True):
        """Adaptively double selected-key budget until certified or exhausted.

        ``reuse_tail_order`` caches the immutable unopened-upper-cap ordering
        and reverse cumulative log-mass. False runs the equivalent repeated
        sort/reduction baseline. Threshold-edge floating rounding may differ;
        both paths use the same conservative log-mass allowance.
        """
        coordinates, seed = _integer(coordinates, "coordinates"), _integer(seed, "seed")
        maximum = self.n if max_values is None else _integer(max_values, "max_values")
        if not 0 <= coordinates <= self.d or not 0 < eta < 1 or seed < 1 or not 1 <= maximum <= self.n:
            raise ValueError("invalid coordinate count, tolerance or budget")
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or q.shape[1] != self.d or len(q) < 1 or q.device != self.device:
            raise ValueError("queries must have shape [heads,64] on the index device")
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32) or not bool(torch.isfinite(q).all()):
            raise ValueError("queries must be finite float16, bfloat16 or float32 tensors")
        q = q.float().contiguous()
        heads, n, d = len(q), self.n, self.d
        order = torch.argsort(q.abs(), dim=1, descending=True, stable=True).contiguous()
        scores = torch.empty((heads, n), device=self.device)
        absolute, lower, upper = [torch.empty_like(scores) for _ in range(3)]
        guard = 64 * torch.finfo(torch.float32).eps * d
        _intervals[(heads, self.blocks)](self.K, q, order, self.minimum, self.maximum,
            scores, absolute, lower, upper, n, d, self.block, coordinates, 1 / math.sqrt(d), guard)
        if not bool(torch.isfinite(lower).all() & torch.isfinite(upper).all()):
            raise ArithmeticError("FP32 interval overflow")
        selected = torch.zeros((heads, n), device=self.device, dtype=torch.bool)
        boundary = torch.arange(n // self.block * self.block, n, device=self.device)
        if len(boundary) > maximum:
            raise ValueError("value budget must fit the partial causal boundary block")
        outputs, bounds, errors, certificates, kl_bounds, margins = [], [], [], [], [], []
        counts, traces = [], []
        for h in range(heads):
            seed_scores = scores[h].clone()
            seed_scores[boundary] = float("inf")
            ids = torch.argsort(seed_scores, descending=True, stable=True)[:min(maximum, max(seed, len(boundary)))]
            stages, profile_items, sort_items, trace = 0, 0, n, []
            tail_order = tail_suffix = None
            tail_cursor = 0
            tail_reduction_slots = tail_preprocessing_slots = tail_suffix_lookups = 0
            while True:
                _complete[(triton.cdiv(len(ids), 32),)](self.K, q, order, ids.contiguous(),
                    scores, absolute, lower, upper, n, d, len(ids), h, coordinates, 1 / math.sqrt(d), guard, 32)
                selected[h, ids] = True
                size = int(selected[h].sum().item())
                # Retained LOWER mass is essential: nominal FP32 logits alone
                # cannot be used as a mathematically exact denominator.
                log_z_low = torch.logsumexp(lower[h].masked_fill(~selected[h], -torch.inf).double(), 0)
                if tail_suffix is None:
                    log_tail = torch.logsumexp(upper[h].masked_fill(selected[h], -torch.inf).double(), 0)
                    tail_reduction_slots += n
                    profile_items += n - size
                elif tail_cursor == len(tail_order):
                    log_tail = torch.full((), -torch.inf, device=self.device, dtype=torch.float64)
                else:
                    log_tail = tail_suffix[tail_cursor]
                    tail_suffix_lookups += 1
                # Float64 logsumexp guard; not a proof of library rounding.
                # The n factor also covers the sequential suffix-accumulation
                # implementation's rounding, as an allowance rather than an
                # IEEE proof. Both execution modes use the same allowance.
                numerical_log_guard = 64 * n * torch.finfo(torch.float64).eps * (1 + log_z_low.abs() + log_tail.nan_to_num().abs())
                log_ratio = log_tail - log_z_low + numerical_log_guard
                bound = torch.sigmoid(log_ratio)
                margin = log_ratio - math.log(eta / (1 - eta))
                if size == n:
                    bound = torch.zeros_like(bound)
                    margin = torch.full_like(margin, -torch.inf)
                    log_ratio = margin
                certified = bool((margin <= 0).item())
                stages += 1
                trace.append({"selected": size, "mass_upper": float(bound.item()), "certified": certified})
                if certified or size == maximum:
                    break
                take = min(maximum - size, max(seed, size))
                if reuse_tail_order:
                    if tail_order is None:
                        # Completing selected keys changes only selected caps.
                        # Every still-unopened cap is immutable: one stable
                        # ordering therefore reproduces repeated upper-cap
                        # sorting, and a suffix sum reproduces its tail mass.
                        remaining = (~selected[h]).nonzero().flatten()
                        permutation = torch.argsort(upper[h, remaining], descending=True, stable=True)
                        tail_order = remaining[permutation]
                        cap_values = upper[h, tail_order].double()
                        tail_suffix = torch.logcumsumexp(cap_values.flip(0), 0).flip(0)
                        tail_preprocessing_slots += len(remaining)
                        profile_items += len(remaining)
                        sort_items += len(remaining)
                    ids = tail_order[tail_cursor:tail_cursor + take]
                    tail_cursor += take
                else:
                    priority = upper[h].masked_fill(selected[h], -torch.inf)
                    ids = torch.argsort(priority, descending=True, stable=True)[:take]
                    sort_items += n
            final_ids = selected[h].nonzero().flatten()
            # Never round an absolute logZ to FP32: at a large common score
            # this erases log(count) and can turn normalized weights into ones.
            # Center before reduction and retain that center in the kernel.
            score_center = scores[h, final_ids].max()
            log_z_centered = torch.logsumexp((scores[h, final_ids] - score_center).double(), 0)
            chunks = triton.cdiv(size, 128)
            pieces = torch.empty((chunks, self.dv), device=self.device)
            _value_reduce[(chunks,)](self.V, final_ids, scores[h], score_center, log_z_centered, pieces, size,
                self.dv, triton.next_power_of_2(self.dv), 128)
            output = pieces.sum(0)
            score_guard = torch.maximum(scores[h, final_ids] - lower[h, final_ids],
                                         upper[h, final_ids] - scores[h, final_ids]).double().max()
            arithmetic_guard = self.value_bound * (64 * torch.finfo(torch.float32).eps *
                (d + self.dv + math.ceil(math.log2(max(n, 2)))))
            output_bound = 2 * self.value_bound * (bound + torch.minimum(score_guard, torch.ones_like(score_guard))) + arithmetic_guard
            outputs.append(output); bounds.append(bound); errors.append(output_bound)
            certificates.append(certified); margins.append(margin)
            kl_bounds.append(torch.nn.functional.softplus(log_ratio))
            counts.append({"key_coordinate_scalars": n * coordinates + size * (d - coordinates),
                "value_scalars": size * self.dv, "selected_keys": size,
                "fully_scored_keys": n if coordinates == d else size,
                "key_read_bytes": (n * coordinates + size * (d - coordinates)) * self.K.element_size(),
                "value_read_bytes": size * self.dv * self.V.element_size(),
                "dense_KV_read_bytes": n * (d * self.K.element_size() + self.dv * self.V.element_size()),
                "bound_evaluations": n, "summary_scalars_read": 2 * self.blocks * d,
                "stages": stages, "tail_item_evaluations": profile_items,
                "tail_reduction_slots": tail_reduction_slots,
                "tail_preprocessing_slots": tail_preprocessing_slots,
                "tail_suffix_lookups": tail_suffix_lookups,
                "retained_mass_reduction_slots": stages * n,
                "sort_items": sort_items,
                "tail_levels": n - size, "score_roundoff_allowance": float(score_guard.item()),
                "output_roundoff_allowance": float(arithmetic_guard.item())})
            traces.append(trace)
        return {"output": torch.stack(outputs), "selected_mask": selected,
                "logits": scores.masked_fill(~selected, torch.nan), "lower": lower, "upper": upper,
                "mass_upper": torch.stack(bounds), "output_error_upper": torch.stack(errors),
                "kl_upper": torch.stack(kl_bounds), "certificate_margin": torch.stack(margins),
                "certified": torch.tensor(certificates, device=self.device), "counts": counts, "trace": traces,
                "tail_mode": "unquantized per-key upper-cap logsumexp (not the CPU 16-band profile)",
                "reuse_tail_order": bool(reuse_tail_order),
                "floating_point_status": "heuristically guarded FP32; dense-oracle verification is not a rounding proof"}
