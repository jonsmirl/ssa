"""Device-only fixed proposal, mass certificate, and conditional dense fallback.

Preparation validates an immutable visible-prefix archive and a query buffer.
``plan.run()`` has fixed shapes and no device-to-host decisions; CUDA graphs may
capture it. Update the original query buffer in-place before replay. Finite,
supported-dtype queries remain a caller obligation after preparation.

The sparse proposal completes a fixed key budget before its certificate. Values
are loaded only after the device decides: selected values for accepted heads,
all values for rejected heads. Rejected heads pay for both proposal key loads
and a full dense key read. Guards are engineering allowances, not an IEEE proof.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .partial_coordinate_gpu import CoordinateGPUIndex, _integer, _intervals, _complete


@triton.jit
def _accepted_values(V, IDS, SCORE, CENTER, LOGZ, ACCEPT, PIECES,
                     H: tl.constexpr, N: tl.constexpr, COUNT: tl.constexpr,
                     DV: tl.constexpr, VD: tl.constexpr,
                     CHUNKS: tl.constexpr, TILE: tl.constexpr):
    h, b = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, VD)
    if tl.load(ACCEPT + h):
        j = b * TILE + tl.arange(0, TILE)
        row = tl.load(IDS + h * COUNT + j, j < COUNT, 0)
        score = tl.load(SCORE + h * N + row, j < COUNT, -float("inf"))
        center = tl.load(CENTER + h)
        log_z = tl.load(LOGZ + h).to(tl.float32)
        weights = tl.exp((score - center) - log_z)
        value = tl.load(V + row[:, None] * DV + d[None, :],
                        (j[:, None] < COUNT) & (d[None, :] < DV), 0).to(tl.float32)
        output = tl.sum(weights[:, None] * value, 0)
    else:
        output = tl.full((VD,), 0, tl.float32)
    tl.store(PIECES + (h * CHUNKS + b) * DV + d, output, d < DV)


@triton.jit
def _fallback_tiles(K, V, Q, ACCEPT, SCORE, LOW, HIGH, MAXIMUM, MASS, NUMERATOR,
                    N: tl.constexpr, D: tl.constexpr, DV: tl.constexpr,
                    VD: tl.constexpr, CHUNKS: tl.constexpr, TILE: tl.constexpr,
                    BETA: tl.constexpr, GUARD: tl.constexpr):
    h, b = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, VD)
    if not tl.load(ACCEPT + h):
        row = b * TILE + tl.arange(0, TILE)
        a = tl.arange(0, D)
        q = tl.load(Q + h * D + a).to(tl.float32) * BETA
        key = tl.load(K + row[:, None] * D + a[None, :], row[:, None] < N, 0).to(tl.float32)
        terms = key * q[None, :]
        score = tl.sum(terms, 1)
        guard = GUARD * tl.sum(tl.abs(terms), 1) + 1.0e-30
        tl.store(SCORE + h * N + row, score, row < N)
        tl.store(LOW + h * N + row, score - guard, row < N)
        tl.store(HIGH + h * N + row, score + guard, row < N)
        score = tl.where(row < N, score, -float("inf"))
        maximum = tl.max(score, 0)
        weights = tl.exp(score - maximum)
        mass = tl.sum(weights, 0)
        value = tl.load(V + row[:, None] * DV + d[None, :],
                        (row[:, None] < N) & (d[None, :] < DV), 0).to(tl.float32)
        numerator = tl.sum(weights[:, None] * value, 0)
    else:
        maximum = 0.0
        mass = 0.0
        numerator = tl.full((VD,), 0, tl.float32)
    tl.store(MAXIMUM + h * CHUNKS + b, maximum)
    tl.store(MASS + h * CHUNKS + b, mass)
    tl.store(NUMERATOR + (h * CHUNKS + b) * DV + d, numerator, d < DV)


class DeviceCoordinateIndex(CoordinateGPUIndex):
    """Same immutable prefix metadata as the measured adaptive GPU reference."""

    def prepare(self, q, coordinates=32, fraction=.25, eta=.1):
        return DeviceCoordinatePlan(self, q, coordinates, fraction, eta)


class DeviceCoordinatePlan:
    """Fixed-shape prepared query; host validation occurs only in construction.

    Query contents may change in-place, but shape, dtype and device must not.
    Replay inputs must remain finite and FP32 calculations must not overflow.
    ``run`` reports a device ``numerically_valid`` mask; invalid results must
    not be treated as certified. It does not synchronize to raise exceptions.
    """

    def __init__(self, index, q, coordinates, fraction, eta):
        coordinates = _integer(coordinates, "coordinates")
        if not 0 <= coordinates <= index.d or not 0 < fraction <= 1 or not 0 < eta < 1:
            raise ValueError("invalid coordinate count, proposal fraction or tolerance")
        if q.ndim == 1:
            q = q[None, :]
        if q.ndim != 2 or q.shape[1] != index.d or len(q) < 1 or q.device != index.device:
            raise ValueError("q must have shape [heads,64] on the index CUDA device")
        if q.dtype not in (torch.float16, torch.bfloat16, torch.float32) or not bool(torch.isfinite(q).all()):
            raise ValueError("q must be finite float16, bfloat16 or float32")
        self.index, self.q = index, q
        self.heads, self.coordinates = len(q), coordinates
        self.fraction, self.eta = float(fraction), float(eta)
        self.count = max(math.ceil(fraction * index.n), index.n % index.block)
        self.guard = 64 * torch.finfo(torch.float32).eps * index.d
        self.boundary_mask = torch.arange(index.n, device=index.device) >= index.n // index.block * index.block
        self.scores = torch.empty((self.heads, index.n), device=index.device)
        self.absolute, self.lower, self.upper = [torch.empty_like(self.scores) for _ in range(3)]
        self.value_chunks = triton.cdiv(self.count, 128)
        self.dense_chunks = triton.cdiv(index.n, 128)
        self.value_pieces = torch.empty((self.heads, self.value_chunks, index.dv), device=index.device)
        self.dense_maximum = torch.empty((self.heads, self.dense_chunks), device=index.device)
        self.dense_mass = torch.empty_like(self.dense_maximum)
        self.dense_numerator = torch.empty((self.heads, self.dense_chunks, index.dv), device=index.device)

    @torch.no_grad()
    def run(self):
        ix, h, n, d, k, r = self.index, self.heads, self.index.n, self.index.d, self.count, self.coordinates
        q = self.q.float().contiguous()
        order = torch.argsort(q.abs(), dim=1, descending=True, stable=True).contiguous()
        _intervals[(h, ix.blocks)](ix.K, q, order, ix.minimum, ix.maximum,
            self.scores, self.absolute, self.lower, self.upper,
            n, d, ix.block, r, 1 / math.sqrt(d), self.guard)
        priority = self.scores.masked_fill(self.boundary_mask[None, :], torch.inf)
        ids = torch.argsort(priority, dim=1, descending=True, stable=True)[:, :k].contiguous()
        proposed_selected = torch.zeros((h, n), device=ix.device, dtype=torch.bool).scatter_(1, ids, True)
        # This loop depends only on prepared head count, never on device data.
        for head in range(h):
            _complete[(triton.cdiv(k, 32),)](ix.K, q, order, ids[head],
                self.scores, self.absolute, self.lower, self.upper,
                n, d, k, head, r, 1 / math.sqrt(d), self.guard, 32)
        selected_scores = self.scores.gather(1, ids)
        selected_lower = self.lower.gather(1, ids)
        selected_upper = self.upper.gather(1, ids)
        centers = selected_scores.amax(1)
        centered_log_z = torch.logsumexp((selected_scores - centers[:, None]).double(), 1)
        retained_lower = torch.logsumexp(selected_lower.double(), 1)
        tail_upper = torch.logsumexp(self.upper.masked_fill(proposed_selected, -torch.inf).double(), 1)
        safe_tail_scale = torch.where(torch.isfinite(tail_upper), tail_upper.abs(), torch.zeros_like(tail_upper))
        log_guard = 64 * n * torch.finfo(torch.float64).eps * (1 + retained_lower.abs() + safe_tail_scale)
        log_ratio = tail_upper - retained_lower + log_guard
        proposed_mass = torch.sigmoid(log_ratio)
        proposal_valid = (torch.isfinite(self.lower).all(1) & torch.isfinite(self.upper).all(1)
                          & torch.isfinite(q).all(1))
        margin = log_ratio - math.log(self.eta / (1 - self.eta))
        accepted = (margin <= 0) & proposal_valid
        _accepted_values[(h, self.value_chunks)](ix.V, ids, self.scores, centers,
            centered_log_z, accepted, self.value_pieces,
            h, n, k, ix.dv, triton.next_power_of_2(ix.dv), self.value_chunks, 128)
        sparse_output = self.value_pieces.sum(1)
        proposal_score_guard = torch.maximum(selected_scores - selected_lower,
                                               selected_upper - selected_scores).amax(1).double()
        # The branch is inside the GPU program, before any fallback K/V load.
        _fallback_tiles[(h, self.dense_chunks)](ix.K, ix.V, q, accepted,
            self.scores, self.lower, self.upper, self.dense_maximum, self.dense_mass, self.dense_numerator,
            n, d, ix.dv, triton.next_power_of_2(ix.dv), self.dense_chunks, 128, 1/math.sqrt(d), self.guard)
        merge_max = self.dense_maximum.amax(1)
        merge_scale = (self.dense_maximum - merge_max[:, None]).exp()
        denominator = (merge_scale * self.dense_mass).sum(1).clamp_min(torch.finfo(torch.float32).tiny)
        dense_output = (merge_scale[:, :, None] * self.dense_numerator).sum(1) / denominator[:, None]
        output = torch.where(accepted[:, None], sparse_output, dense_output)
        selected = proposed_selected | ~accepted[:, None]
        score_guard = torch.maximum(self.scores - self.lower, self.upper - self.scores)
        score_guard = score_guard.masked_fill(~selected, 0).amax(1).double()
        mass = torch.where(accepted, proposed_mass, torch.zeros_like(proposed_mass))
        arithmetic_guard = ix.value_bound * (64 * torch.finfo(torch.float32).eps *
            (d + ix.dv + math.ceil(math.log2(max(n, 2)))))
        output_bound = 2 * ix.value_bound * (mass + score_guard.clamp_max(1)) + arithmetic_guard
        proposed_output_bound = 2 * ix.value_bound * (proposed_mass + proposal_score_guard.clamp_max(1)) + arithmetic_guard
        numerically_valid = (proposal_valid & torch.isfinite(output).all(1)
            & torch.isfinite(self.lower).all(1) & torch.isfinite(self.upper).all(1)
            & torch.isfinite(output_bound))
        fallback = (~accepted).to(torch.int64)
        key_scalars = torch.full((h,), n * r + k * (d - r), dtype=torch.int64, device=ix.device) + fallback * n * d
        value_scalars = torch.where(accepted, k, n) * ix.dv
        counts = {
            "proposal_key_coordinate_scalars": torch.full_like(key_scalars, n * r + k * (d - r)),
            "fallback_key_coordinate_scalars": fallback * n * d,
            "key_coordinate_scalars": key_scalars,
            "value_scalars": value_scalars,
            "proposal_selected_keys": torch.full_like(key_scalars, k),
            "final_selected_keys": torch.where(accepted, k, n),
            "fallback_heads": fallback,
            "key_read_bytes": key_scalars * ix.K.element_size(),
            "value_read_bytes": value_scalars * ix.V.element_size(),
            "dense_KV_read_bytes": torch.full_like(key_scalars, n*(d*ix.K.element_size()+ix.dv*ix.V.element_size())),
            "summary_scalars_read": torch.full_like(key_scalars, 2*ix.blocks*d),
            "bound_evaluations": torch.full_like(key_scalars, n),
            "tail_reduction_slots": torch.full_like(key_scalars, n),
            "sort_items": torch.full_like(key_scalars, n+d),
        }
        return {"output": output, "accepted": accepted, "certified": numerically_valid,
            "proposed_mass_upper": proposed_mass, "mass_upper": mass,
            "proposed_certificate_margin": margin, "output_error_upper": output_bound,
            "proposed_output_error_upper": proposed_output_bound,
            "kl_upper": torch.where(accepted, torch.nn.functional.softplus(log_ratio), torch.zeros_like(log_ratio)),
            "selected_mask": selected, "proposed_selected_mask": proposed_selected,
            "lower": self.lower, "upper": self.upper, "logits": self.scores.masked_fill(~selected, torch.nan),
            "numerically_valid": numerically_valid, "counts": counts,
            "score_roundoff_allowance": score_guard, "output_roundoff_allowance": arithmetic_guard,
            "tail_mode": "fixed-budget unquantized per-key cap; device-selected exact dense fallback",
            "floating_point_status": "heuristic FP32 guards, not an IEEE rounding proof"}
