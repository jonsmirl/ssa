"""Deterministic score-tail certificates for sparse attention.

This module connects a geometry router's selected blocks to the existing
restricted-read certificate without confusing two different claims:

* :class:`RoutingMetricSelection` records what the router selected and whether
  that selection was certified under *its routing metric*.
* :class:`AttentionScoreTailProfile` bounds the exponential mass of unopened
  keys from separately supplied admissible *attention-logit* upper bounds.

The reference reader derives those attention bounds directly from immutable
block means and Euclidean residual radii.  CCC/IVF selections are seeds only;
their current routing certificate is neither consumed nor relabelled as an
attention certificate.  The implementation is CPU/float64 and is checked
against dense oracles.  It is not interval arithmetic and it is not wired into
FlexAttention.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import numpy as np

from .certified_attention import _integer, _outward_radius, _positive_exp


def _readonly(array, dtype=None):
    value = np.asarray(array, dtype=dtype).copy()
    value.setflags(write=False)
    return value


def _host_array(value):
    """Convert NumPy or a CPU/CUDA tensor without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return -np.inf
    top = float(values.max())
    if top == -np.inf:
        return -np.inf
    return top + float(np.log(np.exp(values - top).sum()))


@dataclass(frozen=True)
class RoutingMetricSelection:
    """Blocks supplied by a router, explicitly scoped to its own metric."""

    block_indices: np.ndarray
    metric_name: str
    top_blocks_certified: bool = False

    def __post_init__(self):
        blocks = np.asarray(self.block_indices)
        if blocks.ndim != 1 or not np.issubdtype(blocks.dtype, np.integer):
            raise ValueError("routing block indices must be a one-dimensional integer array")
        if np.any(blocks < 0) or len(np.unique(blocks)) != len(blocks):
            raise ValueError("routing block indices must be distinct and nonnegative")
        if not isinstance(self.metric_name, str) or not self.metric_name:
            raise ValueError("metric_name must name the routing metric")
        object.__setattr__(self, "block_indices", _readonly(blocks, np.int64))

    @classmethod
    def from_ccc_row(cls, kv_num, kv_idx, *, certified=False):
        """Adapt one CCC compact row without upgrading its certificate.

        ``certified=True`` means exact top-block selection under CCC's routing
        metric.  It says nothing about individual attention logits or mass.
        """
        count = operator.index(_host_array(kv_num).item())
        indices = _host_array(kv_idx)[:count]
        return cls(indices.astype(np.int64), "ccc_routing_metric", bool(certified))


@dataclass(frozen=True)
class AttentionScoreTailProfile:
    """Disjoint attention-logit bands with certified key counts.

    ``upper_edges`` are nondecreasing finite logit upper bounds. ``counts[j]``
    is the number of unopened keys assigned to band ``j``; every such key must
    have attention logit at most ``upper_edges[j]``.  The internal ascending
    convention makes assignment use ``searchsorted``.  Reading it in reverse
    gives the descending score-tail profile used in the paper.
    """

    upper_edges: np.ndarray
    counts: np.ndarray
    log_mass_upper: float
    source_items: int

    def __post_init__(self):
        edges = np.asarray(self.upper_edges, dtype=np.float64)
        counts = np.asarray(self.counts)
        if edges.ndim != 1 or counts.ndim != 1 or edges.shape != counts.shape:
            raise ValueError("tail edges and counts must be one-dimensional and aligned")
        if not np.isfinite(edges).all() or np.any(edges[1:] < edges[:-1]):
            raise ValueError("tail edges must be finite and nondecreasing")
        if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
            raise ValueError("tail counts must be nonnegative integers")
        if int(counts.sum()) != operator.index(self.source_items):
            raise ValueError("tail counts must sum to source_items")
        expected = _logsumexp(
            edges[counts > 0] + np.log(counts[counts > 0].astype(np.float64)))
        if not ((expected == self.log_mass_upper)
                or np.isclose(expected, self.log_mass_upper, rtol=2e-15, atol=2e-15)):
            raise ValueError("log_mass_upper does not match the disjoint bands")
        object.__setattr__(self, "upper_edges", _readonly(edges))
        object.__setattr__(self, "counts", _readonly(counts, np.int64))

    @property
    def levels(self):
        return len(self.upper_edges)

    @property
    def cumulative_counts_above(self):
        """Descending cumulative certified counts, for reporting."""
        return np.cumsum(self.counts[::-1])[::-1]

    @property
    def mass_upper(self):
        return 0.0 if self.log_mass_upper == -np.inf else float(np.exp(self.log_mass_upper))


def tail_profile_from_bands(upper_edges, band_count_upper):
    """Build a profile from supplied disjoint-band count upper bounds.

    Soundness requires the caller's bands to cover every unopened key, with
    every key in band ``j`` bounded by ``upper_edges[j]``. Counts may be
    conservative upper bounds rather than exact cardinalities.
    """
    edges = np.asarray(upper_edges, dtype=np.float64)
    raw = np.asarray(band_count_upper)
    if (edges.ndim != 1 or raw.ndim != 1 or edges.shape != raw.shape
            or not np.isfinite(edges).all() or np.any(edges[1:] < edges[:-1])):
        raise ValueError("band edges and counts must be aligned with nondecreasing finite edges")
    if not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0):
        raise ValueError("band count upper bounds must be nonnegative integers")
    counts = raw.astype(np.int64, copy=True)
    nonempty = counts > 0
    log_mass = _logsumexp(edges[nonempty] + np.log(counts[nonempty].astype(np.float64)))
    return AttentionScoreTailProfile(edges, counts, log_mass, int(counts.sum()))


def tail_profile_from_item_caps(score_upper, item_counts=None, *, levels=16,
                                upper_edges=None):
    """Quantize admissible item/group caps into a sound disjoint-band profile.

    Each source item may stand for several keys via ``item_counts``.  If edges
    are supplied, every cap is placed at the first edge not below it. Repeated
    thresholds and empty bands are retained deliberately. Otherwise ``levels``
    equally spaced edges cover the observed cap range. One level is exactly the
    usual ``N * exp(max_score)`` residual-maximum bound.
    """
    caps = np.asarray(score_upper, dtype=np.float64)
    if caps.ndim != 1 or not np.isfinite(caps).all():
        raise ValueError("score_upper must be a finite one-dimensional array")
    if item_counts is None:
        weights = np.ones(len(caps), dtype=np.int64)
    else:
        raw = np.asarray(item_counts)
        if (raw.ndim != 1 or raw.shape != caps.shape
                or not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0)):
            raise ValueError("item_counts must be aligned nonnegative integers")
        weights = raw.astype(np.int64, copy=True)
    total = int(weights.sum())
    if len(caps) == 0:
        if upper_edges is not None and len(np.asarray(upper_edges)):
            edges = np.asarray(upper_edges, dtype=np.float64)
            if edges.ndim != 1 or not np.isfinite(edges).all() or np.any(edges[1:] < edges[:-1]):
                raise ValueError("tail edges must be finite and nondecreasing")
        else:
            edges = np.empty(0, dtype=np.float64)
        return tail_profile_from_bands(edges, np.zeros(len(edges), dtype=np.int64))

    if upper_edges is None:
        levels = _integer(levels, "levels")
        if levels < 1:
            raise ValueError("levels must be positive")
        lo, hi = float(caps.min()), float(caps.max())
        edges = np.linspace(lo, hi, levels, dtype=np.float64)
        edges[-1] = np.nextafter(max(edges[-1], hi), np.inf)
    else:
        edges = np.asarray(upper_edges, dtype=np.float64)
        if (edges.ndim != 1 or len(edges) == 0 or not np.isfinite(edges).all()
                or np.any(edges[1:] < edges[:-1])):
            raise ValueError("upper_edges must be a nonempty nondecreasing finite array")
        if edges[-1] < caps.max():
            raise ValueError("the final tail edge must dominate every source cap")
    band = np.searchsorted(edges, caps, side="left")
    if np.any(band == len(edges)):
        raise ValueError("a source cap was not covered by the tail edges")
    counts = np.zeros(len(edges), dtype=np.int64)
    np.add.at(counts, band, weights)
    profile = tail_profile_from_bands(edges, counts)
    if profile.source_items != total:  # defensive check against integer accumulation errors
        raise AssertionError("band counts did not preserve the supplied source count")
    return profile


def exact_score_histogram(scores):
    """Oracle profile whose band bound equals the supplied scores' true mass."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite one-dimensional array")
    if len(scores) == 0:
        return tail_profile_from_item_caps(scores)
    edges, counts = np.unique(scores, return_counts=True)
    log_mass = _logsumexp(edges + np.log(counts.astype(np.float64)))
    return AttentionScoreTailProfile(edges, counts.astype(np.int64), log_mass, len(scores))


def best_log_mass_upper(*log_caps):
    """The minimum of admissible log-mass caps is again admissible."""
    if not log_caps:
        raise ValueError("at least one admissible cap is required")
    values = np.asarray(log_caps, dtype=np.float64)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("log caps must be finite or negative infinity")
    return float(values.min())


def mass_share_upper_from_logs(log_tail_mass, log_kept_mass):
    if log_tail_mass == -np.inf:
        return 0.0
    if not (np.isfinite(log_tail_mass) and np.isfinite(log_kept_mass)):
        raise ValueError("nonempty tail and kept masses must have finite logarithms")
    return min(1.0, _positive_exp(-float(np.logaddexp(0.0, log_kept_mass - log_tail_mass))))


def certificate_margin(log_tail_mass, log_kept_mass, eta):
    """Exact hard stopping margin; nonpositive iff the mass certificate holds."""
    eta = float(eta)
    if not np.isfinite(eta) or not 0 < eta < 1:
        raise ValueError("eta must lie strictly between zero and one")
    if log_tail_mass == -np.inf:
        return -np.inf
    if not (np.isfinite(log_tail_mass) and np.isfinite(log_kept_mass)):
        raise ValueError("nonempty tail and kept masses must have finite logarithms")
    return float(log_tail_mass - log_kept_mass - math.log(eta / (1 - eta)))


def certificate_softplus_loss(log_tail_mass, log_kept_mass, eta):
    """The requested smooth surrogate ``softplus(M_eta)``.

    Softplus is strictly positive.  Its hard stopping boundary is ``log(2)``:
    ``softplus(M_eta) <= log(2)`` exactly when ``M_eta <= 0``.
    """
    margin = certificate_margin(log_tail_mass, log_kept_mass, eta)
    return float(np.logaddexp(0.0, margin))


@dataclass(frozen=True)
class TailCertifiedRead:
    output: np.ndarray
    indices: np.ndarray
    certified: bool
    mass_upper: float
    kl_upper: float
    output_error_upper: float
    keys_scored: int
    blocks_opened: int
    bounds_evaluated: int
    certificate_checks: int
    value_bounds_evaluated: int
    tail_levels: int
    tail_storage_scalars: int
    tail_query_work: int
    log_kept_mass: float
    log_tail_mass_upper: float
    certificate_margin: float | None
    routing_metric: str
    routing_top_blocks_certified: bool
    attention_score_bound: str


class ScoreTailCertifiedAttention:
    """CPU reference reader seeded by a geometry router.

    Every complete block stores a mean and a supplied Euclidean residual radius.
    For a query, the independent attention-logit bridge is

    ``beta * (q @ mean + ||q|| * radius)``.

    Unopened blocks are compressed into disjoint tail bands.  The reader scores
    routed blocks exactly, then opens further blocks in descending admissible-cap
    order until the hard mass/output request holds or the full fallback is
    reached. A partial causal boundary block is always read exactly.
    """

    def __init__(self, K, V, block_size=32):
        self.block_size = _integer(block_size, "block_size")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        self.K = np.array(K, dtype=np.float64, copy=True)
        self.V = np.array(V, dtype=np.float64, copy=True)
        if (self.K.ndim != 2 or self.V.ndim != 2 or len(self.K) == 0
                or len(self.V) != len(self.K) or self.K.shape[1] == 0 or self.V.shape[1] == 0):
            raise ValueError("K and V must be nonempty matrices with the same row count")
        if not (np.isfinite(self.K).all() and np.isfinite(self.V).all()):
            raise ValueError("K and V must be finite")
        means, radii, value_means, value_radii = [], [], [], []
        for start in range(0, len(self.K), self.block_size):
            keys = self.K[start:start + self.block_size]
            values = self.V[start:start + self.block_size]
            km, vm = keys.mean(0), values.mean(0)
            kr = float(np.linalg.norm(keys - km, axis=1).max())
            vr = float(np.linalg.norm(values - vm, axis=1).max())
            means.append(km)
            radii.append(_outward_radius(kr, self.K.shape[1], exact_zero=bool(np.all(keys == km))))
            value_means.append(vm)
            value_radii.append(_outward_radius(vr, self.V.shape[1], exact_zero=bool(np.all(values == vm))))
        self.key_means = _readonly(means, np.float64)
        self.key_radii = _readonly(radii, np.float64)
        self.value_means = _readonly(value_means, np.float64)
        self.value_radii = _readonly(value_radii, np.float64)
        self.K.setflags(write=False)
        self.V.setflags(write=False)

    def route_by_block_mean(self, q, top_blocks, *, prefix=None):
        """Deterministic CPU stand-in for a geometry router, not a mass certificate."""
        q = np.asarray(q, dtype=np.float64)
        end = len(self.K) if prefix is None else _integer(prefix, "prefix")
        if q.shape != (self.K.shape[1],) or not np.isfinite(q).all():
            raise ValueError("q must be a finite vector matching the key dimension")
        if not 1 <= end <= len(self.K):
            raise ValueError("prefix must select a nonempty prefix")
        full = end // self.block_size
        top_blocks = min(max(0, _integer(top_blocks, "top_blocks")), full)
        ids = np.arange(full)
        scores = self.key_means[:full] @ q
        order = np.lexsort((-ids, -scores))
        return RoutingMetricSelection(order[:top_blocks], "block_mean_inner_product", True)

    def _attention_caps(self, q, beta, full):
        centers = beta * (self.key_means[:full] @ q)
        widths = beta * float(np.linalg.norm(q)) * self.key_radii[:full]
        magnitude = beta * (np.abs(self.key_means[:full]) @ np.abs(q)) + np.abs(widths)
        guard = 32 * np.finfo(float).eps * (self.K.shape[1] + 1) * (magnitude + 1)
        return np.nextafter(centers + widths + guard, np.inf)

    def read(self, q, beta=1.0, *, routing=None, tail_levels=16,
             mass_tol=0.1, error_tol=None, max_blocks=None, prefix=None,
             retained_log_caps=()):
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.K.shape[1],) or not np.isfinite(q).all():
            raise ValueError("q must be a finite vector matching the key dimension")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if mass_tol is None and error_tol is None:
            mass_tol = 0.1
        for name, tol in (("mass_tol", mass_tol), ("error_tol", error_tol)):
            if tol is not None and (not np.isfinite(tol) or tol < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if mass_tol is not None and mass_tol > 1:
            raise ValueError("mass_tol must be at most one")
        end = len(self.K) if prefix is None else _integer(prefix, "prefix")
        if not 1 <= end <= len(self.K):
            raise ValueError("prefix must select a nonempty prefix of K")
        b = self.block_size
        full, partial = divmod(end, b)
        nb = full + bool(partial)
        cap = nb if max_blocks is None else min(nb, _integer(max_blocks, "max_blocks"))
        if cap < 1:
            raise ValueError("max_blocks must be positive")
        routing = routing or RoutingMetricSelection(np.empty(0, dtype=np.int64), "none", False)
        seeds = set(map(int, routing.block_indices))
        if any(c >= nb for c in seeds):
            raise ValueError("routing selection contains a block outside the causal prefix")
        if partial:
            seeds.add(full)
        if len(seeds) > cap:
            raise ValueError("max_blocks cannot be smaller than the routed/partial seed set")

        attention_caps = self._attention_caps(q, beta, full)
        block_ids = np.arange(full)
        order = np.lexsort((-block_ids, -attention_caps))
        order = [int(c) for c in order if int(c) not in seeds]
        opened = set()
        selected = []
        log_z = -np.inf
        output = np.zeros(self.V.shape[1], dtype=np.float64)

        def open_block(c):
            nonlocal log_z, output
            if c in opened:
                return
            ids = np.arange(c * b, min((c + 1) * b, end))
            logits = beta * (self.K[ids] @ q)
            top = float(logits.max())
            weights = np.exp(logits - top)
            block_log_z = top + float(np.log(weights.sum()))
            block_output = (weights / weights.sum()) @ self.V[ids]
            joined = float(np.logaddexp(log_z, block_log_z))
            output = np.exp(log_z - joined) * output + np.exp(block_log_z - joined) * block_output
            log_z = joined
            opened.add(c)
            selected.append(ids)

        for c in sorted(seeds):
            open_block(c)
        if not selected:
            open_block(order.pop(0))

        checks = value_evals = profile_items = 0
        last_profile = None
        last_log_tail = -np.inf
        last_margin = None
        while True:
            checks += 1
            remaining = np.asarray([c for c in range(full) if c not in opened], dtype=np.int64)
            if len(remaining) == 0:
                profile = tail_profile_from_item_caps(np.empty(0), levels=tail_levels)
                log_tail = -np.inf
                mass = kl = error = 0.0
                margin = -np.inf if mass_tol is not None and 0 < mass_tol < 1 else None
            else:
                profile_items += len(remaining)
                profile = tail_profile_from_item_caps(
                    attention_caps[remaining], np.full(len(remaining), b, dtype=np.int64),
                    levels=tail_levels)
                log_tail = best_log_mass_upper(profile.log_mass_upper, *retained_log_caps)
                mass = mass_share_upper_from_logs(log_tail, log_z)
                kl = max(np.nextafter(0.0, 1.0), float(np.logaddexp(0.0, log_tail - log_z)))
                distances = (np.linalg.norm(self.value_means[remaining] - output, axis=1)
                             + self.value_radii[remaining])
                value_evals += len(remaining)
                error = 0.0 if not np.any(distances > 0) else mass * float(distances.max())
                margin = (certificate_margin(log_tail, log_z, mass_tol)
                          if mass_tol is not None and 0 < mass_tol < 1 else None)
            last_profile, last_log_tail, last_margin = profile, log_tail, margin
            certified = ((mass_tol is None or mass <= mass_tol)
                         and (error_tol is None or error <= error_tol))
            if certified or len(opened) >= cap:
                indices = np.concatenate(selected)
                return TailCertifiedRead(
                    output, indices, certified, mass, kl, error, len(indices), len(opened),
                    full, checks, value_evals, last_profile.levels,
                    2 * last_profile.levels,
                    (full * self.K.shape[1] + profile_items
                     + value_evals * self.V.shape[1] + checks * last_profile.levels),
                    log_z, last_log_tail, last_margin, routing.metric_name,
                    routing.top_blocks_certified, "direct_block_mean_plus_residual_radius")
            target = min(cap, max(len(opened) + 1, 2 * len(opened)))
            while len(opened) < target and order:
                open_block(order.pop(0))

    def read_ccc_row(self, q, kv_num, kv_idx, *, ccc_certified=False, **kwargs):
        """Read from one CCC compact row while preserving certificate scopes."""
        routing = RoutingMetricSelection.from_ccc_row(
            kv_num, kv_idx, certified=ccc_certified)
        return self.read(q, routing=routing, **kwargs)
