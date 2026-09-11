"""CPU reference for partial-coordinate *attention-logit* intervals.

This is not a CCC routing-score certificate. A snapshot contains only its
visible causal prefix and supplies coordinate extrema of its own keys. The
real-arithmetic interval follows by bounding each unread q[j]*k[j] between
the corresponding signed coordinate extrema. Float64 guards are engineering
roundoff allowances, not an IEEE interval-arithmetic proof.

The read ledger counts logically accessed key coordinates. NumPy stores the
whole archive and uses dense masks/temporary arrays: this is not a sparse
memory kernel, and its reference refresh work is reported separately.
"""
from __future__ import annotations

import operator
import numpy as np


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _indices(values, size, name):
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must contain integers")
    if np.any(raw < 0) or np.any(raw >= size):
        raise ValueError(f"{name} out of range")
    return np.unique(raw.astype(np.int64))


class CoordinateBounds:
    """Immutable finite-key snapshot; slice K[:prefix] *before* construction.

    ``block_interval`` uses per-block signed minima/maxima. ``global_abs``
    bounds each unread term by |q[j]| max_i |k[i,j]|. Neither assumes that
    layer normalization bounds individual coordinates by one.
    """

    def __init__(self, K, block=64, mode="block_interval"):
        keys = np.array(K, dtype=np.float64, copy=True)
        if keys.ndim != 2 or keys.shape[1] == 0 or not np.isfinite(keys).all():
            raise ValueError("K must be a finite (n, positive dimension) array")
        block = _integer(block, "block")
        if block <= 0:
            raise ValueError("block must be positive")
        if mode not in ("block_interval", "global_abs"):
            raise ValueError("unknown interval mode")
        self.K, self.block, self.mode = keys, block, mode
        self.n, self.d = keys.shape
        self.blocks = (self.n + block - 1) // block
        self.minimum = np.empty((self.blocks, self.d))
        self.maximum = np.empty_like(self.minimum)
        for b in range(self.blocks):
            rows = keys[b * block:(b + 1) * block]
            self.minimum[b] = rows.min(axis=0)
            self.maximum[b] = rows.max(axis=0)
        self.global_abs = np.max(np.abs(keys), axis=0) if self.n else np.zeros(self.d)
        for value in (self.K, self.minimum, self.maximum, self.global_abs):
            value.setflags(write=False)

    @property
    def summary_scalars(self):
        # The reference holds both forms, even when only one is used.
        return int(self.minimum.size + self.maximum.size + self.global_abs.size)

    def start(self, q, beta=1.0):
        return CoordinateRead(self, q, beta)


class CoordinateRead:
    def __init__(self, index, q, beta):
        q = np.array(q, dtype=np.float64, copy=True)
        beta = float(beta)
        if q.shape != (index.d,) or not np.isfinite(q).all():
            raise ValueError("q must be finite and match the key dimension")
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("beta must be finite and positive")
        self.index, self.q, self.beta = index, q, beta
        with np.errstate(over="ignore", invalid="ignore"):
            self.scaled_query = beta * q
        if not np.isfinite(self.scaled_query).all():
            raise ValueError("scaled query overflow")
        self.read_mask = np.zeros((index.n, index.d), dtype=bool)
        self.opened = np.zeros(index.n, dtype=bool)
        self._terms = np.zeros((index.n, index.d), dtype=np.float64)
        self.partial = np.zeros(index.n)
        self.lower = np.full(index.n, -np.inf)
        self.upper = np.full(index.n, np.inf)
        self.coordinate_scalars_read = 0
        self.interval_refreshes = 0
        self._refresh()

    @property
    def exact_mask(self):
        return self.read_mask.all(axis=1)

    @property
    def coordinate_order(self):
        """Largest absolute query coordinates first; coordinate index breaks ties."""
        return np.lexsort((np.arange(self.index.d), -np.abs(self.q)))

    def _refresh(self):
        ix = self.index
        self.interval_refreshes += 1
        with np.errstate(over="ignore", invalid="ignore"):
            if ix.mode == "block_interval":
                block_ids = np.arange(ix.n) // ix.block
                a = ix.minimum[block_ids] * self.scaled_query
                b = ix.maximum[block_ids] * self.scaled_query
                lo, hi = np.minimum(a, b), np.maximum(a, b)
            else:
                radius = ix.global_abs * np.abs(self.scaled_query)
                lo = np.broadcast_to(-radius, self.read_mask.shape)
                hi = np.broadcast_to(radius, self.read_mask.shape)
            lower_terms = np.where(self.read_mask, self._terms, lo)
            upper_terms = np.where(self.read_mask, self._terms, hi)
            lower = lower_terms.sum(axis=1)
            upper = upper_terms.sum(axis=1)
            magnitude = np.maximum(np.abs(lower_terms), np.abs(upper_terms)).sum(axis=1)
            guard = (64 * np.finfo(float).eps * ix.d) * magnitude
            lower = np.nextafter(lower - guard, -np.inf)
            upper = np.nextafter(upper + guard, np.inf)
        if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
            raise ValueError("interval arithmetic overflow")
        self.partial = self._terms.sum(axis=1)
        # Intersections retain the best bound; exact rows use the computed
        # canonical score. These numerical checks do not formalize roundoff.
        self.lower = np.maximum(self.lower, lower)
        self.upper = np.minimum(self.upper, upper)
        exact = self.exact_mask
        self.lower[exact] = self.partial[exact]
        self.upper[exact] = self.partial[exact]
        if np.any(self.lower > self.upper):
            raise ArithmeticError("empty interval after refinement")

    def _read(self, rows, coords):
        if len(rows) == 0 or len(coords) == 0:
            return
        rr, cc = np.ix_(rows, coords)
        missing = ~self.read_mask[rr, cc]
        self.coordinate_scalars_read += int(missing.sum())
        # Already-known terms are left unchanged; no second logical key read.
        selected_rows, selected_cols = np.nonzero(missing)
        r, c = rows[selected_rows], coords[selected_cols]
        with np.errstate(over="ignore", invalid="ignore"):
            terms = self.index.K[r, c] * self.scaled_query[c]
        if not np.isfinite(terms).all():
            raise ValueError("key-coordinate product overflow")
        self._terms[r, c] = terms
        self.read_mask[r, c] = True
        self._refresh()

    def refine_coordinates(self, coords):
        coords = _indices(coords, self.index.d, "coords")
        self._read(np.arange(self.index.n), coords)
        return self

    def open_keys(self, ids):
        """Complete specified keys, returning scores in sorted unique-id order."""
        ids = _indices(ids, self.index.n, "ids")
        self._read(ids, np.arange(self.index.d))
        self.opened[ids] = True
        return self.partial[ids].copy()

    def topk_certificate(self, selected):
        """Strict score-set separation; no claim about omitted attention mass.

        Empty/full selected sets are vacuous. A boundary tie is not a strict
        certificate, even though an algorithm may deterministically break it.
        """
        selected = _indices(selected, self.index.n, "selected")
        other = np.ones(self.index.n, dtype=bool)
        other[selected] = False
        if len(selected) == 0 or not other.any():
            return {"certified": True, "strict_margin": None, "vacuous": True}
        margin = float(self.lower[selected].min() - self.upper[other].max())
        return {"certified": margin > 0, "strict_margin": margin, "vacuous": False}

    def work(self):
        ix = self.index
        return {
            "coordinate_scalars_read": self.coordinate_scalars_read,
            "index_build_key_scalars": int(ix.K.size),
            "index_summary_scalars": ix.summary_scalars,
            "index_archive_bytes": int(ix.K.nbytes),
            "index_summary_bytes": ix.summary_scalars * 8,
            "query_state_bytes": int(self.read_mask.nbytes + self.opened.nbytes
                                     + self._terms.nbytes + self.partial.nbytes
                                     + self.lower.nbytes + self.upper.nbytes
                                     + self.q.nbytes + self.scaled_query.nbytes),
            "interval_refreshes": self.interval_refreshes,
            "reference_refresh_coordinate_slots": self.interval_refreshes * ix.n * ix.d,
            "bound_evaluations": self.interval_refreshes * ix.n,
            "opened_keys": int(self.opened.sum()),
            "fully_scored_keys": int(self.exact_mask.sum()),
        }
