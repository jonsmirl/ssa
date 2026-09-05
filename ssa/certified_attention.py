"""Adaptive subset attention with bounds on dense-attention mass, KL and output error.

NumPy reference, using precomputed key/value balls and log-space partition sums.
The inequalities hold in real arithmetic; float64 evaluation is not an interval proof.
See paper §5.7 and docs/substrate_math_imports.md for the derivation and Lean scope.
Run the deterministic CPU comparison with ``python -m ssa.certified_attention``.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np


@dataclass(frozen=True)
class CertifiedRead:
    output: np.ndarray
    indices: np.ndarray
    certified: bool
    mass_upper: float
    kl_upper: float                 # KL(subset weights || dense weights), natural logs
    output_error_upper: float       # Euclidean norm
    keys_scored: int
    blocks_opened: int
    bounds_evaluated: int
    certificate_checks: int
    value_bounds_evaluated: int


def _integer(value, name):
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _positive_exp(log_value):
    # A nonempty omitted set has positive mass at every finite temperature.
    # Do not let underflow turn a zero-tolerance request into an early stop.
    return float(np.exp(max(log_value, np.log(np.nextafter(0.0, 1.0)))))


class CertifiedBlockAttention:
    """Reusable immutable snapshot of contiguous blocks of keys and values.

    Construction reads all keys/values once, O(n(d + d_v)). Each query evaluates
    O(Bd) key-summary bounds, sorts B blocks, and scores only opened keys. Output
    certificates inspect O(B d_v) summaries per check; geometric batch growth
    limits checks to O(log B). These costs are separate from ``keys_scored``.
    This reference does not implement a GPU kernel or incremental cache updates.
    """

    def __init__(self, K, V, block_size=64):
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
        self.key_means, self.key_radii = self._balls(self.K)
        self.value_means, self.value_radii = self._balls(self.V)
        for a in (self.K, self.V, self.key_means, self.key_radii,
                  self.value_means, self.value_radii):
            if not np.isfinite(a).all():
                raise ValueError("key/value summary arithmetic exceeds float64 range")
            a.setflags(write=False)

    def _balls(self, X):
        means, radii = [], []
        for start in range(0, len(X), self.block_size):
            part = X[start:start + self.block_size]
            mean = part.mean(axis=0)
            means.append(mean)
            radii.append(np.linalg.norm(part - mean, axis=1).max())
        return np.asarray(means), np.asarray(radii)

    def read(self, q, beta=1.0, *, mass_tol=None, error_tol=None,
             max_blocks=None, initial_blocks=(), prefix=None):
        """Open blocks until every requested tolerance holds, or the cap is met.

        With neither tolerance specified, mass_tol defaults to 1e-3. An error-only
        request may omit high-mass blocks whose values already match the output.
        ``initial_blocks`` seeds this read with another router's block selection;
        all visible keys in those blocks are kept. A positive ``max_blocks`` is a
        hard cap including seeds and the partial block. A capped result reports
        ``certified=False`` when its tolerances remain unmet.

        ``prefix`` is an exclusive causal end index. A partially visible block is
        opened first, using only its visible keys; its full-block summaries are
        never consulted. Future keys therefore cannot influence the result.
        """
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self.K.shape[1],) or not np.isfinite(q).all():
            raise ValueError("q must be a finite vector matching the key dimension")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if mass_tol is None and error_tol is None:
            mass_tol = 1e-3
        for name, tol in (("mass_tol", mass_tol), ("error_tol", error_tol)):
            if tol is not None and (not np.isfinite(tol) or tol < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if mass_tol is not None and mass_tol > 1:
            raise ValueError("mass_tol must be at most 1")
        end = len(self.K) if prefix is None else _integer(prefix, "prefix")
        if not 1 <= end <= len(self.K):
            raise ValueError("prefix must select a nonempty prefix of K")
        b = self.block_size
        full, partial = divmod(end, b)
        nb = full + bool(partial)
        cap = nb if max_blocks is None else _integer(max_blocks, "max_blocks")
        if cap < 1:
            raise ValueError("max_blocks must be positive")
        cap = min(cap, nb)
        seeds = {_integer(c, "initial block") for c in initial_blocks}
        if any(c < 0 or c >= nb for c in seeds):
            raise ValueError("initial blocks must be visible block indices")
        if partial:
            seeds.add(full)
        if len(seeds) > cap:
            raise ValueError("max_blocks cannot be smaller than the mandatory initial set")

        centers = beta * (self.key_means[:full] @ q)
        widths = beta * np.linalg.norm(q) * self.key_radii[:full]
        # A small outward cushion limits roundoff at aligned/singleton queries.
        # It is not a claim of directed-rounding certification.
        guard = 32 * np.finfo(float).eps * (self.K.shape[1] + 1) * (
            beta * (np.abs(self.key_means[:full]) @ np.abs(q)) + widths + 1)
        log_upper = np.log(b) + centers + widths + guard
        log_lower = np.log(b) + centers - guard  # Jensen, since centers use means
        if not (np.isfinite(log_upper).all() and np.isfinite(log_lower).all()):
            raise ValueError("query/summary arithmetic exceeds float64 range")
        order = np.asarray([c for c in np.argsort(-log_upper, kind="stable")
                            if c not in seeds], dtype=int)
        # Suffix reductions avoid subtraction/cancellation when removing a block.
        suffix_upper = np.full(len(order) + 1, -np.inf)
        suffix_lower = suffix_upper.copy()
        for j in range(len(order) - 1, -1, -1):
            suffix_upper[j] = np.logaddexp(log_upper[order[j]], suffix_upper[j + 1])
            suffix_lower[j] = np.logaddexp(log_lower[order[j]], suffix_lower[j + 1])

        log_z, output = -np.inf, np.zeros(self.V.shape[1])
        selected = []

        def open_block(c):
            nonlocal log_z, output
            ids = np.arange(c * b, min((c + 1) * b, end))
            logits = beta * (self.K[ids] @ q)
            if not np.isfinite(logits).all():
                raise ValueError("query/key arithmetic exceeds float64 range")
            top = logits.max()
            weights = np.exp(logits - top)
            block_z = top + np.log(weights.sum())
            block_output = (weights / weights.sum()) @ self.V[ids]
            joined = np.logaddexp(log_z, block_z)
            output = np.exp(log_z - joined) * output + np.exp(block_z - joined) * block_output
            log_z = joined
            selected.append(ids)

        for c in sorted(seeds):
            open_block(c)
        pos, checks, value_evals = 0, 0, 0
        if not selected:
            open_block(order[pos])
            pos += 1
        while True:
            checks += 1
            remaining = order[pos:]
            if len(remaining) == 0:
                mass, kl, error = 0.0, 0.0, 0.0
            else:
                log_u = suffix_upper[pos]
                log_mass = -np.logaddexp(0.0, log_z - log_u)
                mass = min(1.0, _positive_exp(log_mass))
                kl = max(np.nextafter(0.0, 1.0), float(np.logaddexp(0.0, log_u - log_z)))
                distances = (np.linalg.norm(self.value_means[remaining] - output, axis=1)
                             + self.value_radii[remaining])
                value_evals += len(remaining)
                positive = distances > 0
                if not positive.any():
                    error = 0.0
                else:
                    # Two valid bounds: max distance * omitted mass, and a
                    # weighted numerator over a Jensen lower partition bound.
                    log_num = np.logaddexp.reduce(
                        log_upper[remaining[positive]] + np.log(distances[positive]))
                    log_error = min(np.log(distances.max()) + log_mass,
                                    log_num - np.logaddexp(log_z, suffix_lower[pos]))
                    error = _positive_exp(log_error)
            certified = ((mass_tol is None or mass <= mass_tol)
                         and (error_tol is None or error <= error_tol))
            if certified or len(selected) >= cap:
                ids = np.concatenate(selected)
                return CertifiedRead(output, ids, certified, mass, kl, error, len(ids),
                                     len(selected), full, checks, value_evals)
            # Geometric batch growth bounds certificate work even on flat logits.
            target = min(cap, max(len(selected) + 1, 2 * len(selected)))
            while len(selected) < target:
                open_block(order[pos])
                pos += 1


def main():
    from .core import dense_read

    rng = np.random.default_rng(0)
    n, d, dv, block = 4096, 32, 8, 64
    q = np.eye(d)[0]
    keys = 0.01 * rng.standard_normal((n, d))
    keys[:block, 0] += 4
    values = rng.standard_normal((n, dv))
    print("CPU reference: n=4096, d=32, d_v=8, block=64, beta=4, seed=0")
    print("geometry       keys  bounds  checks  value_bounds   mass_bound    error_bound   actual_error")
    for name, K, V, kwargs in (
        ("concentrated", keys, values, {"mass_tol": 1e-3}),
        ("flat", np.zeros_like(keys), values, {"mass_tol": 1e-3}),
        ("equal_values", np.zeros_like(keys), np.ones_like(values), {"error_tol": 1e-12}),
    ):
        result = CertifiedBlockAttention(K, V, block).read(q, beta=4, **kwargs)
        dense, _, _ = dense_read(q, K, V, 4)
        print(f"{name:13} {result.keys_scored:5} {result.bounds_evaluated:7} "
              f"{result.certificate_checks:7} {result.value_bounds_evaluated:13} "
              f"{result.mass_upper:12.5g} {result.output_error_upper:14.5g} "
              f"{np.linalg.norm(result.output - dense):14.5g}")


if __name__ == "__main__":
    main()
