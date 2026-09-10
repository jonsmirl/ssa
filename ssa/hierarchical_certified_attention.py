"""Hierarchical adaptive attention certificates.

This is the tree analogue of :mod:`ssa.certified_attention`.  A balanced binary
tree groups contiguous key blocks.  Every frontier node carries key and value
balls, so it bounds the total softmax mass and output contribution of all leaves
below it.  Refinement replaces one admissible node bound by its children; opening
a leaf scores the corresponding keys exactly.  Against a fixed reference point,
each child reach (and therefore its score cap) is no larger than its parent's, so
at a fixed threshold refinement can only enlarge the set certified droppable.

At unit scale, recursively grouped log partitions flatten to ordinary softmax
(``Substrate.Universal.treeSoftmax_unitScale``).  The tree therefore changes only
how the omitted partition is bounded, not the attention distribution being
approximated.  The Lean theorem and the Python composition are separate: the
latter is checked against a dense oracle in the test suite.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np

from .certified_attention import (
    CertifiedBlockAttention,
    CertifiedRead,
    _integer,
    _outward_radius,
    _positive_exp,
)


@dataclass(frozen=True)
class _PeelSummary:
    exposed_indices: np.ndarray
    core_count: int
    core_mean: np.ndarray
    core_radius: float
    core_spread: float
    core_covariance: np.ndarray | None


@dataclass(frozen=True)
class _Node:
    start: int                       # first fixed-size leaf block, inclusive
    end: int                         # last fixed-size leaf block, exclusive
    count: int                       # number of keys below the node
    key_mean: np.ndarray
    key_radius: float
    key_spread: float                # mean squared Euclidean distance from key_mean
    key_covariance: np.ndarray | None
    value_mean: np.ndarray
    value_radius: float
    children: tuple[int, int] | None
    peel: _PeelSummary | None


@dataclass(frozen=True)
class _Bound:
    log_upper: float
    log_lower: float


class CertifiedTreeAttention(CertifiedBlockAttention):
    """Adaptive dense-attention certificate over a binary block tree.

    Radius/trace/covariance construction is linear in the number of keys and
    stores fewer than two nodes per full leaf block. The experimental peeled
    modes rescan and sort each node's descendants; they instantiate the bound
    but are not an optimized index builder. A query evaluates only nodes exposed by the
    best-first traversal.  ``bounds_evaluated`` therefore counts tree nodes,
    unlike :class:`CertifiedBlockAttention`, where it is always the number of
    visible full blocks.

    The public ``read`` contract, including causal prefixes, seed blocks and
    hard block caps, matches ``CertifiedBlockAttention.read``.
    """

    def __init__(self, K, V, block_size=64, mass_bound="radius", peel_count=0):
        super().__init__(K, V, block_size)
        modes = ("radius", "bennett", "bennett_covariance",
                 "bennett_peel", "bennett_peel_covariance")
        if mass_bound not in modes:
            raise ValueError(
                "mass_bound must be radius, bennett, bennett_covariance, "
                "bennett_peel, or bennett_peel_covariance")
        self.peel_count = _integer(peel_count, "peel_count")
        if self.peel_count < 0:
            raise ValueError("peel_count must be nonnegative")
        if mass_bound.startswith("bennett_peel") and self.peel_count == 0:
            raise ValueError("peeled bounds require a positive peel_count")
        self.mass_bound = mass_bound
        self._nodes: list[_Node] = []
        nleaf = len(self.K) // self.block_size
        self._root = self._build_tree(0, nleaf) if nleaf else None
        for node in self._nodes:
            if not (np.isfinite(node.key_mean).all() and np.isfinite(node.key_radius)
                    and np.isfinite(node.key_spread) and node.key_spread >= 0
                    and (node.key_covariance is None
                         or np.isfinite(node.key_covariance).all())
                    and np.isfinite(node.value_mean).all() and np.isfinite(node.value_radius)):
                raise ValueError("key/value tree summary arithmetic exceeds float64 range")
            node.key_mean.setflags(write=False)
            if node.key_covariance is not None:
                node.key_covariance.setflags(write=False)
            if node.peel is not None and not (
                    np.isfinite(node.peel.core_mean).all()
                    and np.isfinite(node.peel.core_radius)
                    and np.isfinite(node.peel.core_spread)
                    and (node.peel.core_covariance is None
                         or np.isfinite(node.peel.core_covariance).all())):
                raise ValueError("peeled key summary arithmetic exceeds float64 range")
            node.value_mean.setflags(write=False)

    @staticmethod
    def _join_ball(left_mean, left_radius, left_count,
                   right_mean, right_radius, right_count):
        count = left_count + right_count
        mean = (left_count / count) * left_mean + (right_count / count) * right_mean
        radius = max(float(np.linalg.norm(left_mean - mean)) + left_radius,
                     float(np.linalg.norm(right_mean - mean)) + right_radius)
        exact_zero = (left_radius == 0 and right_radius == 0
                      and np.array_equal(left_mean, right_mean))
        radius = _outward_radius(radius, mean.shape[0], exact_zero=exact_zero)
        return mean, radius

    def _build_tree(self, start, end):
        if end - start == 1:
            spread, covariance = 0.0, None
            if self.mass_bound in ("bennett", "bennett_covariance"):
                keys = self.K[start * self.block_size:(start + 1) * self.block_size]
                residual = keys - self.key_means[start]
                if self.mass_bound == "bennett":
                    spread = float(np.einsum("ij,ij->", residual, residual) / len(keys))
                    if spread:
                        spread = float(np.nextafter(
                            spread * (1 + 32 * (self.K.shape[1] + 4)
                                      * np.finfo(float).eps), np.inf))
                else:
                    covariance = residual.T @ residual / len(keys)
                    covariance = self._guard_covariance(covariance)
            node = _Node(start, end, self.block_size,
                         self.key_means[start], float(self.key_radii[start]), spread, covariance,
                         self.value_means[start], float(self.value_radii[start]), None, None)
        else:
            mid = (start + end) // 2
            left_id = self._build_tree(start, mid)
            right_id = self._build_tree(mid, end)
            left, right = self._nodes[left_id], self._nodes[right_id]
            key_mean, key_radius = self._join_ball(
                left.key_mean, left.key_radius, left.count,
                right.key_mean, right.key_radius, right.count)
            value_mean, value_radius = self._join_ball(
                left.value_mean, left.value_radius, left.count,
                right.value_mean, right.value_radius, right.count)
            spread, covariance = 0.0, None
            if self.mass_bound == "bennett":
                left_shift = float(np.dot(left.key_mean - key_mean, left.key_mean - key_mean))
                right_shift = float(np.dot(right.key_mean - key_mean, right.key_mean - key_mean))
                spread = ((left.count * (left.key_spread + left_shift)
                           + right.count * (right.key_spread + right_shift))
                          / (left.count + right.count))
                if spread:
                    spread = float(np.nextafter(
                        spread * (1 + 32 * (self.K.shape[1] + 8) * np.finfo(float).eps), np.inf))
            elif self.mass_bound == "bennett_covariance":
                left_delta = left.key_mean - key_mean
                right_delta = right.key_mean - key_mean
                covariance = (
                    left.count * (left.key_covariance + np.outer(left_delta, left_delta))
                    + right.count * (right.key_covariance + np.outer(right_delta, right_delta))
                ) / (left.count + right.count)
                covariance = self._guard_covariance(covariance)
            node = _Node(start, end, left.count + right.count,
                         key_mean, key_radius, spread, covariance, value_mean, value_radius,
                         (left_id, right_id), None)
        if self.mass_bound.startswith("bennett_peel"):
            peel = self._make_peel_summary(node.start, node.end, node.key_mean)
            # Frozen nodes make the experimental summaries immutable snapshots.
            node = _Node(node.start, node.end, node.count, node.key_mean, node.key_radius,
                         node.key_spread, node.key_covariance, node.value_mean, node.value_radius,
                         node.children, peel)
        node_id = len(self._nodes)
        self._nodes.append(node)
        return node_id

    def _make_peel_summary(self, start, end, peel_center):
        lo, hi = start * self.block_size, end * self.block_size
        keys = self.K[lo:hi]
        residual = keys - peel_center
        score = np.einsum("ij,ij->i", residual, residual)
        local_index = np.arange(len(keys))
        order = np.lexsort((local_index, -score))
        t = min(self.peel_count, len(keys) - 1)
        exposed = (lo + order[:t]).astype(int)
        core = keys[order[t:]]
        core_mean = core.mean(axis=0)
        core_residual = core - core_mean
        raw_radius = float(np.linalg.norm(core_residual, axis=1).max())
        core_radius = _outward_radius(
            raw_radius, self.K.shape[1], exact_zero=bool(np.all(core == core_mean)))
        core_spread, core_covariance = 0.0, None
        if self.mass_bound == "bennett_peel":
            core_spread = float(np.einsum("ij,ij->", core_residual, core_residual) / len(core))
            if core_spread:
                core_spread = float(np.nextafter(
                    core_spread * (1 + 32 * (self.K.shape[1] + 4)
                                   * np.finfo(float).eps), np.inf))
        else:
            core_covariance = self._guard_covariance(core_residual.T @ core_residual / len(core))
        exposed.setflags(write=False)
        core_mean.setflags(write=False)
        if core_covariance is not None:
            core_covariance.setflags(write=False)
        return _PeelSummary(exposed, len(core), core_mean, core_radius,
                            core_spread, core_covariance)

    def _guard_covariance(self, covariance):
        """Symmetrize and add a conservative empirical roundoff cushion to the diagonal."""
        covariance = 0.5 * (covariance + covariance.T)
        scale = max(float(np.abs(covariance).sum(axis=1).max()),
                    float(np.finfo(float).tiny))
        cushion = 32 * (self.K.shape[1] + 8) * np.finfo(float).eps * scale
        covariance = covariance.copy()
        covariance.flat[::self.K.shape[1] + 1] += cushion
        return covariance

    @staticmethod
    def _log_expm1_minus_x(x):
        """Accurate ``log(exp(x) - 1 - x)`` for nonnegative finite float64 ``x``."""
        if x == 0:
            return -np.inf
        if x < 1e-3:
            # exp(x)-1-x = x^2/2 * (1+x/3+x^2/12+x^3/60+x^4/360+...).
            polynomial = 1 + x * (1 / 3 + x * (1 / 12 + x * (1 / 60 + x / 360)))
            return float(2 * np.log(x) - np.log(2.0) + np.log(polynomial))
        if x < 50:
            return float(np.log(np.expm1(x) - x))
        return float(x + np.log1p(-(1 + x) * np.exp(-x)))

    def _node_log_upper(self, node, q, beta, qnorm):
        """Guarded radius cap, or its minimum with a trace/covariance Bennett cap."""
        dot_abs = float(np.abs(node.key_mean) @ np.abs(q))
        center = beta * float(node.key_mean @ q)
        rho = qnorm * node.key_radius
        width = beta * rho
        guard = 32 * np.finfo(float).eps * (self.K.shape[1] + 1) * (
            beta * dot_abs + width + np.log(node.count) + 1)
        radius_upper = float(np.log(node.count) + center + width + guard)
        if self.mass_bound == "radius":
            return radius_upper
        if self.mass_bound.startswith("bennett_peel"):
            peel = node.peel
            core_dot_abs = float(np.abs(peel.core_mean) @ np.abs(q))
            core_center = beta * float(peel.core_mean @ q)
            core_rho = qnorm * peel.core_radius
            core_width = beta * core_rho
            core_guard = 32 * np.finfo(float).eps * (self.K.shape[1] + 1) * (
                beta * core_dot_abs + core_width + np.log(peel.core_count) + 1)
            core_upper = float(
                np.log(peel.core_count) + core_center + core_width + core_guard)
            if core_width == 0:
                core_upper = float(np.log(peel.core_count) + core_center + core_guard)
            elif self.mass_bound == "bennett_peel" and peel.core_spread > 0:
                log_ratio = float(
                    np.log(peel.core_spread) - 2 * np.log(peel.core_radius))
                log_term = log_ratio + self._log_expm1_minus_x(core_width)
                correction = float(np.logaddexp(0.0, log_term))
                core_upper = min(core_upper, float(
                    np.log(peel.core_count) + core_center + correction + core_guard))
            elif (self.mass_bound == "bennett_peel_covariance"
                  and (variance := float(q @ peel.core_covariance @ q)) > 0):
                log_ratio = float(np.log(variance) - 2 * np.log(core_rho))
                log_term = log_ratio + self._log_expm1_minus_x(core_width)
                correction = float(np.logaddexp(0.0, log_term))
                core_upper = min(core_upper, float(
                    np.log(peel.core_count) + core_center + correction + core_guard))

            exposed = self.K[peel.exposed_indices]
            logits = beta * (exposed @ q)
            top = float(logits.max())
            exposed_upper = top + float(np.log(np.exp(logits - top).sum()))
            exposed_guard = 32 * np.finfo(float).eps * (self.K.shape[1] + 1) * (
                beta * float((np.abs(exposed) @ np.abs(q)).max())
                + abs(exposed_upper) + np.log(len(exposed)) + 1)
            peeled_upper = float(np.logaddexp(exposed_upper + exposed_guard, core_upper))
            return min(radius_upper, peeled_upper)
        if width == 0:
            return float(np.log(node.count) + center + guard)
        if self.mass_bound == "bennett":
            # A computed zero with positive radius can be underflow rather than
            # a mathematical zero. The radius fallback remains admissible.
            if node.key_spread == 0:
                return radius_upper
            # The query norm cancels from (||q||^2 spread)/(||q|| radius)^2.
            # Computing the ratio this way avoids an unnecessary square overflow.
            log_ratio = float(np.log(node.key_spread) - 2 * np.log(node.key_radius))
        else:
            variance = float(q @ node.key_covariance @ q)
            if variance <= 0:
                return radius_upper
            log_ratio = float(np.log(variance) - 2 * np.log(rho))
        log_term = log_ratio + self._log_expm1_minus_x(width)
        log_correction = float(np.logaddexp(0.0, log_term))
        bennett_upper = float(np.log(node.count) + center + log_correction + guard)
        return min(radius_upper, bennett_upper)

    def read(self, q, beta=1.0, *, mass_tol=None, error_tol=None,
             max_blocks=None, initial_blocks=(), prefix=None):
        """Open best-first tree leaves until all tolerances hold or the cap is met."""
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

        qnorm = float(np.linalg.norm(q))
        frontier: dict[int, _Bound] = {}
        heap: list[tuple[float, int, int]] = []
        bounds_evaluated = 0

        def evaluate(node_id):
            nonlocal bounds_evaluated
            node = self._nodes[node_id]
            center = beta * float(node.key_mean @ q)
            width = beta * qnorm * node.key_radius
            guard = 32 * np.finfo(float).eps * (self.K.shape[1] + 1) * (
                beta * float(np.abs(node.key_mean) @ np.abs(q)) + width + 1)
            bound = _Bound(self._node_log_upper(node, q, beta, qnorm),
                           float(np.log(node.count) + center - guard))
            if not (np.isfinite(bound.log_upper) and np.isfinite(bound.log_lower)):
                raise ValueError("query/summary arithmetic exceeds float64 range")
            bounds_evaluated += 1
            frontier[node_id] = bound
            heapq.heappush(heap, (-bound.log_upper, node.start, node_id))

        def add_visible_cover(node_id):
            node = self._nodes[node_id]
            if node.start >= full:
                return
            if node.end <= full:
                evaluate(node_id)
                return
            for child_id in node.children:
                add_visible_cover(child_id)

        if full:
            add_visible_cover(self._root)

        log_z = -np.inf
        output = np.zeros(self.V.shape[1])
        selected = []
        opened_blocks = set()

        def open_block(c):
            nonlocal log_z, output
            if c in opened_blocks:
                return
            ids = np.arange(c * b, min((c + 1) * b, end))
            logits = beta * (self.K[ids] @ q)
            if not np.isfinite(logits).all():
                raise ValueError("query/key arithmetic exceeds float64 range")
            top = float(logits.max())
            weights = np.exp(logits - top)
            block_z = top + float(np.log(weights.sum()))
            block_output = (weights / weights.sum()) @ self.V[ids]
            joined = float(np.logaddexp(log_z, block_z))
            output = np.exp(log_z - joined) * output + np.exp(block_z - joined) * block_output
            log_z = joined
            selected.append(ids)
            opened_blocks.add(c)

        def expose_seed(c):
            containing = next((node_id for node_id in frontier
                               if self._nodes[node_id].start <= c < self._nodes[node_id].end), None)
            if containing is None:
                return
            del frontier[containing]
            node_id = containing
            while self._nodes[node_id].children is not None:
                left_id, right_id = self._nodes[node_id].children
                if self._nodes[left_id].start <= c < self._nodes[left_id].end:
                    evaluate(right_id)
                    node_id = left_id
                else:
                    evaluate(left_id)
                    node_id = right_id
            open_block(c)

        for c in sorted(seeds):
            if c < full:
                expose_seed(c)
            else:
                open_block(c)       # partially visible boundary block: never use its full summary

        def pop_best_leaf():
            while heap:
                _, _, node_id = heapq.heappop(heap)
                if node_id not in frontier:
                    continue
                del frontier[node_id]
                node = self._nodes[node_id]
                if node.children is None:
                    open_block(node.start)
                    return True
                for child_id in node.children:
                    evaluate(child_id)
            return False

        if not selected:
            pop_best_leaf()

        checks = value_evals = 0
        while True:
            checks += 1
            if not frontier:
                mass = kl = error = 0.0
            else:
                node_ids = list(frontier)
                log_u = float(np.logaddexp.reduce(
                    np.asarray([frontier[node_id].log_upper for node_id in node_ids])))
                log_l = float(np.logaddexp.reduce(
                    np.asarray([frontier[node_id].log_lower for node_id in node_ids])))
                log_mass = -float(np.logaddexp(0.0, log_z - log_u))
                mass = min(1.0, _positive_exp(log_mass))
                kl = max(np.nextafter(0.0, 1.0),
                         float(np.logaddexp(0.0, log_u - log_z)))
                distances = np.asarray([
                    np.linalg.norm(self._nodes[node_id].value_mean - output)
                    + self._nodes[node_id].value_radius for node_id in node_ids])
                value_evals += len(node_ids)
                positive = distances > 0
                if not positive.any():
                    error = 0.0
                else:
                    upper = np.asarray([frontier[node_id].log_upper for node_id in node_ids])
                    log_num = float(np.logaddexp.reduce(upper[positive] + np.log(distances[positive])))
                    log_error = min(float(np.log(distances.max()) + log_mass),
                                    log_num - float(np.logaddexp(log_z, log_l)))
                    error = _positive_exp(log_error)

            certified = ((mass_tol is None or mass <= mass_tol)
                         and (error_tol is None or error <= error_tol))
            if certified or len(opened_blocks) >= cap:
                ids = np.concatenate(selected)
                return CertifiedRead(output, ids, certified, mass, kl, error, len(ids),
                                     len(opened_blocks), bounds_evaluated, checks, value_evals)

            target = min(cap, max(len(opened_blocks) + 1, 2 * len(opened_blocks)))
            while len(opened_blocks) < target:
                if not pop_best_leaf():
                    break
