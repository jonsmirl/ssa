"""Bounded-state, multi-round repair reads for sparse attention.

The reference implementation makes two often-confused operations explicit:

* ``fixed_query_repair`` widens the selected set while always scoring keys with
  the original attention query.  Its streaming state is an exact softmax
  sufficient statistic, so this really recovers mass from one fixed dense read.
* A router may use a different query at every round.  That query only chooses
  which keys to inspect; it does not silently change the attention distribution.

The raw KV archive remains available.  A fixed-size state is therefore a read
controller and accumulator, not a compressed replacement for unseen values.
For fixed round count and per-round budget, the retained-id list is also bounded
independently of context length.  This NumPy module is a CPU reference, not a
kernel and not an IEEE interval proof.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import operator

import numpy as np


def _as_index(value, name):
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass
class StreamingSoftmaxState:
    """Exact (in real arithmetic) softmax state for a disjoint selected union.

    ``weighted_value_sum`` and ``partition`` share the scale ``exp(-maximum)``.
    The persistent numerical state has ``value_dimension + 2`` scalars.  The
    selected ids used to prevent double counting are accounting/router state.
    """

    maximum: float
    partition: float
    weighted_value_sum: np.ndarray
    keys_scored: int = 0

    @classmethod
    def empty(cls, value_dimension):
        value_dimension = _as_index(value_dimension, "value_dimension")
        if value_dimension < 1:
            raise ValueError("value_dimension must be positive")
        return cls(-math.inf, 0.0, np.zeros(value_dimension, dtype=np.float64), 0)

    def add(self, logits, values):
        """Merge one nonempty, disjoint batch without retaining its logits."""
        logits = np.asarray(logits, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if (logits.ndim != 1 or values.ndim != 2 or len(logits) != len(values)
                or values.shape[1:] != self.weighted_value_sum.shape
                or len(logits) == 0):
            raise ValueError("logits and values must be a nonempty aligned batch")
        if not (np.isfinite(logits).all() and np.isfinite(values).all()):
            raise ValueError("logits and values must be finite")
        batch_max = float(logits.max())
        batch_weights = np.exp(logits - batch_max)
        joined_max = max(self.maximum, batch_max)
        old_scale = 0.0 if self.partition == 0 else math.exp(self.maximum - joined_max)
        batch_scale = math.exp(batch_max - joined_max)
        self.weighted_value_sum *= old_scale
        self.weighted_value_sum += batch_scale * (batch_weights @ values)
        self.partition = old_scale * self.partition + batch_scale * float(batch_weights.sum())
        self.maximum = joined_max
        self.keys_scored += len(logits)

    @property
    def output(self):
        if self.partition <= 0:
            raise ValueError("the empty state has no attention output")
        return self.weighted_value_sum / self.partition

    @property
    def log_partition(self):
        if self.partition <= 0:
            return -math.inf
        return self.maximum + math.log(self.partition)


@dataclass(frozen=True)
class RepairRound:
    round_index: int
    indices: np.ndarray
    cumulative_keys: int
    actual_retained_mass: float
    actual_output_error: float


@dataclass(frozen=True)
class RepairRead:
    output: np.ndarray
    indices: np.ndarray
    rounds: tuple[RepairRound, ...]
    attention_query_fixed: bool
    accumulator_scalars: int
    retained_id_capacity: int


def topk_route(routing_query, routing_keys, budget, *, prefix=None, excluded=()):
    """Deterministic exact top-k reference router, with larger ids winning ties."""
    routing_keys = np.asarray(routing_keys, dtype=np.float64)
    routing_query = np.asarray(routing_query, dtype=np.float64)
    if (routing_keys.ndim != 2 or routing_query.shape != (routing_keys.shape[1],)
            or not np.isfinite(routing_keys).all() or not np.isfinite(routing_query).all()):
        raise ValueError("routing query/keys must be finite and dimensionally aligned")
    end = len(routing_keys) if prefix is None else _as_index(prefix, "prefix")
    if not 1 <= end <= len(routing_keys):
        raise ValueError("prefix must select a nonempty causal prefix")
    budget = _as_index(budget, "budget")
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    blocked = np.zeros(end, dtype=bool)
    excluded = np.asarray(tuple(excluded), dtype=np.int64)
    if len(excluded):
        if np.any(excluded < 0) or np.any(excluded >= end):
            raise ValueError("excluded ids must lie in the causal prefix")
        blocked[excluded] = True
    ids = np.arange(end)
    scores = routing_keys[:end] @ routing_query
    order = np.lexsort((-ids, -scores))
    order = order[~blocked[order]]
    return order[:min(budget, len(order))]


def fixed_query_repair(attention_query, keys, values, routed_batches, *, beta=1.0,
                       prefix=None):
    """Accumulate a sequence of sparse reads into exact attention on their union.

    Every key is scored with ``attention_query`` even if a controller used other
    routing queries to produce ``routed_batches``.  Duplicate ids are ignored:
    rereading a key cannot manufacture extra softmax mass.  Dense quantities are
    computed only for reference diagnostics in this CPU implementation.
    """
    keys = np.asarray(keys, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    q = np.asarray(attention_query, dtype=np.float64)
    if (keys.ndim != 2 or values.ndim != 2 or len(keys) != len(values)
            or q.shape != (keys.shape[1],) or len(keys) == 0
            or not np.isfinite(beta)):
        raise ValueError("attention query, keys, values, and beta must be aligned and finite")
    if not (np.isfinite(keys).all() and np.isfinite(values).all() and np.isfinite(q).all()):
        raise ValueError("attention query, keys, and values must be finite")
    end = len(keys) if prefix is None else _as_index(prefix, "prefix")
    if not 1 <= end <= len(keys):
        raise ValueError("prefix must select a nonempty causal prefix")

    dense_logits = float(beta) * (keys[:end] @ q)
    dense_max = float(dense_logits.max())
    dense_weights = np.exp(dense_logits - dense_max)
    dense_weights /= dense_weights.sum()
    dense_output = dense_weights @ values[:end]

    state = StreamingSoftmaxState.empty(values.shape[1])
    selected = np.zeros(end, dtype=bool)
    rounds = []
    capacity = 0
    for round_index, raw_ids in enumerate(routed_batches):
        raw_ids = np.asarray(raw_ids)
        if raw_ids.ndim != 1 or not np.issubdtype(raw_ids.dtype, np.integer):
            raise ValueError("every routed batch must be a one-dimensional integer array")
        ids = raw_ids.astype(np.int64, copy=False)
        if np.any(ids < 0) or np.any(ids >= end):
            raise ValueError("routed ids must lie in the causal prefix")
        capacity += len(ids)
        # Stable first occurrence, then global exclusion across prior rounds.
        _, first = np.unique(ids, return_index=True)
        ids = ids[np.sort(first)]
        ids = ids[~selected[ids]]
        if len(ids):
            state.add(float(beta) * (keys[ids] @ q), values[ids])
            selected[ids] = True
        chosen = np.flatnonzero(selected)
        retained = float(dense_weights[selected].sum())
        error = (float(np.linalg.norm(dense_output - state.output))
                 if state.partition > 0 else math.inf)
        frozen_ids = ids.copy(); frozen_ids.setflags(write=False)
        rounds.append(RepairRound(round_index, frozen_ids, len(chosen), retained, error))
    if state.partition <= 0:
        raise ValueError("at least one routed batch must contain a visible key")
    chosen = np.flatnonzero(selected); chosen.setflags(write=False)
    output = state.output.copy(); output.setflags(write=False)
    return RepairRead(output, chosen, tuple(rounds), True, values.shape[1] + 2, capacity)


def route_repair_rounds(attention_query, keys, values, routing_keys, initial_routing_query,
                        controller, *, rounds, budget, beta=1.0, prefix=None,
                        initial_state=None):
    """Run a bounded controller that proposes a routing query after each read.

    ``controller(state, sparse_attention_output, round_index)`` returns
    ``(new_state, next_routing_query)``.  The controller observes only opened
    attention output, so any improvement requires useful information in that
    output or prior state.  The final attention read remains the fixed-query
    union above.
    """
    rounds = _as_index(rounds, "rounds")
    budget = _as_index(budget, "budget")
    if rounds < 1 or budget < 1:
        raise ValueError("rounds and budget must be positive")
    routing_keys = np.asarray(routing_keys, dtype=np.float64)
    rq = np.asarray(initial_routing_query, dtype=np.float64)
    end = len(routing_keys) if prefix is None else _as_index(prefix, "prefix")
    keys_array = np.asarray(keys, dtype=np.float64)
    state = initial_state
    excluded, batches = [], []
    values_array = np.asarray(values, dtype=np.float64)
    q_attention = np.asarray(attention_query, dtype=np.float64)
    if (keys_array.ndim != 2 or values_array.ndim != 2 or len(keys_array) != len(values_array)
            or q_attention.shape != (keys_array.shape[1],)):
        raise ValueError("attention query, keys, and values must be aligned")
    attention_state = StreamingSoftmaxState.empty(values_array.shape[1])
    for r in range(rounds):
        ids = topk_route(rq, routing_keys, budget, prefix=end, excluded=excluded)
        if not len(ids):
            break
        batches.append(ids)
        excluded.extend(map(int, ids))
        attention_state.add(float(beta) * (keys_array[ids] @ q_attention), values_array[ids])
        state, rq = controller(state, attention_state.output.copy(), r)
        rq = np.asarray(rq, dtype=np.float64)
        if rq.shape != (routing_keys.shape[1],) or not np.isfinite(rq).all():
            raise ValueError("controller returned an invalid routing query")
    return fixed_query_repair(attention_query, keys, values, batches, beta=beta, prefix=end)
