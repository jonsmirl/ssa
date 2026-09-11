"""Sharp fixed-cell-summary recovery diagnostic, in CPU float64 arithmetic.

For fixed numerical target weights p, cell-indicator sums Av, and exact reads S,
the optimal arbitrary-decoder worst-case scalar error over v in [-1, 1]^n is
sum_{i not in S} |p_i - median(p in i's unread cell)|. This specializes the
fixed-channel theorem; the self-contained public argument is in
``docs/summary_recovery_experiments.md``.
Balanced integer witnesses have exactly zero cell sums and selected coordinates.
Their pairing with p and the decoder arithmetic are floating-point diagnostics,
not an IEEE/interval certificate for real-model softmax. Witness values need not
be realizable by a model. The median decoder's weights need not be normalized.
The result is for the supplied S, not an optimization over all read policies.
"""

from __future__ import annotations

import numpy as np


def _ids(value, name, size=None):
    result = np.asarray(value)
    if result.ndim != 1 or (result.size and result.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a vector of integer indices")
    if result.size and (np.any(result < 0) or (size is not None and np.any(result >= size))):
        raise ValueError(f"{name} indices are out of range")
    if result.size and np.any(result > np.iinfo(np.int64).max):
        raise ValueError(f"{name} indices exceed int64")
    return result.astype(np.int64, copy=True)


def _partition(cell_ids, cells):
    ids = _ids(cell_ids, "cell_ids")
    required = int(ids.max()) + 1 if ids.size else 0
    if cells is None:
        cells = required
    if (not isinstance(cells, (int, np.integer)) or isinstance(cells, (bool, np.bool_))
            or cells < required or cells < 0):
        raise ValueError("cells must be a nonnegative integer covering every cell id")
    return ids, int(cells)


def cell_summary_minimax(weights, cell_ids, selected=(), *, cells=None):
    """Return optimal median coefficients and an attained balanced dual witness.

    ``selected`` is a unique index vector; its supplied order is preserved for
    the decoder. Empty/unread-empty cells have coefficient zero. In even cells
    the midpoint of the two central weights is used. For witness construction,
    ascending (weight, original index) order resolves ties deterministically.
    ``mean_radius`` measures the alternative *unread* mean-weight decoder, not
    normalized sparse attention. All array outputs own their data.

    For scalar |v_i| <= B, worst-case error is B * radius. For vector values
    ||v_i||_2 <= B the same expression bounds output L2 error; for a coordinate
    cube it bounds each coordinate, hence L2 by sqrt(dv) * B * radius.
    """
    p = np.asarray(weights, dtype=np.float64)
    ids, cells = _partition(cell_ids, cells)
    if p.ndim != 1 or p.shape != ids.shape or not np.isfinite(p).all():
        raise ValueError("weights must be a finite vector aligned with cell_ids")
    selected = _ids(selected, "selected", len(p))
    if np.unique(selected).size != selected.size:
        raise ValueError("selected indices must be unique")
    unread = np.ones(len(p), dtype=bool)
    unread[selected] = False
    coefficients = np.zeros(cells)
    means = np.zeros(cells)
    h = np.zeros(len(p), dtype=np.int8)
    counts = np.bincount(ids[unread], minlength=cells)
    for cell in range(cells):
        members = np.flatnonzero(unread & (ids == cell))
        if not members.size:
            continue
        ordered = members[np.lexsort((members, p[members]))]
        half = len(ordered) // 2
        if len(ordered) % 2:
            coefficients[cell] = p[ordered[half]]
        else:
            lower, upper = p[ordered[half - 1]], p[ordered[half]]
            # Halves avoid overflow; clipping keeps subnormal rounding inside
            # the median interval (including two identical smallest floats).
            coefficients[cell] = np.clip(lower / 2 + upper / 2, lower, upper)
        means[cell] = np.sum(p[members] / len(members))
        if half:
            h[ordered[:half]] = -1
            h[ordered[-half:]] = 1
    residual = p - coefficients[ids]
    residual[selected] = 0
    mean_residual = p - means[ids]
    mean_residual[selected] = 0
    radius = float(np.abs(residual).sum())
    mean_radius = float(np.abs(mean_residual).sum())
    pairing = float(p @ h)
    if not np.isfinite([radius, mean_radius, pairing]).all():
        raise FloatingPointError("coefficient residual or pairing overflowed")
    kernel_sums = np.zeros(cells, dtype=np.int64)
    np.add.at(kernel_sums, ids, h.astype(np.int64))
    return {
        "weights": p.copy(), "cell_ids": ids, "selected": selected,
        "coefficients": coefficients, "mean_coefficients": means,
        "residual": residual, "mean_residual": mean_residual,
        "radius": radius, "mean_radius": mean_radius,
        "dual_witness": h, "dual_pairing": pairing,
        "primal_dual_deficit": radius - pairing,
        "kernel_sums": kernel_sums, "kernel_exact": bool(np.all(kernel_sums == 0)),
        "selected_witness_exact": bool(np.all(h[selected] == 0)),
        "unread_counts": counts,
    }


def summarize_cells(values, cell_ids, *, cells=None):
    """Sum scalar or vector values in each disjoint cell, including selected keys."""
    ids, cells = _partition(cell_ids, cells)
    v = np.asarray(values, dtype=np.float64)
    if v.ndim not in (1, 2) or len(v) != len(ids) or not np.isfinite(v).all():
        raise ValueError("values must be finite scalar/vector rows aligned with cell_ids")
    sums = np.zeros((cells,) + v.shape[1:])
    np.add.at(sums, ids, v)
    if not np.isfinite(sums).all():
        raise FloatingPointError("cell sums overflowed")
    return sums


def decode_cell_summary(profile, cell_sums, selected_values, *, coefficients=None):
    """Decode from *only* supplied full cell sums and aligned selected values.

    The selected correction subtracts their summary contribution before adding
    their exact target contribution. Passing ``profile['mean_coefficients']``
    selects the mean-weight control. No unselected values are accessed here.
    """
    p = np.asarray(profile["weights"], dtype=np.float64)
    ids = np.asarray(profile["cell_ids"])
    selected = np.asarray(profile["selected"])
    c = np.asarray(profile["coefficients"] if coefficients is None else coefficients,
                   dtype=np.float64)
    sums = np.asarray(cell_sums, dtype=np.float64)
    exact = np.asarray(selected_values, dtype=np.float64)
    if c.ndim != 1 or c.shape != np.shape(profile["coefficients"]) or not np.isfinite(c).all():
        raise ValueError("coefficients must be a finite vector with one entry per cell")
    if sums.ndim not in (1, 2) or sums.shape[0] != len(c) or not np.isfinite(sums).all():
        raise ValueError("cell_sums must have one finite scalar/vector row per cell")
    if exact.shape != (len(selected),) + sums.shape[1:] or not np.isfinite(exact).all():
        raise ValueError("selected_values must match selected order and value dimension")
    result = c @ sums + (p[selected] - c[ids[selected]]) @ exact
    if not np.isfinite(result).all():
        raise FloatingPointError("decoder overflowed")
    return result
