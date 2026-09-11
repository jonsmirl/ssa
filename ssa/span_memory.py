"""CPU references for linear shared memory and training-only cell summaries.

Rows are observations: X has shape (m, r), Y (m, dv), and W (r, dv).
These are floating-point experiments, not interval-certified implementations of
the Substrate results. Work/storage diagnostics are explicit algebraic estimates,
not measured runtime or allocator peaks. Joint fitting reads all supplied rows;
it is not a constant-cost streaming update.
"""

from __future__ import annotations

import numpy as np


def _matrix(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite matrix")
    return result


def _inputs(W, X, Y, rate):
    W, X, Y = (_matrix(v, n) for v, n in ((W, "W"), (X, "X"), (Y, "Y")))
    if X.shape[1] != W.shape[0] or Y.shape != (X.shape[0], W.shape[1]):
        raise ValueError("expected W(r,dv), X(m,r), Y(m,dv)")
    if W.shape[0] == 0 or W.shape[1] == 0:
        raise ValueError("feature and value dimensions must be positive")
    rate = float(rate)
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("rate must be finite and in [0,1]")
    return W, X, Y, rate


def _factor(X, rcond):
    rcond = float(rcond)
    if not np.isfinite(rcond) or not 0 <= rcond < 1:
        raise ValueError("rcond must be finite and in [0,1)")
    u, singular, vt = np.linalg.svd(X, full_matrices=False)
    keep = singular > rcond * singular[0] if singular.size else np.zeros(0, dtype=bool)
    inverse = np.zeros_like(singular)
    np.divide(1., singular, out=inverse, where=keep)
    pinv = (vt.T * inverse) @ u.T
    return pinv, singular, keep


def _diagnostics(W, updated, X, Y, pinv, singular, keep, method):
    m, r = X.shape
    dv = Y.shape[1]
    rank = int(keep.sum())
    before, after = Y - X @ W, Y - X @ updated
    floor = Y - X @ (pinv @ Y)
    # Dense multiply-add counts use two operations per scalar product term.
    fit_ops = (6 * m * r * dv + r * dv if method == "joint" else 4 * m * r * dv + 2 * m * r)
    # SVD counts are order proxies: algorithm, blocking, and shape matter.
    svd_proxy = int(m * r * min(m, r))
    return {
        "method": method,
        "residual_before_fro": float(np.linalg.norm(before)),
        "residual_after_fro": float(np.linalg.norm(after)),
        "residual_before_row_norm_sum": float(np.linalg.norm(before, axis=1).sum()),
        "residual_after_row_norm_sum": float(np.linalg.norm(after, axis=1).sum()),
        "residual_before_squared": float(np.sum(before * before)),
        "residual_after_squared": float(np.sum(after * after)),
        "joint_irreducible_residual_fro": float(np.linalg.norm(floor)),
        "numerical_rank": rank,
        "retained_condition": float(singular[keep][0] / singular[keep][-1]) if rank else None,
        "rows_read": m,
        "parameter_bytes": int(W.nbytes),
        "supplied_batch_bytes": int(X.nbytes + Y.nbytes),
        "fit_arithmetic_ops_estimate": int(fit_ops),
        "svd_work_proxy": svd_proxy,
        "svd_required_by_fit": method == "joint",
        "diagnostic_dense_ops_estimate": int(8 * m * r * dv),
        "temporary_bytes_estimate": int(8 * (2 * m * r + 3 * m * dv + r * dv + min(m, r) * (m + r + 2))),
        "cost_note": "Estimates exclude BLAS/SVD workspace and Python overhead. Parameter bytes are frozen decoder W only, not batch inputs, a feature basis, or online fit statistics. Sequential SVD is diagnostic only; joint SVD is also fit work. Diagnostics reread rows and are not free.",
    }


def joint_update(W, X, Y, rate=1., rcond=1e-10):
    """Joint least-squares correction; preserves directions orthogonal to rows.

    Small singular values are discarded at ``rcond * largest_singular``. The
    reported irreducible residual therefore concerns this numerical subspace.
    Under realizability, exact arithmetic and no truncation of required modes,
    all supplied residuals contract by ``1-rate``. No such statement is made
    about rows not supplied to the update.
    """
    W, X, Y, rate = _inputs(W, X, Y, rate)
    pinv, singular, keep = _factor(X, rcond)
    updated = W + rate * (pinv @ (Y - X @ W))
    if not np.isfinite(updated).all():
        raise FloatingPointError("joint update overflowed")
    return updated, _diagnostics(W, updated, X, Y, pinv, singular, keep, "joint")


def sequential_update(W, X, Y, rate=1.):
    """One ordered pass of normalized delta updates; zero rows do nothing."""
    W, X, Y, rate = _inputs(W, X, Y, rate)
    updated = W.copy()
    for x, y in zip(X, Y):
        scale = float(np.max(np.abs(x)))
        if scale:
            # Scaling avoids underflow of ||x||² for small nonzero features.
            unit = x / scale
            updated += rate * np.outer(unit, (y - x @ updated) / scale) / (unit @ unit)
    if not np.isfinite(updated).all():
        raise FloatingPointError("sequential update overflowed")
    pinv, singular, keep = _factor(X, 1e-10)
    return updated, _diagnostics(W, updated, X, Y, pinv, singular, keep, "sequential")


def _assign(K, centers):
    distances = np.sum(K * K, axis=1)[:, None] + np.sum(centers * centers, axis=1)[None, :] - 2 * K @ centers.T
    # argmin selects the smallest cell id at a tie.
    return np.argmin(distances, axis=1)


def fit_cells(K, Y, cells, seed=0, iterations=10):
    """Fit deterministic seeded k-means on supplied training keys only.

    Return centers, counts, value sums, assignments. Empty cells retain their
    previous center and have count/sum zero. No evaluation keys or targets enter
    this API. Lloyd iterations are fixed-budget, not a convergence guarantee.
    """
    K, Y = _matrix(K, "K"), _matrix(Y, "Y")
    if K.shape[0] != Y.shape[0] or not all(K.shape) or Y.shape[1] == 0:
        raise ValueError("nonempty aligned training keys and values required")
    if not isinstance(cells, (int, np.integer)) or isinstance(cells, bool) or not 1 <= cells <= len(K):
        raise ValueError("cells must be an integer between one and training count")
    if not isinstance(iterations, (int, np.integer)) or isinstance(iterations, bool) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    centers = K[np.random.default_rng(seed).choice(len(K), cells, replace=False)].copy()
    for _ in range(iterations):
        assignment = _assign(K, centers)
        for cell in range(cells):
            members = K[assignment == cell]
            if len(members):
                centers[cell] = members.mean(axis=0)
    assignment = _assign(K, centers)
    counts = np.bincount(assignment, minlength=cells)
    sums = np.zeros((cells, Y.shape[1]), dtype=np.float64)
    np.add.at(sums, assignment, Y)
    return centers, counts, sums, assignment


def predict_cells(K, centers, counts, sums):
    """Nearest-cell value mean; empty cells predict zero, without target access."""
    K, centers, sums = (_matrix(v, n) for v, n in ((K, "K"), (centers, "centers"), (sums, "sums")))
    counts = np.asarray(counts, dtype=np.float64)
    if not len(centers) or K.shape[1] != centers.shape[1] or sums.shape[0] != len(centers):
        raise ValueError("incompatible key and cell shapes")
    if counts.shape != (len(centers),) or not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("counts must be nonnegative finite integers per cell")
    means = np.zeros_like(sums)
    np.divide(sums, counts[:, None], out=means, where=counts[:, None] > 0)
    return means[_assign(K, centers)]


def linear_obstruction(K, V, *, prime=2147483647):
    """Exact modular witness ruling out a linear map on stored float values.

    A (d+1)-rank augmented matrix over this prime field proves a nonzero rational
    augmented minor, hence rules out K w = V[:, 0] over the reals. Float64 values
    are interpreted as their exact dyadic rationals; the odd prime never divides
    their denominators. Failing to find this witness is inconclusive: only the
    first d+1 rows and first value coordinate are inspected, and reduction modulo
    a prime can lose rank. This does not prove anything about unrounded model
    activations or nonlinear decoders.
    """
    K, V = _matrix(K, "K"), _matrix(V, "V")
    if K.shape[0] != V.shape[0] or K.shape[1] == 0 or V.shape[1] == 0:
        raise ValueError("expected aligned keys and values with positive dimensions")
    if isinstance(prime, (bool, np.bool_)) or not isinstance(prime, (int, np.integer)) or prime != 2147483647:
        raise ValueError("only the known prime 2147483647 is supported")
    prime = int(prime)
    rows = min(len(K), K.shape[1] + 1)
    augmented = np.concatenate((K[:rows], V[:rows, :1]), axis=1)
    field = []
    for row in augmented:
        field.append([
            (numerator % prime) * pow(denominator % prime, -1, prime) % prime
            for numerator, denominator in (float(value).as_integer_ratio() for value in row)
        ])
    rank = 0
    for column in range(K.shape[1] + 1):
        pivot = next((index for index in range(rank, rows) if field[index][column]), None)
        if pivot is None:
            continue
        field[rank], field[pivot] = field[pivot], field[rank]
        inverse = pow(field[rank][column], -1, prime)
        field[rank] = [value * inverse % prime for value in field[rank]]
        for index in range(rank + 1, rows):
            factor = field[index][column]
            if factor:
                field[index] = [(value - factor * basis) % prime for value, basis in zip(field[index], field[rank])]
        rank += 1
        if rank == rows:
            break
    return {
        "rows": rows,
        "value_column": 0,
        "prime": prime,
        "rank": rank,
        "obstruction": bool(rank > K.shape[1]),
        "scope": "Exact dyadic values of supplied float64 arrays, first min(n,d+1) rows and first value coordinate. Rank>d certifies no real linear key-to-value map on these observations. No obstruction is inconclusive, not evidence of realizability. No claim about unrounded model activations, nonlinear memory, or attention-output error.",
    }
