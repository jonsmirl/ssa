"""Float64 reference for a persistent, zero-centred all-prefix ridge fit.

The fixed penalty applies to the *sum* of squared errors, not their mean:
``||Y - X W||_F**2 + penalty * ||W||_F**2``. Appending rows retains
``X.T @ X`` and ``X.T @ Y``; solving always recomputes the penalized minimizer.
This is not a sequence of penalized residual corrections. It does not retain
the design or values, provide an accessible attention-weight summary, or
certify floating-point errors. Statistics and factorization work are charged.
"""

from __future__ import annotations

import numpy as np


def _dimension(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _matrix(value, name, columns):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != columns or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite matrix with {columns} columns")
    return value


class PersistentRidge:
    """Append-only sufficient statistics and a cached ridge decoder.

    Inputs and exposed arrays are never retained by alias. Failed appends do
    not modify the state. ``append`` invalidates the decoder; ``solve``,
    ``weights`` and ``predict`` lazily refresh it. The number of feature/value
    coordinates and the strictly positive penalty cannot be changed through
    the public API. Batch partition invariance holds in real arithmetic;
    floating-point reductions may differ by roundoff.
    """

    def __init__(self, features, values, penalty):
        self._features = _dimension(features, "features")
        self._values = _dimension(values, "values")
        if isinstance(penalty, (bool, np.bool_)):
            raise ValueError("penalty must be finite and strictly positive")
        self._penalty = float(penalty)
        if not np.isfinite(self._penalty) or self._penalty <= 0:
            raise ValueError("penalty must be finite and strictly positive")
        self._gram = np.zeros((self.features, self.features), dtype=np.float64)
        self._cross = np.zeros((self.features, self.values), dtype=np.float64)
        self._weights = np.zeros_like(self._cross)
        self._rows = 0
        self._dirty = False

    @property
    def features(self):
        return self._features

    @property
    def values(self):
        return self._values

    @property
    def penalty(self):
        return self._penalty

    @property
    def rows(self):
        return self._rows

    @property
    def gram(self):
        return self._gram.copy()

    @property
    def cross(self):
        return self._cross.copy()

    @property
    def weights(self):
        return self.solve()

    def copy(self):
        result = PersistentRidge(self.features, self.values, self.penalty)
        result._gram = self._gram.copy()
        result._cross = self._cross.copy()
        result._weights = self._weights.copy()
        result._rows, result._dirty = self._rows, self._dirty
        return result

    def append(self, X, Y):
        """Admit exactly these rows, without solving or rereading old rows."""
        X, Y = _matrix(X, "X", self.features), _matrix(Y, "Y", self.values)
        if len(X) != len(Y):
            raise ValueError("X and Y must have equal row counts")
        m, r, d = len(X), self.features, self.values
        if m:
            with np.errstate(over="ignore", invalid="ignore"):
                gram = self._gram + X.T @ X
                cross = self._cross + X.T @ Y
            if not np.isfinite(gram).all() or not np.isfinite(cross).all():
                raise FloatingPointError("ridge sufficient statistics overflowed")
            self._gram, self._cross = gram, cross
            self._rows += m
            self._dirty = True
        return {
            "rows_read": m,
            "total_rows": self.rows,
            "arithmetic_ops_estimate": int(2 * m * r * (r + d) + (r * r + r * d if m else 0)),
            "supplied_batch_bytes": int(X.nbytes + Y.nbytes),
            "temporary_bytes_estimate": int(16 * (r * r + r * d) if m else 0),
            "note": "Dense multiply/add estimate. Atomic-update temporaries exclude BLAS workspace and caller conversion buffers; supplied inputs are separate.",
        }

    def _normal(self):
        normal = self._gram.copy()
        normal.flat[::self.features + 1] += self.penalty
        if not np.isfinite(normal).all():
            raise FloatingPointError("ridge normal matrix overflowed")
        return normal

    def solve(self):
        """Return a copy of the persistent minimizer (a dense solve reference)."""
        if self._dirty:
            candidate = np.linalg.solve(self._normal(), self._cross)
            if not np.isfinite(candidate).all():
                raise FloatingPointError("ridge solve produced nonfinite weights")
            self._weights = candidate
            self._dirty = False
        return self._weights.copy()

    def predict(self, X):
        X = _matrix(X, "X", self.features)
        with np.errstate(over="ignore", invalid="ignore"):
            result = X @ self.solve()
        if not np.isfinite(result).all():
            raise FloatingPointError("ridge prediction overflowed")
        return result

    def reader_gain(self, a):
        """Return ``||a (Gram+penalty I)^-1 X.T||_2`` from Gram alone.

        With ``b=(Gram+penalty I)^-1 a``, its square is ``b.T Gram b``.
        This is the sharp fixed-reader Frobenius mismatch gain in exact
        arithmetic, not a bound on unknown mismatch or target bias. Computing
        the reader ``a`` is the caller's separately charged responsibility.
        Tiny negative roundoff is clamped; substantial negativity is rejected.
        """
        a = np.asarray(a, dtype=np.float64)
        if a.shape != (self.features,) or not np.isfinite(a).all():
            raise ValueError("a must be a finite feature vector")
        b = np.linalg.solve(self._normal(), a)
        with np.errstate(over="ignore", invalid="ignore"):
            terms = b * (self._gram @ b)
            square = float(np.sum(terms))
            tolerance = 64 * np.finfo(np.float64).eps * float(np.sum(np.abs(terms)))
        if not np.isfinite(square) or not np.isfinite(tolerance) or square < -tolerance:
            raise FloatingPointError("invalid ridge reader gain quadratic form")
        return float(np.sqrt(max(0., square)))

    def state_accounting(self):
        """Explicit float64-equivalent state and dense work proxies.

        Feature maps/normalization, raw archives, batch inputs and the Python
        object allocator are outside this object and must be charged by its
        caller. Decoder storage is included even before the first solve.
        ``count`` and ``penalty`` each consume one scalar-equivalent; array
        shape and cache-validity metadata are included in excluded Python
        overhead. Work proxies are not exact operation counts or measured peaks.
        """
        r, d = self.features, self.values
        scalars = r * r + 2 * r * d + 2
        return {
            "gram_scalars": r * r,
            "cross_scalars": r * d,
            "decoder_scalars": r * d,
            "bookkeeping_scalars": 2,
            "persistent_scalars": scalars,
            "persistent_bytes": 8 * scalars,
            "solve_cubic_work_proxy": r ** 3,
            "solve_rhs_work_proxy": r * r * d,
            "solve_temporary_bytes_estimate": 8 * (2 * r * r + 2 * r * d),
            "reader_gain_cubic_work_proxy": r ** 3,
            "reader_gain_quadratic_work_proxy": 3 * r * r,
            "reader_gain_temporary_bytes_estimate": 8 * (2 * r * r + 4 * r),
            "cost_note": "Dense solve reference refactors for each reader gain. Temporary estimates exclude solver/BLAS internal workspace, caller inputs, copies of returned arrays, feature maps, archives and Python overhead. No subquadratic attention or interval-certificate claim.",
        }
