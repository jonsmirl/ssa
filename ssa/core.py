"""
Retrieval-margin demonstrator — core.

A controlled (empirical) test of the content-addressable sparse-attention
theory (paper §3):

  recovery weight  rho(beta, gap, mu) = 1 / (1 + mu * e^{-beta*gap}) = sigma(beta*gap - log mu)
  detectability    recovered (mass > 1/2)  <=>  beta*gap > log mu
  length-gen       margin needed grows only as log(context)
  truncation       ||sparse_read - dense_read|| <= 2 V_max * missed_mass  (exp-small in the gap)
  capacity         near-orthogonal keys recover up to  n < e^{beta*(1-eps)}
  composition      an h-hop chain succeeds ~ rho^h  (recall != reasoning)

and the open engineering crux — the *selector*: can a sublinear index assembled from
known ANN parts (SimHash-LSH; routing/centroid clustering) capture the top-k WITHOUT
scoring all n keys, losslessly, when the keys are separated (the regime training drives
keys toward)?  This module is NumPy-only and uses controlled synthetic geometry so that
gap, separation, dimension, and count are all set exactly.
"""
from __future__ import annotations
import numpy as np


# --------------------------------------------------------------------------------------
# The theory (the predictions, as plain formulas).
# --------------------------------------------------------------------------------------

def recovery_weight(beta: float, gap: float, mu: float) -> float:
    """rho = 1 / (1 + mu * exp(-beta*gap)). recovery weight (paper §3)."""
    return 1.0 / (1.0 + mu * np.exp(-beta * gap))


def threshold_n(beta: float, gap: float) -> float:
    """Detectability threshold on the distractor count: recovery (mass>1/2) iff (n-1) < e^{beta*gap}."""
    return float(np.exp(beta * gap))


# --------------------------------------------------------------------------------------
# The read.
# --------------------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def dense_read(q, K, V, beta):
    """Full softmax attention over all n keys. Returns (output, weights, keys_scored)."""
    s = beta * (K @ q)
    p = softmax(s)
    return p @ V, p, len(K)


def read_over(q, K, V, beta, cand, k):
    """Exact softmax over a candidate set `cand`, renormalized on its top-k.
    Returns (output, selected_indices, weights_on_selected)."""
    cand = np.asarray(cand)
    if cand.size == 0:
        return np.zeros(V.shape[1]), cand, np.array([])
    s = beta * (K[cand] @ q)
    if k < cand.size:
        loc = np.argpartition(s, -k)[-k:]
        cand = cand[loc]
    p = softmax(beta * (K[cand] @ q))
    return p @ V[cand], cand, p


# --------------------------------------------------------------------------------------
# Selectors — produce a candidate set, and report keys_scored (the cost we care about).
# --------------------------------------------------------------------------------------

class ExactSelector:
    """Scores all n keys (O(n)). The quality reference for any sparse selector."""
    name = "exact"

    def build(self, K):
        self.K = K
        return self

    def select(self, q, beta, k):
        s = beta * (self.K @ q)
        idx = np.argpartition(s, -k)[-k:] if k < len(self.K) else np.arange(len(self.K))
        return idx, len(self.K)


class CentroidSelector:
    """Routing/clustering selector (Roy et al. 'Routing Transformer'-style): partition keys into
    B basins with centroids, score the B centroids, descend into the top clusters until the budget
    is met. Cost ~ B + (#keys in chosen clusters) ~ O(sqrt n + k) for B=sqrt(n)."""
    name = "centroid"

    def __init__(self, B=None, lloyd=4, seed=0):
        self.B, self.lloyd, self.seed = B, lloyd, seed

    def build(self, K):
        n, d = K.shape
        B = self.B or max(1, int(round(np.sqrt(n))))
        rng = np.random.default_rng(self.seed)
        cent = K[rng.choice(n, B, replace=False)].copy()
        assign = np.zeros(n, dtype=int)
        for _ in range(self.lloyd):
            assign = np.argmax(K @ cent.T, axis=1)
            for b in range(B):
                m = assign == b
                if m.any():
                    c = K[m].mean(0)
                    nc = np.linalg.norm(c)
                    cent[b] = c / nc if nc > 1e-12 else cent[b]
        self.K, self.cent, self.B = K, cent, B
        self.members = [np.where(assign == b)[0] for b in range(B)]
        return self

    def select(self, q, beta, k):
        cs = self.cent @ q                       # score B centroids
        order = np.argsort(cs)[::-1]
        cand, cost = [], self.B
        for b in order:
            m = self.members[b]
            cand.extend(m.tolist())
            cost += len(m)
            if len(cand) >= k:
                break
        cand = np.asarray(cand, dtype=int)
        s = beta * (self.K[cand] @ q)
        idx = np.argpartition(s, -k)[-k:] if k < cand.size else np.arange(cand.size)
        return cand[idx], cost


class LSHSelector:
    """SimHash (sign-of-random-projection) LSH. For unit-norm keys, max inner product = max cosine,
    so SimHash buckets near-neighbors directly (Indyk-Motwani; Charikar SimHash). Candidates = keys
    colliding with the query in any of L tables. Cost ~ L*bits + (#candidates)."""
    name = "lsh"

    def __init__(self, L=10, bits=10, seed=0):
        self.L, self.bits, self.seed = L, bits, seed

    def build(self, K):
        n, d = K.shape
        rng = np.random.default_rng(self.seed)
        self.planes = [rng.standard_normal((self.bits, d)).astype(np.float32) for _ in range(self.L)]
        self.w = (1 << np.arange(self.bits))
        self.tables = []
        for P in self.planes:
            codes = (K @ P.T > 0).astype(np.int64) @ self.w
            tab = {}
            for i, c in enumerate(codes):
                tab.setdefault(int(c), []).append(i)
            self.tables.append(tab)
        self.K = K
        return self

    def select(self, q, beta, k):
        cand, cost = set(), self.L * self.bits
        for P, tab in zip(self.planes, self.tables):
            c = int((q @ P.T > 0).astype(np.int64) @ self.w)
            for i in tab.get(c, ()):
                cand.add(i)
        cand = np.fromiter(cand, dtype=int) if cand else np.array([], dtype=int)
        cost += cand.size
        if cand.size == 0:
            return cand, cost
        s = beta * (self.K[cand] @ q)
        idx = np.argpartition(s, -k)[-k:] if k < cand.size else np.arange(cand.size)
        return cand[idx], cost


# --------------------------------------------------------------------------------------
# Synthetic geometry.
# --------------------------------------------------------------------------------------

def random_unit_keys(n, d, seed=0):
    """Random unit-norm keys; coherence (max off-diagonal inner product) ~ sqrt(2 log n / d)."""
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((n, d)).astype(np.float32)
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    return K


def clustered_keys(n, d, B, spread, seed=0):
    """B basins; key = basin_center + spread * noise, renormalized. Small spread = high separation
    (low within-basin coherence) — the regime training drives keys toward."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((B, d)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    assign = rng.integers(0, B, n)
    K = centers[assign] + spread * rng.standard_normal((n, d)).astype(np.float32)
    K /= np.linalg.norm(K, axis=1, keepdims=True)
    return K, assign


def coherence(K, sample=2000, seed=0):
    """Estimate the max off-target inner product (coherence eps) over a random sample of pairs."""
    rng = np.random.default_rng(seed)
    n = len(K)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    G = K[idx] @ K[idx].T
    np.fill_diagonal(G, -np.inf)
    return float(G.max())
