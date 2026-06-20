"""Fast, robust pytest assertions for the retrieval-margin predictions and the selector."""
import numpy as np
import pytest
from ssa.core import (
    recovery_weight, threshold_n, softmax, dense_read, read_over,
    ExactSelector, CentroidSelector, LSHSelector, random_unit_keys, clustered_keys)


def test_recovery_weight_is_exact_target_mass():
    """The homogeneous-distractor softmax mass equals recoveryWeight to float precision."""
    beta, gap = 8.0, 0.5
    for n in (10, 1000, 100000):
        scores = np.concatenate([[0.0], -gap * np.ones(n - 1)])
        p = softmax(beta * scores)
        assert abs(p[0] - recovery_weight(beta, gap, n - 1)) < 1e-9


def test_recovery_threshold_at_half():
    """rho crosses 1/2 exactly at beta*gap = log mu."""
    beta, gap = 5.0, 0.7
    mu_star = np.exp(beta * gap)
    assert recovery_weight(beta, gap, mu_star * 0.5) > 0.5
    assert recovery_weight(beta, gap, mu_star * 2.0) < 0.5


def test_length_generalization_logarithmic():
    """Required gap to hold recovery at 1-eps grows only logarithmically in the count."""
    beta, eps = 10.0, 0.01
    gap_needed = lambda mu: (np.log(mu) + np.log((1 - eps) / eps)) / beta
    g100, g10000 = gap_needed(100), gap_needed(10000)
    # 100x more context costs only an additive log(100)/beta of gap
    assert abs((g10000 - g100) - np.log(100) / beta) < 1e-9


def test_truncation_bound_holds():
    """||sparse_read - dense_read|| <= 2 Vmax * missed_mass on every trial."""
    rng = np.random.default_rng(0)
    n, d, beta = 8000, 96, 18.0
    K = random_unit_keys(n, d, seed=0)
    V = rng.standard_normal((n, 12)).astype(np.float32)
    Vmax = float(np.linalg.norm(V, axis=1).max())
    sel = ExactSelector().build(K)
    for _ in range(40):
        t = rng.integers(n)
        q = K[t]
        o, p, _ = dense_read(q, K, V, beta)
        for k in (4, 32, 256):
            cand, _ = sel.select(q, beta, k)
            hat, selidx, _ = read_over(q, K, V, beta, cand, k)
            m = 1.0 - p[selidx].sum()
            assert np.linalg.norm(hat - o) <= 2 * Vmax * m + 1e-5


def test_centroid_selector_sublinear_and_lossless_under_separation():
    """Under good separation, the centroid selector captures the target at sublinear cost."""
    rng = np.random.default_rng(1)
    d, beta, k, trials = 96, 22.0, 48, 120
    n = 16000
    K, _ = clustered_keys(n, d, B=int(np.sqrt(n)), spread=0.15, seed=1)
    sel = CentroidSelector(seed=1).build(K)
    hits, cost = 0, 0
    for _ in range(trials):
        t = rng.integers(n)
        q = K[t] + 0.02 * rng.standard_normal(d).astype(np.float32)
        q /= np.linalg.norm(q)
        cand, c = sel.select(q, beta, k)
        cost += c
        hits += int(t in set(cand.tolist()))
    recall, frac = hits / trials, cost / trials / n
    assert recall >= 0.9, f"recall {recall} too low under good separation"
    assert frac < 0.5, f"cost fraction {frac} not sublinear"


def test_recall_degrades_under_crowding():
    """The selector's recall drops as separation worsens (the doubling-dimension dependence)."""
    rng = np.random.default_rng(2)
    d, beta, k, trials = 96, 22.0, 48, 120
    n = 16000

    def recall_at(spread):
        K, _ = clustered_keys(n, d, B=int(np.sqrt(n)), spread=spread, seed=2)
        sel = CentroidSelector(seed=2).build(K)
        hits = 0
        for _ in range(trials):
            t = rng.integers(n)
            q = K[t] + 0.02 * rng.standard_normal(d).astype(np.float32)
            q /= np.linalg.norm(q)
            cand, _ = sel.select(q, beta, k)
            hits += int(t in set(cand.tolist()))
        return hits / trials

    assert recall_at(0.10) > recall_at(0.60) + 0.05


def test_reasoning_decays_as_rho_pow_h():
    """With an imperfect cue, single-hop recall rho<1 and an h-hop chain succeeds ~ rho^h:
    recall generalizes, composition does not. (Vacuous unless rho<1, so use a noisy cue.)"""
    rng = np.random.default_rng(3)
    d, trials, noise = 64, 500, 0.18
    n = 8000
    K = random_unit_keys(n, d, seed=3)

    def one_hop():
        t = rng.integers(n)
        q = K[t] + noise * rng.standard_normal(d).astype(np.float32)
        q /= np.linalg.norm(q)
        return int(np.argmax(K @ q) == t)

    rho = np.mean([one_hop() for _ in range(trials)])
    assert 0.6 < rho < 0.98, f"need a non-trivial rho to test compounding, got {rho}"
    succ = sum(int(all(one_hop() for _h in range(3))) for _ in range(trials))
    assert abs(succ / trials - rho ** 3) < 0.10
