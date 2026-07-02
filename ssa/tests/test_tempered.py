"""Numerically check the tempered-routing identities + the log-sum-exp sandwich (CPU)."""
import numpy as np
from ssa.tempered_routing import _tempered


def test_limits_and_second_order():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(64)
    assert abs(_tempered(s, 1.0, "centroid") - s.mean()) < 1e-9          # β→0 is the mean
    assert _tempered(s, 1.0, "oracle") == float(s.max())                 # β→∞ is the max
    # The "second" score μ + (β/2)σ² is the 2nd-order cumulant expansion of the escort
    # log-partition β⁻¹·log(mean e^{βs}) = μ + (β/2)σ² + O(β²κ₃). Check it INDEPENDENTLY against
    # the EXACT tempered score (the LSE branch, minus the (log m)/β sum-vs-mean offset) in the
    # small-β regime — not by re-deriving the same formula on both sides.
    beta, m = 0.05, len(s)
    exact_normalized = _tempered(s, beta, "exact") - np.log(m) / beta    # μ + (β/2)σ² + O(β²)
    second = _tempered(s, beta, "second")
    assert abs(second - exact_normalized) < 1e-3                         # matches the LSE to 2nd order
    # and it must be a strictly better approximation than the 0th-order centroid (μ alone)
    assert abs(second - exact_normalized) < abs(s.mean() - exact_normalized)


def test_tempered_sandwich():
    """max ≤ β⁻¹logΣe^{βs} ≤ max + log(n)/β  (temperedLogPartition_max_sandwich, numerically)."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        s = rng.standard_normal(rng.integers(2, 80))
        for beta in (0.5, 1.0, 2.0, 5.0):
            t = _tempered(s, beta, "exact")
            assert s.max() <= t + 1e-9
            assert t <= s.max() + np.log(len(s)) / beta + 1e-9


def test_higher_temperature_tightens_to_max():
    """The slack |tempered − max| is antitone in β (temperedLogPartition_slack_antitone)."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        s = rng.standard_normal(50)
        gaps = [_tempered(s, b, "exact") - s.max() for b in (0.5, 1.0, 4.0, 16.0)]
        assert all(gaps[i] >= gaps[i + 1] - 1e-9 for i in range(len(gaps) - 1))   # shrinking
