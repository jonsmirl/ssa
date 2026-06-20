"""Numerically check the tempered-routing identities + the log-sum-exp sandwich (CPU)."""
import numpy as np
from ssa.tempered_routing import _tempered


def test_limits_and_second_order():
    rng = np.random.default_rng(0)
    s = rng.standard_normal(64)
    assert abs(_tempered(s, 1.0, "centroid") - s.mean()) < 1e-9          # β→0 is the mean
    assert _tempered(s, 1.0, "oracle") == float(s.max())                 # β→∞ is the max
    assert abs(_tempered(s, 3.0, "second") - (s.mean() + 1.5 * s.var())) < 1e-9   # μ + (β/2)σ²


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
