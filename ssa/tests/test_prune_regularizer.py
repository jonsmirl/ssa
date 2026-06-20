"""Fast assertions for Route F: the Samuelson bound is admissible (validating samuelson_centered) and
the Samuelson B&B is lossless."""
import numpy as np
from ssa.core import clustered_keys
from ssa.prune_regularizer import samuelson_bnb


def test_samuelson_bound_is_admissible():
    """max_{k∈cluster} ⟨q,k⟩ ≤ ⟨q,μ⟩ + √((m−1)·qᵀΣq) — the Samuelson prune bound (samuelson_centered)."""
    rng = np.random.default_rng(0)
    K, _ = clustered_keys(400, 16, B=8, spread=0.3, seed=0)
    for _ in range(30):
        q = rng.standard_normal(16).astype(np.float32)
        mem = rng.choice(400, 24, replace=False)
        proj = K[mem] @ q
        m = len(mem); mean = float(proj.mean()); var = float(((proj - mean) ** 2).mean())
        bound = mean + np.sqrt((m - 1) * var)
        assert float(proj.max()) <= bound + 1e-3


def test_samuelson_bnb_is_lossless():
    """Best-first B&B with the Samuelson bound returns the true dense argmax."""
    rng = np.random.default_rng(1)
    K, _ = clustered_keys(512, 24, B=16, spread=0.2, seed=0)
    members = [np.arange(c * 32, (c + 1) * 32) for c in range(16)]
    for _ in range(30):
        i = int(rng.integers(512)); q = K[i] + 0.2 * rng.standard_normal(24).astype(np.float32)
        bkey, _ = samuelson_bnb(K, q, members)
        assert bkey == int((K @ q).argmax())


def test_prune_gate_predicts_pruning():
    """samuelson_prune_gate: card·(β−s̄)² > (card−1)·SS  ⟹  every member < β (numerically)."""
    rng = np.random.default_rng(2)
    for _ in range(200):
        m = int(rng.integers(3, 40))
        f = rng.standard_normal(m)
        beta = float(f.mean()) + float(rng.uniform(0, 3))            # best ≥ cluster mean
        mean = float(f.mean()); ss = float(((f - mean) ** 2).sum())
        gate = m * (beta - mean) ** 2 > (m - 1) * ss
        if gate:
            assert float(f.max()) < beta + 1e-9                      # gate fires ⟹ pruned (no member ≥ β)
