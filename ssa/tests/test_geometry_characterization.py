"""Fast assertion for the entry-magnitude → rank characterization (no training)."""
import numpy as np
from ssa.geometry_characterization import eff_rank


def test_eff_rank_grows_with_entry_magnitude():
    """exp(B·KKᵀ) is low-rank at small entry scale B and full-rank at large B — the split between the
    linear-attention route (small B) and the selection route (large B)."""
    rng = np.random.default_rng(0)
    K = rng.standard_normal((128, 16)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    ranks = [eff_rank(np.exp(B * (K @ K.T))) for B in (0.25, 1.0, 4.0, 16.0)]
    assert all(ranks[i] < ranks[i + 1] for i in range(len(ranks) - 1))   # monotone in B
    assert ranks[0] < 5.0 and ranks[-1] > 40.0                            # low at small B, high at large
