"""Numerically validate the anisotropic bound: both the isotropic and ellipsoidal bounds are admissible
(upper bounds on the cluster's best score)."""
import numpy as np
from ssa.core import clustered_keys
from ssa.adaptive import kmeans
from ssa.anisotropic_bound import cluster_stats


def test_both_bounds_are_admissible():
    """ellipsoidal_search_bound + the isotropic bound both upper-bound max_{k∈cluster} ⟨q,k⟩."""
    K, _ = clustered_keys(600, 16, B=8, spread=0.3, seed=0)
    members, mu, _ = kmeans(K, 8, seed=0)
    members = [np.asarray(m) for m in members]
    stats = cluster_stats(K, members, mu)
    rng = np.random.default_rng(1)
    for _ in range(25):
        q = rng.standard_normal(16).astype(np.float32)
        qn = float(np.linalg.norm(q))
        for b, st in enumerate(stats):
            if st is None:
                continue
            mu_b, R_iso, half, R_maha = st
            mem = members[b]
            true_max = float((K[mem] @ q).max())
            iso = float(mu_b @ q) + qn * R_iso
            ellip = float(mu_b @ q) + R_maha * float(np.linalg.norm(half @ q))
            assert true_max <= iso + 1e-3                       # isotropic bound valid
            assert true_max <= ellip + 1e-3                     # ellipsoidal bound valid (the theorem)
