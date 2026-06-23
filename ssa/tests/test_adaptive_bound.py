"""Regression test for the adaptive.py admissible-bound bug: the UB must keep the ‖q‖ factor, else it
under-estimates whenever ‖q‖ > 1 and branch-and-bound can prune the true argmax. (Was UB = mu@q + R.)"""
import numpy as np
from ssa.adaptive import kmeans


def test_admissible_bound_holds_when_qnorm_gt_one():
    rng = np.random.default_rng(0)
    K = rng.standard_normal((600, 16)).astype(np.float32)
    members, mu, R = kmeans(K, 8, seed=0)
    q = (4.0 * rng.standard_normal(16)).astype(np.float32)        # ‖q‖ ≈ 16 ≫ 1
    qn = float(np.linalg.norm(q))
    assert qn > 1.0

    buggy_violations = 0
    for b in range(8):
        m = np.asarray(members[b])
        if m.size == 0:
            continue
        true_max = float((K[m] @ q).max())
        ub_fixed = float(mu[b] @ q + qn * R[b])                  # the FIXED Cauchy–Schwarz bound (eq 5.1)
        ub_buggy = float(mu[b] @ q + R[b])                       # the bug (dropped ‖q‖)
        assert ub_fixed >= true_max - 1e-3, f"cluster {b}: fixed bound {ub_fixed:.3f} < true {true_max:.3f}"
        buggy_violations += int(ub_buggy < true_max - 1e-3)
    # the dropped-‖q‖ bound is genuinely inadmissible here (would prune real argmaxes):
    assert buggy_violations >= 1
