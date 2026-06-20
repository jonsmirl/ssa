"""Fast assertions for the real-key selector logic (no model load — synthetic geometry only).

These lock in the two structural claims the GPT-2 experiment measures:
  (1) exact admissible branch-and-bound is LOSSLESS by construction (recall = dense ceiling), and
  (2) the prune fraction is a function of key GEOMETRY — clumped keys prune, spread keys do not —
      which is the trilemma (`SearchTradeoff.cluster_prunable`) realized as a compute cost.
The live GPT-2 run (`python3 -m ssa.real_keys`) is the actual experiment;
it is not run in CI (heavy model load)."""
import numpy as np
from ssa.core import clustered_keys, random_unit_keys
from ssa.real_keys import bandb_exact, bandb_budget


def _queries_from(K, cos=0.6, seed=1):
    """Queries that each 'want' the same-index key (a findable target at cosine `cos`)."""
    rng = np.random.default_rng(seed)
    n, d = K.shape
    Q = np.empty_like(K)
    for i in range(n):
        t = K[i]
        z = rng.standard_normal(d).astype(np.float32)
        z -= z.dot(t) * t
        z /= np.linalg.norm(z) + 1e-9
        Q[i] = cos * t + np.sqrt(max(0.0, 1 - cos * cos)) * z
    return Q


def test_exact_bandb_is_lossless():
    """Admissible-bound B&B always returns the true argmax (recall = 1) on any geometry."""
    K, _ = clustered_keys(400, 32, B=8, spread=0.15, seed=0)
    Q = _queries_from(K)
    _, recall, _ = bandb_exact(K, Q, B=16, causal=False, max_queries=120)
    assert recall == 1.0
    Kr = random_unit_keys(400, 32, seed=3)
    Qr = _queries_from(Kr)
    _, recall_r, _ = bandb_exact(Kr, Qr, B=16, causal=False, max_queries=120)
    assert recall_r == 1.0


def test_clumped_keys_prune_more_than_spread():
    """The prune fraction is set by geometry: tight clusters let the bound prune; uniform keys do not."""
    K, _ = clustered_keys(600, 32, B=10, spread=0.10, seed=0)   # benign / clumped
    Kr = random_unit_keys(600, 32, seed=0)                       # worst case / spread
    cost_clumped, _, _ = bandb_exact(K, _queries_from(K), B=20, causal=False, max_queries=120)
    cost_spread, _, _ = bandb_exact(Kr, _queries_from(Kr), B=20, causal=False, max_queries=120)
    assert cost_clumped < cost_spread          # clumping turns pruning on
    assert cost_spread > 0.9                    # spread keys: essentially no pruning


def test_budget_recall_monotone_and_benign_beats_random():
    """Approximate top-k recall rises with budget, and clumped (benign) geometry beats random at a
    fixed sublinear budget — the SubQ regime."""
    K, _ = clustered_keys(600, 32, B=10, spread=0.12, seed=0)
    Kr = random_unit_keys(600, 32, seed=0)
    Q, Qr = _queries_from(K), _queries_from(Kr)
    r05 = bandb_budget(K, Q, B=20, budget_frac=0.05, causal=False, max_queries=120)
    r20 = bandb_budget(K, Q, B=20, budget_frac=0.20, causal=False, max_queries=120)
    rr05 = bandb_budget(Kr, Qr, B=20, budget_frac=0.05, causal=False, max_queries=120)
    assert r20 >= r05                           # more budget never hurts recall
    assert r05 > rr05                           # benign geometry beats random at the same budget
