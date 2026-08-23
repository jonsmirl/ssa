"""Tests for `ssa.group_bound` — shared block selection over a group of queries.

The dangerous failure here is not a crash. A group bound that is secretly too large prunes nothing
and passes every admissibility test vacuously, and a group bound that is too small still returns the
right argmax most of the time — it fails only on the queries whose block it dropped, which is exactly
the case a small random test misses. So every section pairs a positive check with a NEGATIVE one, and
the negatives are built to fire.

The invariants under test are the operational face of
`Substrate/Universal/Potential/ChainedPrune.lean`: safety for a group is domination of the group's
TOP CLAIM, the top claim is the least such bound, and the shared threshold is the group's WEAKEST
incumbent — using the strongest silently loses the argmax for every other member.
"""
import numpy as np
import pytest

from ssa.bound_floor import bound_samuelson, kmeans_blocks
from ssa.group_bound import (group_bound_exact, group_bound_isotropic, group_bound_lowrank,
                             group_oracle, group_top_claim, key_radius, make_case,
                             per_query_select, query_summary, shared_select, true_argmax)

GROUP_BOUNDS = {"top_claim": group_bound_exact,
                "lowrank": lambda K, m, Q: group_bound_lowrank(K, m, Q, 8),
                "isotropic": group_bound_isotropic}


# ---------------------------------------------------------------------------- admissibility
@pytest.mark.parametrize("qspread", [0.02, 0.10, 0.30, 1.00])
@pytest.mark.parametrize("name", sorted(GROUP_BOUNDS))
def test_group_bound_is_admissible(name, qspread):
    """Every group bound must sit at or above `max_q max_k <q,k>` for its block. This is the whole
    correctness content: below it, a block that holds some query's argmax can be dropped."""
    K, Q, members = make_case(qspread=qspread, mQ=16, n=1024, d=32, B=32)
    for mem in members:
        assert GROUP_BOUNDS[name](K, mem, Q) >= group_oracle(K, mem, Q) - 1e-9


@pytest.mark.parametrize("qspread", [0.02, 0.30])
def test_admissibility_is_not_vacuous(qspread):
    """NEGATIVE — the floor must actually bind somewhere. A bound that always exceeded the block's
    true max by a wide margin would pass the test above while pruning nothing, so at least one block
    must sit close to its own floor."""
    K, Q, members = make_case(qspread=qspread, mQ=16, n=1024, d=32, B=32)
    slack = [group_bound_exact(K, m, Q) - group_oracle(K, m, Q) for m in members]
    assert min(slack) < 1.0, "no block is anywhere near its floor — the bound cannot be pruning"


# ---------------------------------------------------------------------------- the top claim
@pytest.mark.parametrize("qspread", [0.02, 0.30])
def test_top_claim_dominates_every_member(qspread):
    """`safe_for_every_stage_iff_safe_over_the_top_claim`: safety for the group is exactly domination
    of the top claim, so the top claim stands at or above every member's own bound."""
    K, Q, members = make_case(qspread=qspread, mQ=12, n=1024, d=32, B=32)
    for mem in members:
        tc = group_top_claim(K, mem, Q, per_query=bound_samuelson)
        assert all(tc >= bound_samuelson(K, mem, q) - 1e-9 for q in Q)


def test_top_claim_is_the_least_safe_bound():
    """`no_safe_bound_holds_below_the_top_claim` — NEGATIVE. Anything strictly below the top claim on
    some block is unsafe: it drops that block at a threshold some member of the group still needs."""
    K, Q, members = make_case(qspread=0.30, mQ=12, n=1024, d=32, B=32)
    mem = max(members, key=len)
    tc = group_top_claim(K, mem, Q, per_query=bound_samuelson)
    below = tc - 1e-6
    assert any(bound_samuelson(K, mem, q) > below for q in Q), \
        "a value below the top claim must be beaten by some member, or the claim was not the max"


# ---------------------------------------------------------------------------- exactness
@pytest.mark.parametrize("qspread", [0.02, 0.10, 0.30])
@pytest.mark.parametrize("name", sorted(GROUP_BOUNDS))
def test_shared_selection_is_exact(name, qspread):
    """Admissible bound plus the correct threshold: the shared selection returns the TRUE argmax for
    every query in the group, not an approximation of it."""
    K, Q, members = make_case(qspread=qspread, mQ=16, n=2048, d=32, B=32)
    keys, _, _ = shared_select(K, Q, members, GROUP_BOUNDS[name])
    assert (keys == true_argmax(K, Q)).all()


def test_threshold_must_be_the_weakest_incumbent():
    """NEGATIVE — the bug this file exists to prevent. Breaking out when the block's bound falls
    under the group's BEST incumbent is wrong: the block may still be the argmax's home for a weaker
    member of the group. The strongest-incumbent rule must lose the argmax on some case."""
    def bad_select(K, Q, members, bound):
        ubs = [(float(bound(K, mem, Q)), mem) for mem in members]
        best = np.full(len(Q), -1e30)
        bkey = np.full(len(Q), -1, dtype=int)
        for ub, mem in sorted(ubs, key=lambda t: -t[0]):
            if ub <= float(best.max()):                      # the bug: strongest, not weakest
                break
            sc = Q @ K[mem].T
            j = sc.argmax(1)
            v = sc[np.arange(len(Q)), j]
            upd = v > best
            best = np.where(upd, v, best)
            bkey = np.where(upd, np.asarray(mem)[j], bkey)
        return bkey

    lost = 0
    for seed in range(8):
        K, Q, members = make_case(spread=0.05, qspread=0.30, mQ=32, n=4096, d=64, B=64, seed=seed)
        gt = true_argmax(K, Q)
        assert (shared_select(K, Q, members, group_bound_exact)[0] == gt).all(), \
            "the weakest-incumbent rule must stay exact"
        lost += int(not (bad_select(K, Q, members, group_bound_exact) == gt).all())
    assert lost > 0, "the strongest-incumbent rule must lose the argmax somewhere, or the test is blind"


def test_a_bound_safe_for_one_member_is_not_safe_for_the_group():
    """NEGATIVE, and the operational twin of `a_prune_can_drop_the_greatest_part_of_a_second_objective`.

    Prune the group with a bound that is perfectly admissible for ONE member — the key-side bound at
    `Q[0]` — and the other members lose their argmax. This is the whole reason the group needs the
    top claim rather than any member's own bound, and it is the failure a per-query test cannot see.

    Note what does NOT work as a negative: scaling every block's bound down by one positive factor
    leaves the visiting ORDER untouched and only moves the stopping point, so the search still finds
    the true argmax before it stops. The negative has to break the ordering, not the level.
    """
    lost = 0
    for seed in range(6):
        K, Q, members = make_case(spread=0.05, qspread=0.30, mQ=32, n=4096, d=64, B=64, seed=seed)
        gt = true_argmax(K, Q)
        one = lambda K, m, Q: float(bound_samuelson(K, m, Q[0]))
        keys, _, _ = shared_select(K, Q, members, one)
        lost += int(not (keys == gt).all())
    assert lost > 0, "a single member's bound must fail some other member, or the test is blind"


# ---------------------------------------------------------------------------- the radius
def test_key_radius_bounds_the_deviation():
    """`R_c` must bound `max_k <q, k - mu_c>` for every unit query, which is what makes it a valid
    shared additive term."""
    K, Q, members = make_case(qspread=0.30, mQ=8, n=1024, d=32, B=32)
    for mem in members:
        mu = K[mem].mean(0)
        R = key_radius(K, mem)
        assert ((K[mem] - mu) @ Q.T).max() <= R + 1e-9


def test_key_radius_is_query_independent():
    """NEGATIVE — it must not read the queries at all, or it could not be precomputed at index
    build, which is the entire reason for its shape."""
    import inspect
    params = list(inspect.signature(key_radius).parameters)
    assert params == ["K", "mem"], f"key_radius must take no query: {params}"


# ---------------------------------------------------------------------------- the group's cost
def test_group_size_costs_retention():
    """`one_more_stage_drops_within_the_chain_drop` — a larger group can only read MORE. Adding
    members raises the top claim, and a raised bound drops a subset."""
    K, _, members = make_case(qspread=0.10, mQ=64, n=2048, d=32, B=32)
    _, Qbig, _ = make_case(qspread=0.10, mQ=64, n=2048, d=32, B=32)
    for mem in members[:6]:
        vals = [group_top_claim(K, mem, Qbig[:m], per_query=bound_samuelson) for m in (4, 16, 64)]
        assert vals[0] <= vals[1] + 1e-9 <= vals[2] + 1e-9


def test_the_query_side_relaxation_loosens_with_group_size():
    """The mechanism behind the REFUTED cost claim, pinned so it is not re-proposed.

    The query-side Samuelson term carries `sqrt(mQ - 1)`, so the relaxed rungs loosen as the group
    grows even when the group's own geometry is unchanged. That is why the two cheap rungs lose to
    the top claim at every measured group size instead of overtaking it: the O(1)-per-block
    evaluation they save is smaller than the pruning they give up.
    """
    K, Q, members = make_case(qspread=0.10, mQ=256, n=2048, d=32, B=32)
    mem = max(members, key=len)
    small = group_bound_isotropic(K, mem, Q[:8])
    large = group_bound_isotropic(K, mem, Q[:256])
    assert large > small, "the query-side term must grow with the group, or the refutation's mechanism is gone"


def test_shared_selection_beats_per_query_on_a_tight_group():
    """The POSITIVE cost claim, and the only one made: sharing the block CHOICE pays when the group
    is tight. Measured on total work — keys scored plus bound evaluations."""
    K, Q, members = make_case(spread=0.05, qspread=0.02, mQ=64, n=8192, d=64, B=128)
    _, pc, pb = per_query_select(K, Q, members, bound_samuelson)
    _, sc, sb = shared_select(K, Q, members, group_bound_exact)
    assert (pc + pb) / (sc + sb) > 1.4


def test_a_spread_group_is_not_helped():
    """NEGATIVE, and the quantitative face of `nothing_is_free_to_drop_when_every_part_is_within_one_reach`:
    when the group spreads, sharing a selection stops paying. Reported rather than hidden, because a
    cost claim without its failing regime is not a measurement."""
    K, Q, members = make_case(spread=0.05, qspread=0.30, mQ=64, n=8192, d=64, B=128)
    _, pc, pb = per_query_select(K, Q, members, bound_samuelson)
    _, sc, sb = shared_select(K, Q, members, group_bound_exact)
    assert (pc + pb) / (sc + sb) < 1.0


def test_query_summary_is_the_centred_group():
    mu, Y, mQ = query_summary(np.array([[1.0, 0.0], [3.0, 2.0]]))
    assert np.allclose(mu, [2.0, 1.0]) and np.allclose(Y, [[-1.0, -1.0], [1.0, 1.0]]) and mQ == 2
