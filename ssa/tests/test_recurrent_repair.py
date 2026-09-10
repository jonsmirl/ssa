"""Dense-oracle tests for bounded-state recurrent sparse-read repair."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ssa.recurrent_repair import (
    StreamingSoftmaxState,
    fixed_query_repair,
    route_repair_rounds,
    topk_route,
)


def _direct(q, K, V, ids, beta=1.0):
    logits = beta * (K[ids] @ q)
    weights = np.exp(logits - logits.max())
    return (weights / weights.sum()) @ V[ids]


def test_streaming_state_equals_direct_union_for_any_batching_and_order():
    rng = np.random.default_rng(0)
    K, V, q = rng.normal(size=(41, 7)), rng.normal(size=(41, 5)), rng.normal(size=7)
    ids = rng.permutation(41)
    result = fixed_query_repair(q, K, V, [ids[:3], ids[3:19], ids[19:]], beta=1.7)
    np.testing.assert_allclose(result.output, _direct(q, K, V, ids, 1.7), rtol=2e-14, atol=2e-14)
    assert result.accumulator_scalars == V.shape[1] + 2
    assert result.rounds[-1].actual_retained_mass == pytest.approx(1.0)


def test_duplicate_retries_do_not_double_count_mass():
    rng = np.random.default_rng(1)
    K, V, q = rng.normal(size=(12, 3)), rng.normal(size=(12, 2)), rng.normal(size=3)
    once = fixed_query_repair(q, K, V, [[1, 4, 8]])
    repeated = fixed_query_repair(q, K, V, [[1, 4, 8], [8, 4, 1, 1]])
    np.testing.assert_allclose(repeated.output, once.output)
    np.testing.assert_array_equal(repeated.indices, once.indices)
    assert repeated.rounds[-1].cumulative_keys == 3


@pytest.mark.parametrize("prefix", [1, 7, 19, 30])
def test_arbitrary_causal_prefix_and_full_refinement(prefix):
    rng = np.random.default_rng(prefix)
    K, V, q = rng.normal(size=(30, 4)), rng.normal(size=(30, 6)), rng.normal(size=4)
    ids = np.arange(prefix)[::-1]
    result = fixed_query_repair(q, K, V, np.array_split(ids, 3), prefix=prefix)
    np.testing.assert_allclose(result.output, _direct(q, K, V, np.arange(prefix)), atol=2e-14)
    assert result.rounds[-1].actual_output_error <= 3e-14


def test_disjoint_static_rounds_equal_one_shot_at_equal_total_budget():
    rng = np.random.default_rng(3)
    routing_keys = rng.normal(size=(50, 6)); rq = rng.normal(size=6)
    first = topk_route(rq, routing_keys, 5)
    second = topk_route(rq, routing_keys, 5, excluded=first)
    one = topk_route(rq, routing_keys, 10)
    np.testing.assert_array_equal(np.r_[first, second], one)
    K, V, q = rng.normal(size=(50, 6)), rng.normal(size=(50, 4)), rng.normal(size=6)
    staged = fixed_query_repair(q, K, V, [first, second])
    one_shot = fixed_query_repair(q, K, V, [one])
    np.testing.assert_allclose(staged.output, one_shot.output, atol=2e-14)


def test_opened_clue_can_steer_next_read_to_high_attention_target():
    # Routing and attention geometry are deliberately separate.  The first
    # route finds a clue; its value is the routing direction of a target that
    # has almost all mass under the fixed attention query.
    n, d = 20, 4
    routing = np.zeros((n, d)); routing[0, 0] = 4; routing[13, 2] = 4
    K = np.zeros((n, d)); K[13, 3] = 12
    V = np.zeros((n, d)); V[0, 2] = 1; V[13, 1] = 1
    q_attention = np.array([0, 0, 0, 1.0])

    def controller(state, sparse_output, _round):
        clue = sparse_output
        return clue, clue

    repaired = route_repair_rounds(
        q_attention, K, V, routing, np.array([1, 0, 0, 0.]), controller,
        rounds=2, budget=1)
    np.testing.assert_array_equal(repaired.rounds[0].indices, [0])
    np.testing.assert_array_equal(repaired.rounds[1].indices, [13])
    assert repaired.rounds[1].actual_retained_mass > 0.999


def test_no_observation_of_unopened_target_means_no_target_specific_repair():
    # The first read is identical in both worlds.  A deterministic controller
    # therefore issues the same second query and cannot select both possible
    # targets at unit budget: the executable AddressLoss witness.
    routing_a = np.eye(4)
    routing_b = routing_a.copy()
    values = np.zeros((4, 4)); values[0, 0] = 1
    K = np.zeros((4, 4)); K[2, 3] = K[3, 3] = 8

    def controller(state, sparse_output, _round):
        return state, np.array([0, 1, 0, 0.])

    args = (np.array([0, 0, 0, 1.]), K, values)
    a = route_repair_rounds(*args, routing_a, np.array([1, 0, 0, 0.]), controller,
                            rounds=2, budget=1)
    # Swap the meanings/locations of the two unread worlds; the observation at
    # id 0 remains byte-identical and so does the chosen second id.
    routing_b[[2, 3]] = routing_b[[3, 2]]
    b = route_repair_rounds(*args, routing_b, np.array([1, 0, 0, 0.]), controller,
                            rounds=2, budget=1)
    np.testing.assert_array_equal(a.rounds[0].indices, b.rounds[0].indices)
    np.testing.assert_array_equal(a.rounds[1].indices, b.rounds[1].indices)


def test_invalid_or_empty_reads_fail_closed():
    with pytest.raises(ValueError):
        StreamingSoftmaxState.empty(0)
    with pytest.raises(ValueError):
        fixed_query_repair([1.0], [[1.0]], [[2.0]], [[]])


def test_raw_loss_has_no_gradient_through_hard_address_but_route_surrogate_does():
    scores = torch.tensor([[0.1, 0.4, -0.2]], requires_grad=True)
    target = torch.tensor([0])
    selected = scores.argmax(-1)
    archive_values = 8.0 * torch.eye(3)
    raw = F.cross_entropy(archive_values[selected], target) + 0.0 * scores.sum()
    raw.backward()
    assert torch.count_nonzero(scores.grad) == 0

    scores.grad = None
    F.cross_entropy(scores, target).backward()
    assert torch.count_nonzero(scores.grad) > 0
