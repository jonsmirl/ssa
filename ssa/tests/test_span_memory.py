import numpy as np
import pytest

from ssa.span_memory import fit_cells, joint_update, linear_obstruction, predict_cells, sequential_update


def test_substrate_realizable_single_key_interference_witness():
    X = np.array([[1., 0.], [2., 1.]])
    Y = np.zeros((2, 1))
    W = np.array([[1.], [-3.]])
    updated, info = sequential_update(W, X[:1], Y[:1], rate=.5)
    assert np.linalg.norm(Y - X @ W, axis=1).sum() == 2
    assert np.linalg.norm(Y - X @ updated, axis=1).sum() == 2.5
    assert np.sum((Y - X @ W) ** 2) == 2
    assert np.sum((Y - X @ updated) ** 2) == 4.25
    assert info["residual_after_squared"] == .25
    assert np.array_equal(X @ np.zeros_like(W), Y)  # joint realizability


def test_joint_realizable_residual_contraction_and_orthogonal_invariance():
    X = np.array([[1., 0., 0.], [2., 1., 0.]])
    W = np.array([[1., 2.], [-3., 4.], [7., 9.]])
    Y = X @ np.array([[3., 4.], [5., 6.], [-9., 2.]])
    updated, info = joint_update(W, X, Y, rate=.5)
    np.testing.assert_allclose(Y - X @ updated, .5 * (Y - X @ W), atol=1e-14)
    np.testing.assert_array_equal(updated[2], W[2])
    assert info["numerical_rank"] == 2
    assert info["joint_irreducible_residual_fro"] < 1e-13


def test_inconsistent_targets_leave_exact_least_squares_floor():
    X = np.array([[1.], [1.]])
    Y = np.array([[0.], [2.]])
    updated, info = joint_update(np.array([[7.]]), X, Y)
    np.testing.assert_allclose(updated, [[1.]], atol=1e-14)
    np.testing.assert_allclose(info["residual_after_squared"], 2.)
    np.testing.assert_allclose(info["joint_irreducible_residual_fro"], np.sqrt(2.))


def test_inconsistent_joint_update_only_contracts_representable_residual():
    X = np.array([[1.], [1.]])
    Y = np.array([[0.], [2.]])
    W = np.array([[7.]])
    updated, _ = joint_update(W, X, Y, rate=.25)
    floor = Y - X @ np.linalg.pinv(X) @ Y
    np.testing.assert_allclose(Y - X @ updated, floor + .75 * (Y - X @ W - floor))
    # Irreducible association error must not be advertised as contracting.
    assert not np.allclose(Y - X @ updated, .75 * (Y - X @ W))


def test_unobserved_orthogonal_target_can_remain_arbitrarily_wrong():
    W = np.zeros((2, 1))
    updated, info = joint_update(W, [[1., 0.]], [[2.]])
    assert info["residual_after_fro"] == 0
    assert float((np.array([[0., 1.]]) @ updated)[0, 0]) == 0
    # Perfect fit of read associations supplies no bound for this unseen one.
    assert abs(1e6 - (np.array([[0., 1.]]) @ updated)[0, 0]) == 1e6


@pytest.mark.parametrize("method", [joint_update, sequential_update])
def test_rank_deficient_zero_empty_and_no_mutation(method):
    W = np.array([[3.], [4.]])
    X = np.array([[1., 0.], [2., 0.], [0., 0.]])
    Y = np.array([[2.], [4.], [5.]])
    originals = [v.copy() for v in (W, X, Y)]
    updated, info = method(W, X, Y)
    np.testing.assert_allclose(updated, [[2.], [4.]])
    assert info["numerical_rank"] == 1 and info["rows_read"] == 3
    assert info["residual_after_squared"] == pytest.approx(25.)
    for original, current in zip(originals, (W, X, Y)):
        np.testing.assert_array_equal(original, current)
    zero, info = method(W, np.zeros((2, 2)), np.ones((2, 1)))
    np.testing.assert_array_equal(zero, W)
    assert info["retained_condition"] is None
    empty, info = method(W, np.empty((0, 2)), np.empty((0, 1)))
    np.testing.assert_array_equal(empty, W)
    assert info["rows_read"] == 0


def test_singular_value_cutoff_is_reported_not_hidden():
    W = np.zeros((2, 1))
    X = np.diag([1., 1e-12])
    Y = np.ones((2, 1))
    updated, info = joint_update(W, X, Y)
    np.testing.assert_array_equal(updated, [[1.], [0.]])
    assert info["numerical_rank"] == 1
    assert info["joint_irreducible_residual_fro"] == 1


@pytest.mark.parametrize("method", [joint_update, sequential_update])
@pytest.mark.parametrize("rate", [-1., 1.1, float("nan"), float("inf")])
def test_invalid_rates(method, rate):
    with pytest.raises(ValueError):
        method(np.zeros((2, 1)), np.ones((3, 2)), np.ones((3, 1)), rate=rate)


@pytest.mark.parametrize("method", [joint_update, sequential_update])
def test_invalid_shapes_and_nonfinite(method):
    with pytest.raises(ValueError):
        method(np.zeros((2, 1)), np.ones((3, 2)), np.ones((2, 1)))
    with pytest.raises(ValueError):
        method(np.zeros((2, 1)), [[float("nan"), 0.]], [[1.]])


def test_sequential_order_matters_and_joint_is_order_invariant():
    X, Y, W = np.array([[1., 0.], [1., 1.]]), np.array([[1.], [0.]]), np.zeros((2, 1))
    a, _ = sequential_update(W, X, Y)
    b, _ = sequential_update(W, X[::-1], Y[::-1])
    assert not np.allclose(a, b)
    a, _ = joint_update(W, X, Y)
    b, _ = joint_update(W, X[::-1], Y[::-1])
    np.testing.assert_allclose(a, b, atol=1e-14)


def test_cell_fit_deterministic_targets_do_not_change_partition():
    K = np.array([[-2.], [-1.], [1.], [2.]])
    Y = np.array([[1.], [3.], [8.], [10.]])
    originals = K.copy(), Y.copy()
    first = fit_cells(K, Y, 2, seed=8)
    second = fit_cells(K, Y, 2, seed=8)
    changed_targets = fit_cells(K, Y + 100, 2, seed=8)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    for index in (0, 1, 3):
        np.testing.assert_array_equal(first[index], changed_targets[index])
    np.testing.assert_array_equal(K, originals[0])
    np.testing.assert_array_equal(Y, originals[1])
    centers, counts, sums, assignment = first
    np.testing.assert_allclose(sums.sum(axis=0), Y.sum(axis=0))
    assert counts.sum() == len(Y)
    np.testing.assert_allclose(predict_cells(K, centers, counts, sums), [[2.], [2.], [9.], [9.]])
    assert np.array_equal(np.bincount(assignment, minlength=2), counts)


def test_cell_ties_empty_cells_and_empty_queries_are_defined():
    K, Y = np.zeros((3, 1)), np.arange(3.)[:, None]
    centers, counts, sums, assignment = fit_cells(K, Y, 3)
    np.testing.assert_array_equal(assignment, [0, 0, 0])
    np.testing.assert_array_equal(counts, [3, 0, 0])
    np.testing.assert_allclose(predict_cells([[0.]], centers, counts, sums), [[1.]])
    np.testing.assert_array_equal(predict_cells([[2.]], [[0.], [2.]], [1, 0], [[3.], [0.]]), [[0.]])
    assert predict_cells(np.empty((0, 1)), centers, counts, sums).shape == (0, 1)


def test_no_lookahead_update_only_depends_on_supplied_rows():
    W = np.zeros((2, 1))
    train_X, train_Y = np.eye(2)[:1], np.array([[2.]])
    for method in (joint_update, sequential_update):
        a, _ = method(W, train_X, train_Y)
        # Unobserved association may be arbitrary without affecting this state.
        future_Y = np.array([[999.]])
        b, _ = method(W, train_X, train_Y)
        np.testing.assert_array_equal(a, b)
        assert future_Y[0, 0] != (np.eye(2)[1:] @ a)[0, 0]


@pytest.mark.parametrize("method", [joint_update, sequential_update])
def test_zero_rate_returns_independent_unchanged_state(method):
    W = np.array([[1.], [2.]])
    updated, _ = method(W, [[1., 2.]], [[9.]], rate=0.)
    np.testing.assert_array_equal(updated, W)
    assert not np.shares_memory(updated, W)


def test_exact_modular_linear_obstruction():
    K = np.array([[1., 0.], [0., 1.], [1., 1.]])
    result = linear_obstruction(K, [[1.], [2.], [4.]])
    assert result["obstruction"] and result["rank"] == 3
    assert result["rows"] == 3 and result["value_column"] == 0
    realizable = linear_obstruction(K, [[1.], [2.], [3.]])
    assert not realizable["obstruction"] and realizable["rank"] == 2


def test_exact_modular_obstruction_handles_dyadic_and_tiny_values():
    K = np.array([[.5, 0.], [0., .25], [.5, .25]])
    assert linear_obstruction(K, [[.125], [.0625], [.25]])["obstruction"]
    # Neither conversion nor elimination rounds the smallest positive subnormal.
    tiny = np.nextafter(0., 1.)
    assert linear_obstruction([[tiny], [tiny]], [[0.], [tiny]])["obstruction"]


def test_absence_of_modular_obstruction_is_inconclusive():
    assert not linear_obstruction(np.eye(2), [[1.], [2.]])["obstruction"]
    assert not linear_obstruction(np.empty((0, 2)), np.empty((0, 1)))["obstruction"]
    # Reduction modulo the prime can hide a genuine rational obstruction.
    assert not linear_obstruction([[1.], [1.]], [[0.], [2147483647.]])["obstruction"]
    # The first coordinate agrees, while an uninspected coordinate need not.
    assert not linear_obstruction([[1.], [1.]], [[0., 0.], [0., 1.]])["obstruction"]


def test_modular_obstruction_input_validation_and_nonmutation():
    K, V = np.array([[1.], [1.]]), np.array([[0.], [1.]])
    before = K.copy(), V.copy()
    linear_obstruction(K, V)
    np.testing.assert_array_equal(K, before[0])
    np.testing.assert_array_equal(V, before[1])
    for prime in (4, 2147483647., True):
        with pytest.raises(ValueError):
            linear_obstruction(K, V, prime=prime)
    with pytest.raises(ValueError):
        linear_obstruction([[float("inf")]], [[1.]])
