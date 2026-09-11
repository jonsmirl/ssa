"""Dense-oracle checks for the persistent positive-penalty construction."""

import numpy as np
import pytest

from ssa.persistent_ridge import PersistentRidge


def test_all_prefix_fit_matches_offline_and_batch_partition():
    rng = np.random.default_rng(21)
    X, Y = rng.normal(size=(29, 5)), rng.normal(size=(29, 3))
    once, chunks = PersistentRidge(5, 3, .7), PersistentRidge(5, 3, .7)
    once.append(X, Y)
    for start, stop in ((0, 3), (3, 12), (12, 29)):
        chunks.append(X[start:stop], Y[start:stop])
        expected = np.linalg.solve(X[:stop].T @ X[:stop] + .7 * np.eye(5), X[:stop].T @ Y[:stop])
        np.testing.assert_allclose(chunks.solve(), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(once.solve(), chunks.solve(), rtol=1e-12, atol=1e-12)
    assert once.rows == chunks.rows == 29


def test_inputs_exposed_arrays_and_copies_do_not_alias_or_leak_future():
    rng = np.random.default_rng(22)
    X, Y = rng.normal(size=(12, 4)), rng.normal(size=(12, 2))
    model = PersistentRidge(4, 2, 2.)
    model.append(X[:5], Y[:5])
    before = model.solve()
    clone = model.copy()
    clone.append(X[5:], Y[5:])
    X[:] = 1e6
    Y[:] = -1e6
    model.gram[:] = 0
    model.cross[:] = 0
    model.weights[:] = 0
    np.testing.assert_array_equal(model.solve(), before)
    assert model.rows == 5 and clone.rows == 12
    assert not np.allclose(clone.solve(), before)


def test_objective_excess_identity_rank_deficient_design():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(8, 4))
    X[:, 3] = X[:, 1]
    Y, W = rng.normal(size=(8, 3)), rng.normal(size=(4, 3))
    penalty = 1.3
    model = PersistentRidge(4, 3, penalty)
    model.append(X, Y)
    optimum = model.solve()
    objective = lambda w: np.linalg.norm(Y - X @ w) ** 2 + penalty * np.linalg.norm(w) ** 2
    delta = W - optimum
    excess = np.linalg.norm(X @ delta) ** 2 + penalty * np.linalg.norm(delta) ** 2
    np.testing.assert_allclose(objective(W) - objective(optimum), excess, rtol=1e-13)
    np.testing.assert_allclose((X.T @ X + penalty * np.eye(4)) @ optimum, X.T @ Y, atol=1e-12)


def test_reader_gain_matches_direct_map_and_sharp_mismatch():
    rng = np.random.default_rng(24)
    X, Y, a = rng.normal(size=(11, 5)), rng.normal(size=(11, 3)), rng.normal(size=5)
    model = PersistentRidge(5, 3, .4)
    model.append(X, Y)
    G = np.linalg.solve(X.T @ X + .4 * np.eye(5), X.T)
    row = a @ G
    gain = model.reader_gain(a)
    np.testing.assert_allclose(gain, np.linalg.norm(row), rtol=1e-12)
    assert gain <= np.linalg.norm(a) / (2 * np.sqrt(.4)) + 1e-12
    mismatch = np.outer(row / gain, np.array([0., 2., 0.]))
    np.testing.assert_allclose(np.linalg.norm(a @ G @ mismatch), 2 * gain, rtol=1e-12)


def test_noiseless_ridge_bias_identity():
    rng = np.random.default_rng(25)
    X, target = rng.normal(size=(7, 4)), rng.normal(size=(4, 2))
    model = PersistentRidge(4, 2, .8)
    model.append(X, X @ target)
    normal = X.T @ X + .8 * np.eye(4)
    np.testing.assert_allclose(normal @ (model.solve() - target), -.8 * target, atol=1e-12)


def test_repeated_residual_ridge_is_not_persistent_ridge():
    model = PersistentRidge(1, 1, 1.)
    model.append([[1.]], [[1.]])
    first = model.solve().item()
    second = first + .5 * (1 - first)
    objective = lambda w: (1 - w) ** 2 + w ** 2
    assert first == .5 and second == .75
    assert objective(first) == .5 and objective(second) == .625
    assert model.solve().item() == .5  # Solving again does not apply a residual correction.


def test_penalty_is_fixed_sum_objective_not_batch_or_mean_penalty():
    model = PersistentRidge(1, 1, 1.)
    model.append([[1.]], [[1.]])
    assert model.solve().item() == .5
    model.append([[1.]], [[1.]])
    np.testing.assert_allclose(model.solve(), [[2 / 3]])
    assert model.penalty == 1.


def test_empty_and_zero_design_are_well_defined():
    model = PersistentRidge(3, 2, .2)
    np.testing.assert_array_equal(model.solve(), np.zeros((3, 2)))
    assert model.reader_gain(np.ones(3)) == 0
    diag = model.append(np.empty((0, 3)), np.empty((0, 2)))
    assert diag["rows_read"] == diag["arithmetic_ops_estimate"] == 0
    model.append(np.zeros((7, 3)), np.ones((7, 2)))
    assert model.rows == 7
    np.testing.assert_array_equal(model.predict(np.ones((2, 3))), np.zeros((2, 2)))


@pytest.mark.parametrize("features,values,penalty", [(0, 2, 1), (2, 0, 1), (True, 2, 1), (2., 2, 1), (2, 2, 0), (2, 2, -1), (2, 2, np.nan), (2, 2, np.inf), (2, 2, True)])
def test_constructor_rejects_invalid_parameters(features, values, penalty):
    with pytest.raises(ValueError):
        PersistentRidge(features, values, penalty)


@pytest.mark.parametrize("X,Y", [([[1., 2.]], [[1., 2.]]), ([[1.]], [[1.]]), ([[1., 2.], [3., 4.]], [[1.]]), ([[np.nan, 2.]], [[1.]]), ([[1., 2.]], [[np.inf]])])
def test_invalid_append_is_atomic(X, Y):
    model = PersistentRidge(2, 1, 1.)
    model.append([[2., 3.]], [[4.]])
    before = model.solve()
    with pytest.raises(ValueError):
        model.append(X, Y)
    assert model.rows == 1
    np.testing.assert_array_equal(model.solve(), before)


def test_overflow_append_is_atomic():
    model = PersistentRidge(2, 1, 1.)
    with pytest.raises(FloatingPointError):
        model.append([[1e308, 0.]], [[1.]])
    assert model.rows == 0
    np.testing.assert_array_equal(model.gram, np.zeros((2, 2)))


def test_state_and_work_accounting_charges_statistics_and_decoder():
    model = PersistentRidge(3, 2, .4)
    account = model.state_accounting()
    assert account["persistent_scalars"] == 9 + 6 + 6 + 2
    assert account["persistent_bytes"] == 8 * 23
    diag = model.append(np.ones((4, 3)), np.ones((4, 2)))
    assert diag["rows_read"] == diag["total_rows"] == 4
    assert diag["arithmetic_ops_estimate"] == 2 * 4 * 3 * (3 + 2) + 9 + 6
    assert model.state_accounting() == account


@pytest.mark.parametrize("a", [[1.], [1., np.nan], [[1., 2.]]])
def test_reader_rejects_invalid_vectors(a):
    with pytest.raises(ValueError):
        PersistentRidge(2, 1, 1.).reader_gain(a)
