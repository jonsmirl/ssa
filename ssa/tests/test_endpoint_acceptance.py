import pytest
import torch
from ssa.endpoint_acceptance import accept_logits


def test_acceptance_preserves_reference_without_target_labels():
    r = torch.tensor([[3., 1., 0.], [3., 1., 0.], [3., 3., 0.], [3., 1., 0.]])
    c = torch.tensor([[4., 1., 0.], [1., 4., 0.], [4., 1., 0.], [4., 4., 0.]])
    out, info = accept_logits(r, c)
    assert info["accepted"].tolist() == [True, False, False, False]
    assert torch.equal(out.argmax(-1), r.argmax(-1))
    assert torch.equal(out[1:], r[1:])


def test_subset_does_not_claim_untested_vocabulary_preservation():
    r = torch.tensor([[4., 2., 1.]])
    c = torch.tensor([[3., 2., 9.]])
    out, info = accept_logits(r, c, [0, 1])
    assert info["accepted"].item() and out.argmax(-1).item() == 2
    _, full = accept_logits(r, c)
    assert not full["accepted"].item()


def test_nonfinite_candidate_fails_closed_and_bad_reference_errors():
    r = torch.tensor([[2., 0., 1.]])
    for bad in (float("nan"), float("inf"), -float("inf")):
        c = torch.tensor([[3., 0., bad]])
        out, info = accept_logits(r, c, [0, 1])
        assert not info["accepted"].item() and torch.equal(out, r)
    with pytest.raises(ValueError):
        accept_logits(torch.tensor([[float("nan"), 0.]]), torch.zeros(1, 2))


@pytest.mark.parametrize("indices", [[0, 0], [-1, 1], [0, 3], [0], [0., 1.]])
def test_invalid_candidate_sets(indices):
    with pytest.raises(ValueError):
        accept_logits(torch.ones(2, 3), torch.ones(2, 3), indices)


def test_extreme_finite_logits_do_not_require_subtraction():
    f = torch.finfo(torch.float32).max
    r = torch.tensor([[f, -f]])
    out, info = accept_logits(r, r)
    assert info["accepted"].item() and torch.equal(out, r)
