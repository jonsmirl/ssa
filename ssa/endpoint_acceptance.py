"""Charged, exact-endpoint acceptance for computed floating-point logits.

This compares two already executed models. It is NOT an instantiation of the
uniform-derivative/path certificate, nor a certificate about exact-real logits.
The decision uses no target label and preserves the reference's unique winner
on precisely the supplied candidate set. Reference correctness is not implied.
Both trajectories must be maintained independently if used autoregressively.
"""
import torch


def accept_logits(reference, candidate, candidates=None):
    """Return selected logits and per-row acceptance metadata.

    Leading dimensions index independent decisions. Nonfinite reference logits
    are an error; nonfinite candidates fail closed. Reference/candidate ties fail
    closed. A restricted candidate set does not protect untested vocabulary.
    """
    if (reference.shape != candidate.shape or reference.ndim < 1
            or not reference.is_floating_point() or not candidate.is_floating_point()
            or reference.device != candidate.device):
        raise ValueError("aligned floating-point logits on the same device required")
    if not bool(torch.isfinite(reference).all()):
        raise ValueError("nonfinite reference logits cannot supply a reference prediction")
    if candidates is None:
        r, c = reference, candidate
    else:
        indices = torch.as_tensor(candidates, device=reference.device)
        if (indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64)
                or indices.numel() != indices.unique().numel()
                or bool((indices < 0).any()) or bool((indices >= reference.shape[-1]).any())):
            raise ValueError("candidate indices must be distinct valid integer indices")
        r, c = reference.index_select(-1, indices.long()), candidate.index_select(-1, indices.long())
    if r.shape[-1] < 2:
        raise ValueError("at least two candidates required by this reference implementation")
    reference_choice, candidate_choice = r.argmax(-1), c.argmax(-1)
    rt, ct = r.topk(2, dim=-1).values, c.topk(2, dim=-1).values
    reference_strict = rt[..., 0] > rt[..., 1]
    candidate_strict = ct[..., 0] > ct[..., 1]
    accepted = (torch.isfinite(candidate).all(-1) & reference_strict & candidate_strict
                & (reference_choice == candidate_choice))
    selected = torch.where(accepted[..., None], candidate, reference)
    return selected, {"accepted": accepted, "reference_choice": reference_choice,
                      "candidate_choice": candidate_choice, "reference_strict": reference_strict,
                      "candidate_strict": candidate_strict}
