"""Mean-consistent residual mass and bounded-influence output correction.

For an omitted cell T with m keys, Jensen gives
  m exp(beta <q, mean_T(k)> + g) <= sum_T exp(beta <q,k>)
when g <= 0. A fixed assignment prototype need not obey that inequality.
This is a LOWER bound, not the upper cap required by the certified reader.

With tail mean u and selected output oS, output = oS + alpha*(u-oS).
Capping alpha at rho bounds movement by rho*||u-oS||. It does not prove
improvement: error improves iff 2<o-oS,u-oS> >= alpha*||u-oS||^2 (alpha>0).
"""
import torch


def jensen_cell_logmass(q, counts, key_sums, log_gain=0.0):
    """Aligned [...,d], [...,cells], [...,cells,d]; real-arithmetic lower mass."""
    gain = torch.as_tensor(log_gain, dtype=q.dtype, device=q.device).clamp_max(0)
    means = key_sums / counts.clamp_min(1)[..., None]
    score = (q[..., None, :] * means).sum(-1) / q.shape[-1]**0.5
    return (score + gain + counts.clamp_min(1).log()).masked_fill(counts <= 0, -torch.inf)


def mix_tail_read(sparse, log_z, log_mass, means, max_tail_share=None):
    """Positive mixed output and actual applied share; no quality certificate."""
    if max_tail_share is not None:
        if not 0 <= max_tail_share <= 1:
            raise ValueError("max_tail_share must lie in [0,1]")
        if max_tail_share == 0:
            return sparse, torch.zeros_like(log_z)
        if max_tail_share < 1:
            # logsumexp of an all -inf row has undefined derivatives, even
            # when a later clamp/mask makes its forward contribution zero.
            # Use a constant finite row only for empty-tail reductions.
            has_tail = (log_mass != -torch.inf).any(-1)
            reduction_mass = log_mass.masked_fill(~has_tail[..., None], 0)
            # Form the scalar mixture directly. Subtracting a huge cap shift
            # from each huge log mass can erase logit(rho) in float32 and
            # silently violate the requested influence limit.
            alpha = (reduction_mass.logsumexp(-1) - log_z).sigmoid()
            alpha = alpha.clamp_max(max_tail_share).masked_fill(~has_tail, 0)
            tail_output = (reduction_mass.softmax(-1)[..., None] * means).sum(-2)
            return (1 - alpha[..., None]) * sparse + alpha[..., None] * tail_output, alpha
    weights = torch.cat((log_z[..., None], log_mass), -1).softmax(-1)
    output = weights[..., :1] * sparse + (weights[..., 1:, None] * means).sum(-2)
    return output, weights[..., 1:].sum(-1)
