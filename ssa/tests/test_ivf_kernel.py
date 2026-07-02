"""GPU+faiss-gated correctness for the IVF-routed SSA kernel (ssa.ivf_kernel).

Mirrors test_ssa_kernel.py: full-budget == dense, causal+sparse selection, plus IVF-specific agreement
with the flat block_route on clustered geometry. The decode-step primitive test lives in
test_ivf_decode.py (it exercises ssa.ivf_decode alongside this kernel)."""
import importlib.util
import pytest
import torch

cuda = torch.cuda.is_available()
has_faiss = importlib.util.find_spec("faiss") is not None
skip = pytest.mark.skipif(not (cuda and has_faiss), reason="IVF kernel needs CUDA + faiss-gpu + FlexAttention")


def _clustered_keys(B, H, n, d, block, seed=0):
    """Keys whose per-block means are clustered (the geometry where block routing is meaningful — on
    isotropic randn, agreement between any two routers is vacuously low; see faiss_router.py)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    nb = n // block
    nc = max(4, int(nb ** 0.5))
    centers = torch.randn(nc, d, generator=g, device="cuda")
    assign = torch.randint(0, nc, (nb,), generator=g, device="cuda")
    block_centers = centers[assign]                                   # (nb, d)
    k = block_centers.repeat_interleave(block, 0) + 0.1 * torch.randn(n, d, generator=g, device="cuda")
    return k.to(torch.float16).view(1, 1, n, d).expand(B, H, n, d).contiguous()


@skip
def test_ivf_full_budget_matches_dense():
    """Exhaustive IVF (every causal block selectable) == dense causal attention."""
    from ssa.ivf_kernel import ssa_flex_ivf
    from ssa.ssa_kernel import dense, BLOCK
    torch.manual_seed(0)
    n = 16 * BLOCK
    nb = n // BLOCK
    with torch.no_grad():
        q = torch.randn(1, 1, n, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q); v = torch.randn_like(q)
        out_dense = dense(q, k, v)
        out_ivf = ssa_flex_ivf(q, k, v, top_c=nb, local=nb, nprobe=nb, search_k=nb)   # select everything
    assert torch.allclose(out_ivf, out_dense, atol=2e-2)


@skip
def test_ivf_selection_is_causal_and_owns_its_block():
    """A tight budget selects a strict causal subset; every query block keeps at least its own block."""
    from ssa.ivf_kernel import ivf_route
    from ssa.ssa_kernel import BLOCK
    torch.manual_seed(0)
    n = 8 * BLOCK
    nb = n // BLOCK
    with torch.no_grad():
        q = torch.randn(1, 1, n, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        kv_num, kv_idx = ivf_route(q, k, top_c=2, local=1)
    qi = torch.arange(nb, device="cuda")
    assert (kv_num >= 1).all()                                        # own block always present
    assert kv_num.float().mean().item() < (nb + 1) / 2               # strictly sparse
    for i in range(nb):
        kn = int(kv_num[0, 0, i]); ids = kv_idx[0, 0, i, :kn]
        assert (ids <= i).all()                                       # no future block
        assert i in ids.tolist()                                      # own block kept
        assert len(set(ids.tolist())) == kn                           # unique


@skip
def test_ivf_agreement_with_flat_router():
    """On clustered geometry, IVF block selection overlaps the flat block_route selection (Jaccard)."""
    from ssa.ivf_kernel import ivf_route
    from ssa.ssa_kernel import block_route, BLOCK
    torch.manual_seed(0)
    n = 64 * BLOCK
    nb = n // BLOCK
    top_c = 8
    q = _clustered_keys(1, 1, n, 64, BLOCK, seed=1)                   # queries also clustered
    k = _clustered_keys(1, 1, n, 64, BLOCK, seed=2)
    with torch.no_grad():
        kv_num, kv_idx = ivf_route(q, k, top_c=top_c, local=1, nprobe=8)
        _, _, sel = block_route(q, k, top_c=top_c, local=1)
    inter = 0.0
    rows = 0
    for i in range(top_c, nb):                                        # rows with a real choice to make
        ivf_set = set(kv_idx[0, 0, i, :int(kv_num[0, 0, i])].tolist())
        flat_set = set(sel[0, 0, i].nonzero().flatten().tolist())
        inter += len(ivf_set & flat_set) / max(1, len(ivf_set | flat_set))
        rows += 1
    jaccard = inter / rows
    assert jaccard >= 0.6, f"IVF/flat Jaccard {jaccard:.3f} < 0.6"
