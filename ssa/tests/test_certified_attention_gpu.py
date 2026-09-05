"""Check the CPU certificate against CUDA SDPA on identical input values.

Float64 math SDPA is the independent dense oracle. This verifies the certificate's
mathematics on CUDA; it is not a GPU implementation or a low-precision error bound.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ssa.certified_attention import CertifiedBlockAttention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="certificate oracle needs CUDA")
@pytest.mark.parametrize("geometry", ["concentrated", "flat", "equal_values"])
@pytest.mark.parametrize("prefix", [1024, 997])
def test_certificate_bounds_cuda_dense_attention(geometry, prefix):
    rng = np.random.default_rng(42)
    n, d, dv, block, beta = 1024, 32, 8, 32, 4.0
    K = 0.01 * rng.normal(size=(n, d))
    K[:block, 0] += 4
    V = rng.normal(size=(n, dv))
    q = np.eye(d)[0]
    if geometry != "concentrated":
        K[:] = 0
    if geometry == "equal_values":
        V[:] = 1
    kwargs = {"error_tol": 1e-12} if geometry == "equal_values" else {"mass_tol": 1e-3}
    result = CertifiedBlockAttention(K, V, block).read(q, beta, prefix=prefix, **kwargs)
    assert result.certified
    Qg = torch.tensor(q, device="cuda").view(1, 1, 1, d)
    Kg = torch.tensor(K[:prefix], device="cuda").view(1, 1, prefix, d)
    Vg = torch.tensor(V[:prefix], device="cuda").view(1, 1, prefix, dv)
    ids = torch.tensor(result.indices, device="cuda")
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        # Explicit prefix slicing: is_causal=True with q_len=1 would keep only key 0.
        dense = F.scaled_dot_product_attention(Qg, Kg, Vg, scale=beta)
        sparse = F.scaled_dot_product_attention(
            Qg, Kg.index_select(2, ids), Vg.index_select(2, ids), scale=beta)
    np.testing.assert_allclose(sparse.cpu().numpy().ravel(), result.output, atol=2e-12)
    error = np.linalg.norm(dense.cpu().numpy().ravel() - result.output)
    assert error <= result.output_error_upper + 2e-12
    weights = torch.softmax(beta * (Kg[0, 0] @ Qg[0, 0, 0]), dim=0)
    omitted = torch.ones(prefix, dtype=torch.bool, device="cuda")
    omitted[ids] = False
    assert float(weights[omitted].sum()) <= result.mass_upper + 2e-12
    kl = -float(torch.log(weights[ids].sum()))
    assert kl <= result.kl_upper + 2e-12
    if geometry == "flat":
        assert result.keys_scored == prefix
    else:
        assert result.keys_scored <= block + prefix % block
