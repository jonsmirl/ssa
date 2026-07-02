"""GPU+faiss-gated test that the decode primitive (ssa.ivf_decode.decode_attend) reproduces the IVF
kernel's prefill row for the same selected blocks — the correctness pin under the decode benchmark."""
import importlib.util
import pytest
import torch

cuda = torch.cuda.is_available()
has_faiss = importlib.util.find_spec("faiss") is not None
skip = pytest.mark.skipif(not (cuda and has_faiss), reason="needs CUDA + faiss-gpu + FlexAttention")


@skip
def test_decode_step_matches_prefill_last_row():
    from ssa.ivf_kernel import ivf_route, ssa_flex_ivf
    from ssa.ivf_decode import decode_attend
    from ssa.ssa_kernel import BLOCK
    torch.manual_seed(0)
    n = 12 * BLOCK
    with torch.no_grad():
        q = torch.randn(1, 1, n, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q); v = torch.randn_like(q)
        prefill = ssa_flex_ivf(q, k, v, top_c=4, local=1)             # (1,1,n,64)
        kv_num, kv_idx = ivf_route(q, k, top_c=4, local=1)            # same routing the prefill used
        last = n // BLOCK - 1
        blocks = kv_idx[0, 0, last, :int(kv_num[0, 0, last])].tolist()
        row = decode_attend(q[0, 0, n - 1], k[0, 0], v[0, 0], blocks, pos=n - 1, block=BLOCK)
    assert torch.allclose(row, prefill[0, 0, n - 1], atol=2e-2), (row - prefill[0, 0, n - 1]).abs().max()
