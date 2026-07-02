"""GPU+faiss-gated SOUNDNESS tests for the CCC certificate/escalation/outlier machinery. The hard gate is
`test_certificate_soundness`: every CERTIFIED query-block's selection must contain the exact top-κ routing
blocks — zero violations, on clustered AND random geometry, with a real (non-flat-scan) index."""
import importlib.util
import pytest
import torch

cuda = torch.cuda.is_available()
has_faiss = importlib.util.find_spec("faiss") is not None
skip = pytest.mark.skipif(not (cuda and has_faiss), reason="needs CUDA + faiss-gpu")

BLOCK, SUB = 128, 32


def _clustered(n, d=64, seed=0, spread=0.1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    nb4 = n // SUB
    nc = max(8, int(nb4 ** 0.5))
    centers = torch.randn(nc, d, generator=g, device="cuda")
    assign = torch.randint(0, nc, (nb4,), generator=g, device="cuda")
    base = centers[assign].repeat_interleave(SUB, 0)
    k = (base + spread * torch.randn(n, d, generator=g, device="cuda")).half()
    q = _clustered_q(centers, n, g)
    return q.view(1, 1, n, d), k.view(1, 1, n, d)


def _clustered_q(centers, n, g):
    nbq = n // BLOCK
    nc, d = centers.shape
    qc = centers[torch.randint(0, nc, (nbq,), generator=g, device="cuda")]
    return (qc.repeat_interleave(BLOCK, 0) + 0.2 * torch.randn(n, d, generator=g, device="cuda")).half()


def _random(n, d=64, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(1, 1, n, d, generator=g, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, n, d, generator=g, device="cuda", dtype=torch.float16)
    return q, k


def _build_cascade(k, q, top_c=6, local=1, nprobe=3, chunk_blocks=8, force_build=96, **kw):
    """Append k chunk-by-chunk so the IVF index actually builds (small threshold), returning the cascade
    after the last append, ready to route the final chunk."""
    from ssa.cascade_router import CausalCascade
    n = k.shape[2]
    cc = CausalCascade(64, block=BLOCK, sub=SUB, top_c=top_c, local=local, nprobe=nprobe,
                       chunk_blocks=chunk_blocks, n_hint=n, **kw)
    cc.build_threshold = force_build
    cb = chunk_blocks * BLOCK
    chunks = list(range(0, n, cb))
    for t in chunks[:-1]:
        cc.append(k[0, 0, t:t + cb])
        cc.route(q[0, 0, t:t + cb], qpos=t)                        # advance staging/commit
    return cc, chunks[-1]


def _bruteforce_topc(cc, qr, qb_start):
    """Exact top-κ routing blocks per query-block over ALL past sub-means (committed + staged), causal."""
    nbq = qr.shape[0]
    means = cc.means[:cc.n_sub]                                     # (n_sub, d_r) routing space
    spb = cc.spb
    nsub = means.shape[0]
    par = torch.arange(nsub, device="cuda") // spb
    s = qr @ means.T                                               # (nbq, nsub)
    nb = par.max().item() + 1
    qblk = qb_start + torch.arange(nbq, device="cuda")
    out = []
    for j in range(nbq):
        sj = s[j].clone()
        sj[par >= qblk[j]] = float("-inf")                        # strictly-past
        blk = torch.full((nb,), float("-inf"), device="cuda")
        blk.scatter_reduce_(0, par, sj, reduce="amax", include_self=True)
        top = blk.topk(min(cc.top_c, int(qblk[j]))).indices if int(qblk[j]) > 0 else torch.tensor([], device="cuda")
        out.append(set(t for t in top.tolist() if blk[t] > float("-inf")))
    return out


@skip
@pytest.mark.parametrize("geom", ["clustered", "random"])
def test_certificate_soundness(geom):
    n = 96 * BLOCK
    q, k = (_clustered(n, seed=1) if geom == "clustered" else _random(n, seed=1))
    with torch.no_grad():
        cc, last = _build_cascade(k, q)
        assert cc.index is not None, "index must build for a meaningful certificate test"
        cc.append(k[0, 0, last:])
        kn, ki, cert, _ = cc.route(q[0, 0, last:], qpos=last, certify=True)
        qr = cc.pj_q(__import__("ssa.ivf_kernel", fromlist=["block_means"]).block_means(q[0, 0, last:], BLOCK))
        exact = _bruteforce_topc(cc, qr, last // BLOCK)
    viol = 0
    for j in range(len(exact)):
        if not bool(cert[j]):
            continue
        sel = set(ki[j, :int(kn[j])].tolist())
        if not exact[j] <= sel:                                    # certified ⇒ no top-κ block missed
            viol += 1
    assert viol == 0, f"{geom}: {viol} certified rows missed a top-κ block"


@skip
def test_radius_admissible():
    """Incremental Rc upper-bounds the true per-cell max residual; exact after a rebuild."""
    n = 200 * BLOCK
    q, k = _clustered(n, seed=2)
    with torch.no_grad():
        cc, last = _build_cascade(k, q, chunk_blocks=8)
        cc.append(k[0, 0, last:]); cc._commit()
        X = cc.means[:cc.committed]
        a = cc._assign(X)
        true_R = torch.zeros(cc.nlist, device="cuda")
        true_R.scatter_reduce_(0, a, (X - cc.centroids[a]).norm(dim=1), reduce="amax", include_self=True)
    assert (cc.Rc + 1e-4 >= true_R).all(), (cc.Rc - true_R).min().item()


@skip
def test_escalation_monotone():
    """More escalation rounds never shrink the certified set."""
    n = 96 * BLOCK
    q, k = _clustered(n, seed=4)
    rates = []
    for me in (0, 1, 2):
        with torch.no_grad():
            cc, last = _build_cascade(k, q, nprobe=2, max_escalations=me)
            cc.append(k[0, 0, last:])
            _, _, cert, _ = cc.route(q[0, 0, last:], qpos=last, certify=True)
        rates.append(float(cert.float().mean()))
    assert rates[0] <= rates[1] + 1e-6 <= rates[2] + 1e-6, rates


@skip
def test_outlier_catches_norm_spike():
    """A high-norm key hidden in a block (k=c·q) is missed by sub-block routing at a tight budget but
    caught by the outlier side-channel."""
    from ssa.cascade_router import CausalCascade
    from ssa.ivf_kernel import block_means
    n = 64 * BLOCK
    g = torch.Generator(device="cuda").manual_seed(6)
    q = torch.randn(1, 1, n, 64, generator=g, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, n, 64, generator=g, device="cuda", dtype=torch.float16)
    u = torch.nn.functional.normalize(torch.randn(64, generator=g, device="cuda"), dim=0).half()
    q[0, 0, n - BLOCK:] = u                                        # last query block points at u
    target_blk = 5
    k[0, 0, target_blk * BLOCK + 17] = 12.0 * u                   # one hidden high-norm key in block 5

    def selects(outlier_cap):
        cc = CausalCascade(64, block=BLOCK, sub=SUB, top_c=2, local=0, chunk_blocks=n // BLOCK,
                           outlier_cap=outlier_cap, search_k=64)
        cc.append(k[0, 0])
        kn, ki, _, _ = cc.route(q[0, 0], qpos=0, search_k=64)
        last = n // BLOCK - 1
        return target_blk in ki[last, :int(kn[last])].tolist()

    with torch.no_grad():
        assert not selects(0), "sub-block routing alone should miss the hidden norm spike at tight budget"
        assert selects(4), "the outlier side-channel should recover the hidden norm spike"


@skip
def test_rebuild_preserves_index():
    """A warm-start rebuild keeps every committed vector searchable and re-tightens radii."""
    n = 400 * BLOCK
    q, k = _clustered(n, seed=8)
    with torch.no_grad():
        cc = _build_cascade(k, q, chunk_blocks=8, retrain_every=None)[0]
        before = cc.committed
        cc._rebuild()
    assert cc.index.ntotal == before
    X = cc.means[:cc.committed]
    a = cc._assign(X)
    true_R = torch.zeros(cc.nlist, device="cuda")
    true_R.scatter_reduce_(0, a, (X - cc.centroids[a]).norm(dim=1), reduce="amax", include_self=True)
    assert torch.allclose(cc.Rc, true_R, atol=1e-4)               # exact after rebuild
