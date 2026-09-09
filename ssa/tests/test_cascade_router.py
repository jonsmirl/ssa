"""GPU+faiss-gated mechanics for the CCC selector (ssa.cascade_router): full-budget==dense, causal/own
/unique selection, exhaustive==brute-force sub-block max-pool, chunking invariance, decode==prefill, and
the sub-block flag on block_route_budget. Certificate soundness lives in test_ccc_certificates.py."""
import importlib.util
import pytest
import torch

cuda = torch.cuda.is_available()
has_faiss = importlib.util.find_spec("faiss") is not None
skip = pytest.mark.skipif(not (cuda and has_faiss), reason="CCC needs CUDA + faiss-gpu + FlexAttention")


def _qkv(n, d=64, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(1, 1, n, d, generator=g, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, n, d, generator=g, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 1, n, d, generator=g, device="cuda", dtype=torch.float16)
    return q, k, v


def test_head_consensus_is_bounded_broadcast_union():
    from ssa.streaming_qwen import StreamingQwenPrefill
    # Block 7 is independently selected by three heads; block 9 by one. The own/local blocks 10/11
    # are unanimous. With cap=3, consensus retains 10, 11, and 7, then broadcasts them without dupes.
    idx = torch.tensor([[[[10, 11, 7, 0]], [[10, 11, 7, 0]],
                         [[10, 11, 7, 0]], [[10, 11, 9, 0]]]], dtype=torch.int32)
    num = torch.full((1, 4, 1), 3, dtype=torch.int32)
    extra_num, extra_idx = StreamingQwenPrefill._head_consensus(num, idx, 16, 3, 2)
    assert int(extra_num[0, 0]) == 3
    assert set(extra_idx[0, 0, :3].tolist()) == {7, 10, 11}
    out_num, out_idx = StreamingQwenPrefill._augment_plan(num, idx, extra_num, extra_idx, 16)
    for head in range(4):
        got = out_idx[0, head, 0, :int(out_num[0, head, 0])].tolist()
        assert len(got) == len(set(got))
        assert {7, 10, 11}.issubset(got)


def _bruteforce_subblock_topc(q, k, block, sub, top_c, local):
    """Reference selection: block score = max over sub-block ⟨q̄, μ_sub⟩, causal, top_c + own + local."""
    n, d = q.shape[2], q.shape[3]
    nb, spb = n // block, block // sub
    qb = q[0, 0].view(nb, block, d).float().mean(1)                  # (nb,d)
    mus = k[0, 0].view(nb * spb, sub, d).float().mean(1)            # (nb*spb, d) sub means
    s_sub = qb @ mus.T                                             # (nb, nb*spb)
    s_par = s_sub.view(nb, nb, spb).amax(-1)                       # max-pool -> (nb, nb) block scores
    qi = torch.arange(nb, device="cuda")
    routable = qi[:, None] > qi[None, :]
    s_par = s_par.masked_fill(~routable, float("-inf"))
    sel = torch.zeros(nb, nb, dtype=torch.bool, device="cuda")
    idx = s_par.topk(min(top_c, nb), dim=1).indices
    sel.scatter_(1, idx, s_par.gather(1, idx) > float("-inf"))
    for L in range(local + 1):
        sel |= (qi[:, None] - L == qi[None, :]) & (qi[:, None] >= qi[None, :])
    sel &= qi[:, None] >= qi[None, :]
    return sel


@skip
def test_topc_exact_ties_are_broken_by_larger_parent_index():
    """The certified target remains a named set when routing scores tie exactly."""
    from ssa.cascade_router import CausalCascade

    cc = CausalCascade(64, top_c=2, outlier_cap=0)
    scores = torch.ones(1, 4, device="cuda")
    parents = torch.tensor([[0, 3, 1, 2]], device="cuda")
    keep, tau, count = cc._select_topc(scores, parents, 1, SENT=4)
    assert keep[keep < 4].tolist() == [3, 2]
    assert float(tau[0]) == 1.0
    assert int(count[0]) == 2


@skip
def test_ccc_full_budget_matches_dense():
    from ssa.cascade_router import ccc_prefill
    from ssa.ssa_kernel import dense, BLOCK
    torch.manual_seed(0)
    n = 16 * BLOCK
    nb = n // BLOCK
    with torch.no_grad():
        q, k, v = _qkv(n)
        out_dense = dense(q, k, v)
        out, kv_num, kv_idx, _ = ccc_prefill(q, k, v, top_c=nb, local=nb, nprobe=nb,
                                             search_k=nb * 4, chunk_blocks=nb)
    assert torch.allclose(out, out_dense, atol=2e-2), (out - out_dense).abs().max().item()


@skip
def test_ccc_selection_causal_owns_block_unique():
    from ssa.cascade_router import ccc_prefill
    from ssa.ssa_kernel import BLOCK
    torch.manual_seed(0)
    n = 12 * BLOCK
    nb = n // BLOCK
    with torch.no_grad():
        q, k, v = _qkv(n)
        _, kv_num, kv_idx, _ = ccc_prefill(q, k, v, top_c=2, local=1, chunk_blocks=4)
    assert (kv_num[0, 0] >= 1).all()                                # own block always kept
    assert kv_num[0, 0].float().mean().item() < (nb + 1) / 2       # strictly sparse
    for i in range(nb):
        kn = int(kv_num[0, 0, i]); ids = kv_idx[0, 0, i, :kn]
        assert (ids <= i).all(), (i, ids.tolist())                 # no future block
        assert i in ids.tolist()                                    # own block kept
        assert len(set(ids.tolist())) == kn                         # unique


@skip
def test_ccc_exhaustive_equals_bruteforce():
    """Exhaustive probe (flat-scan regime at this n) == brute-force sub-block max-pool top-κ selection."""
    from ssa.cascade_router import CausalCascade
    from ssa.ssa_kernel import BLOCK
    torch.manual_seed(0)
    n, top_c, local, sub = 24 * BLOCK, 3, 1, 32
    nb = n // BLOCK
    q, k, v = _qkv(n, seed=3)
    with torch.no_grad():
        cc = CausalCascade(64, block=BLOCK, sub=sub, top_c=top_c, local=local, chunk_blocks=nb,
                           search_k=nb * (BLOCK // sub), outlier_cap=0)   # pure routing selection
        cc.append(k[0, 0])
        kn, ki, _, _ = cc.route(q[0, 0], qpos=0, search_k=nb * (BLOCK // sub))
        ref = _bruteforce_subblock_topc(q, k, BLOCK, sub, top_c, local)
    for i in range(nb):
        got = set(ki[i, :int(kn[i])].tolist())
        want = set(ref[i].nonzero().flatten().tolist())
        assert got == want, (i, sorted(got), sorted(want))


@skip
def test_ccc_chunking_invariance():
    """Same keys appended as small chunks vs one chunk, exhaustive probe ⇒ identical selections."""
    from ssa.cascade_router import CausalCascade
    from ssa.ssa_kernel import BLOCK
    n, top_c, sub = 16 * BLOCK, 4, 32
    nb = n // BLOCK
    q, k, v = _qkv(n, seed=5)
    sk = nb * (BLOCK // sub)

    def run(chunk_blocks):
        cc = CausalCascade(64, block=BLOCK, sub=sub, top_c=top_c, local=1, chunk_blocks=chunk_blocks,
                           search_k=sk, outlier_cap=0)              # routing selection is chunk-invariant
        W_ = top_c + 2 + 4
        kv_num = torch.empty(nb, dtype=torch.int32, device="cuda")
        kv_idx = torch.zeros(nb, top_c + 2 + 4, dtype=torch.int32, device="cuda")
        cb = chunk_blocks * BLOCK
        for t in range(0, n, cb):
            e = min(n, t + cb)
            cc.append(k[0, 0, t:e])
            kn, ki, _, _ = cc.route(q[0, 0, t:e], qpos=t, search_k=sk)
            kv_num[t // BLOCK:e // BLOCK] = kn
            kv_idx[t // BLOCK:e // BLOCK, :ki.shape[1]] = ki
        return kv_num, kv_idx

    with torch.no_grad():
        n1, i1 = run(nb)          # one chunk
        n2, i2 = run(2)           # 2-block chunks
    assert torch.equal(n1, n2)
    for i in range(nb):
        assert set(i1[i, :int(n1[i])].tolist()) == set(i2[i, :int(n2[i])].tolist()), i


@skip
def test_streaming_gqa_router_matches_whole_tensor_driver():
    """The >10M streaming surface is exactly the same selector contract as ccc_route_gqa."""
    from ssa.cascade_router import StreamingGQARouter, ccc_route_gqa
    from ssa.ssa_kernel import BLOCK
    g = torch.Generator(device="cuda").manual_seed(17)
    n, hq, hkv, d, cb = 8 * BLOCK, 4, 2, 64, 2 * BLOCK
    q = torch.randn(1, hq, n, d, generator=g, device="cuda", dtype=torch.float16)
    k = torch.randn(1, hkv, n, d, generator=g, device="cuda", dtype=torch.float16)
    cfg = dict(top_c=3, local=1, sub=32, chunk_blocks=2, search_k=32,
               build_threshold=10_000, outlier_cap=0)
    with torch.no_grad():
        want_n, want_i, _ = ccc_route_gqa(q, k, block=BLOCK, **cfg)
        stream = StreamingGQARouter(1, hq, hkv, n, d, block=BLOCK, **cfg)
        got_n = torch.empty_like(want_n)
        got_i = torch.empty_like(want_i)
        for start in range(0, n, cb):
            stop = start + cb
            kn, ki, _ = stream.route_chunk(q[:, :, start:stop], k[:, :, start:stop], start)
            got_n[:, :, start // BLOCK:stop // BLOCK] = kn
            got_i[:, :, start // BLOCK:stop // BLOCK] = ki
    assert torch.equal(got_n, want_n)
    for h in range(hq):
        for b in range(n // BLOCK):
            count = int(got_n[0, h, b])
            assert set(got_i[0, h, b, :count].tolist()) == \
                   set(want_i[0, h, b, :count].tolist())


@skip
def test_torch_tree_backend_full_budget_is_exact_and_causal():
    """The Blackwell fallback hierarchy recovers the complete causal prefix at full budget."""
    from ssa.cascade_router import ccc_route_gqa
    from ssa.ssa_kernel import BLOCK
    n, nb = 16 * BLOCK, 16
    q, k, _ = _qkv(n, seed=19)
    with torch.no_grad():
        kn, ki, _ = ccc_route_gqa(
            q, k, backend="tree", top_c=nb, local=nb, sub=32,
            chunk_blocks=4, search_k=nb * 4, tree_beam=nb * 4, outlier_cap=0,
        )
    for i in range(nb):
        got = set(ki[0, 0, i, :int(kn[0, 0, i])].tolist())
        assert got == set(range(i + 1)), (i, sorted(got))


@skip
def test_torch_tree_max_pools_query_and_key_subblocks():
    """A shared 128-query mask represents a union, so routing must max-pool both sides."""
    from ssa.cascade_router import ccc_route_gqa
    from ssa.ssa_kernel import BLOCK
    n, nb, sub, top_c = 12 * BLOCK, 12, 32, 3
    q, k, _ = _qkv(n, seed=29)
    with torch.no_grad():
        kn, ki, _ = ccc_route_gqa(
            q, k, backend="tree", top_c=top_c, local=0, sub=sub, query_sub=sub,
            chunk_blocks=nb, search_k=nb * (BLOCK // sub),
            tree_beam=nb * (BLOCK // sub), outlier_cap=0,
        )
        qsub = q[0, 0].view(nb, BLOCK // sub, sub, -1).float().mean(2)
        ksub = k[0, 0].view(nb, BLOCK // sub, sub, -1).float().mean(2)
        score = torch.einsum("qrd,ksd->qkrs", qsub, ksub).amax((2, 3))
        causal = torch.arange(nb, device="cuda")[:, None] > torch.arange(nb, device="cuda")[None]
        score.masked_fill_(~causal, float("-inf"))
    for query_block in range(nb):
        got = set(ki[0, 0, query_block, :int(kn[0, 0, query_block])].tolist())
        want = {query_block}
        if query_block:
            count = min(top_c, query_block)
            want.update(score[query_block].topk(count).indices.tolist())
        assert got == want, (query_block, sorted(got), sorted(want))


@skip
def test_ccc_decode_step_matches_prefill_row():
    from ssa.cascade_router import CausalCascade, ccc_prefill
    from ssa.ivf_decode import decode_attend
    from ssa.ssa_kernel import BLOCK
    torch.manual_seed(0)
    n = 12 * BLOCK
    nb = n // BLOCK
    sk = nb * 4
    with torch.no_grad():
        q, k, v = _qkv(n, seed=7)
        out, kv_num, kv_idx, _ = ccc_prefill(q, k, v, top_c=4, local=1, chunk_blocks=nb, search_k=sk)
        last = nb - 1
        blocks = kv_idx[0, 0, last, :int(kv_num[0, 0, last])].tolist()
        row = decode_attend(q[0, 0, n - 1], k[0, 0], v[0, 0], blocks, pos=n - 1, block=BLOCK)
    assert torch.allclose(row, out[0, 0, n - 1], atol=2e-2), (row - out[0, 0, n - 1]).abs().max().item()


@skip
def test_block_route_budget_sub_flag_identity_and_recall():
    """sub=None byte-identical to current; sub=32 keeps invariants and finds a planted sub-block needle."""
    from ssa.ssa_kernel import block_route_budget, BLOCK
    torch.manual_seed(0)
    n = 8 * BLOCK
    q, k, v = _qkv(n, seed=9)
    a = block_route_budget(q, k, top_c=2, local=1)
    b = block_route_budget(q, k, top_c=2, local=1, sub=None)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])     # sub=None == default
    # planted: one coherent 32-sub-block of block 1 aligned with the (coherent) last query block.
    # Sub-block routing sees a sub-mean of ~u (score ~1); 128-block routing dilutes it 4× (score ~0.25).
    q2, k2, _ = _qkv(n, seed=11)
    u = torch.nn.functional.normalize(torch.randn(64, device="cuda"), dim=0).half()
    q2[0, 0, n - BLOCK:] = u                                        # whole last query block points at u
    k2[0, 0, BLOCK:BLOCK + 32] = u                                  # block 1, sub-block 0 fully aligned
    qblk = (n - 1) // BLOCK
    _, _, sub_sel = block_route_budget(q2, k2, top_c=1, local=0, sub=32)
    _, _, blk_sel = block_route_budget(q2, k2, top_c=1, local=0, sub=None)
    assert bool(sub_sel[0, 0, qblk, 1]), "sub-block routing should find the coherent sub-block's parent"
    # the sub-block score (~1.0) strictly exceeds the diluted 128-block score (~0.25)


@skip
def test_starved_probe_pads_never_reach_the_mask():
    """faiss pads a starved probe (probed lists jointly holding < search_k vectors) with id -1 and score
    ~-3.4e38 — NOT -inf, so it passed the scores>NEG guard and could emit parent block -1 into the
    BlockMask. Starve deterministically (fixed one-hot centroids, 4 members/cell, nprobe=1, search_k=16)
    and pin: pads are masked to NEG with non-negative ids, D_last follows the exhaustive convention,
    and route() emits only real past blocks."""
    from ssa.cascade_router import CausalCascade, NEG
    d, nlist, per = 32, 8, 4
    cc = CausalCascade(d=d, block=128, sub=32, top_c=8, local=1, nprobe=1,
                       nlist=nlist, max_escalations=0, search_k=16)
    C = torch.eye(nlist, d, device="cuda")                          # one centroid per one-hot direction
    g = torch.Generator(device="cuda").manual_seed(0)
    X = C.repeat_interleave(per, 0) + 0.01 * torch.randn(nlist * per, d, generator=g, device="cuda")
    cc._kmeans = lambda X, iters=6: C                               # fixed centroids -> exactly 4 per cell
    cc.means = X.float(); cc.n_sub = X.shape[0]; cc.committed = X.shape[0]
    cc._build_index()
    dvals, ids, dlast = cc._search_committed(C[:1].clone(), search_k=16, nprobe=1)
    assert (dvals == NEG).any(), "construction failed to starve the probe"
    assert (ids >= 0).all()
    assert dlast[0] == NEG                                          # exhaustive: nothing lost to truncation
    q_chunk = torch.randn(cc.block, d, generator=g, device="cuda", dtype=torch.float16) + 4 * C[0].half()
    kv_num, kv_idx, cert, _ = cc.route(q_chunk, qpos=cc.n_sub * cc.sub, certify=True)
    nb_global = cc.n_sub * cc.sub // cc.block + 1
    assert (kv_idx >= 0).all() and (kv_idx < nb_global).all()
    assert not bool(cert[0])                                        # starved + tiny budget: fails closed


@skip
def test_outlier_reservoir_has_a_fixed_global_cap():
    """A fixed reservoir prevents the optional outlier lookup from growing with context length."""
    from ssa.cascade_router import CausalCascade
    from ssa.ssa_kernel import BLOCK
    _, k, _ = _qkv(8 * BLOCK, seed=13)
    cc = CausalCascade(64, block=BLOCK, sub=32, top_c=2, chunk_blocks=2,
                       outlier_rate=0.25, outlier_cap=2, outlier_store_cap=7)
    with torch.no_grad():
        for start in range(0, 8 * BLOCK, 2 * BLOCK):
            cc.append(k[0, 0, start:start + 2 * BLOCK])
            assert cc.n_out <= 7
    assert cc.n_out == 7
    assert cc.O.shape[0] == cc.O_parent.shape[0] == cc.O_pos.shape[0] == cc.O_score.shape[0] == 7
