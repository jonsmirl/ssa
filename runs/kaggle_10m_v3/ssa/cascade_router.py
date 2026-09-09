"""
The Certified Causal Cascade (CCC) selector — one streaming, chunked-causal selector that composes the
five ingredients a quality-preserving cheap selector needs, each individually evidenced by P0–P6:

  1. a shared (optionally trained, low-dim) ROUTING SPACE — `proj`;
  2. SUB-BLOCK max-pooled summaries — routing metric s(i,B) = max_{sub∈B} ⟨q̄_i, μ_sub⟩ (sub=32 means
     a spike is divided by 32, not 128, before it is averaged away — 4× more visible, zero kernel cost);
  3. a CHUNKED-CAUSAL streaming index — the faiss index holds only COMMITTED PAST sub-block means, so
     selection is causal by construction; the in-flight chunk is scored by an exact causally-masked flat
     GEMM, and the partial tail block by the own+local OR. Prefill and decode share one code path;
  4. an OUTLIER side-channel (Phase B) — high-leverage keys indexed exactly, defeating the k=c·q
     impossibility construction (does NOT rescue unit-norm isolated needles — that's 2+5);
  5. per-query CERTIFICATES + escalation (Phase B) — an admissible bound over unprobed cells; certified
     ⇒ the selected top-κ parent blocks provably equal the exact top-κ UNDER THE ROUTING METRIC (not
     attention-output error); uncertified ⇒ escalate that query only.

Emits the same compressed `(kv_num, kv_idx)` contract as `ivf_kernel._route_head`, consumed by the same
`_build_mask` (from_kv_blocks, compute_q_blocks=False) and the same compiled `_flex`. `ivf_kernel.py` is
the untouched baseline this is compared against. faiss stays outside compiled regions.

Run:  python3 -m ssa.cascade_router                 # -> paper/figures/ccc_kernel.json
"""
from __future__ import annotations
import time
import torch
from torch.nn.attention.flex_attention import BlockMask
from ssa.ssa_kernel import BLOCK, _causal_mod, _flex, dense
from ssa.ivf_kernel import block_means, _build_mask

try:
    import faiss
    import faiss.contrib.torch_utils
except ImportError as e:                      # faiss optional — keep ssa_kernel import-clean
    raise ImportError("ssa.cascade_router needs faiss-gpu (pip install faiss-gpu-cu12).") from e

DEV = "cuda" if torch.cuda.is_available() else "cpu"
_RES = None
NEG = float("-inf")


def _gpu_res(temp_mb=512):
    global _RES
    if _RES is None:
        _RES = faiss.StandardGpuResources()
        _RES.setTempMemory(temp_mb * 1024 * 1024)
    return _RES


def sub_block_means(x, block=BLOCK, sub=32, chunk=1 << 20):
    """(n, d) fp16 CUDA -> (n//sub, d) fp32 sub-block means (fp32-accumulated, chunked — no fp32 copy)."""
    return block_means(x, block=sub, chunk=chunk)                    # sub-blocks are just smaller blocks


def _as_projectors(proj, d):
    """Normalize `proj` to (project_q, project_k, d_r). proj is None | (d,d_r) tensor | obj with
    project_q/project_k. A bare tensor is symmetric (applied to both q and k)."""
    if proj is None:
        return (lambda t: t), (lambda t: t), d
    if hasattr(proj, "project_q"):
        d_r = proj.project_k(torch.zeros(1, d, device=DEV)).shape[-1]
        return proj.project_q, proj.project_k, d_r
    W = proj.to(DEV).float()
    return (lambda t: t @ W), (lambda t: t @ W), W.shape[1]


class CausalCascade:
    """Streaming selector. Usage: for each chunk of `chunk_blocks` 128-blocks, call `append(k_chunk)`
    then `route(q_chunk, qpos)`. `ccc_prefill` drives the loop and returns one BlockMask + one flex call."""

    def __init__(self, d, block=BLOCK, sub=32, top_c=8, local=1, nprobe=4,
                 chunk_blocks=2048, retrain_every=None, d_r=None, proj=None, n_hint=None,
                 outlier_rate=1e-3, outlier_cap=4, cert_margin=0.0, max_escalations=2,
                 escalate_factor=4, search_k=None, nlist=None, res=None, dtype=torch.float16,
                 build_threshold=None, outlier_store_cap=None):
        self.d, self.block, self.sub = d, block, sub
        self.spb = block // sub                                      # sub-blocks per 128-block (=4)
        self.top_c, self.local, self.nprobe = top_c, local, nprobe
        self.chunk_blocks = chunk_blocks
        self.retrain_every = retrain_every
        self.outlier_rate, self.outlier_cap = outlier_rate, outlier_cap
        self.outlier_store_cap = outlier_store_cap
        self.cert_margin, self.max_escalations, self.escalate_factor = cert_margin, max_escalations, escalate_factor
        self.search_k = search_k or 4 * top_c
        self.res = res or _gpu_res()
        self.dtype = dtype
        self.pj_q, self.pj_k, self.d_r = _as_projectors(proj, d)
        self.dr = self.d_r
        # nlist frozen from n_hint (drift-only rebuilds) or grown lazily
        if nlist is not None:
            self.nlist = nlist
        elif n_hint is not None:
            self.nlist = max(4, int((n_hint / sub) ** 0.5))
        else:
            self.nlist = None
        self.build_threshold = (max(4 * (self.nlist or 4), 16384)
                                if build_threshold is None
                                else max(int(build_threshold), self.nlist or 4))
        if self.build_threshold < 1:
            raise ValueError("build_threshold must be positive")
        # state
        self.means = None                                           # (n_sub_max, d_r) fp32 routing-space sub-means
        self.n_sub = 0                                              # sub-means computed
        self.committed = 0                                          # sub-means in the index / flat-committed
        self.index = None
        self.centroids = None                                       # (nlist, d_r) our torch copy (certificates)
        self.Rc = None                                              # (nlist,) per-cell radius (admissible)
        self.cell_of = None                                         # (n_sub,) cell assignment
        self.committed_at_rebuild = 0
        self.chunks_since_rebuild = 0
        # outlier side-channel buffers (Phase B populates)
        self.O = None                                               # (s, d_r) routing-space outlier keys
        self.O_parent = None                                        # (s,) parent 128-block id
        self.O_pos = None                                           # (s,) token position
        self.O_score = None                                         # (s,) leverage used by capped retention
        self.n_out = 0

    # -- summaries / staging / commit ------------------------------------------------------------

    def _ensure_means(self, add_sub):
        need = self.n_sub + add_sub
        if self.means is None:
            self.means = torch.empty(need, self.dr, device=DEV, dtype=torch.float32)
        elif need > self.means.shape[0]:
            new = torch.empty(int(need * 1.3) + 1, self.dr, device=DEV, dtype=torch.float32)
            new[:self.n_sub] = self.means[:self.n_sub]
            self.means = new

    def append(self, k_chunk):
        """Commit the previously-staged chunk into the index, then summarize + stage this chunk."""
        self._commit()
        m4 = k_chunk.shape[0] // self.sub                           # sub-blocks in this chunk
        mu = self.pj_k(sub_block_means(k_chunk, self.block, self.sub))   # (m4, d_r) routing space
        self._ensure_means(m4)
        self.means[self.n_sub:self.n_sub + m4] = mu
        self._extract_outliers(k_chunk, mu)                         # Phase B (no-op in A)
        self.n_sub += m4

    def _commit(self):
        """Push staged sub-means [committed:n_sub] into the index (or flat-committed region)."""
        if self.n_sub <= self.committed:
            return
        X = self.means[self.committed:self.n_sub]
        if self.index is None:
            if self.n_sub >= self.build_threshold:
                self.committed = self.n_sub
                self._build_index()                                # build over everything committed
            else:
                self.committed = self.n_sub                        # flat-scan regime: exhaustive, no index
                return
        else:
            a = self._assign(X)                                    # my centroids == faiss's ⇒ same assignment
            self.index.add(X)
            self.committed = self.n_sub
            self._update_radii(X, a)                               # incremental (upper bound until rebuild)
            self.chunks_since_rebuild += 1
            self._maybe_rebuild()

    def _assign(self, X):
        """Coarse assignment = argmax IP against OUR centroids. Because faiss's index is built via copyFrom
        with exactly these centroids (never faiss-trained), faiss's IVF-IP coarse quantizer assigns the same
        way (same centroids, same max-IP rule, same lowest-index tie-break) — so radii cover exactly the
        members faiss searches, and the certificate's probed set matches faiss's. This equality is the
        soundness pin (test_certificate_soundness). Caveat: it assumes exact fp tie-parity between torch's
        and faiss's GEMM reductions — duplicated / near-tie keys can land a vector's radius in a different
        cell than the list faiss searches; set cert_margin > 0 to absorb near-ties on adversarial data."""
        return (X @ self.centroids.T).argmax(1)

    def _kmeans(self, X, iters=6):
        """IP-Lloyd, warm-started from current centroids (drift-only rebuilds) or random rows (cold)."""
        if self.centroids is not None and self.centroids.shape[0] == self.nlist:
            C = self.centroids.clone()
        else:
            g = torch.Generator(device=DEV).manual_seed(0)
            C = X[torch.randperm(X.shape[0], generator=g, device=DEV)[:self.nlist]].clone()
        for _ in range(iters):
            a = (X @ C.T).argmax(1)
            sums = torch.zeros_like(C).index_add_(0, a, X)
            cnt = torch.zeros(self.nlist, device=DEV).index_add_(0, a, torch.ones(X.shape[0], device=DEV))
            nz = cnt > 0
            C[nz] = sums[nz] / cnt[nz, None]
        return C

    def _build_index(self, iters=6):
        """(Re)build the GPU IVF over committed means with OUR centroids (torch IP-kmeans -> CPU shell ->
        copyFrom -> bulk add), then compute exact per-cell radii. Used for the first build and every
        warm-start rebuild — one code path, so faiss never trains and our centroids stay authoritative."""
        X = self.means[:self.committed]
        if self.nlist is None:
            self.nlist = max(4, int(self.committed ** 0.5))
        C = self._kmeans(X, iters)
        quant = faiss.IndexFlatIP(self.dr)
        quant.add(C.detach().cpu().contiguous())                   # centroids on CPU for the shell
        cpu = faiss.IndexIVFFlat(quant, self.dr, self.nlist, faiss.METRIC_INNER_PRODUCT)
        cpu.is_trained = True                                       # centroids provided -> skip faiss train()
        gpu = faiss.GpuIndexIVFFlat(self.res, self.dr, self.nlist, faiss.METRIC_INNER_PRODUCT)
        gpu.copyFrom(cpu)
        gpu.add(X); gpu.nprobe = min(self.nprobe, self.nlist)
        self.index = gpu
        self.centroids = C
        a = self._assign(X)
        self.cell_of = torch.full((self.committed,), -1, device=DEV, dtype=torch.long)
        self.cell_of[:] = a
        self.Rc = torch.zeros(self.nlist, device=DEV)
        self._update_radii(X, a)                                    # exact per-cell radius
        self.committed_at_rebuild = self.committed
        self.chunks_since_rebuild = 0

    def _update_radii(self, X, a):
        r = (X - self.centroids[a]).norm(dim=1)                     # ‖μ_sub − c‖ per member
        self.Rc.scatter_reduce_(0, a, r, reduce="amax", include_self=True)

    def _maybe_rebuild(self):
        if self.retrain_every is not None and self.retrain_every <= 0:
            return
        due = (self.chunks_since_rebuild >= self.retrain_every) if self.retrain_every is not None \
            else (self.committed >= 2 * max(1, self.committed_at_rebuild))
        if due:
            self._rebuild()

    def _rebuild(self):
        """Warm-start recluster + exact radii refresh (incremental radii only upper-bound between rebuilds)."""
        if self.index is None:
            return
        self._build_index()

    # -- outliers (Phase B) ----------------------------------------------------------------------

    def _extract_outliers(self, k_chunk, mu):
        """Leverage = ‖k − μ_sub‖ in ROUTING space (free here: μ_sub is exactly the routing summary; catches
        k=c·q at ≈c‖q‖ — a pure-norm rule would miss mean-cancelled keys). Keep a per-chunk quota so the
        buffer is temporally uniform. Stored: (routing-space key vector, parent 128-block id, token pos)."""
        if self.outlier_rate <= 0 or self.outlier_cap <= 0:
            return
        m_tok = k_chunk.shape[0]
        base = self.n_sub * self.sub                               # global token index of this chunk's start
        kr = self.pj_k(k_chunk.float())                            # (m_tok, d_r)
        resid = (kr - mu[torch.arange(m_tok, device=DEV) // self.sub]).norm(dim=1)
        quota = max(1, int(self.outlier_rate * m_tok))
        val, top = resid.topk(min(quota, m_tok))
        vecs = kr[top]
        par = (base + top) // self.block
        pos = base + top
        self.O = vecs if self.O is None else torch.cat([self.O, vecs])
        self.O_parent = par if self.O_parent is None else torch.cat([self.O_parent, par])
        self.O_pos = pos if self.O_pos is None else torch.cat([self.O_pos, pos])
        self.O_score = val if self.O_score is None else torch.cat([self.O_score, val])
        self.n_out = self.O.shape[0]
        if self.outlier_store_cap is not None and self.n_out > self.outlier_store_cap:
            keep = self.O_score.topk(self.outlier_store_cap).indices
            self.O = self.O[keep]
            self.O_parent = self.O_parent[keep]
            self.O_pos = self.O_pos[keep]
            self.O_score = self.O_score[keep]
            self.n_out = self.outlier_store_cap

    # -- routing ---------------------------------------------------------------------------------

    def _search_committed(self, qr, search_k, nprobe):
        """Candidates from the committed past for the given rows. (nbq, kc) scores + sub ids; strictly-past
        ⇒ causal. Returns D_last (the search's smallest returned score per row) for the certificate."""
        nbq = qr.shape[0]
        if self.committed == 0:
            z = torch.empty(nbq, 0, device=DEV)
            return z, z.long(), torch.full((nbq,), NEG, device=DEV)
        kc = min(search_k, self.committed)
        if self.index is None:                                     # flat-scan regime: exhaustive
            sc = qr @ self.means[:self.committed].T
            v, i = sc.topk(kc, dim=1)
            return v, i, torch.full((nbq,), NEG, device=DEV)       # exhaustive ⇒ nothing truncated
        old = self.index.nprobe
        self.index.nprobe = min(nprobe, self.nlist)
        d, i = self.index.search(qr, kc)
        self.index.nprobe = old
        pad = i < 0                                                # a starved probe (probed lists jointly
        if pad.any():                                              # hold < kc vectors) is padded by faiss
            d = d.masked_fill(pad, NEG)                            # with id -1 / score ~-3.4e38, which is
            i = i.masked_fill(pad, 0)                              # NOT -inf: mask to NEG so route()'s
        # scores>NEG guard drops the slot (id 0 is inert), and D_last=NEG is sound — the search returned
        # every member of the probed lists, so nothing was lost to truncation (the exhaustive convention).
        return d, i.long(), d[:, -1]

    def _search_staged(self, qr, qb_start):
        """Candidates from the in-flight chunk (staged sub-means), exact GEMM, causally masked
        (staged parent block strictly before the query block). (nbq, ks) scores + sub ids."""
        st = self.means[self.committed:self.n_sub]                 # (ns, d_r)
        ns = st.shape[0]
        nbq = qr.shape[0]
        if ns == 0:
            z = torch.empty(nbq, 0, device=DEV)
            return z, z.long()
        sc = qr @ st.T                                             # (nbq, ns)
        sub_global = self.committed + torch.arange(ns, device=DEV)  # global sub id
        par = sub_global // self.spb                               # global parent block
        qblk = qb_start + torch.arange(nbq, device=DEV)            # global query block per row
        routable = par[None, :] < qblk[:, None]                    # strictly-past parent
        sc = sc.masked_fill(~routable, NEG)
        ks = min(self.search_k, ns)
        v, j = sc.topk(ks, dim=1)
        return v, sub_global[j]

    def route(self, q_chunk, qpos, certify=False, search_k=None):
        """Route a chunk of query-blocks. Returns (kv_num (nbq,), kv_idx (nbq,W), cert (nbq,) or None, stats).
        Escalation: rows whose certificate fails are re-searched with a higher nprobe (≤ max_escalations)."""
        search_k0 = search_k or self.search_k
        qb = block_means(q_chunk, self.block)                      # (nbq, d) 128-granular query summaries
        qr = self.pj_q(qb)
        qn = qr.norm(dim=1)
        qb_start = qpos // self.block
        nbq = qr.shape[0]
        nb_global = qb_start + nbq
        SENT = nb_global
        qblk = qb_start + torch.arange(nbq, device=DEV)
        st_v, st_sub = self._search_staged(qr, qb_start)           # staged: exhaustive ⇒ no cert failure
        staged_best = st_v[:, 0] if st_v.shape[1] else torch.full((nbq,), NEG, device=DEV)

        W = min(self.top_c + self.local + 1 + self.outlier_cap, nb_global)
        kv_num = torch.zeros(nbq, dtype=torch.int32, device=DEV)
        kv_idx = torch.zeros(nbq, W, dtype=torch.int32, device=DEV)
        cert = torch.zeros(nbq, dtype=torch.bool, device=DEV)
        rounds = torch.zeros(nbq, dtype=torch.int32, device=DEV)
        active = torch.arange(nbq, device=DEV)
        nprobe, sk = self.nprobe, search_k0

        for rnd in range(self.max_escalations + 1):
            qa, qna = qr[active], qn[active]
            sc_v, sc_sub, D_last = self._search_committed(qa, sk, nprobe)
            scores = torch.cat([sc_v, st_v[active]], dim=1)
            pars = torch.cat([sc_sub, st_sub[active]], dim=1) // self.spb
            pars = torch.where(scores > NEG, pars, torch.full_like(pars, SENT))
            keep_pars, tau_a, ndist = self._select_topc(scores, pars, active.shape[0], SENT)
            opar = self._outlier_parents(qa, tau_a, qblk[active], SENT)
            loc = qblk[active][:, None] - torch.arange(self.local + 1, device=DEV)[None, :]
            loc = torch.where(loc >= 0, loc, torch.full_like(loc, SENT))
            kn, ki = self._pack(torch.cat([keep_pars, loc, opar], dim=1), SENT, W)
            kv_num[active] = kn
            kv_idx[active] = ki
            if not certify:
                break
            c = self._certify(qa, qna, tau_a, D_last, ndist, qblk[active], nprobe)
            cert[active] = c
            rounds[active] = rnd
            active = active[~c]
            if active.numel() == 0 or self.index is None:
                break
            nprobe = min(nprobe * self.escalate_factor, self.nlist)
            sk = min(sk * 2, self.committed)

        stats = {"nbq": nbq, "rounds": rounds,
                 "cert_rate": float(cert.float().mean()) if certify else None}
        return kv_num, kv_idx, (cert if certify else None), stats

    def _select_topc(self, scores, pars, nbq, SENT):
        """Sort candidates by score desc, keep the first top_c DISTINCT parents (= max-pool over sub-blocks
        of a parent, since the highest-scoring sub-mean appears first). Returns (keep_pars, tau, n_distinct)."""
        order = scores.argsort(dim=1, descending=True)
        ps = torch.gather(pars, 1, order)
        ss = torch.gather(scores, 1, order)
        K = ps.shape[1]
        eq = ps[:, :, None] == ps[:, None, :]                      # (nbq,K,K); K=2·search_k is small
        earlier = torch.tril(torch.ones(K, K, device=DEV, dtype=torch.bool), diagonal=-1)
        dup = (eq & earlier[None]).any(dim=2)
        valid = (ps < SENT) & ~dup
        keep = valid & (valid.cumsum(1) <= self.top_c)
        n_distinct = keep.sum(1)
        tau = self._min_kept(ss, keep, nbq)                        # the κ-th (smallest kept) block score
        return torch.where(keep, ps, torch.full_like(ps, SENT)), tau, n_distinct

    def _outlier_parents(self, qa, tau, qblk_a, SENT):
        """Extra parents whose stored high-leverage key would out-score the κ-th selected block (score > τ).
        Augments the routing selection (does not consume top_c slots); ≤ outlier_cap, causal (parent<query)."""
        nbq = qa.shape[0]
        if self.n_out == 0:
            return torch.full((nbq, self.outlier_cap), SENT, device=DEV, dtype=torch.long)
        So = qa @ self.O[:self.n_out].T                            # (nbq, s)
        opar = self.O_parent[:self.n_out]
        hit = (opar[None, :] < qblk_a[:, None]) & (So > tau[:, None])
        So_h = So.masked_fill(~hit, NEG)
        cap = min(self.outlier_cap, self.n_out)
        ov, oj = So_h.topk(cap, dim=1)
        sel = torch.where(ov > NEG, opar[oj], torch.full_like(opar[oj], SENT))
        if cap < self.outlier_cap:
            pad = torch.full((nbq, self.outlier_cap - cap), SENT, device=DEV, dtype=sel.dtype)
            sel = torch.cat([sel, pad], dim=1)
        return sel

    @staticmethod
    def _min_kept(scores, keep, nbq):
        masked = torch.where(keep, scores, torch.full_like(scores, float("inf")))
        m = masked.amin(dim=1)
        return torch.where(torch.isinf(m), torch.full_like(m, NEG), m)

    def _pack(self, cand, SENT, W):
        """Dedupe (two-sort sentinel trick) and pack to (kv_num, kv_idx int32), pads -> block 0."""
        cand, _ = cand.sort(1)
        dup = torch.zeros_like(cand, dtype=torch.bool)
        dup[:, 1:] = cand[:, 1:] == cand[:, :-1]
        cand = torch.where(dup, torch.full_like(cand, SENT), cand)
        cand, _ = cand.sort(1)
        cand = cand[:, :W]
        kv_num = (cand < SENT).sum(1).to(torch.int32)
        kv_idx = torch.where(cand < SENT, cand, torch.zeros_like(cand)).to(torch.int32)
        return kv_num, kv_idx

    def _certify(self, qa, qn, tau, D_last, ndist, qblk, nprobe):
        """Sound certificate: the selected top-κ parent blocks equal the exact top-κ under the routing
        metric s(i,B)=max_{sub∈B}⟨q̄_i,μ_sub⟩ over the past. Three conditions (staged region is exhaustive
        so it never fails): (1) no UNPROBED cell can hold a sub-mean ≥ τ — admissible bound
        ⟨q̄,c⟩+‖q̄‖·R_c < τ (Cauchy–Schwarz; R_c upper-bounds every member, exact after rebuild);
        (2) the search bottomed out below τ (D_last < τ) so probed cells lost nothing ≥ τ to truncation;
        (3) a full budget was found (ndist ≥ min(top_c, #past blocks))."""
        n_routable = qblk.to(ndist.dtype)                          # blocks strictly before (= global index)
        full_budget = ndist >= torch.minimum(torch.full_like(ndist, self.top_c), n_routable)
        has_tau = tau > NEG
        if self.index is None:                                     # committed exhaustively scored (flat scan)
            return full_budget & (has_tau | (n_routable == 0))
        cs = qa @ self.centroids.T                                 # (nbq, nlist)
        UB = cs + qn[:, None] * self.Rc[None, :]
        _, probed = cs.topk(min(nprobe, self.nlist), dim=1)
        UB = UB.scatter(1, probed, torch.full_like(UB, NEG))       # bound only the UNPROBED cells
        cond1 = UB.amax(1) < tau - self.cert_margin
        cond2 = D_last < tau - self.cert_margin
        return full_budget & has_tau & cond1 & cond2

    # -- decode ----------------------------------------------------------------------------------

    def step(self, q_new, k_new, v_new, kbuf, vbuf, pos):
        """One decode step (Phase A minimal): append k_new to the running summaries as blocks complete,
        route the single query, attend via ivf_decode.decode_attend over the selected blocks."""
        from ssa.ivf_decode import decode_attend
        # committed sub-means already hold all complete past blocks; the tail is covered by own+local.
        kn, ki, _, _ = self.route(q_new.view(1, self.d).expand(self.block, self.d)[:self.block],
                                  qpos=(pos // self.block) * self.block)
        blocks = ki[0, :int(kn[0])].tolist()
        return decode_attend(q_new, kbuf, vbuf, blocks, pos, self.block)


def ccc_prefill(q, k, v, block=BLOCK, certify=False, **cfg):
    """Chunked-causal routing (index holds only committed past) assembled into ONE global BlockMask +
    ONE flex call. Honest label: routing is chunked-causal; the single flex call is benchmark convenience.
    q,k,v: (1,1,n,d)."""
    B, H, n, d = q.shape
    assert B == 1 and H == 1, "single-head rig"
    nb = n // block
    cc = CausalCascade(d, block=block, n_hint=n, **cfg)
    W = min(cc.top_c + cc.local + 1 + cc.outlier_cap, nb)
    kv_num = torch.empty(1, 1, nb, device=q.device, dtype=torch.int32)
    kv_idx = torch.zeros(1, 1, nb, W, device=q.device, dtype=torch.int32)
    cb = cc.chunk_blocks * block
    cert_all = []
    for t in range(0, n, cb):
        e = min(n, t + cb)
        cc.append(k[0, 0, t:e])
        kn, ki, cert, _ = cc.route(q[0, 0, t:e], qpos=t, certify=certify)
        b0, b1 = t // block, e // block
        kv_num[0, 0, b0:b1] = kn
        kv_idx[0, 0, b0:b1, :ki.shape[1]] = ki
        if cert is not None:
            cert_all.append(cert)
    out = _flex(q, k, v, block_mask=_build_mask(kv_num, kv_idx, n, block))
    cert = torch.cat(cert_all) if cert_all else None
    return out, kv_num, kv_idx, cert


class CausalTree:
    """Pure-PyTorch online hierarchy with admissible center-radius node bounds.

    Complete fanout groups are recursively summarized; the at-most ``fanout-1`` remainder nodes per
    level form an exact cover of the committed past.  Query search expands that cover coarse-to-fine,
    pruning to a fixed beam by ``q·center + ||q|| radius``.  The bound is the Cauchy--Schwarz tree bound
    used by :mod:`ssa.hierarchical_certified_attention`; a fixed beam makes this a router rather than a
    full certificate.  It exists primarily for CUDA architectures unsupported by FAISS GPU.
    """

    def __init__(self, d, block=BLOCK, sub=32, query_sub=None, top_c=8, local=1,
                 search_k=None, n_hint=None,
                 outlier_rate=1e-3, outlier_cap=4, outlier_store_cap=1024,
                 tree_fanout=16, tree_beam=None, chunk_blocks=128, **_ignored):
        if n_hint is None or n_hint % sub:
            raise ValueError("CausalTree needs a sub-block-aligned n_hint")
        query_sub = block if query_sub is None else int(query_sub)
        if block % sub or block % query_sub:
            raise ValueError("sub and query_sub must divide block")
        self.d, self.block, self.sub, self.query_sub = d, block, sub, query_sub
        self.spb = block // sub
        self.qspb = block // query_sub
        self.top_c, self.local = top_c, local
        self.chunk_blocks = chunk_blocks
        self.search_k = search_k or 4 * top_c
        self.beam = tree_beam or self.search_k
        self.fanout = int(tree_fanout)
        if self.fanout < 2:
            raise ValueError("tree_fanout must be at least two")
        self.outlier_rate, self.outlier_cap = outlier_rate, outlier_cap
        self.outlier_store_cap = outlier_store_cap
        self.levels, self.radii, self.counts, self.sizes = [], [], [], []
        cap, size = n_hint // sub, 1
        while True:
            self.levels.append(torch.empty(cap, d, device=DEV, dtype=torch.float32))
            self.radii.append(torch.empty(cap, device=DEV, dtype=torch.float32))
            self.counts.append(0); self.sizes.append(size)
            if cap <= 1:
                break
            cap = (cap + self.fanout - 1) // self.fanout
            size *= self.fanout
        self.stage = None
        self.n_sub = self.committed = 0
        self.O = self.O_parent = self.O_pos = self.O_score = None
        self.n_out = 0

    def append(self, k_chunk):
        self._commit()
        mu = sub_block_means(k_chunk, self.block, self.sub)
        self._extract_outliers(k_chunk, mu)
        self.stage = mu
        self.n_sub += mu.shape[0]

    def _commit(self):
        if self.stage is None:
            return
        X = self.stage
        start = self.counts[0]
        self.levels[0][start:start + X.shape[0]] = X
        self.radii[0][start:start + X.shape[0]] = 0
        self.counts[0] += X.shape[0]
        for level in range(len(self.levels) - 1):
            have = self.counts[level + 1]
            complete = self.counts[level] // self.fanout
            if complete <= have:
                break
            children = self.levels[level][have * self.fanout:complete * self.fanout]
            children = children.view(complete - have, self.fanout, self.d)
            center = children.mean(1)
            child_r = self.radii[level][have * self.fanout:complete * self.fanout]
            child_r = child_r.view(complete - have, self.fanout)
            radius = ((children - center[:, None]).norm(dim=-1) + child_r).amax(1)
            self.levels[level + 1][have:complete] = center
            self.radii[level + 1][have:complete] = radius
            self.counts[level + 1] = complete
        self.committed += X.shape[0]
        self.stage = None

    def _extract_outliers(self, k_chunk, mu):
        if self.outlier_rate <= 0 or self.outlier_cap <= 0:
            return
        m = k_chunk.shape[0]
        base = self.n_sub * self.sub
        kr = k_chunk.float()
        resid = (kr - mu[torch.arange(m, device=k_chunk.device) // self.sub]).norm(dim=1)
        quota = max(1, int(self.outlier_rate * m))
        val, ids = resid.topk(min(quota, m))
        vec, par, pos = kr[ids], (base + ids) // self.block, base + ids
        self.O = vec if self.O is None else torch.cat((self.O, vec))
        self.O_parent = par if self.O_parent is None else torch.cat((self.O_parent, par))
        self.O_pos = pos if self.O_pos is None else torch.cat((self.O_pos, pos))
        self.O_score = val if self.O_score is None else torch.cat((self.O_score, val))
        self.n_out = self.O.shape[0]
        if self.outlier_store_cap is not None and self.n_out > self.outlier_store_cap:
            keep = self.O_score.topk(self.outlier_store_cap).indices
            self.O, self.O_parent = self.O[keep], self.O_parent[keep]
            self.O_pos, self.O_score = self.O_pos[keep], self.O_score[keep]
            self.n_out = self.outlier_store_cap

    def _frontier(self):
        levels, ids, offset = [], [], 0
        for level in range(len(self.levels) - 1, -1, -1):
            size = self.sizes[level]
            while offset + size <= self.committed:
                levels.append(level); ids.append(offset // size); offset += size
        if offset != self.committed:
            raise AssertionError("tree frontier did not cover the committed prefix")
        return levels, ids

    def _node_data(self, level, idx):
        center = torch.zeros(*idx.shape, self.d, device=idx.device)
        radius = torch.zeros(idx.shape, device=idx.device)
        valid = level >= 0
        for lev in range(len(self.levels)):
            mask = level == lev
            if mask.any():
                center[mask] = self.levels[lev][idx[mask]]
                radius[mask] = self.radii[lev][idx[mask]]
        return center, radius, valid

    def _search_committed(self, q):
        roots_l, roots_i = self._frontier()
        nq = q.shape[0]
        if not roots_l:
            z = torch.empty(nq, 0, device=q.device)
            return z, z.long()
        level = torch.tensor(roots_l, device=q.device).expand(nq, -1).clone()
        idx = torch.tensor(roots_i, device=q.device).expand(nq, -1).clone()
        qnorm = q.norm(dim=1)

        def prune(level, idx, cap):
            center, radius, valid = self._node_data(level, idx)
            upper = (q[:, None] * center).sum(-1) + qnorm[:, None] * radius
            upper = upper.masked_fill(~valid, NEG)
            keep = min(cap, upper.shape[1])
            pick = upper.topk(keep, dim=1).indices
            return level.gather(1, pick), idx.gather(1, pick)

        level, idx = prune(level, idx, self.beam)
        ar = torch.arange(self.fanout, device=q.device)
        for lev in range(len(self.levels) - 1, 0, -1):
            expand = level == lev
            if not expand.any():
                continue
            child_l = (level[..., None] - 1).expand(-1, -1, self.fanout).clone()
            child_i = idx[..., None] * self.fanout + ar
            # Nodes below this level survive in slot zero; the other slots become sentinels.
            child_l = torch.where(expand[..., None], child_l, torch.full_like(child_l, -1))
            child_i = torch.where(expand[..., None], child_i, torch.zeros_like(child_i))
            child_l[..., 0] = torch.where(expand, child_l[..., 0], level)
            child_i[..., 0] = torch.where(expand, child_i[..., 0], idx)
            level, idx = prune(child_l.flatten(1), child_i.flatten(1), self.beam)
        center = self.levels[0][idx]
        score = (q[:, None] * center).sum(-1).masked_fill(level != 0, NEG)
        keep = min(self.search_k, score.shape[1])
        val, pick = score.topk(keep, dim=1)
        return val, idx.gather(1, pick)

    def _compact_distinct_topc(self, scores, parents, rows, SENT):
        """Return fixed-width, scored top-parent rows.

        Candidate search is performed separately for every query sub-block.  Compacting each result
        before taking their union keeps the final distinct-parent reduction small: a parent in the
        top-k of ``max_r score(r, parent)`` must be top-k for at least one representative ``r``.
        """
        keep, _, _ = CausalCascade._select_topc(self, scores, parents, rows, SENT)
        valid = keep < SENT
        # ``keep`` is aligned with scores sorted by descending score inside _select_topc. Recreate that
        # ordering so the compact values retain the score belonging to each selected parent.
        order = scores.argsort(dim=1, descending=True)
        sorted_scores = scores.gather(1, order)
        selected_scores = torch.where(valid, sorted_scores, torch.full_like(sorted_scores, NEG))
        width = min(self.top_c, selected_scores.shape[1])
        values, pick = selected_scores.topk(width, dim=1)
        selected = keep.gather(1, pick)
        selected = torch.where(values > NEG, selected, torch.full_like(selected, SENT))
        return values, selected

    def _outlier_parents_reps(self, qr, tau, qblocks, SENT):
        """Outlier side channel under max-over-query-representatives block scoring."""
        nbq = qr.shape[0]
        if self.n_out == 0:
            return torch.full((nbq, self.outlier_cap), SENT, device=qr.device, dtype=torch.long)
        score = torch.einsum("brd,sd->brs", qr, self.O[:self.n_out]).amax(1)
        parents = self.O_parent[:self.n_out]
        hit = (parents[None] < qblocks[:, None]) & (score > tau[:, None])
        score = score.masked_fill(~hit, NEG)
        cap = min(self.outlier_cap, self.n_out)
        values, pick = score.topk(cap, dim=1)
        selected = torch.where(values > NEG, parents[pick], torch.full_like(parents[pick], SENT))
        if cap < self.outlier_cap:
            pad = torch.full((nbq, self.outlier_cap - cap), SENT,
                             device=qr.device, dtype=selected.dtype)
            selected = torch.cat((selected, pad), dim=1)
        return selected

    def route(self, q_chunk, qpos, certify=False, search_k=None):
        if search_k is not None and search_k != self.search_k:
            raise ValueError("CausalTree search_k is fixed at construction")
        # A BlockMask row is shared by every query in a parent block.  Therefore its routing score must
        # cover the union of those queries' needs.  Averaging all 128 queries is not max-preserving;
        # score smaller query summaries independently and max-pool their selected parent candidates.
        qr = sub_block_means(q_chunk, self.block, self.query_sub)
        nbq, qb_start = q_chunk.shape[0] // self.block, qpos // self.block
        qr = qr.view(nbq, self.qspb, self.d)
        SENT = qb_start + nbq
        qblocks = qb_start + torch.arange(nbq, device=qr.device)
        representative_scores, representative_parents = [], []
        for rep in range(self.qspb):
            query = qr[:, rep]
            old_v, old_sub = self._search_committed(query)
            if self.stage is None:
                stage_v = torch.empty(nbq, 0, device=qr.device)
                stage_sub = stage_v.long()
            else:
                stage_v = query @ self.stage.T
                sub_ids = self.committed + torch.arange(self.stage.shape[0], device=qr.device)
                stage_parent = sub_ids // self.spb
                stage_v = stage_v.masked_fill(stage_parent[None] >= qblocks[:, None], NEG)
                keep = min(self.search_k, self.stage.shape[0])
                stage_v, pick = stage_v.topk(keep, dim=1)
                stage_sub = sub_ids[pick]
            scores = torch.cat((old_v, stage_v), dim=1)
            parents = torch.cat((old_sub, stage_sub), dim=1) // self.spb
            parents = torch.where(scores > NEG, parents, torch.full_like(parents, SENT))
            values, selected = self._compact_distinct_topc(scores, parents, nbq, SENT)
            representative_scores.append(values)
            representative_parents.append(selected)
        scores = torch.cat(representative_scores, dim=1)
        parents = torch.cat(representative_parents, dim=1)
        keep_pars, tau, _ = CausalCascade._select_topc(self, scores, parents, nbq, SENT)
        outliers = self._outlier_parents_reps(qr, tau, qblocks, SENT)
        local = qblocks[:, None] - torch.arange(self.local + 1, device=qr.device)
        local = torch.where(local >= 0, local, torch.full_like(local, SENT))
        width = min(self.top_c + self.local + 1 + self.outlier_cap, SENT)
        kn, ki = CausalCascade._pack(self, torch.cat((keep_pars, local, outliers), 1), SENT, width)
        cert = torch.zeros(nbq, dtype=torch.bool, device=qr.device) if certify else None
        return kn, ki, cert, {"nbq": nbq, "backend": "torch_tree", "cert_rate": None}

    _min_kept = staticmethod(CausalCascade._min_kept)


class StreamingGQARouter:
    """Stateful, strict-causal CCC routing for token-chunked GQA execution.

    Unlike :func:`ccc_route_gqa`, this surface does not require the full query/key tensors to coexist.
    ``route_chunk`` consumes one aligned chunk at a time and returns only that chunk's compact routing
    plan.  It is the selector used by the >10M-token layer-streaming transformer demo.
    """

    def __init__(self, batch, query_heads, kv_heads, n, d, block=BLOCK, certify=False,
                 backend="faiss", **cfg):
        if kv_heads < 1 or query_heads % kv_heads:
            raise ValueError("query_heads must be divisible by a positive kv_heads")
        if n % block:
            raise ValueError("StreamingGQARouter requires a block-padded total length")
        self.batch, self.hq, self.hkv = batch, query_heads, kv_heads
        self.n, self.d, self.block = n, d, block
        self.groups = query_heads // kv_heads
        self.certify = certify
        self.cfg = cfg
        cascade_cls = {"faiss": CausalCascade, "tree": CausalTree}.get(backend)
        if cascade_cls is None:
            raise ValueError("backend must be 'faiss' or 'tree'")
        self.backend = backend
        self.cascades = [[cascade_cls(d, block=block, n_hint=n, **cfg)
                          for _ in range(kv_heads)] for _ in range(batch)]
        probe = self.cascades[0][0]
        self.width = min(probe.top_c + probe.local + 1 + probe.outlier_cap, n // block)
        self.next_pos = 0

    def route_chunk(self, q, k, start):
        """Append and route one block-aligned chunk.

        ``q`` is ``(B,Hq,m,d)``, ``k`` is unrepeated ``(B,Hkv,m,d)``, and chunks must arrive in
        increasing contiguous order.  Returned indices are global key-block ids.
        """
        if start != self.next_pos:
            raise ValueError(f"expected chunk at {self.next_pos}, got {start}")
        if q.shape[:2] != (self.batch, self.hq) or k.shape[:2] != (self.batch, self.hkv):
            raise ValueError("chunk batch/head geometry does not match the router")
        if q.shape[2:] != k.shape[2:] or q.shape[-1] != self.d:
            raise ValueError("q and k chunks must agree in sequence length and head dimension")
        m = q.shape[2]
        if m % self.block or start + m > self.n:
            raise ValueError("chunks must be block aligned and remain within the declared length")
        nbq = m // self.block
        kv_num = torch.empty(self.batch, self.hq, nbq, dtype=torch.int32, device=q.device)
        kv_idx = torch.zeros(self.batch, self.hq, nbq, self.width,
                             dtype=torch.int32, device=q.device)
        cert_out = (torch.zeros(self.batch, self.hq, nbq, dtype=torch.bool, device=q.device)
                    if self.certify else None)
        for batch in range(self.batch):
            for hk in range(self.hkv):
                cc = self.cascades[batch][hk]
                cc.append(k[batch, hk])
                first = hk * self.groups
                for hq in range(first, first + self.groups):
                    kn, ki, cert, _ = cc.route(q[batch, hq], qpos=start, certify=self.certify)
                    kv_num[batch, hq] = kn
                    kv_idx[batch, hq, :, :ki.shape[1]] = ki
                    if cert_out is not None:
                        cert_out[batch, hq] = cert
        self.next_pos += m
        return kv_num, kv_idx, cert_out


def ccc_route_gqa(q, k, block=BLOCK, certify=False, **cfg):
    """Strictly-causal streaming CCC routing for grouped-query attention.

    ``q`` is ``(B,Hq,n,d)`` and unrepeated ``k`` is ``(B,Hkv,n,d)``.  A cascade is built once per
    KV head, then shared by the query heads in its GQA group.  Earlier query chunks are routed and
    frozen before later key chunks enter the index, so changing future keys cannot change an earlier
    selection.  Returns the compact ``(kv_num, kv_idx, cert)`` BlockMask contract.
    """
    B, Hq, n, d = q.shape
    if k.shape[0] != B or k.shape[2:] != (n, d):
        raise ValueError("q and k must agree in batch, sequence, and head dimensions")
    Hkv = k.shape[1]
    if Hkv < 1 or Hq % Hkv:
        raise ValueError("the query-head count must be a positive multiple of the KV-head count")
    if n % block:
        raise ValueError("ccc_route_gqa requires a block-padded sequence")
    nb = n // block
    stream = StreamingGQARouter(B, Hq, Hkv, n, d, block=block, certify=certify, **cfg)
    W = stream.width
    kv_num = torch.empty(B, Hq, nb, dtype=torch.int32, device=q.device)
    kv_idx = torch.zeros(B, Hq, nb, W, dtype=torch.int32, device=q.device)
    cert_out = torch.zeros(B, Hq, nb, dtype=torch.bool, device=q.device) if certify else None
    cb = stream.cascades[0][0].chunk_blocks * block
    for start in range(0, n, cb):
        stop = min(n, start + cb)
        kn, ki, cert = stream.route_chunk(q[:, :, start:stop], k[:, :, start:stop], start)
        b0, b1 = start // block, stop // block
        kv_num[:, :, b0:b1] = kn
        kv_idx[:, :, b0:b1] = ki
        if cert_out is not None:
            cert_out[:, :, b0:b1] = cert
    return kv_num, kv_idx, cert_out


# ---------------------------------------------------------------------------------------------------
# benchmark: the selector-share decomposition
# ---------------------------------------------------------------------------------------------------

def _gen(n, d, geometry, g):
    """(1,1,n,d) q,k,v. 'random' = the speed rig; 'clustered' = where certificates fire."""
    if geometry == "random":
        q = torch.empty(1, 1, n, d, device=DEV, dtype=torch.float16)
        k = torch.empty_like(q); v = torch.empty_like(q)
        for t in (q, k, v):
            t.view(-1).normal_(generator=g)
        return q, k, v
    nb4 = n // 32
    nc = max(8, int(nb4 ** 0.5))
    ctr = torch.randn(nc, d, generator=g, device=DEV)
    a = torch.randint(0, nc, (nb4,), generator=g, device=DEV)
    k = (ctr[a].repeat_interleave(32, 0) + 0.1 * torch.randn(n, d, generator=g, device=DEV)).half()
    nbq = n // BLOCK
    qc = ctr[torch.randint(0, nc, (nbq,), generator=g, device=DEV)]
    q = (qc.repeat_interleave(BLOCK, 0) + 0.2 * torch.randn(n, d, generator=g, device=DEV)).half()
    v = torch.randn(n, d, generator=g, device=DEV, dtype=torch.float16)
    return q.view(1, 1, n, d), k.view(1, 1, n, d), v.view(1, 1, n, d)


def _sync_ms(fn):
    torch.cuda.synchronize(); s = time.time(); r = fn(); torch.cuda.synchronize()
    return (time.time() - s) * 1000, r


@torch.no_grad()
def decompose_ccc(n, d=64, geometry="random", certify=True, chunk_blocks=1024, g=None, attn_reps=4, **cfg):
    """Single-head streaming decomposition: append (summaries+index-maintain) / route (search+cert+outlier)
    / maskbuild / attention (the floor). selector_share = everything but attention. One streaming pass."""
    nb = n // BLOCK
    q, k, v = _gen(n, d, geometry, g)
    torch.cuda.reset_peak_memory_stats()
    cc = CausalCascade(d, block=BLOCK, n_hint=n, chunk_blocks=chunk_blocks, **cfg)
    W = min(cc.top_c + cc.local + 1 + cc.outlier_cap, nb)
    kv_num = torch.empty(1, 1, nb, dtype=torch.int32, device=DEV)
    kv_idx = torch.zeros(1, 1, nb, W, dtype=torch.int32, device=DEV)
    cb = chunk_blocks * BLOCK
    append_ms = route_ms = 0.0
    certs, rounds = [], []
    rebuilds = 0
    for t in range(0, n, cb):
        e = min(n, t + cb)
        cbr = cc.committed_at_rebuild
        a_ms, _ = _sync_ms(lambda: cc.append(k[0, 0, t:e]))
        r_ms, out = _sync_ms(lambda: cc.route(q[0, 0, t:e], qpos=t, certify=certify))
        kn, ki, cert, st = out
        if cc.committed_at_rebuild != cbr and t > 0:
            rebuilds += 1
        kv_num[0, 0, t // BLOCK:e // BLOCK] = kn
        kv_idx[0, 0, t // BLOCK:e // BLOCK] = ki
        append_ms += a_ms; route_ms += r_ms
        if cert is not None:
            certs.append(cert.float().mean().item())
            rounds.append(st["rounds"].float().mean().item())
    mb_ms, bm = _sync_ms(lambda: _build_mask(kv_num, kv_idx, n, BLOCK))
    at_ms = _t_attn(q, k, v, bm, attn_reps)
    selector = append_ms + route_ms + mb_ms
    total = selector + at_ms
    peak = torch.cuda.max_memory_allocated() / 1e9
    del q, k, v, kv_num, kv_idx, bm, cc
    torch.cuda.empty_cache()
    import numpy as np
    return {"n": n, "nb": nb, "geometry": geometry, "certify": certify, "rebuilds": rebuilds,
            "append_ms": append_ms, "route_ms": route_ms, "maskbuild_ms": mb_ms, "attention_ms": at_ms,
            "total_ms": total, "selector_ms": selector,
            "selector_share": selector / total if total else None,
            "amortized_share_L24": selector / (selector + 24 * at_ms) if at_ms else None,
            "cert_rate": float(np.mean(certs)) if certs else None,
            "mean_rounds": float(np.mean(rounds)) if rounds else None, "peak_mem_gb": peak}


def _t_attn(q, k, v, bm, reps):
    for _ in range(2):
        _flex(q, k, v, block_mask=bm)
    torch.cuda.synchronize(); s = time.time()
    for _ in range(reps):
        _flex(q, k, v, block_mask=bm)
    torch.cuda.synchronize()
    return (time.time() - s) / reps * 1000


def main():
    import argparse
    import json
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[262144, 1048576, 4194304, 8388608, 12582912])
    ap.add_argument("--geometry", default="clustered", choices=["random", "clustered"])
    ap.add_argument("--no-cert", action="store_true")
    ap.add_argument("--out", default="paper/figures/ccc_kernel.json")
    args = ap.parse_args()
    torch._dynamo.config.cache_size_limit = 64
    g = torch.Generator(device=DEV).manual_seed(0)
    decompose_ccc(262144, geometry=args.geometry, certify=not args.no_cert, chunk_blocks=1024, g=g)  # warm compile
    print("=" * 100)
    print(f"THE CERTIFIED CAUSAL CASCADE — selector-share decomposition, single-head ({args.geometry} keys)")
    print("=" * 100)
    print(f"  {'n':>10} {'append':>8} {'route':>8} {'maskbld':>8} {'attn':>8} {'total':>9} "
          f"{'sel.share':>9} {'amort/L24':>9} {'cert':>6} {'peak':>6}")
    rows = []
    for n in args.ns:
        r = decompose_ccc(n, geometry=args.geometry, certify=not args.no_cert,
                          chunk_blocks=1024, g=g, attn_reps=4 if n <= (1 << 21) else 2)
        rows.append(r)
        cr = f"{r['cert_rate']:.2f}" if r["cert_rate"] is not None else "—"
        print(f"  {n:>10} {r['append_ms']:>8.1f} {r['route_ms']:>8.1f} {r['maskbuild_ms']:>8.2f} "
              f"{r['attention_ms']:>8.1f} {r['total_ms']:>9.1f} {r['selector_share']:>9.3f} "
              f"{r['amortized_share_L24']:>9.3f} {cr:>6} {r['peak_mem_gb']:>6.2f}", flush=True)
        import os
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"meta": {"H": 1, "d": 64, "block": BLOCK, "sub": 32, "geometry": args.geometry,
                            "certify": not args.no_cert, "seed": 0, "gpu": "RTX 4080 16GB",
                            "note": "single-head; selector_share = (append+route+maskbuild)/total; "
                                    "amortized_share_L24 = selector/(selector+24·attention) is an ARITHMETIC "
                                    "composition on the single-head rig (real multi-layer = Qwen leg); "
                                    "certificates certify selector==routing-metric top-κ, not attention error"},
                   "rows": rows}, open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
