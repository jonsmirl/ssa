"""
Bounded-candidate treecode router (see docs/bounded_treecode_scope.md).

P1  build_tree       — positional F-ary tree over key-blocks; each node carries (mu, recursive R,
                       diagonal var, lo). Contiguous grouping ⇒ children implicit (p -> [p*F : p*F+F]).
P2  descend_beam     — fixed-width-W beam descent: scan the small top, then each level gather the W·F
                       children (FIXED width — the invariant that kills the √n gather), score by the
                       admissible bound <q,mu>+||q||R (+ optional cumulant ½β·qᵀΣq), causal-mask, keep
                       top-W; at the leaves return top_c blocks. All tensors are (BH, Q, W·F, …) — linear
                       in n, GPU-regular, no (n/b)² matrix ever built.

`main` validates both: the admissible invariant (P1) and that the beam's recall climbs to the full-scan
ceiling as W grows (P2). Pure tensor ops, batched over (B*H).

Run: python3 -m ssa.treecode
"""
from __future__ import annotations
from dataclasses import dataclass
import torch

NEG = float("-inf")


@dataclass
class TreeLevel:
    mu: torch.Tensor    # (BH, n_l, d)  node means
    R: torch.Tensor     # (BH, n_l)     recursive radii
    var: torch.Tensor   # (BH, n_l, d)  diagonal variance (cumulant term)
    lo: torch.Tensor    # (n_l,)        earliest token index in the node's span
    n: int              # node count at this level


# ----------------------------------------------------------------------------- P1: build
def build_tree(k: torch.Tensor, block: int, F: int = 8, top_cap: int = 8):
    """k: (B,H,n,d). Returns (levels, F): levels[0] = leaf key-blocks, ascending to the (<=top_cap) top."""
    B, H, n, d = k.shape
    BH = B * H
    nb = n // block
    kr = k.reshape(BH, nb, block, d)
    # memory-lean leaf stats: chunk over blocks so no (BH·nb·block·d) fp32 temporary is materialized
    mu = torch.empty(BH, nb, d, device=k.device)
    var = torch.empty(BH, nb, d, device=k.device)
    R = torch.empty(BH, nb, device=k.device)
    CH = max(1, min(nb, 2048))
    for i in range(0, nb, CH):
        kc = kr[:, i:i + CH].float()                       # (BH, c, block, d)
        m = kc.mean(2)
        mu[:, i:i + CH] = m
        var[:, i:i + CH] = kc.var(2, unbiased=False)
        R[:, i:i + CH] = (kc - m.unsqueeze(2)).norm(dim=-1).amax(dim=-1)
    lo = torch.arange(nb, device=k.device) * block
    levels = [TreeLevel(mu, R, var, lo, nb)]
    BIG = float(nb * block + 1)
    while levels[-1].n > top_cap:
        ch = levels[-1]
        nc, npar = ch.n, (ch.n + F - 1) // F
        pad = npar * F - nc

        def grp(x, fill):
            if pad:
                x = torch.cat([x, x.new_full((x.shape[0], pad, *x.shape[2:]), fill)], dim=1)
            return x.reshape(x.shape[0], npar, F, *x.shape[2:])

        cmu, cvar, cR = grp(ch.mu, 0.0), grp(ch.var, 0.0), grp(ch.R, 0.0)
        cnt = torch.ones(npar * F, device=k.device)
        if pad:
            cnt[-pad:] = 0.0
        cnt = cnt.reshape(1, npar, F, 1)
        wsum = cnt.sum(2)
        mu_p = (cmu * cnt).sum(2) / wsum
        var_p = ((cvar + cmu ** 2) * cnt).sum(2) / wsum - mu_p ** 2
        dist = (cmu - mu_p.unsqueeze(2)).norm(dim=-1)
        R_p = (dist + cR).masked_fill(cnt.squeeze(-1) == 0, NEG).amax(2)
        lo_c = ch.lo if not pad else torch.cat([ch.lo, ch.lo.new_full((pad,), BIG)])
        lo_p = lo_c.reshape(npar, F).amin(1)
        levels.append(TreeLevel(mu_p, R_p, var_p, lo_p, npar))
    return levels, F


# ----------------------------------------------------------------------------- P2: descend
def _gather_rows(tab, idx):
    """tab:(BH,n[,d]); idx:(BH,M) -> (BH,M[,d])."""
    if tab.dim() == 2:
        return torch.gather(tab, 1, idx)
    return torch.gather(tab, 1, idx.unsqueeze(-1).expand(-1, -1, tab.shape[2]))


@torch.no_grad()
def descend_beam(levels, F, q, W, top_c, q_pos=None, beta=0.0):
    """q:(BH,Q,d). Returns (BH,Q,top_c) leaf-block indices (-1 = empty). Fixed beam width W."""
    BH, Q, d = q.shape
    L = len(levels) - 1
    qn = q.norm(dim=-1)
    ar = torch.arange(F, device=q.device)

    def score(mu_g, R_g, var_g):
        s = torch.einsum('bqd,bqmd->bqm', q, mu_g) + qn[:, :, None] * R_g
        if beta:
            s = s + 0.5 * beta * torch.einsum('bqd,bqmd->bqm', q * q, var_g)
        return s

    # top level: scan all n_top nodes, seed the width-W frontier
    top = levels[L]
    bt = score(top.mu.unsqueeze(1).expand(BH, Q, top.n, d),
               top.R.unsqueeze(1).expand(BH, Q, top.n),
               top.var.unsqueeze(1).expand(BH, Q, top.n, d))
    if q_pos is not None:
        bt = bt.masked_fill(top.lo[None, None, :] > q_pos[:, :, None], NEG)
    k0 = min(W, top.n)
    tv, frontier = bt.topk(k0, dim=-1)
    if k0 < W:
        frontier = torch.cat([frontier, frontier[..., :1].expand(BH, Q, W - k0)], dim=-1)
        tv = torch.cat([tv, tv.new_full((BH, Q, W - k0), NEG)], dim=-1)
    fvalid = tv > NEG

    for l in range(L - 1, -1, -1):
        nL = levels[l].n
        raw = frontier[..., None] * F + ar                       # (BH,Q,W,F)
        vchild = fvalid[..., None] & (raw < nL)
        child = raw.clamp(max=nL - 1).reshape(BH, Q, W * F)
        vchild = vchild.reshape(BH, Q, W * F)
        cidx = child.reshape(BH, Q * W * F)
        mu_g = _gather_rows(levels[l].mu, cidx).reshape(BH, Q, W * F, d)
        R_g = _gather_rows(levels[l].R, cidx).reshape(BH, Q, W * F)
        var_g = _gather_rows(levels[l].var, cidx).reshape(BH, Q, W * F, d)
        b = score(mu_g, R_g, var_g).masked_fill(~vchild, NEG)
        if q_pos is not None:
            b = b.masked_fill(levels[l].lo[child] > q_pos[:, :, None], NEG)
        tv, ti = b.topk(W, dim=-1)
        frontier = torch.gather(child, 2, ti)
        fvalid = tv > NEG

    tc = min(top_c, W)
    sel_v, sel_i = tv.topk(tc, dim=-1)
    sel = torch.gather(frontier, 2, sel_i)
    return sel.masked_fill(sel_v == NEG, -1)


# ----------------------------------------------------------------------------- validation
@torch.no_grad()
def check_admissible(k, levels, F, block, trials=32, seed=0):
    B, H, n, d = k.shape
    nb = n // block
    kf = k.reshape(B * H, n, d).float()
    g = torch.Generator(device=k.device).manual_seed(seed)
    worst = float("inf")
    for _ in range(trials):
        q = torch.randn(B * H, d, generator=g, device=k.device)
        leafmax = torch.einsum('bnd,bd->bn', kf, q).reshape(B * H, nb, block).amax(-1)
        qn = q.norm(dim=-1)
        for l, lvl in enumerate(levels):
            bound = torch.einsum('bnd,bd->bn', lvl.mu, q) + qn[:, None] * lvl.R
            span = F ** l
            pad = lvl.n * span - nb
            lm = leafmax if pad == 0 else torch.cat([leafmax, leafmax.new_full((B * H, pad), NEG)], 1)
            truemax = lm.reshape(B * H, lvl.n, span).amax(-1)
            worst = min(worst, (bound - truemax).min().item())
    return worst


@torch.no_grad()
def check_recall(k, levels, F, block, W, top_c, trials=400, noise=0.6, beta=0.0, seed=1):
    """Recall@top_c: fraction of corrupted-cue queries whose TRUE best block lands in the selection,
    for the beam vs a flat full-scan of all blocks (the ceiling)."""
    B, H, n, d = k.shape
    nb = n // block
    BH = B * H
    kf = k.reshape(BH, n, d).float()
    mu0, R0 = levels[0].mu, levels[0].R
    g = torch.Generator(device=k.device).manual_seed(seed)
    hit = fhit = tot = 0
    for _ in range(trials):
        t = torch.randint(0, n, (BH,), generator=g, device=k.device)
        kt = kf[torch.arange(BH), t]
        q = (kt + noise * torch.randn(BH, d, generator=g, device=k.device))
        true_block = torch.einsum('bnd,bd->bn', kf, q).argmax(1) // block
        sel = descend_beam(levels, F, q.unsqueeze(1), W, top_c, beta=beta).squeeze(1)   # (BH,top_c)
        hit += (sel == true_block[:, None]).any(1).sum().item()
        qn = q.norm(dim=-1)
        bflat = torch.einsum('bnd,bd->bn', mu0, q) + qn[:, None] * R0
        fsel = bflat.topk(top_c, dim=-1).indices
        fhit += (fsel == true_block[:, None]).any(1).sum().item()
        tot += BH
    return hit / tot, fhit / tot


def main():
    torch.manual_seed(0)
    print("=" * 78)
    print("TREECODE P1+P2 — tree build (admissible) + fixed-width beam descent (recall)")
    print("=" * 78)

    # P1: admissible invariant
    B, H, d, block, F = 1, 4, 32, 64, 8
    print("\nP1  admissible bound (min slack = bound - true max key score; must be >= 0):")
    for n, kind in ((4096, "random"), (4096, "clustered")):
        if kind == "random":
            k = torch.randn(B, H, n, d)
        else:
            nb = n // block
            k = (torch.randn(B, H, nb, 1, d) * 3 + 0.2 * torch.randn(B, H, nb, block, d)).reshape(B, H, n, d)
        levels, _ = build_tree(k, block, F, top_cap=1)
        sl = check_admissible(k, levels, F, block)
        print(f"    {kind:>9} | levels {' -> '.join(str(lv.n) for lv in levels):>14} | min slack {sl:>7.3f}"
              f"  [{'PASS' if sl >= -1e-3 else 'FAIL'}]")

    # P2: recall vs beam width W
    n, block, F, top_c = 8192, 64, 8, 4
    nb = n // block
    k = (torch.randn(1, 4, nb, 1, d) * 3 + 0.3 * torch.randn(1, 4, nb, block, d)).reshape(1, 4, n, d)
    levels, _ = build_tree(k, block, F, top_cap=8)
    shape = " -> ".join(str(lv.n) for lv in levels)
    print(f"\nP2  beam descent on clustered keys (n={n}, nb={nb}, levels {shape}, top_c={top_c}):")
    print(f"    {'W':>3} {'beam recall':>12} {'flat ceiling':>13} {'beam evals':>11} {'flat evals':>11}")
    Ld = len(levels) - 1
    for W in (1, 2, 4, 8, 16):
        r, rf = check_recall(k, levels, F, block, W, top_c)
        evals = levels[-1].n + Ld * W * F
        print(f"    {W:>3} {r:>12.3f} {rf:>13.3f} {evals:>11} {nb:>11}")
    r0, _ = check_recall(k, levels, F, block, 8, top_c, beta=0.0)
    rb, _ = check_recall(k, levels, F, block, 8, top_c, beta=1.0)
    print(f"    cumulant term (W=8): recall {r0:.3f} (beta=0) -> {rb:.3f} (beta=1)  [the (a) recall knob]")
    print("\n  Beam recall climbs to the full-scan ceiling as W grows, at W·F·L evals vs nb (the win grows")
    print("  with nb). Fixed width W ⇒ no √n gather, no (n/b)² matrix — the P4 no-2M-regression invariant.")


if __name__ == "__main__":
    main()
