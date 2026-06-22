"""
Bounded-candidate treecode router — Phase 1: the tree build (see docs/bounded_treecode_scope.md).

Positional F-ary tree over key-blocks. Each node carries (mu, R, var, lo):
  * mu   = mean of the node's keys (mean of child means above the leaves),
  * R     = RECURSIVE radius  R_parent = max_child(||mu_child - mu_parent|| + R_child)  — the bound
            SearchTradeoff.subtree_radius_bound licenses, so  <q,k> <= <q,mu> + ||q||*R  for EVERY
            key k under the node (the admissible upper bound the beam descends by),
  * var  = diagonal variance (law of total variance up the tree) — the cumulant recall term,
  * lo    = earliest token index in the node's span — causality is then a range check (lo <= q_end).
Contiguous grouping makes children implicit: node p at level l has children [p*F : p*F+F] at level l-1.

Pure tensor ops, device-agnostic, batched over (B*H). `main` builds and VALIDATES the admissible-bound
invariant (every node's bound >= the true max key score under it) before P2 relies on it.

Run: python3 -m ssa.treecode
"""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class TreeLevel:
    mu: torch.Tensor    # (BH, n_l, d)  node means
    R: torch.Tensor     # (BH, n_l)     recursive radii
    var: torch.Tensor   # (BH, n_l, d)  diagonal variance (cumulant term)
    lo: torch.Tensor    # (n_l,)        earliest token index in each node's span
    n: int              # node count at this level


def build_tree(k: torch.Tensor, block: int, F: int = 8, top_cap: int = 1):
    """k: (B,H,n,d). Returns (levels, F): levels[0] = leaf key-blocks, ascending to the (<=top_cap) top."""
    B, H, n, d = k.shape
    nb = n // block
    kb = k.reshape(B * H, nb, block, d).float()
    mu = kb.mean(2)                                              # (BH, nb, d)
    var = kb.var(2, unbiased=False)                             # (BH, nb, d)
    R = (kb - mu.unsqueeze(2)).norm(dim=-1).amax(dim=-1)        # (BH, nb)
    lo = torch.arange(nb, device=k.device) * block             # (nb,)
    levels = [TreeLevel(mu, R, var, lo, nb)]
    BIG = float(nb * block + 1)

    while levels[-1].n > top_cap:
        ch = levels[-1]
        nc = ch.n
        npar = (nc + F - 1) // F
        pad = npar * F - nc

        def grp(x, fill):                                       # (BH, nc, *) -> (BH, npar, F, *)
            if pad:
                x = torch.cat([x, x.new_full((x.shape[0], pad, *x.shape[2:]), fill)], dim=1)
            return x.reshape(x.shape[0], npar, F, *x.shape[2:])

        cmu = grp(ch.mu, 0.0)                                   # (BH, npar, F, d)
        cvar = grp(ch.var, 0.0)
        cR = grp(ch.R, 0.0)                                     # (BH, npar, F)
        cnt = torch.ones(npar * F, device=k.device)
        if pad:
            cnt[-pad:] = 0.0
        cnt = cnt.reshape(1, npar, F, 1)                        # valid-child mask

        wsum = cnt.sum(2)                                       # (1, npar, 1)
        mu_p = (cmu * cnt).sum(2) / wsum                        # mean of child means
        var_p = ((cvar + cmu ** 2) * cnt).sum(2) / wsum - mu_p ** 2   # total variance E[var]+Var[E]
        dist = (cmu - mu_p.unsqueeze(2)).norm(dim=-1)          # (BH, npar, F)
        R_p = (dist + cR).masked_fill(cnt.squeeze(-1) == 0, float("-inf")).amax(2)

        lo_c = ch.lo
        if pad:
            lo_c = torch.cat([lo_c, lo_c.new_full((pad,), BIG)])
        lo_p = lo_c.reshape(npar, F).amin(1)                    # earliest token under the parent

        levels.append(TreeLevel(mu_p, R_p, var_p, lo_p, npar))
    return levels, F


@torch.no_grad()
def check_admissible(k: torch.Tensor, levels, F: int, block: int, trials: int = 32, seed: int = 0):
    """For random queries, the worst-case slack (node bound - true max key score under it). Must be >= 0."""
    B, H, n, d = k.shape
    nb = n // block
    kf = k.reshape(B * H, n, d).float()
    g = torch.Generator(device=k.device).manual_seed(seed)
    worst = float("inf")
    for _ in range(trials):
        q = torch.randn(B * H, d, generator=g, device=k.device)
        ks = torch.einsum('bnd,bd->bn', kf, q)                  # (BH, n) per-key score
        leafmax = ks.reshape(B * H, nb, block).amax(-1)         # (BH, nb)
        qn = q.norm(dim=-1)
        for l, lvl in enumerate(levels):
            bound = torch.einsum('bnd,bd->bn', lvl.mu, q) + qn[:, None] * lvl.R   # (BH, n_l)
            span = F ** l
            padn = lvl.n * span - nb
            lm = leafmax if padn == 0 else torch.cat(
                [leafmax, leafmax.new_full((B * H, padn), float("-inf"))], dim=1)
            truemax = lm.reshape(B * H, lvl.n, span).amax(-1)   # (BH, n_l) true max key under each node
            worst = min(worst, (bound - truemax).min().item())
    return worst


def main():
    torch.manual_seed(0)
    print("=" * 74)
    print("TREECODE P1 — positional F-ary tree build + admissible-bound validation")
    print("=" * 74)
    B, H, d, block, F = 1, 4, 32, 64, 8
    for n, keytype in ((4096, "random"), (4096, "clustered")):
        if keytype == "random":
            k = torch.randn(B, H, n, d)
        else:
            nb = n // block
            centers = torch.randn(B, H, nb, 1, d) * 3.0
            k = (centers + 0.15 * torch.randn(B, H, nb, block, d)).reshape(B, H, n, d)
        levels, _ = build_tree(k, block, F, top_cap=1)
        slack = check_admissible(k, levels, F, block)
        shape = " -> ".join(str(lv.n) for lv in levels)
        ok = "PASS" if slack >= -1e-3 else "FAIL"
        print(f"  n={n:>5} {keytype:>9} keys | levels {shape:>16} | "
              f"min slack (bound-true) = {slack:>8.3f}  [{ok}]")
        rbar = levels[0].R.mean().item()
        rtop = levels[-1].R.mean().item()
        print(f"            leaf radius ~{rbar:.2f}, top-node radius ~{rtop:.2f}  "
              f"(tighter leaves => the beam can discriminate)")
    print("\n  Admissible invariant holds => the bound never under-estimates a key, so a beam that keeps")
    print("  high-bound nodes cannot silently drop the true argmax (recall is set by the beam width W,")
    print("  not by missed-but-reachable keys). P2 descends this tree with a fixed-width beam.")


if __name__ == "__main__":
    main()
