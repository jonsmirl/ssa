# Subquadratic Sparse Attention: Content-Routed Exact Attention for Long Contexts

## Abstract

Dense self-attention costs $O(n^2)$ in the sequence length $n$, which is the binding
constraint on long-context language models. This paper presents **Subquadratic Sparse Attention (SSA)**, an
attention mechanism that, for each query, (i) routes to a small content-dependent set of key blocks using
only per-block summary statistics, (ii) adds a local window, and (iii) performs *exact* softmax attention
over the selected keys. The per-query work is $O(\kappa)$ in a fixed budget $\kappa \ll n$ plus a sublinear
routing cost: a flat router runs the layer in $O(n\sqrt{n})$ (at block size $b=\sqrt n$, where the budget
grows as $\sqrt n$), and a hierarchical router runs it near-linearly at fixed $b$ and fixed $\kappa$ — the
configuration in which retrieval is also flat in $n$ (Section 4.4 keeps the two regimes apart).

A supporting theory establishes when this is sound. A retrieval-margin analysis shows that softmax
attention recovers a target key with weight $\sigma(\beta\Delta - \log \mu)$, where $\Delta$ is the score
margin and $\mu$ the number of competing distractors; selection works because it cuts $\mu$ from $n$ to
$\kappa$, making recovery *flat in $n$*. The analysis gives the admissible routing bound that licenses summary-only
selection, a tempered (cumulant) routing score that — unlike centroid routing — sees in-block outliers, and
a closed-form variance prune test from Samuelson's inequality. The limit is then proved: cheap, lossless,
length-robust selection cannot hold simultaneously for arbitrary keys (a one-line probe argument), so SSA's
subquadratic *exactness* is licensed by the **benign geometry** of trained representations — geometry that
training can be made to manufacture, via a routability regularizer that shrinks off-target spread. Finally,
length generalization (rotary position + staged continued training reaches $32\times$ the trained length
at $0.98$ recall for $\sim\!800$ adaptation steps) and a construction pipeline that converts a dense
pretrained model into a subquadratic one by swapping the attention and briefly adapting (recovering to within
$+1.2$ perplexity of a dense model given equal training while attending $38\%$ of keys) are demonstrated.
At operational scale, a fixed-beam center-radius router and sparse GQA execute every layer and head of a
frozen Qwen2.5-0.5B over 10,000,128 tokens on one RTX Pro 6000 in 713.2 s with 25.72 GB peak allocation. A
4K dense-equivalence gate, dense and streamed 128K retrieval gates, and one semantic 10M retrieval ranking
pass. The 10M result establishes the complete execution conjunction; one retrieval instance does not
establish broad quality preservation.

---

A separate adaptive reference bounds omitted softmax mass, the subset-to-dense KL divergence, and
attention-output error from key and value summaries, with a full-scan fallback when the requested
tolerance cannot be certified sparsely (see §5.7).

### Evidence status

| Claim | Status | Scope |
|---|---|---|
| Complete transformer execution beyond 10M | demonstrated | one frozen 0.5B model, all layers and heads, one GPU |
| Subquadratic router and sparse kernel at 10M | demonstrated | bounded fixed-beam tree; exact softmax over the selected set |
| Semantic retrieval at 10M | narrow evidence | one NIAH ranking; no broad task suite or dense 10M baseline |
| Near-floor kernel scaling to 12M | demonstrated | single-head synthetic IVF benchmark |
| Omitted-mass and output-error certificates | reference implementation | CPU, adaptive, worst-case full scan |
| Supporting routing invariants | proved in this paper | self-contained statements and proofs in Appendix B; private Lean audit is corroborating provenance |
| Cheap worst-case losslessness | impossible under the stated model | arbitrary isolated targets force reads outside any strict sublinear budget |

The paper reports the system as it exists at the stated measurement points. Experiment fixtures are kept
separate when they establish different claims; results from synthetic kernel timing, real-model routing
quality, and complete-transformer execution are not combined into an unmeasured speedup or quality claim.

## 1. Introduction

A transformer layer computes, for queries $Q\in\mathbb{R}^{n\times d}$, keys $K\in\mathbb{R}^{n\times d}$,
and values $V\in\mathbb{R}^{n\times d}$,
$$
\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d}}\Big)\,V .
$$
The $n\times n$ score matrix makes both time and memory $\Theta(n^2 d)$. For contexts of $10^6$–$10^7$
tokens this is prohibitive, yet most of the matrix is near-zero: for a given query only a small set of keys
carries appreciable weight. The question is whether one can **find** that set without forming all $n^2$
scores.

Two subquadratic families answer differently. **Kernel / linear attention** replaces $\exp(\langle q,k\rangle)$
by a factorizable feature map $\phi(q)^\top\phi(k)$, giving $O(n)$ cost but a low-rank (smoothed) attention
matrix. **Sparse / selective attention** keeps the exact softmax but evaluates it only on a chosen subset of
keys. SSA is in the second family, with three design commitments:

1. **Content-dependent selection.** The chosen keys depend on the query, not only on position; this is what
   lets a query reach the one relevant block a million tokens back.
2. **Summary-only routing.** Selection uses per-block statistics (a mean, a covariance), so it does not read
   every key — that is where the subquadratic saving comes from.
3. **Exact attention over the selected set.** Within the selected keys the softmax is computed exactly, so
   there is no kernel-approximation error on the keys that matter.

The paper is organized around one tension. Selection is cheap only if it can judge a block from its summary;
it is *correct* only if those summaries do not hide a key that mattered. Sections 3–5 make both sides precise;
Section 6 shows they cannot both hold in the worst case, so something must give; Section 7 shows what gives —
the data geometry, which training shapes. Sections 8–9 turn the mechanism into a usable recipe.

---

## 2. Notation and the retrieval view

Fix a single query $q\in\mathbb{R}^d$ and keys $k_1,\dots,k_n$. Write the **logit** (score) of key $j$ as
$$
a_j \;=\; \langle q, k_j\rangle, \qquad
w_j \;=\; \frac{e^{\beta a_j}}{\sum_{l=1}^n e^{\beta a_l}}, \qquad \beta = \tfrac{1}{\sqrt d},
$$
so $w$ is the attention distribution and $\beta$ its inverse temperature. The output is $o = \sum_j w_j v_j$.

A long-context layer is, operationally, an **associative recall**: a query must place most of its weight on
the key(s) that hold the information it needs and little on the rest. Attention is therefore analyzed by the
weight it puts on a designated **target** key $k_\star$ relative to the **distractors** $\{k_j\}_{j\neq\star}$.

---

## 3. Retrieval margin and the recovery weight

Define the **margin** of the target as the gap between its logit and the typical distractor logit,
$$
\Delta \;=\; a_\star - \bar a_{\text{dist}}, \qquad \bar a_{\text{dist}} = \text{(typical } a_j,\ j\neq\star).
$$
Suppose there are $\mu$ effective distractors with logit $\approx a_\star-\Delta$. Then the target weight is
$$
w_\star \;=\; \frac{e^{\beta a_\star}}{e^{\beta a_\star}+\mu\,e^{\beta(a_\star-\Delta)}}
\;=\; \frac{1}{1+\mu\,e^{-\beta\Delta}}
\;=\; \sigma\!\big(\beta\Delta-\log\mu\big),
\tag{3.1}
$$
where $\sigma(x)=1/(1+e^{-x})$ is the logistic. Equation (3.1) is the **recovery weight**. Two consequences:

- **Detectability threshold.** The target is recovered ($w_\star>\tfrac12$) iff
  $$
  \boxed{\;\beta\,\Delta \;>\; \log\mu\;}
  \tag{3.2}
  $$
  Recovery degrades only *logarithmically* in the number of distractors.

- **Why selection helps, and why it is flat in $n$.** Dense attention pays $\mu=n$. A selector that attends
  only a budget of $\kappa$ keys pays $\mu=\kappa$, replacing the threshold $\log n$ by $\log\kappa$. If
  $\kappa$ is held *fixed* as the context grows, the threshold (3.2) does not move with $n$: retrieval
  accuracy becomes **independent of context length**. This is the entire value proposition of selection, and
  it is why a fixed-budget selector can answer single-target queries at $10^6$–$10^7$ tokens that dense
  attention, with its $\log n$ erosion, increasingly struggles with.

The catch is hidden in the phrase "a selector that attends $\kappa$ keys": the selector must *contain the
target in its budget*. Sections 4–6 are about exactly that.

---

## 4. The algorithm

### 4.1 Block partition and summaries

Partition the $n$ keys into $B$ contiguous **blocks** of size $b$ (so $Bb=n$). For block $c$ precompute, once,
the summary statistics
$$
\mu_c=\frac1b\sum_{j\in c}k_j \quad(\text{mean}),\qquad
\Sigma_c=\frac1b\sum_{j\in c}(k_j-\mu_c)(k_j-\mu_c)^\top \quad(\text{covariance}),
\tag{4.1}
$$
and a radius $R_c=\max_{j\in c}\lVert k_j-\mu_c\rVert$. In practice $\Sigma_c$ is kept diagonal,
$\sigma_c^2=\mathrm{diag}\Sigma_c$, so a block's summary is $O(d)$ numbers. (The diagonal form is fine for
the *heuristic* routing score, but it voids the Samuelson prune *certificate* unless the surrogate bound is
used — see the diagonal-summary caveat in Section 5.3.)

### 4.2 Routing score

For a query $q$, the **routing score** of block $c$ estimates the largest logit the block can contribute. The
exact quantity is the block's log-sum-exp $\mathrm{LSE}_c(q)=\log\sum_{j\in c}e^{\beta\langle q,k_j\rangle}$,
which requires reading every key. SSA uses its **second-order (cumulant) surrogate**, computable from the
summary alone:
$$
r_c(q) \;=\; \langle q,\mu_c\rangle \;+\; \tfrac{\beta}{2}\,q^\top\Sigma_c\,q
\;\;\approx\;\; \langle q,\mu_c\rangle + \tfrac{\beta}{2}\,\langle q^2,\sigma_c^2\rangle .
\tag{4.2}
$$
Equation (4.2) is the Taylor/cumulant expansion of the tempered mean
$\beta^{-1}\log\frac1b\sum_{j\in c}e^{\beta\langle q,k_j\rangle}$ about $\beta=0$: the first cumulant is the
mean logit $\langle q,\mu_c\rangle$; the second adds the **variance of the logit across the block**,
$q^\top\Sigma_c q$. The variance term is what makes routing see an in-block outlier (Section 5.2).

### 4.3 Selection and exact attention

For each query $q_i$ (at position $i$):
1. Compute $r_c(q_i)$ for all blocks $c$ whose keys are causally visible ($c$ ends at or before $i$).
2. Select the top-$k$ blocks by $r_c$, **union** a local window of the $w$ most recent blocks (recency is
   always relevant and cheap to include), giving a key set $S_i$ with $\lvert S_i\rvert=\kappa\le (k+w)b$.
3. Compute exact softmax attention restricted to $S_i$:
   $$
   o_i \;=\; \sum_{j\in S_i}\frac{e^{\beta\langle q_i,k_j\rangle}}{\sum_{l\in S_i}e^{\beta\langle q_i,k_l\rangle}}\,v_j .
   \tag{4.3}
   $$

```
Algorithm 1  SSA forward (one head)
input: Q, K, V ∈ R^{n×d}; block size b; budgets k (global), w (local)
precompute: for each block c: μ_c, Σ_c (diag)            # O(n d)
for each query q_i:                                       # parallel over i
    for each causally-visible block c:                    # routing
        r_c ← ⟨q_i, μ_c⟩ + (β/2) · ⟨q_i², σ_c²⟩
    T ← top-k blocks by r_c  ∪  w most-recent blocks      # selection
    S_i ← keys of blocks in T, with position ≤ i          # causal
    o_i ← softmax_{j∈S_i}(⟨q_i,k_j⟩/√d) · V[S_i]          # exact attention
return O
```

### 4.4 Complexity — two regimes, and which claim lives where

Routing is $O(n\,B\,d)$ if every query scores every block, and attention is $O(n\,\kappa\,d)$. Two parameter
regimes must be kept apart, because the cost claim and the retrieval-flatness claim of Section 3 do not live
in the same one:

- **Cost-optimal flat router**, $B=\sqrt n$, $b=\sqrt n$, fixed block-counts $k,w$: routing $O(n^{1.5}d)$,
  attention $O(n\,\kappa\,d)=O(n^{1.5}d)$; total $O(n^{1.5}d)$ — already a large saving over $n^2$. But here
  the key budget $\kappa=(k+w)\sqrt n$ **grows with $n$**: the detectability threshold $\log\kappa$ of
  Section 3 rises as $\tfrac12\log n$ (the $\log n$ erosion returns at half rate), and the $1/b$ mean
  attenuation of Section 5.2 worsens as $1/\sqrt n$. Subquadratic, but **not** retrieval-flat.
- **Retrieval-flat flat router**, $b$ fixed (the measured setting, $b=128$), fixed $k,w$: $\kappa$ is fixed,
  so recovery is flat in $n$ (Section 3) — but scan-all-blocks routing is $O(n\,B\,d)=O(n^2d/b)$, quadratic
  up to a constant (the measured $n^{1.76}$ router and $n^{2.12}$ maskbuild walls of Section 10).
- **Hierarchical router at fixed $b$ and fixed $\kappa$** — the configuration that delivers both. Organize
  blocks into a tree of summaries and descend it per query with branch-and-bound pruning (Section 5.1), so
  each query inspects $O(k\log B)$ nodes rather than all $B$ — a benign-geometry cost, not a worst-case
  guarantee. Routing drops toward $O(n\log n\,d)$ while $\kappa$ stays fixed. Two practical approximate
  instances are measured: the IVF router of Section 10 (0.93–0.97 block agreement), which places the isolated
  12M kernel at $\sim\!2.9\times$ the $n\,\kappa$ floor, and the fixed-beam center-radius tree used in the
  complete 10M transformer. Both occupy the cheap+length-robust-but-approximate corner of Section 6, not the
  certified one.

So "$O(n\sqrt n)$ per layer" and "recovery flat in $n$" are claims about **different** configurations, and the
hierarchical (or IVF) router at fixed $b,\kappa$ is what reconciles them. Either way the dominant $n^2$ term
is gone. In a block-sparse kernel implementation the practical speedup over a dense exact kernel was
$20.6\times$ at $n=262{,}144$ on a single accelerator (Section 10).

---

## 5. Routing theory: when a summary is enough

The routing score (4.2) is a *heuristic* surrogate. This section gives the *certified* objects — upper bounds
that make selection provably lossless — and the prune test SSA uses.

### 5.1 The admissible bound and branch-and-bound exactness

For any key $k$ in block $c$, decompose its logit around the block mean and apply Cauchy–Schwarz:
$$
\langle q,k\rangle = \langle q,\mu_c\rangle + \langle q,\,k-\mu_c\rangle
\;\le\; \langle q,\mu_c\rangle + \lVert q\rVert\,\lVert k-\mu_c\rVert
\;\le\; \underbrace{\langle q,\mu_c\rangle + \lVert q\rVert\,R_c}_{\displaystyle U_c(q)} .
\tag{5.1}
$$
$U_c(q)$ is an **admissible upper bound**: no key in block $c$ has a logit exceeding it, and it depends only on
the summary $(\mu_c,R_c)$. This licenses exact selection at adaptive cost:

> **Branch-and-bound selection.** Maintain the best *actual* logit found so far, $s^\star$. Open blocks in
> decreasing $U_c(q)$. Stop as soon as the next block has $U_c(q)\le s^\star$: by (5.1) no unopened block can
> contain a key beating $s^\star$. The result is identical to scanning all keys, but only the blocks whose
> bound clears $s^\star$ are ever read.

The cost of branch-and-bound is the number of blocks whose bound exceeds the true best — a quantity governed
entirely by how *tight* the bound (5.1) is, i.e. by the geometry of the keys (Section 6–7).

**Anisotropic refinement.** The isotropic radius $R_c$ is loose when a block's keys are spread unevenly across
directions. Using the covariance ellipsoid instead,
$$
\langle q,\,k-\mu_c\rangle \;\le\; \sqrt{\,q^\top \Sigma_c\, q\,}\cdot \rho_c,\qquad
\rho_c=\max_{j\in c}\big\lVert \Sigma_c^{-1/2}(k_j-\mu_c)\big\rVert,
\tag{5.2}
$$
which replaces $\lVert q\rVert R_c$ by the **directional radius** $\sqrt{q^\top\Sigma_c q}\cdot\rho_c$. When the
keys are anisotropic (the usual trained case), $\sqrt{q^\top\Sigma_c q}$ can be far smaller than
$\lVert q\rVert R_c$ for queries pointing along thin directions — a tighter, query-specific bound, and exactly
the quantity the cumulant score (4.2) already trades on.

**The regularised form is not the same bound.** (5.2) is admissible in exact arithmetic, but $\Sigma_c$ is
singular whenever a block holds fewer keys than the head dimension — the usual case — so every implementation
forms $S_c=\Sigma_c+\varepsilon I$ and takes $\rho_c=\max_j\lVert S_c^{-1/2}(k_j-\mu_c)\rVert$. *The
quadratic form must then be $S_c$'s as well*: Cauchy–Schwarz reads
$\langle q,k-\mu_c\rangle=\langle S_c^{1/2}q,\,S_c^{-1/2}(k-\mu_c)\rangle\le\sqrt{q^\top S_c q}\cdot\rho_c$,
and $q^\top S_c q = q^\top\Sigma_c q+\varepsilon\lVert q\rVert^2$. Pairing a radius measured in the $S_c$
metric with the *unregularised* $\sqrt{q^\top\Sigma_c q}$ yields a quantity that can fall below the block's
own maximum, and a bound that does so can discard the block holding the argmax — at which point losslessness
is no longer a property of the construction. The deficit is small and direction-dependent: on 32-dimensional
blocks of ~33 keys it appeared on 5 of 128 block–query pairs, by at most $2.8\times10^{-3}$, and *only* for
queries aligned with a block's leading direction. That last clause is the practical warning, because it is the
realistic case — a router's queries are not orthogonal to the keys they route to — and it is why a
random-query admissibility test at a single geometry does not detect the error.

### 5.2 Why second order: centroid routing is blind to outliers

Centroid routing uses only $r_c=\langle q,\mu_c\rangle$. A single key $k_\star$ in a block of size $b$
contributes $\tfrac1b k_\star$ to $\mu_c$, so its signal in the mean is attenuated by $1/b$: a lone target in a
large block is invisible to centroid routing. The variance term in (4.2) repairs this, because an outlier
aligned with $q$ inflates $q^\top\Sigma_c q$. Concretely, if block $c$ holds one key with $\langle
q,k_\star\rangle=a_\star$ and $b-1$ keys with logit $\approx 0$, then $\langle q,\mu_c\rangle\approx a_\star/b$
but $q^\top\Sigma_c q\approx a_\star^2/b$, so the second-order score $r_c\approx a_\star/b +
(\beta/2)a_\star^2/b$ carries the quadratic outlier signal the mean discards.

**The tempered family and its bias.** It is the *normalized* tempered mean
$g_c^{(\beta)}=\beta^{-1}\log\frac1b\sum_{j\in c}e^{\beta\langle q,k_j\rangle}$ — the object expanded in
(4.2)/A.2 — that interpolates between the mean logit ($\beta\to0$) and the block max ($\beta\to\infty$). The
unnormalized log-sum-exp $r_c^{(\beta)}=\beta^{-1}\log\sum_{j\in c}e^{\beta\langle q,k_j\rangle}
= g_c^{(\beta)}+(\log b)/\beta$ differs from it only by a block-size constant (irrelevant when ranking
equal-size blocks, where the two orderings coincide) and is sandwiched by
$$
\max_{j\in c}\langle q,k_j\rangle \;\le\; r_c^{(\beta)} \;\le\; \max_{j\in c}\langle q,k_j\rangle + \frac{\log b}{\beta}.
\tag{5.3}
$$
So the tempered score estimates the block's *best* logit — exactly what selection wants — with bias at most
$(\log b)/\beta$. Small $\beta$ smooths over outliers (the centroid failure); large $\beta$ removes the bias
but amplifies noise and loses the averaging that makes summaries stable. The cumulant form (4.2) is the
second-order Taylor truncation of $g_c^{(\beta)}$ about $\beta=0$, used at the measured optimum
$\beta\approx 2$; its truncation error is the Lagrange remainder $(\beta^2/6)\,K_c'''(\xi)$ for some
$\xi\in(0,\beta)$, with $K_c$ the cumulant generating function of the in-block logits — a third-cumulant
(skew) term that vanishes for light-tailed blocks and is *positive* for a positively-skewed (spiked) block,
so the second-order score under-estimates exactly the outlier blocks it most needs to see at moderate
$\beta$. That residual gap is the mechanism behind the isolated-needle failures measured in Section 10, and
the reason the implementation offers an Edgeworth variant that adds the diagonal third-cumulant term.

### 5.3 The Samuelson prune test

A closed-form, summary-only test for *discarding* a block uses **Samuelson's inequality**: for any reals
$s_1,\dots,s_m$ with mean $\bar s$ and population variance $\mathrm{Var}=\frac1m\sum_j(s_j-\bar s)^2$, every
element obeys
$$
(s_i-\bar s)^2 \;\le\; (m-1)\,\mathrm{Var}, \qquad\text{equivalently}\qquad
\max_j s_j \;\le\; \bar s + \sqrt{(m-1)\,\mathrm{Var}} .
\tag{5.4}
$$
Apply it to the in-block logits $s_j=\langle q,k_j\rangle$, whose mean is $\langle q,\mu_c\rangle$ and whose
variance is $q^\top\Sigma_c q$. Then the block's best logit is bounded by
$\langle q,\mu_c\rangle+\sqrt{(b-1)\,q^\top\Sigma_c q}$, giving the **prune gate**: block $c$ can be safely
discarded against a threshold $\tau=s^\star$ whenever
$$
\boxed{\;(s^\star-\langle q,\mu_c\rangle)^2 \;>\; (b-1)\,q^\top\Sigma_c q\quad\text{and}\quad \langle q,\mu_c\rangle<s^\star\;}
\tag{5.5}
$$
i.e. when the **margin** of the current best over the block mean exceeds $\sqrt{(b-1)\cdot\text{spread}}$. The
test needs only $(\mu_c,\Sigma_c)$. It is *sufficient* (it never wrongly prunes) but not necessary. It
sharpens the radius bound (5.1) when the block's spread along $q$ is small relative to its worst-case radius
($\sqrt{(b-1)\,q^\top\Sigma_c q}\ll\lVert q\rVert R_c$); neither bound dominates in general —
$\sqrt{(b-1)\,q^\top\Sigma_c q}$ can exceed $\lVert q\rVert R_c$ by up to a factor $\sqrt{b-1}$ for
spread-out blocks — so the implementation takes the minimum of the two. Equation (5.5) is the
operational core of cheap exact selection: it fires — and the block is skipped — precisely when the off-target
spread $q^\top\Sigma_c q$ is small, which is the benign-geometry condition of Section 7.

**Diagonal-summary caveat.** (5.4)–(5.5) are sound with the *full* quadratic form
$q^\top\Sigma_c q=\mathrm{Var}_{j\in c}\langle q,k_j\rangle$. With the diagonal summary of (4.1), the proxy
$\langle q^2,\sigma_c^2\rangle$ *under-estimates* the true logit variance whenever cross-covariances are
positive, and the gate can then wrongly prune: **run on diagonal summaries, (5.5) is a heuristic, not a
certificate.** A sound $O(d)$-summary surrogate exists: by the triangle inequality in $L^2$ over the block,
$\mathrm{Var}_{j\in c}\langle q,k_j\rangle\le\big(\sum_i\lvert q_i\rvert\,\sigma_{c,i}\big)^2$, so
substituting $(\sum_i\lvert q_i\rvert\,\sigma_{c,i})^2$ for $q^\top\Sigma_c q$ in (5.5) keeps the gate
admissible at the same summary cost (looser, so it fires less often). The prune-rate measurements reported in
this paper compute the per-member logit variance directly — the full quadratic form — so they certify the
full-covariance gate, not the diagonal shortcut.

---

### 5.4 The summary-only floor: Samuelson is the tightest, and how far it is from the partition's own limit

Equation (prune) is labelled *sufficient*. That labelling is right, and it leaves an obvious question
unanswered: could a *cleverer* function of the same summary prune more? It could not, and the reason is that
Samuelson's inequality is not merely valid but **attained**.

> **Proposition (Samuelson is the tightest summary-only bound).** Fix `m ≥ 2`, a mean `s̄` and a population
> variance `Var > 0`. The configuration `s₁ = s̄ + √((m−1)Var)`, `s₂ = … = s_m = s̄ − √(Var/(m−1))` has exactly
> mean `s̄` and variance `Var`, and its maximum equals `s̄ + √((m−1)Var)`. Consequently any bound
> `U < s̄ + √((m−1)Var)` is violated by a block whose summary `(μ_c, Σ_c, b)` the router cannot distinguish
> from the one it read. **No admissible bound computable from `(μ_c, Σ_c, b)` alone is tighter.**

The arithmetic is immediate and is verified numerically at `m ∈ {2,4,8,16,64,256}` in `ssa/bound_floor.py`.
The content is the quantifier: the bound is not conservative *given the summary*, so every improvement in
pruning must come from a better **partition** or from **reading keys** — never from a better formula on the
same statistics. This is the necessary-side companion to the benign-geometry condition, which is sufficient
only.

**What the partition costs, and what the summary costs.** The tightest admissible bound that exists at all is
the *oracle* `U_c = max_{k∈c}⟨q,k⟩`; it is not a router (computing it reads the block), but it is the floor
the partition imposes on *any* correct branch-and-bound, so
`cost(oracle) ≤ cost(ellipsoidal) ≤ cost(Samuelson)`.

Measured on clustered synthetic keys (`n=4096`, `d=64`, `B=64` k-means blocks, 120 queries per row; all four
bounds admissible, so recall is 1.000 throughout):

| spread | oracle | ellipsoidal | Samuelson | summary ÷ floor |
|---:|---:|---:|---:|---:|
| 0.02 | 89.0 | 203.2 | 720.3 | 8.1× |
| 0.05 | 81.0 | 543.6 | 975.4 | 12.0× |
| 0.10 | 74.6 | 1765.9 | 2165.0 | 29.0× |
| 0.20 | 70.2 | 3564.3 | 3685.8 | 52.5× |
| 0.40 | 64.1 | 4022.8 | 4045.3 | 63.1× |

Three readings. First, the routability programme has a **measurable ceiling**: driving the geometry benign
moves the summary price from 63× the floor to 8×, monotonically — but not to 1×, and the proposition says why
it cannot. Second, **reading keys is worth roughly 3.5×** at benign geometry (203.2 against 720.3 at spread
0.02), a quantitative justification for the anisotropic refinement; `ρ_c` is not a minor sharpening but
most of the distance to the floor — *at these shapes*, a qualification the next paragraph makes precise. Third, the partition price is nearly flat in the spread (89.0 → 64.1) while
the summary price moves by a factor of six: **the geometry is a fact about summaries, not about partitions.**

**`ρ_c` inverts when a block holds far fewer keys than dimensions.** The table is `d = 64` with blocks of
~64 keys. On real Gemma-2 layer-6 keys (`d = 2304`, blocks of 64–256) the same stored scalar **costs**
rather than buys — 0.7×, 0.9×, 1.0× at `n = 4096/16384/65536` — because `Σ_c` is then deeply rank-deficient,
`S_c = Σ_c + εI` is dominated by `ε`, and the honest `√(qᵀΣ_c q + ε‖q‖²)` is loose exactly where the whitened
radius is large. The anisotropic refinement is therefore a recommendation **scoped to `b ≳ d`**. Omitting
the regularization term produces an inadmissible estimate and the wrong ordering of the two bounds on real
keys.

**Scope.** k-means blocks on synthetic clustered keys at one `(n,d,B)` and one query-noise level — the
adaptive/IVF regime, not the contiguous-position blocking of the flat kernel; the oracle is a reference and
not an achievable router; and the floor is about *lossless* selection, so a budget-κ lossy router may sit
below it and SSA's does. The proposition itself is geometry-free and carries none of these caveats.

**Ellipsoidal refinement and the remaining room.** `ρ_c` and `R_c` are *query-independent*, hence precomputed
and stored as one scalar per block: the ellipsoidal bound is summary-only at query time too, and the
hierarchy is about summary *size* rather than summaries versus keys. That strengthens the reading — at spread
0.02 one extra stored scalar takes the bound from 8.7× to 2.5× the floor, i.e. the implemented bound reads
5.33% of keys against a floor of 2.10%. Measurements of two further refinements show no general gain: seeding
the incumbent with precomputed block representatives saves *exactly* zero (B&B opens blocks in
decreasing `U_c`, so the first block opened already sets `s★` to the true maximum — cost is bound-driven, not
incumbent-driven), and an axis-aligned box bound in a shared basis is *worse* than the ellipsoid, with `min` of
the two buying 8% at the most benign geometry alone. The remaining lever is the **partition**.

**Relation to the formalized ceiling.** Maximum-score estimation has a related obstruction in the substrate
development: a single score standing for a whole fibre of completions carries an error floor set by the spread
of the objective across that fibre (`Universal/Potential/PartialScore.lean`,
`score_error_ge_of_reach_split` and `not_exactOnReach_of_reach_split`). Partial object = the summary,
completions = blocks carrying it, objective = the block's true maximum logit. The instantiation is stated in
prose and is **not** machine-checked; see `docs/substrate_math_imports.md`. This score-approximation floor
is distinct from the attained-upper-bound argument above; variation among individual logits in a fixed
block does not by itself preclude exact identification of that block's maximum or correct block selection.

The abstract skeleton is machine-checked, axiom-pure, in `Universal/Potential/AdmissibleBound.lean`: a
pointwise tighter bound provably drops a superset of the parts (`a_higher_bound_drops_a_subset`,
`dropped_card_mono`) — §4.1's qualitative claim as a theorem; the family maximum is the least admissible
bound (`familyMax_le_of_isAdmissible`); an attained bound admits nothing smaller
(`no_bound_below_an_attained_one`), the proposition above in abstract form; and the minimum of two admissible
bounds is admissible (`min_isAdmissible`), which licenses the implementation's `min` of the isotropic and
ellipsoidal bounds. Finally,
`at_the_floor_a_maximal_threshold_drops_every_part` predicts the null above: at the floor a maximal threshold
drops every part, so a search still reading parts is paying for bounds above the floor and a better incumbent
cannot help it.

### 5.5 The cost of asking: what a summary bound charges on real keys

Everything above counts *keys read*. That is the right metric for §5.4, where only the bound varies, and
it is the wrong one as soon as the **block size** varies, because evaluating the bound is neither free
nor cheap.

**An exact summary bound cannot beat a full scan, at any block size.** Samuelson's bound needs
$q^\top\Sigma_c q$. Dense that is $O(d^2)$ per block; from the rank-$b$ centred factor $X_c$ it is
$\lVert X_c q\rVert^2/b$, i.e. $O(bd)$ — *exactly the cost of scoring the block*. Over $B=n/b$ blocks:

$$
B(1+b) \;=\; \tfrac{n}{b}(1+b) \;=\; n + \tfrac{n}{b} \;>\; n \qquad \text{for every } b .
\tag{5.5}
$$

Arithmetic, with no hypothesis about the keys. A search that evaluates the exact summary bound on every
block **has already paid for a full pass over the data before it skips anything.** Measured against a
scan on real Gemma-2 layer-6 keys at $n=65536$: 1.062× at $b=16$ falling to 1.004× at $b=256$ — never
below one, and by (5.5) it cannot be.

**The routing is excellent; the price of asking is the problem.** The same measurement, counting keys
alone, has contiguous blocks at $b=16$ under the exact bound reading **204 of 65536 keys** — a 321×
reduction — with k-means at 1519. Real transformer keys *route extremely well*. Every difficulty is on
the cost side of the ledger, not the geometry side.

**The escape, and its absence.** A rank-$r$ sketch of $\Sigma_c$, made admissible by the
discarded-eigenvalue tail
$q^\top\Sigma_c q \le \sum_{i\le r}\lambda_i\langle v_i,q\rangle^2 + \lambda_{r+1}\lVert q\rVert^2$,
costs $B(1+r) = \frac{n}{b}(1+r)$, below a scan exactly when $r<b$. So the whole question is one number:
how small can $r$ be and still prune? On real keys it cannot be smaller than $b$ at all. At $b=16$ with
contiguous blocks:

| rank $r$ | 2 | 4 | 6 | 8 | 10 | 12 | **16** |
|---|---|---|---|---|---|---|---|
| % of keys read | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | **0.3** |

A step function, not a gradient.

**Why: the block spectrum does not decay.** The tail term is exactly zero at $r=b$ and positive below it,
so the step is a statement about the spectrum of $\Sigma_c$ — and it was measured directly. Across
$b\in\{16,32,64,128\}$ and both partitions, $\lambda_2/\lambda_1$ lies in **[0.70, 0.84]** — the *first*
eigenvalue drop is already small, where a genuinely low-rank block would sit near 0.01 — and the
participation ratio $(\sum\lambda)^2/\sum\lambda^2$ never falls below a quarter of the block size,
reaching $0.72b$ for contiguous blocks at $b=16$. Between a quarter and three quarters of all available
directions carry real variance. **There is no small tail to discard**, so no truncation is both cheap and
tight.

**What this does and does not settle.** Together: on real transformer keys, *lossless selection driven by
a $(\mu_c,\Sigma_c,b)$ summary costs more than reading every key*, and no rank truncation of that
summary escapes it. It does **not** say summarising a block is hopeless — it bounds this family of
bounds. Nor does it touch budget-$\kappa$ **lossy** routing, a different object with a different
accounting and what SSA actually ships; §5.4's floor was always about lossless selection and this is its
cost-side companion. The measurements are one model at one layer; the arithmetic of (5.5) is the only
part that is geometry-free.


### 5.6 Sharing one selection across a group of queries

Everything above prunes for *one* query. A long context does not have one query: the number of query positions
grows with the context exactly as the number of keys does, and kernels therefore select blocks once for a
group — a decode chunk, the query heads of a GQA group, a tile of positions — and read the selected blocks for
every member. That is a different object, and it is not automatically lossless: **a bound admissible for one
query says nothing about another query's argmax.**

Write $Q$ for the group, $|Q|=m_Q$. A shared bound $U$ is safe for every member iff it dominates the group's
**top claim** $\max_{q\in Q}U_c(q)$ blockwise, and the top claim is the *least* such bound. Machine-checked in
`Universal/Potential/ChainedPrune.lean`: `safe_for_every_stage_iff_safe_over_the_top_claim`,
`no_safe_bound_holds_below_the_top_claim`, and the negative
`a_prune_can_drop_the_greatest_part_of_a_second_objective` — a prune beyond reproach for its own objective can
remove a second objective's best part outright, and the part is then gone rather than approximated.

Two consequences carry into the implementation. The shared **threshold must be the group's weakest incumbent**,
not its strongest: a block whose bound has fallen under the best member's incumbent may still hold a weaker
member's argmax. And **group size costs retention monotonically** — each member admitted raises the top claim,
and a raised bound prunes strictly less — so in the limit where every block is above threshold for *some*
member, the safe prune drops nothing at all.

**What it buys.** Total work is keys scored plus bound evaluations, since the arms differ in how many bounds
they evaluate and counting keys alone credits a saving paid for elsewhere. $n=8192$, $d=64$, $B=128$, key
spread $0.05$; every arm returns the true argmax for every member.

| $q$-spread | $m_Q$ | per-query | top claim | rank-8 | isotropic |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 8   | 9,014   | **1.99×** | 2.31× | 1.13× |
| 0.02 | 32  | 36,293  | **2.01×** | 1.09× | 0.63× |
| 0.02 | 128 | 142,663 | **1.97×** | 0.55× | 0.43× |
| 0.02 | 512 | 572,668 | **1.98×** | 0.18× | 0.17× |
| 0.10 | 32  | 42,729  | 0.51× | 0.16× | 0.16× |
| 0.30 | 32  | 112,660 | 0.43× | 0.43× | 0.43× |

Sharing the block *choice* is worth about 2×, and — the useful part — **flat in group size** up to $m_Q=512$,
because the blocks a tight group wants are the same blocks. As the group spreads it stops paying and then
costs: the quantitative face of the drop-nothing limit above.

**A negative result: summarising the query side does not pay.** The obvious way to make the top claim cheap is
to apply this paper's own hierarchy to the queries — the top claim costs $O(m_Q)$ per block, whereas
$\max_{q\in Q}\langle q,\mu_c\rangle\le\langle\mu_Q,\mu_c\rangle+\sqrt{(m_Q-1)\,\mu_c^\top\Sigma_Q\,\mu_c}$
costs $O(d)$ once the group's summary is formed. The resulting bounds are admissible and exact, and they are
*worse at every group size measured*, degrading from 1.13× to 0.17× as $m_Q$ grows.

The mechanism is structural and is the point worth keeping. §5.4's proposition establishes Samuelson's bound as
tightest *because it is attained* — by one member at $\bar{x}+s\sqrt{m-1}$ with the rest at $\bar{x}$. That is a
worst-case tightness, and a worst case is a statement about the adversarial configuration, not the typical one.
On the key side the adversarial block is exactly what a bound must survive. A *tight query group* is the
opposite shape: its members cluster near their mean, which is where the $\sqrt{m_Q-1}$ factor is furthest from
the truth. The same inequality that is tight where it was proved is loose where it is reused, and the
$O(1)$-per-block evaluation it saves is smaller than the pruning it gives up. **The room in shared selection is
in the block choice, not in the query summary.**

### 5.7 Certifying omitted mass and attention-output error

An exact argmax or routing-metric top-$\kappa$ certificate does not
control the total softmax weight of omitted keys. In particular, many
individually small logits can carry most of the partition sum. For a
nonempty kept set $S$, let $D=S^c$ within the visible causal prefix, and
define

$$
Z_S=\sum_{i\in S}e^{\beta a_i},\qquad
 Z_D=\sum_{i\in D}e^{\beta a_i},\qquad
 \widehat o=Z_S^{-1}\sum_{i\in S}e^{\beta a_i}v_i .
$$

 Suppose $D$ is
partitioned into unopened blocks. Each block has size $b_c$, key mean
$\mu_c$, key radius $R_c$, value mean $\nu_c$, and value radius $r_c^V$.
For $\beta\ge0$ set

$$
L_c=b_c e^{\beta\langle q, \mu_c\rangle},\qquad
 A_c=b_c e^{\beta(\langle q, \mu_c\rangle+\lVert q\rVert R_c)},\qquad
 L=\sum_{c\subseteq D}L_c,\quad A=\sum_{c\subseteq D}A_c .
$$

 Jensen’s
inequality and the admissible radius bound give $L_c\le Z_c\le A_c$
without scoring the unopened keys. Other admissible maximum-logit bounds
can replace the radius term, with their evaluation cost charged
separately.

**Proposition** (Subset-attention certificate). Let $p$ be the dense
softmax distribution and $\widehat p$ its renormalized restriction to
$S$, extended by zero outside $S$. With $\delta=Z_D/(Z_S+Z_D)$,


$$
\begin{aligned}
 \operatorname{TV}(\widehat p,p)&=\delta\le\overline\delta:=\frac{A}{Z_S+A},\\
 \operatorname{KL}(\widehat p\Vert p)&=-\log(1-\delta)
       \le\log(1+A/Z_S).
\end{aligned}
$$

 For $d_c=\lVert\nu_c-\widehat o\rVert+r_c^V$ and
nonempty $D$, the output obeys

$$
\lVert o-\widehat o\rVert\le
 \min\left\{\overline\delta\max_{c\subseteq D}d_c,\quad
       \frac{\sum_{c\subseteq D}A_c d_c}{Z_S+L}\right\}.
$$

 All three
bounds are zero when $D$ is empty.

*Proof.* On $S$, $\widehat p_i=p_i/(1-\delta)$, so the absolute weight
difference sums to $\delta$ on each of $S$ and $D$. The likelihood ratio
on $S$ is constant, giving the KL identity. Both expressions increase
with $Z_D$, which is at most $A$. For the value bound, subtract
$\widehat o$ from the full weighted average:


$$
o-\widehat o=\frac{\sum_{i\in D}e^{\beta a_i}(v_i-\widehat o)}{Z_S+Z_D}.
$$


Each omitted value in $c$ lies within $d_c$ of $\widehat o$. Bounding
all distances by their maximum gives the first term; bounding each block
numerator above by $A_c d_c$ and the denominator below by $Z_S+L$ gives
the second. ◻

The direction of KL is essential: for finite logits and a proper
restriction, $\operatorname{KL}(p\Vert\widehat p)=+\infty$. At equal
logits, $\delta=1-|S|/n$ and
$\operatorname{KL}(\widehat p\Vert p)=\log(n/|S|)$, the
nested-uniform-support identity formalized by `klDiv_uniformSupport` in
`Universal/Potential/Entropy/UniformSupport.lean`. The finite-vector
Pinsker theorem `totalVariation_le_sqrt_klDiv` in the same directory’s
`TotalVariation.lean` allows zeros in its first distribution and a
strictly positive second distribution; it applies in this direction.
The mass identity above directly computes the
TV quantity for a restriction, so using Pinsker would give a weaker
bound. The partition bound consumes the reasoning of
`logSumExp_max_sandwich`; the value bound uses the barycenter-truncation
reasoning of `Universal/Generator/Dissipative/SelectionGeometry.lean`.
These source theorems are machine-checked in Substrate. Their
composition into
the proposition above is proved here in ordinary
mathematics; the SSA instantiation and Python implementation are not
Lean proofs.

#### Adaptive implementation and cost.

`ssa/certified_attention.py` builds immutable contiguous-block key/value
summaries and opens blocks in decreasing $\log A_c$, optionally starting
with blocks selected by another router. It evaluates attention exactly
on opened keys, checks every requested mass or output tolerance, and
doubles the number of opened blocks after a failed check. A block cap
returns an explicitly uncertified result if the tolerance remains unmet.
A partially visible causal block is opened using only its visible keys;
its full-block summary is excluded from the decision. Log-space
partition sums and suffix reductions avoid exponential overflow and
subtraction of nearly equal masses. Bounds are evaluated in float64 with
a small outward score cushion; this is not an interval-arithmetic
guarantee.

Construction costs $O(n(d+d_v))$. For $B$ blocks, each query pays
$O(Bd)$ for key bounds, $O(B\log B)$ for ordering, $O(Bd_v\log B)$ for
value-certificate checks in the worst case, and $O(|S|(d+d_v))$ for
opened keys and values. The reference reports key scores, key-bound
evaluations, certificate checks, and value-bound evaluations separately.
It can scan all keys on diffuse geometry and does not establish a
sublinear per-query cost at fixed block size or a GPU speedup. The exact
covariance factor whose cost is analyzed in
§5.5 is not evaluated by this
radius-based reference.

| geometry                  | keys scored |   $\overline\delta$ |        output bound |       measured error |
|:--------------------------|------------:|--------------------:|--------------------:|---------------------:|
| concentrated              |        $64$ | $9.54\times10^{-6}$ | $4.62\times10^{-5}$ |  $1.79\times10^{-6}$ |
| flat logits               |      $4096$ |                 $0$ |                 $0$ | $8.58\times10^{-17}$ |
| equal values, flat logits |        $64$ |          $0.984375$ |                 $0$ |                  $0$ |

The deterministic CPU fixture uses $n=4096$, $d=32$, $d_v=8$, $b=64$,
$\beta=4$, seed zero. The first two rows request omitted mass at most
$10^{-3}$; the last requests only output error at most $10^{-12}$. Every
row evaluates $64$ key bounds. Certificate checks number $1,7,1$, and
value-bound evaluations number $63,321,63$, respectively. The full-scan
residual is floating-point roundoff. Equal values permit exact output
even when almost all attention mass is omitted; conversely, a small
omitted weight can matter when its value is sufficiently distinct. These
are controlled mechanism measurements, reproducible with
`python -m ssa.certified_attention`. Six additional checks compare the certificates against
float64 CUDA SDPA on an RTX 4080, using concentrated logits, flat logits, and equal values at
$n=1024$, $d=32$, $d_v=8$, $b=32$, with visible prefixes of $1024$ and $997$ keys.
These validate numerical agreement with an independent attention implementation, including partial
causal blocks. The selector runs on CPU; GPU routing speed and real-model quality remain unmeasured.

## 6. The trilemma and the impossibility

Call a selector **cheap** if it reads $o(n)$ keys, **lossless** if it attends every key dense attention would
weight non-negligibly, and **length-robust** if its accuracy is flat in $n$. The bounds above suffice to state the
fundamental limit.

> **Proposition (no free selection).** No selector can be simultaneously cheap and lossless for *arbitrary*
> keys.
>
> *Proof.* Suppose a selector reads a set $\mathcal{R}$ of keys with $\lvert\mathcal{R}\rvert<n$, and let $j_0\notin\mathcal{R}$.
> Construct a probe input identical on $\mathcal{R}$ but with $k_{j_0}=c\,q$ for $c$ large. Dense attention puts
> weight $\to 1$ on $j_0$, so the correct output is $v_{j_0}$. The selector, never having read $j_0$, returns
> the same output as on the unmodified input, which is independent of $v_{j_0}$. Hence it is lossy on this
> input. $\square$
>
> The argument covers any *deterministic* selector, adaptive or not: run it, let $\mathcal R$ be the keys its
> execution actually read, and perturb an unread one — the execution, hence the output, is unchanged. The
> adaptive case is machine-checked (`lossless_adaptive_reads_every_key`, over an explicit
> decision-tree probe model in which the read set is the per-input queried path). A
> *randomized* selector reading $o(n)$ keys misses a uniformly-planted spike with probability $1-o(1)$, so
> the conclusion survives in expectation; we state that extension as a remark, not a formalized claim.

> **Note on formalization.** The proofs given in this paper, including the complete routing-invariant
> statements in Appendix B, are the public mathematical arguments. Their
> formal counterparts — `subquadratic_forces_skip`, `flat_router_work`, `lossless_selector_reads_every_key`,
> `lossless_adaptive_reads_every_key`, `capacity_pigeonhole_tension`, `read_capacity_le_dim`,
> `hierarchical_prune`, and the rest of the `(proved)` results — are **machine-checked in a separate Lean 4
> development** (namespace `Substrate.Inference.Algebra.PhaseTransition`, sources under `Substrate/Inference/Substrate/`
> and inference recognition modules, with supporting results in `Substrate.Universal`; Lean + Mathlib), each confirmed
> `sorry`-free and axiom-clean (`#print axioms` → only `[propext, Classical.choice, Quot.sound]`). That
> development is **not bundled in this repository**, so a reader of this artifact alone cannot re-run the
> checker, but no routing-invariant claim is available only by private reference. The formal statements are
> deliberately modest — finite-counting / probe-model lower bounds and
> sufficient conditions, not the grander informal reading (e.g. `subquadratic_forces_skip` proves only that
> sub-`Q·B` work must skip some block, not that any specific system achieves a quality-preserving 1,000×).

The proposition says losslessness for *worst-case* keys forces $\lvert\mathcal{R}\rvert=n$ — no summary suffices, because a
summary can always hide a spike. A quantitative companion holds under fine-grained complexity assumptions
(SETH): even *approximating* the attention output requires $n^{2-o(1)}$ time once entries reach
$\omega(\sqrt{\log n})$, and the truly subquadratic algorithm that exists in the bounded-entry regime is
itself an approximation (a low-rank polynomial method), not exact computation (Alman–Song). What is *proved*
here is the two-way impossibility — cheap $\wedge$ lossless, worst case. The three-way reading below is an
organizing **taxonomy** whose third axis is measured in Sections 7–10, not a proved trichotomy: at most
**two** of {cheap, lossless, length-robust} are available at once:

| keep | give up | what it is |
|---|---|---|
| lossless + length-robust | cheap | dense attention — reads every key |
| cheap + length-robust | lossless | approximate selection — fast, flat, but can miss a worst-case spike |
| cheap + lossless | length-robust | branch-and-bound on **benign** keys — bounds tight, cost stays low (this section's escape) |

Note the third row concedes length-robustness of **cost**, not of accuracy: admissible branch-and-bound never
returns a wrong answer — off benign geometry it degrades by *reading more* (toward everything), whereas row 2
trades accuracy. "Length-robust" in the first two rows is the accuracy sense of Section 3; keeping the two
currencies straight is part of the taxonomy's honesty.

The escape from the trilemma is the third row: cheapness *and* losslessness are jointly available **when the
bounds (5.1)/(5.2)/(5.5) are tight**, which is a property of the key geometry, not of the algorithm. SSA is
therefore not a universal subquadratic exact attention — no such thing exists — but a mechanism that is cheap,
lossless, *and* length-robust **on benign geometry**, and merely cheap-and-length-robust-but-approximate
otherwise. The next section is about making the geometry benign.

---

## 7. Manufacturing routability

### 7.1 Benign geometry, precisely

The branch-and-bound cost and the prune gate (5.5) are governed by the **off-target spread** $q^\top\Sigma_c
q$ for the blocks a query does *not* need, relative to the **margin** $\Delta$ to the block it does. Geometry
is *benign* for a query $q$ when, for every non-target block $c$,
$$
\langle q,\mu_c\rangle + \sqrt{(b-1)\,q^\top\Sigma_c q}\;<\; s^\star(q),
\tag{7.1}
$$
i.e. the prune gate (5.5) fires everywhere except the target's block. Then exactly one block (plus the local
window) is opened, branch-and-bound reads $O(b)$ keys, and selection is cheap *and* lossless. Isotropic keys
violate (7.1) — every block's bound is large and uninformative, so nothing prunes and cost reverts to dense.

### 7.2 The routability regularizer

Benign geometry is not given; it can be *trained in*. Add to the language-model loss a penalty that shrinks
the off-target spread along the directions queries actually use:
$$
\mathcal{L} \;=\; \mathcal{L}_{\text{LM}} \;+\; \lambda \sum_{i}\sum_{c\,\neq\, \text{target}(i)} q_i^\top \Sigma_c\, q_i .
\tag{7.2}
$$
Minimizing $\sum_{c\neq\text{target}} q^\top\Sigma_c q$ is exactly minimizing the quantity that appears under
the square root in the prune gate (5.5), so as training proceeds the gate fires for more non-target blocks and
the certified bound (5.1)/(5.2) tightens. Measured: co-training with (7.2) drove the *lossless*
branch-and-bound cost from $26.5\%$ of keys to $4.2\%$ — a $6\times$ reduction — at **no** loss of retrieval
accuracy. Training is what manufactures the geometry that the impossibility result (Section 6) says cheap
exactness requires.

### 7.3 The entry-magnitude split: selection vs. linear attention

A complementary fact decides *which* subquadratic route a given layer should take. Let $B$ be the
characteristic magnitude of the logits $\langle q,k\rangle$ (the "entry scale"). The exponentiated score
matrix $[\,e^{B\langle q_i,k_j\rangle}\,]$ has effective rank that **grows with $B$**: for small $B$ it is
nearly low-rank (so a kernel/linear-attention factorization $\phi(q)^\top\phi(k)$ is accurate and $O(n)$); for
large $B$ it is full-rank (so no linear map approximates it, and one must *select*). Empirically, the effective
rank of $e^{B\,KK^\top}$ on $256$ keys rose from $2.7$ at $B=0.5$ to $256$ (full) by $B=8$.

The selection route, by contrast, is **scale-invariant**: in the prune gate (5.5) both the squared margin and
the spread scale as $B^2$, so the gate's truth value is unchanged by $B$. Selection therefore works at *any*
entry scale, whereas linear attention works only in the small-$B$ (smooth) regime. Sharp, long-range retrieval
is the large-$B$ regime — which is why selection, not linearization, is the route for long-context exact
recall.

---

## 8. Length generalization and staged extension

A selector that is flat in $n$ (Section 3) still needs the *model* to behave at lengths beyond those it was
trained on. The position encoding decides whether it can.

### 8.1 Position-invariant routing under rotary embeddings

With rotary position embeddings (RoPE), a query/key pair interacts only through their **relative** offset:
the logit $\langle q_i,k_j\rangle$ depends on $i-j$, not on $i,j$ absolutely. A model that has learned
*content* routing — match a query to the key whose content binds it, at whatever offset — therefore transfers
to offsets it never saw, because (i) the decisive relative structure (e.g. a key-to-value offset of $+1$) is
constant at any length and (ii) content matching is position-free. A model with *learned absolute* position
embeddings cannot: its embeddings past the trained length are untrained. This predicts, and experiments confirm,
zero-shot extrapolation of $\sim\!2$–$4\times$ for RoPE and immediate collapse for learned-absolute position.

### 8.2 The staging ladder

Zero-shot extrapolation buys a factor, not an unbounded range. To go further, **stage**: extend the context,
briefly continue-train (adapt) at the new length, extend again — doubling each rung. The economics are
favorable precisely because routing is position-invariant: each rung extrapolates only $2\times$ from the last
*adapted* length, so every adaptation starts from the $\sim\!2\times$ zero-shot recall and only has to clean it
up. The adaptation cost is small and roughly *flat in length* rather than growing.

Measured (a $6$-layer model on a synthetic multi-query associative-recall task; base trained at $48$ pairs):

| rung | multiple | zero-shot recall | adapt steps | recall after adapt | recall with SSA |
|---|---|---|---|---|---|
| 96   | 2×  | 0.993 | 100 | 1.000 | 1.000 |
| 192  | 4×  | 0.904 | 100 | 0.999 | 1.000 |
| 384  | 8×  | 0.810 | 100 | 1.000 | 0.995 |
| 768  | 16× | 0.672 | 100 | 0.996 | 0.996 |
| 1536 | 32× | 0.516 | 400 | 0.982 | 0.979 |

The base schedule cost $7600$ steps; the *entire* climb from $16\times$ (where zero-shot recall is already
near the floor) to $32\times$ at recall $0.982$ cost only $800$ additional steps — about $10\%$ of the base,
and roughly constant per rung. The final column confirms that the same adapted model runs under SSA selection
($\le15\%$ of keys attended) with essentially no recall loss at every rung.

---

## 9. The construction pipeline: dense → subquadratic

SSA can be retrofitted onto an existing dense pretrained model, which is how a frontier long-context system is
built without pretraining from scratch:

1. **Start** from a dense pretrained base.
2. **Swap** dense attention for SSA (Algorithm 1) in every layer.
3. The swap **degrades** quality, because the base was trained expecting every key.
4. **Adapt** by continued pretraining; the keys co-adapt to the sparse routing (and, with the regularizer
   (7.2), become routable).
5. **Stage** the context extension (Section 8).

Measured on a $124$M-parameter dense base, held-out perplexity (lower is better):

| condition | perplexity |
|---|---|
| dense base (off-domain start) | 32.5 |
| + SSA swap, no adaptation | 45.8 |
| dense base + equal in-domain continued training (control) | 23.9 |
| **SSA-swapped + equal in-domain continued training** | **25.1** |

The swap costs $+13.2$ perplexity; an *equal-budget* adaptation recovers $94\%$ of that gap, landing within
$+1.2$ of a dense model given the **same** continued training — while attending only $\sim\!38\%$ of keys at
this configuration. The control matters: continued training lowers perplexity for either attention (domain
adaptation), so "recovery" must be measured against a dense model given the same steps, not against the
off-domain start. The residual is the price of sparsity at this small budget and shrinks as the budget grows.

---

## 10. The compute floor and a sub-linear router

![Attention compute versus context length (lower is faster). Dense O(n²) rises at the top; our flat-router
kernel (measured) stays well below it but its speedup is capped by the argsort BlockMask build; the measured
faiss-GPU IVF router drops the kernel onto the n·κ floor. SubQ's two published speedups and its 1,000×@12M
claim are shown as compute = dense/speedup. Measured solid, projection dashed; one 16 GB GPU.](figures/unified_scaling.png)

Selection caps the per-query work at a budget κ keys, so the irreducible cost of the layer is the **attention
floor** `n·κ` — linear in n. A measured kernel sits above this floor by exactly its router cost. Decomposing
one forward of the §9 kernel into router (the `(n/b)²` block-score GEMM), `BlockMask` construction, and
attention, and fitting each over n ∈ [16K, 524K], gives attention ~ n^1.02 (the floor), router ~ n^1.76, and —
largest — the argsort-based mask build ~ n^2.12. Extrapolated to 12M tokens the floor is ~1% of the forward: a
**128× gap**, entirely the two `(n/b)²` terms. A router that emits the selected block indices directly removes
both.

**Lowering the floor.** The floor `n·κ` itself drops if fewer keys recover the target. The smallest budget
κ_min reaching recall ≥ 0.9 is κ_min/n = 3% for tight clusters, 50% for diffuse or adversarial geometry (no
compression), and the routability regularizer of §7 drives it from 25% (λ=0) to **0.4%** (λ=64) — a **60×
reduction**, the dominant lever, on benign geometry only.

**Closing the gap.** The necessity argument of §6 forces a sub-linear, examine-o(B) index. A bake-off on benign
co-trained keys (recall vs keys scored) ranks an IVF (inverted-file) router and a recursive-radius treecode
ahead of LSH, which fails to reach recall 0.9. An IVF over the block means scores O(√(n/b)) blocks at 0.93–0.97
block-selection agreement with the flat router and emits `kv_idx` directly. Timed on a single 16 GB GPU with
both routers on-device (no host transfer):

| context n | flat (n/b)² GEMM | faiss-GPU IVF | winner |
|---|---|---|---|
| 256K | 0.21 ms | 7.3 ms | flat 35× |
| 2M | 14.0 ms | 19.5 ms | flat 1.4× |
| 4M | 52.6 ms (runs) | 31.4 ms | IVF 1.7× |
| 8M | OOM (nb²=17.2 GB) | 63.7 ms | IVF only |

The single-head GEMM's constant wins below ~3M; it OOMs only at 8M (17 GB matrix), while the kernel's actual
router (block_route, H heads + argsort) OOMs near 1M; the IVF runs linearly past both and is measured to **12M
(94 ms)**. Crossover ~3M (IVF 1.7× faster at 4M).

**The gap, closed end-to-end (measured).** Wiring the IVF router into the FlexAttention kernel — emitting the
`from_kv_blocks` contract directly and building the mask with `compute_q_blocks=False` (which skips a dense
`(n_b,n_b{+}1)` transpose that would need 38.7 GB at $n_b{=}98{,}304$) — lets the whole forward be measured,
single-head, to 12M on one 16 GB GPU:

| context $n$ | router (ms) | maskbuild (ms) | attention = floor (ms) | total (ms) | gap to floor | peak |
|---|---|---|---|---|---|---|
| 1M | 13.5 | 0.002 | 3.2 | 16.1 | 5.0× | 0.55 GB |
| 4M | 36.4 | 0.003 | 13.5 | 52.0 | 3.9× | 2.18 GB |
| 12M | 101.4 | 0.003 | 47.5 | **139.5** | **2.9×** | 6.55 GB |

The direct-index mask build is **0.003 ms**, compared with a 40.7 s extrapolation for the argsort path, and the
residual gap to the floor is a *measured* 2.9×. The scope is **single-head** (H=8 does not fit at 12M) and
**synthetic keys** (a speed result), and the whole story is conditional on benign geometry — adversarial or multi-hop
retrieval returns κ_min to the 50% floor, where no speedup exists. Both ingredients of a quality-preserving
large-context speedup — a floor-lowering training stage and a sub-linear indexer — are thus exhibited, and the
indexer is shown driving a live kernel to ~the floor, under exactly that benign-geometry condition.

## 11. Experiments

The evidence below is organized by current claim and fixture. Synthetic timing establishes kernel scaling;
controlled training establishes mechanism behavior; frozen pretrained models establish integration and
retrieval behavior. Only the complete 10M experiment combines real-model geometry, every layer and head,
long context, subquadratic routing, sparse execution, and an observed quality outcome.

**Routing.** Second-order (cumulant) routing recovers targets where centroid routing collapses, matching the
$1/b$ outlier-attenuation analysis of Section 5.2; routing quality peaks near $\beta\approx2$, consistent with
the bias–variance reading of (5.3).

**Kernel speedup.** A block-sparse implementation of Algorithm 1 achieved a $20.6\times$ wall-clock speedup
over a dense exact kernel at $n=262{,}144$ on a single accelerator, with the crossover well below that length.

**The IVF kernel, measured to 12M.** Wiring the sub-linear IVF router into the kernel (Section 10) and measuring
the whole forward single-head on one 16 GB GPU runs a **12M-token forward in 139 ms and 6.55 GB**, at a *measured*
$2.9\times$ the $n\!\cdot\!\kappa$ floor (versus the $128\times$ gap the flat kernel projected); the argsort mask
build collapses to sub-millisecond because the IVF emits block indices directly. The autoregressive decode step
is **flat in $n$** ($\sim\!0.6$ ms from 1M to 12M at fixed $\kappa$) while a fair fp16 flash-decode step's prefix
read grows with $n$ ($0.5\to5.3$ ms) — a $9\times$ per-step gap at 12M with the crossover near 1M–2M, both
measured. The comparison uses the fair fp16 flash-decode row; a naive fp32-upcasting reference remains in the
benchmark only as a diagnostic. This is a single-head synthetic-key speed result; selection quality is a
separate benign-geometry claim.

**Multi-hop composition.** A chained retrieval through the same budgeted block selector obeys the composition law
$\text{chain}\approx\prod_j\rho_j$: benign single needles hold at $1.00$ while a *mixed* two-hop chain (one benign
hop, one isolated hop) collapses to $0.02$ at $n=65{,}536$ — the isolated hop's $\rho\!\approx\!0.02$ divides the
product. The multi-hop sag is the single-needle result read $h$ times, reproducing the NIAH$\gg$MRCR benchmark
split as a prediction of the same theory rather than an anomaly.

**The fused kernel inside a real model.** Swapping the kernel into a pretrained Qwen2.5-0.5B (`impl="flex"`) and
measuring at $8$K–$128$K preserves single-needle NIAH at $1.00$ while giving a $1.5$–$1.6\times$ prefill speedup
at $32$K (budget $0.06$–$0.12$), the speedup growing with $n$ and with a tighter budget — the synthetic-key
crossover shape inside a real model. At matched budget the analytic $O(n^2)$ path needs $10.7$ GB and $3.5$ s
where the fused kernel needs $1.4$ GB and $130$ ms, and the analytic path OOMs before $64$K while the kernel
reaches $128$K (under YaRN). The real-model two-hop chain shows the predicted budget-sensitivity. This
configuration jointly measures a real model, long context, a subquadratic kernel, and quality at $0.5$B scale.

**Complete pretrained transformer beyond 10M.** The memory-bounded Qwen2.5-0.5B path processes
**10,000,128 tokens** through all 24 layers, 14 query heads, and 2 KV heads on one RTX Pro 6000. It completes
in **713.2 s** at **14,022 token/s** with **25.72 GB** peak CUDA allocation. The fixed-beam center-radius tree
routes on pre-RoPE content geometry; native GQA sparse attention scores the selected blocks with post-RoPE Q/K.
Layer-1 cross-head consensus and a bounded 128-block cross-layer reservoir retain route evidence while keeping
the per-query/head budget at no more than 198 blocks, an upper selected fraction of **0.506%**. The 4K
dense-equivalence gate, dense and streamed 128K NIAH gates, and the 10M semantic ranking pass (`walnut` 5.844,
best distractor 4.438). The model is frozen and uses static YaRN at approximately 306× its training range.
Consequently this experiment demonstrates the full execution conjunction and one successful quality outcome,
not dense-equivalent output or general long-context quality.

**An optimal selector: the Certified Causal Cascade.** Composing five ingredients — a shared low-dim routing
space, sub-block max-pool summaries, a chunked-causal streaming index, an exact outlier side-channel, and
per-query admissible certificates with escalation — into one streaming selector, and measuring which pay off.
The certificate is sound (certified $\Rightarrow$ the selected top-$\kappa$ blocks equal the exact top-$\kappa$
under the routing metric; zero violations on clustered and random geometry; fire-rate $0.89$ / $0.50$).
This certifies the routing metric, not the omitted softmax mass or attention-output error; the latter
have a separate reference certificate in §5.7. The
component table is the trilemma made concrete: sub-block granularity and the outlier channel rescue high-norm
spikes, but **isolated unit-norm needles stay unretrievable for every cheap selector** (recall $0.05$) — the
impossibility of Section 6 in miniature. On the selector's cost: per-layer routing is $\sim\!59\%$ of a
Qwen-0.5B prefill (at DSA's reported $58\%$), and the lever that makes it cheap is **cross-layer sharing from a
mid donor layer** — measured cutting it to $\sim\!6\%$ with single-needle retrieval preserved, consistent with
the analytic $\div 5$ estimate; sharing from layer 0 fails. A trained $d_r{=}16$ routing projection reaches
$0.65$ block agreement on real keys versus $0.32$ untrained, but is itself too lossy
to drive retrieval — the honest boundary. This gives a falsifiable signature for any production selector: cheap
$\Leftrightarrow$ shared from a mid layer, preserving single-needle recall while sagging on isolated/multi-hop.

**The compression corner, measured.** The trilemma has a second corner — a fixed- or growing-state memory
written at inference time (the "zero attention" / DeltaNet/Titans family). Small exact reference memories,
measured against Lean predictions, place it. The READ rule sets the capacity class: a contracted linear read
$o = S q$ is rank-$d$ capped (recall collapses at $m\approx d$; `read_capacity_le_dim` / `rank_d_read_wall`
prove the $\le d$ ceiling) while a softmax read over the same pairs holds
far past $m=d$ (measured to $m=512$; `softmax_capacity` gives the exponential form) — capacity is a
property of the read, not the substrate, with machine-checked results on both sides of the contrast. The load-bearing measurement (empirical, no theorem) is that
**compression $\neq$ selection**: a needle salient only at read time is lost by a surprise-gated fixed memory
(recall $0.10$) where selection recovers it ($1.00$) — write-time compression cannot keep what the future
query has not yet made relevant. A distribution shift is a fold a fixed memory cannot track
(`fold_not_hopfield`: its pre-shift recall decays $0.90\to0.10$; a growing slot-birth state holds
$0.65$), and the proved composition bound $\prod\rho \le \min$-hop (`chain_le_weakest`) is reproduced,
with the measured joint chain sagging below $\prod\rho$. So the
NIAH-$\gg$-multi-hop split is architecture-independent: it holds whichever corner of the trilemma one builds.

**The compression corner, trained.** The reference memories above are untrained; the zero-attention recipe's
load-bearing half is *training-dependent* — a learned write gate and an auxiliary future-prediction objective.
The trained fixture is a small micro-LM (d=128, head_dim $d_h=16$) trained end-to-end on MQAR with a token-mixer
swappable between the two corners at matched state (DeltaNet state $d_h$ vs an SSA budget $\kappa\approx d_h$).
Three measurements. (i) *Capacity:* trained selection (dense, SSA) is flat in load, while the trained DeltaNet
groks the task and holds to $m\approx d_h$ then walls at the same rank-$d_h$ boundary — training moves the wall,
it does not remove it. (ii) *The learned write gate has no measured benefit:* on write-salient MQAR (keep-worthy
pairs use reserved marker keys, identifiable at write time) the no-gate delta rule already solves it — training
shapes the $\le d_h$ keepable keys itself; on read-salient MQAR nothing lifts the compression wall, gate or no
gate. (iii) *The future-prediction auxiliary loss is flat in its weight* on the read-salient wall. So the
training-dependent half of the recipe does not close the gap: where keeping is possible training already does
it, and where relevance is read-time-only no write policy can serve it — the trained mirror of the
$0.10$-vs-$1.00$ split, and the composition sag persists for the compression corner under training as well.
The bottom line: dropping attention does not dissolve the trilemma but relocates within it — a compression
memory *works* where relevance is fixed at write time and within its state (write-salient recall, NIAH), and
*fails*, by capacity rather than by any trainable objective, on read-time-only relevance, past-capacity
retrieval, and multi-hop chains. This is why the frontier long-context state-space models remain *hybrids*
with interleaved attention.

**Routability.** The regularizer (7.2) reduced lossless branch-and-bound selection cost from $26.5\%$ to
$4.2\%$ of keys at zero accuracy cost (Section 7.2). The reduction was robust across head dimension and showed
no capacity trade-off down to $d=$ (number of clusters), since query-specific anisotropy needs only $\sim\!1$
dimension per cluster — real head dimensions ($64$–$256$) sit well above this.

**Length generalization and staging.** Table in Section 8.2: $32\times$ the trained length at recall $0.982$
for $\sim\!800$ adaptation steps, SSA-preserved.

**Construction pipeline.** Table in Section 9: swap $+13.2$ perplexity, $94\%$ recovered to within $+1.2$ of
the dense-adapted control at $\sim\!38\%$ of keys.

**Retrieval regime boundaries.** The complete 10M run supplies one successful semantic ranking. The controlled
sweep below isolates the geometry behind such single-target results by measuring retrieval as a function of
context length at fixed budget $\kappa\approx10^3$ and margin $\Delta=0.55$:

| context $n$ | dense | SSA, isolated target | SSA, benign target |
|---|---|---|---|
| 1024   | 1.00 | 1.00 | 1.00 |
| 4096   | 1.00 | 0.47 | 1.00 |
| 16384  | 1.00 | 0.20 | 1.00 |
| 65536  | 0.90 | 0.10 | 1.00 |
| 262144 | 0.83 | 0.00 | 1.00 |

An **isolated** target — a lone spike with no correlated neighbors — *collapses* with length: cheap moment
routing averages it into its block and loses it among the fluctuations of the growing number of blocks (the
$1/b$ attenuation of Section 5.2, competing against more and more random blocks). This is the impossibility
of Section 6 in miniature. A **benign** target — one accompanied by a coherent span of query-aligned neighbors,
as a real answer is by its surrounding context — lifts its whole block's score and stays flat at $1.00$,
*beating dense* at long $n$ because selection caps the effective distractor count. The same separation appears
across margins: an isolated target barely fires at any margin, while a benign one tracks dense once the margin
clears the budget floor $\sqrt{2\log\kappa/d}$. So the headline numbers certify the **easy and benign** regime
(single, high-margin, accuracy not losslessness) — the regime everyone agrees is achievable — and do not
certify lossless selection in the worst case, which Section 6 shows is unavailable cheaply.

---

## 12. Discussion and limitations

SSA is best understood as the resolution of a constrained problem rather than a universal accelerator. The
recovery-weight law (3.1) shows selection makes retrieval flat in $n$; the admissible bound (5.1) and the
prune gate (5.5) show summaries can certify that selection losslessly; the trilemma (Section 6) shows this can
be cheap **only** on benign geometry; and the regularizer (7.2) shows training can supply that geometry. The
construction pipeline (Section 9) and the staging ladder (Section 8) turn the mechanism into a recipe that
retrofits a dense model and extends its context a rung at a time.

The current limitations follow directly from the theory and evidence. (i) **Worst-case losslessness is not available cheaply** —
a sufficiently adversarial or genuinely low-margin target can always evade summary routing; SSA's guarantees
are conditional on the (trained, measured) benign geometry. (ii) **Multi-needle and low-margin retrieval** are
the hard regime the headline single-needle numbers do not address. (iii) The selection budget $\kappa$ sets a
floor margin $\sqrt{2\log\kappa/d}$; targets below it are missed regardless of $n$. (iv) The complete 10M
experiment uses a frozen 0.5B model, static YaRN far outside its training range, one semantic NIAH ranking,
and no dense 10M baseline; it is execution evidence rather than a broad quality result. (v) The 12M IVF
near-floor timing is single-head and synthetic, while the real-model 128K timing and quality measurements are
smaller-scale. Broad retrieval, perplexity, multi-hop evaluation, trained long-context models, and
frontier-model validation remain open. (vi) The complete implementation is not formally verified end to end.
Appendix B gives self-contained statements and proofs of the supporting exact-arithmetic invariants; access to
the separate formalization is not required to inspect them. As corroborating provenance, private Substrate
commit `908ec0d6d` machine-checks the corresponding results for recursive
real-valued ball containment, conditional 9-vote retention by a 128-slot highest-count reservoir at the
14-selector/70-item route bounds, and survival with a uniform cardinality cap under union with a fixed carrier.
It also proves that the radial pairing cap is attained under alignment plus a realizable boundary member and
that the per-centre refinement agrees there. Those alignment hypotheses are sufficient, not shown necessary.
The per-centre cap is universally no larger; a plane witness exhibits a strict gap as large as the whole cap,
but no converse says nonalignment forces strictness or equality forces alignment. These results do not verify
the Python/CUDA mapping, float32 outward rounding, fixed-beam quality, or the unrecorded premise that the
measured target had nine pre-consensus base-route votes.

---

## Appendix A. Derivations

**A.1 Recovery weight (3.1).** With one target at logit $a_\star$ and $\mu$ distractors at $a_\star-\Delta$,
$w_\star=e^{\beta a_\star}/(e^{\beta a_\star}+\mu e^{\beta(a_\star-\Delta)})$. Divide numerator and denominator
by $e^{\beta a_\star}$ to get $1/(1+\mu e^{-\beta\Delta})=\sigma(\beta\Delta-\log\mu)$, since
$1/(1+e^{-x})=\sigma(x)$ with $x=\beta\Delta-\log\mu$.

**A.2 Cumulant score (4.2).** Let $g(\beta)=\beta^{-1}\log\frac1b\sum_{j\in c}e^{\beta\langle q,k_j\rangle}$.
As $\beta\to0$, $g(\beta)=\mathbb{E}_j\langle q,k_j\rangle+\frac{\beta}{2}\mathrm{Var}_j\langle
q,k_j\rangle+O(\beta^2)$, the cumulant generating function expansion. The mean is $\langle q,\mu_c\rangle$ and
the variance is $q^\top\Sigma_c q$, giving (4.2).

**A.3 Log-sum-exp sandwich (5.3).** For any reals $x_1,\dots,x_b$ with $M=\max_j x_j$:
$e^{\beta M}\le\sum_j e^{\beta x_j}\le b\,e^{\beta M}$. Take $\log$, divide by $\beta$:
$M\le\beta^{-1}\log\sum_j e^{\beta x_j}\le M+\beta^{-1}\log b$.

**A.4 Samuelson bound (5.4).** Center the data, $d_j=s_j-\bar s$, so $\sum_j d_j=0$. Fix index $i$. By
Cauchy–Schwarz over the other $m-1$ indices, $d_i^2=(\sum_{j\neq i}d_j)^2\le(m-1)\sum_{j\neq
i}d_j^2=(m-1)(\sum_j d_j^2-d_i^2)$. Hence $d_i^2\,m\le(m-1)\sum_j d_j^2$, i.e.
$(s_i-\bar s)^2\le(m-1)\mathrm{Var}$. Taking the max over $i$ and adding $\bar s$ gives the stated bound on
$\max_j s_j$, and (5.5) is its contrapositive against the threshold $s^\star$.

---

## Appendix B. Self-contained routing invariants

This appendix contains the mathematical content used to justify the 10M router's center-radius hierarchy,
cross-head reservoir, and fixed cross-layer carrier. It is included so the public artifact does not depend on
access to the separate Lean repository. The formal audit is useful corroboration, but the definitions,
statements, proofs, counterexamples, and scope needed to assess the claims are all below. Every geometric
statement is over an exact real inner-product space; floating-point consequences require a separate outward-
rounding argument.

### B.1 Finite families of key regions

Let $I$ be finite and nonempty. In a real inner-product space, let key $x_i$ lie in the closed ball with centre
$c_i$ and radius $r_i$:
$$
\lVert x_i-c_i\rVert\le r_i .
$$
For a reference point $p$, define the **radial reach** and its score cap by
$$
R=\max_{i\in I}\bigl(\lVert c_i-p\rVert+r_i\bigr),
\qquad U_R(q)=\langle q,p\rangle+\lVert q\rVert R .
\tag{B.1}
$$

**Proposition B.1 (admissibility and radial minimality).** For every $i$ and $q$,
$$
\lVert x_i-p\rVert\le R,
\qquad \langle q,x_i\rangle\le U_R(q).
\tag{B.2}
$$
Moreover, $R$ is attained by some index and is the least number satisfying
$\lVert c_i-p\rVert+r_i\le R$ for all $i$.

**Proof.** The triangle inequality gives
$$
\lVert x_i-p\rVert
\le \lVert x_i-c_i\rVert+\lVert c_i-p\rVert
\le r_i+\lVert c_i-p\rVert\le R.
$$
Then
$\langle q,x_i\rangle-\langle q,p\rangle
=\langle q,x_i-p\rangle
\le\lVert q\rVert\lVert x_i-p\rVert\le\lVert q\rVert R$
by Cauchy–Schwarz. Attainment and minimality follow directly because (B.1) is the maximum of a finite,
nonempty family. $\square$

Radial minimality does **not** make $U_R$ the sharpest score bound available from the same data. Define the
per-centre cap
$$
U_C(q)=\max_{i\in I}\bigl(\langle q,c_i\rangle+\lVert q\rVert r_i\bigr).
\tag{B.3}
$$

**Proposition B.2 (per-centre refinement).** Every key is bounded by its own centre,
$$
\langle q,x_i\rangle\le\langle q,c_i\rangle+\lVert q\rVert r_i,
$$
and $U_C(q)\le U_R(q)$ for every query.

**Proof.** Apply Cauchy–Schwarz to $x_i-c_i$ for the first inequality. For the second, apply it to
$c_i-p$:
$$
\langle q,c_i\rangle+\lVert q\rVert r_i
\le \langle q,p\rangle+\lVert q\rVert
   \bigl(\lVert c_i-p\rVert+r_i\bigr)
\le U_R(q),
$$
then maximize the left side over $i$. $\square$

The comparison can be strict. In $\mathbb R^2$, take one region with $p=(0,0)$, $q=(1,0)$,
$c_1=(0,1)$, $r_1=0$, and $x_1=c_1$. Then $R=1$ and $U_R(q)=1$, whereas
$U_C(q)=\langle q,x_1\rangle=0$. Thus the gap can equal the whole reach term. This is one witness, not a
theorem that nonalignment always makes the inequality strict.

For the equality case, write $e_q=\lVert q\rVert^{-1}q$, using $e_0=0$. Say two vectors are on the same ray
when one is a nonnegative scalar multiple of the other; then
$\langle q,v\rangle=\lVert q\rVert\lVert v\rVert$.

**Proposition B.3 (aligned realizable attainment).** Suppose an index $i_\star$ attains the reach,
$\lVert c_{i_\star}-p\rVert+r_{i_\star}=R$, the vectors $q$ and $c_{i_\star}-p$ lie on the same ray, and
$$
x_{i_\star}=c_{i_\star}+r_{i_\star}e_q .
\tag{B.4}
$$
Then
$$
\langle q,x_{i_\star}\rangle=U_R(q).
\tag{B.5}
$$
Consequently, if $B$ bounds every $\langle q,x_i\rangle$ for this fixed family and query, then
$U_R(q)\le B$; no smaller skip bound is admissible there. Under the reach-attainment and same-ray hypotheses
alone—without the boundary-member hypothesis (B.4)—$U_C(q)=U_R(q)$.

**Proof.** Same-ray equality and (B.4) give
$$
\begin{aligned}
\langle q,x_{i_\star}\rangle
 &=\langle q,p\rangle+\langle q,c_{i_\star}-p\rangle
   +r_{i_\star}\langle q,e_q\rangle\\
 &=\langle q,p\rangle+\lVert q\rVert
   \bigl(\lVert c_{i_\star}-p\rVert+r_{i_\star}\bigr)=U_R(q).
\end{aligned}
$$
Any common bound $B$ must bound this member, proving the next claim. For the cap comparison, the
$i_\star$ term in (B.3) equals $U_R(q)$ under alignment, while Proposition B.2 supplies the reverse
inequality. $\square$

No $q\ne0$ hypothesis is needed: at $q=0$, $e_q=0$ and both sides of (B.5) vanish. Nor does equality require
$r_{i_\star}\ge0$. That sign condition is needed only to make the constructed boundary member admissible:
$\lVert r_{i_\star}e_q\rVert\le r_{i_\star}$ when $r_{i_\star}\ge0$. The hypotheses above are sufficient;
neither attainment nor equality of the two caps is proved to force alignment.

### B.2 Recursive ball containment

Define a ball tree recursively. A leaf stores a centre and radius. A node stores a centre $c_t$ and children
$u$, and its radius is
$$
\rho_t=\max\!\left(0,\max_{u\text{ child of }t}
  \bigl(\lVert c_u-c_t\rVert+\rho_u\bigr)\right).
\tag{B.6}
$$
Let $s\preceq t$ mean that $s=t$ or $s$ is below $t$ through a finite chain of child edges.

**Proposition B.4 (containment and ancestor cap).** Every child ball lies inside its parent ball. More
generally, if $s\preceq t$ and $\lVert x-c_s\rVert\le\rho_s$, then
$$
\lVert x-c_t\rVert\le\rho_t.
\tag{B.7}
$$
For every reference point $p$, ancestor reach is monotone:
$$
\lVert c_s-p\rVert+\rho_s
\le \lVert c_t-p\rVert+\rho_t.
\tag{B.8}
$$
Consequently, if each key lies in a descendant ball below $t$ and
$\lVert c_t-p\rVert+\rho_t\le R_t$, then
$$
\langle q,x_i\rangle\le\langle q,p\rangle+\lVert q\rVert R_t
$$
for every such key.

**Proof.** Equation (B.6) directly gives
$\lVert c_u-c_t\rVert+\rho_u\le\rho_t$ for each child. If $x$ lies in $u$, the triangle inequality yields
$\lVert x-c_t\rVert\le\lVert x-c_u\rVert+\lVert c_u-c_t\rVert\le\rho_t$.
Induction on the child path proves (B.7). The same induction, now applying the triangle inequality to
$c_u-p$, proves (B.8). Proposition B.1 applied at node $t$ gives the score cap. $\square$

This establishes exact-real containment at arbitrary depth. It proves neither that a node radius is minimal
nor that the fixed-beam search visits the correct branch. In float32, (B.7) additionally requires radii to be
rounded outward or inflated enough to cover accumulated error.

### B.3 Uniform and position-dependent orthogonal actions

**Proposition B.5 (what orthogonality preserves).** If one orthogonal map $A$ is applied to the query and both
keys, then
$$
\langle Aq,Ax\rangle<\langle Aq,Ay\rangle
\quad\Longleftrightarrow\quad
\langle q,x\rangle<\langle q,y\rangle.
$$
Different maps at different key positions need not preserve the order.

**Proof.** Orthogonality gives $\langle Au,Av\rangle=\langle u,v\rangle$. For the negative statement in
$\mathbb R^2$, take $q=x=(1,0)$, $y=(1/2,0)$, $A=-I$, and $B=I$. Initially
$\langle q,y\rangle=1/2<1=\langle q,x\rangle$, but
$\langle q,Ax\rangle=-1<1/2=\langle q,By\rangle$. $\square$

RoPE is position-dependent, so the second statement explains why pre-RoPE and post-RoPE rankings are not
identities. It is only an existence counterexample: it says nothing about how often rankings change or which
ranking gives better retrieval quality.

### B.4 Cross-head consensus and top-count retention

Let $H$ selectors choose finite sets $S_h$, each of cardinality at most $W$. Let
$U=\bigcup_h S_h$, let $\nu(a)=|\{h:a\in S_h\}|$, and define
$C_v=\{a\in U:\nu(a)\ge v\}$.

**Proposition B.6 (consensus counting).** For every $v$,
$$
|C_v|v\le HW,
$$
and for $v>0$, $|C_v|\le\lfloor HW/v\rfloor$.

**Proof.** Double-count selector–element incidences:
$$
\sum_h|S_h|=\sum_{a\in U}\nu(a).
$$
The left side is at most $HW$, while elements of $C_v$ contribute at least $|C_v|v$ to the right side.
$\square$

A **top-count reservoir** of capacity $k$ is a set $T\subseteq U$ such that
$|T|=\min(k,|U|)$ and every outsider has count no larger than every member:
$$
a\in U\setminus T,\ b\in T\quad\Longrightarrow\quad\nu(a)\le\nu(b).
\tag{B.9}
$$
Such a reservoir exists by sorting the finite pool by $\nu$; ties may be broken arbitrarily.

**Proposition B.7 (consensus retention).** If $v>0$ and $\lfloor HW/v\rfloor\le k$, then every top-count
reservoir of capacity $k$ contains $C_v$.

**Proof.** Suppose $a\in C_v\setminus T$. By (B.9), every $b\in T$ has
$\nu(b)\ge\nu(a)\ge v$, so $T\cup\{a\}\subseteq C_v$. Hence
$|T|+1\le|C_v|\le\lfloor HW/v\rfloor\le k$, giving $|T|<k$. The fullness condition
$|T|=\min(k,|U|)$ then forces $|T|=|U|$. Since $T\subseteq U$, this implies $T=U$, contradicting
$a\notin T$. $\square$

For SSA's route limits, $H=14$, $W=70$, and $v=9$, so
$$
|C_9|\le\left\lfloor\frac{14\cdot70}{9}\right\rfloor=108<128.
$$
Therefore every nine-vote item is retained by any full 128-slot highest-count reservoir, independent of tie
breaking. This is conditional on the item actually receiving nine **pre-reservoir** selections. It says
nothing about items at the implementation's two-vote admission threshold, for which the same bound is
$\lfloor980/2\rfloor=490$.

The fullness equality in the reservoir definition is essential: merely requiring $|T|\le k$ allows the empty
set, which retains nothing. The multiplied counting bound is attained in general when all $H$ selectors choose
the same $W$ items and $v=H$. No construction here is claimed to attain the numeric ceiling 108 at
$(H,W,v)=(14,70,9)$.

### B.5 Accumulating runs and a fixed cross-layer carrier

Let $S_\ell$ be the selection made at round (or layer) $\ell$, and let $P$ be a carried finite set. There are
two different constructions:
$$
A_0=P,\qquad A_{n+1}=S_n\cup A_n,
\qquad\text{and}\qquad
F_\ell=S_\ell\cup P.
\tag{B.10}
$$

**Proposition B.8 (retention and capacity).** The accumulating run satisfies
$$
A_n=P\cup\bigcup_{\ell<n}S_\ell,
\qquad A_n\subseteq A_m\ (n\le m),
\qquad |A_n|\le|P|+nW
$$
when $|S_\ell|\le W$. If additionally $|P|\le C$, the fixed-carrier state satisfies, for every $\ell$,
$$
P\subseteq F_\ell,
\qquad |F_\ell|\le W+C.
\tag{B.11}
$$

**Proof.** The closed form and monotonicity follow by induction from (B.10). The cardinality claims use
$|A\cup B|\le|A|+|B|$, once per induction step for $A_n$ and once directly for $F_\ell$. The inclusion in
(B.11) is immediate from the union. $\square$

The constructions must not be conflated. With $P=\varnothing$ and $S_\ell=\{\ell\}$,
$F_0=\{0\}$ is not a subset of $F_1=\{1\}$, while $A_1=\{0\}\ne F_1$. Thus the fixed carrier preserves
$P$, not every earlier layer's transient selection, and its $W+C$ cap is uniform precisely because it does not
accumulate those selections.

Finally, if “past-bounded at $t$” means every member of a set is at most $t$, a union is past-bounded only when
both components are. This property is preserved by either construction under the corresponding hypotheses;
the union does not create it: at $t=0$, $P=\{0\}$ is past-bounded but $S_0=\{5\}$ and
$S_0\cup P$ are not. None of these set identities specifies how a selector admits an item, proves a causal
mask, or proves that the measured target belonged to $P$.

---

## References

1. A. Vaswani et al. *Attention Is All You Need.* NeurIPS, 2017.
2. R. Child, S. Gray, A. Radford, I. Sutskever. *Generating Long Sequences with Sparse Transformers.* 2019.
3. I. Beltagy, M. E. Peters, A. Cohan. *Longformer: The Long-Document Transformer.* 2020.
4. M. Zaheer et al. *Big Bird: Transformers for Longer Sequences.* NeurIPS, 2020.
5. N. Kitaev, Ł. Kaiser, A. Levskaya. *Reformer: The Efficient Transformer.* ICLR, 2020.
6. A. Roy, M. Saffar, A. Vaswani, D. Grangier. *Efficient Content-Based Sparse Attention with Routing
   Transformers.* TACL, 2021.
7. A. Katharopoulos, A. Vyas, N. Pappas, F. Fleuret. *Transformers are RNNs: Fast Autoregressive Transformers
   with Linear Attention.* ICML, 2020.
8. K. Choromanski et al. *Rethinking Attention with Performers.* ICLR, 2021.
9. J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* 2021.
10. L. Greengard, V. Rokhlin. *A Fast Algorithm for Particle Simulations.* J. Comput. Phys., 1987.
11. J. Barnes, P. Hut. *A Hierarchical $O(N\log N)$ Force-Calculation Algorithm.* Nature, 1986.
12. P. A. Samuelson. *How Deviant Can You Be?* J. Amer. Statist. Assoc., 1968.
13. A. Keles, P. Wijewardena, C. Hegde. *On the Computational Complexity of Self-Attention.* ALT, 2023.
14. J. Alman, Z. Song. *Fast Attention Requires Bounded Entries.* NeurIPS, 2023.
15. H. Ramsauer et al. *Hopfield Networks Is All You Need.* ICLR, 2021.
16. S. Arora et al. *Zoology: Measuring and Improving Recall in Efficient Language Models.* 2023.
