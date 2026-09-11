# Variational coarse reads on finite partitions

Public specification imported from Substrate commit `206193290e5aa1b0465d518f12625c10952b0953`.
The finite mathematical statements are machine-checked there; this public copy requires no access to
the private repository. SSA implementations and floating-point execution are not formally verified.

This specification uses finite real arithmetic and contains the definitions and proof arguments needed to reproduce the construction without any software-library dependency. All logarithms are natural.

## Finite carrier, cells and the coarse distribution

Let \(I\) be a nonempty finite set, let \(s_i\in\mathbb R\), and let \(P\) be a partition of \(I\) into nonempty cells. Define

\[
 Z=\sum_{i\in I}e^{s_i},\qquad p_i=e^{s_i}/Z.
\]

For each cell \(C\in P\), define its count and actual mean logit by

\[
 m_C=|C|,\qquad a_C=\frac1{m_C}\sum_{i\in C}s_i.
\]

The coarse partition function and distribution are

\[
 Z_P=\sum_{C\in P}m_Ce^{a_C},\qquad
 q_P(i)=\frac{e^{a_C}}{Z_P}\quad(i\in C).
\]

Both partition functions are strictly positive. The mass of a cell is
\(q_P(C)=m_Ce^{a_C}/Z_P\); its conditional distribution is uniform.

Let \(\mathcal F_P\) contain every distribution \(r\) with \(r_i\ge0\), \(\sum_i r_i=1\), and \(r_i=r_j\) whenever \(i,j\) lie in the same cell. Cells may have zero competitor mass. For a strictly positive distribution \(t\), use

\[
 \operatorname{KL}(r\Vert t)=\sum_i r_i\log(r_i/t_i),
 \qquad 0\log(0/t_i):=0.
\]

## Exact projection and its uniqueness

For every \(r\in\mathcal F_P\),

\[
 \boxed{\operatorname{KL}(r\Vert p)
 =\operatorname{KL}(r\Vert q_P)+\log(Z/Z_P).}
\]

To prove the identity, subtract the two KL expressions. At a coordinate with \(r_i>0\), the difference is

\[
 r_i\log\frac{q_P(i)}{p_i}
 =r_i\bigl(a_C-s_i+\log(Z/Z_P)\bigr).
\]

At \(r_i=0\), both contributions are zero. Writing the constant value of \(r_i\) on cell \(C\) as \(r_C\), the cell contribution from the first two terms is

\[
 r_C\left(m_Ca_C-\sum_{i\in C}s_i\right)=0.
\]

The remaining constant contributes \(\log(Z/Z_P)\), because \(\sum_i r_i=1\). Thus the argument includes zero-mass cells without taking their logarithms.

Taking \(r=q_P\) gives

\[
 \operatorname{KL}(q_P\Vert p)=\log(Z/Z_P).
\]

Finite KL nonnegativity, with equality exactly when its distributions agree, now gives

\[
 \operatorname{KL}(r\Vert p)\ge\operatorname{KL}(q_P\Vert p),
 \qquad
 \operatorname{KL}(r\Vert p)=\operatorname{KL}(q_P\Vert p)
 \iff r=q_P.
\]

The optimum is over the specified cell-uniform family and uses the direction \(\operatorname{KL}(\text{approximation}\Vert\text{dense})\). It is not an optimum over all approximation families or for the opposite KL direction. A fixed assignment prototype need not have logit \(a_C\); the theorem requires the actual mean.

## Lower mass bound, equality and refinement

Finite Jensen applied separately to each cell gives

\[
 m_Ce^{a_C}\le\sum_{i\in C}e^{s_i},\qquad Z_P\le Z.
\]

Strict convexity of the exponential gives equality in a cell exactly when all its logits are equal. The full equality \(Z_P=Z\) therefore holds exactly when the logits are constant within every cell. No distinctness hypothesis is needed, and singleton cells satisfy the condition automatically.

Suppose \(P'\) refines \(P\). For a cell \(C\), its mean is the count-weighted mean of the means of its subcells \(D\subseteq C\). Jensen therefore gives

\[
 m_Ce^{a_C}\le\sum_{D\subseteq C}m_De^{a_D}.
\]

Consequently,

\[
 Z_P\le Z_{P'}\le Z,
 \qquad
 \operatorname{KL}(q_{P'}\Vert p)\le\operatorname{KL}(q_P\Vert p).
\]

The singleton partition has \(Z_P=Z\) and \(q_P=p\). These results order the stated KL objective. They do not imply monotone actual output error, opposite-direction KL, or downstream cross-entropy.

## Selected singletons and an exact streaming realization

For any selected set \(S\subseteq I\), form the refinement

\[
 P_S=\{\{i\}:i\in S\}
 \;\cup\;\{C\setminus S:C\in P,\ C\setminus S\ne\varnothing\}.
\]

Empty omitted cells contribute nothing. Each selected singleton uses its exact logit. Every omitted cell uses its actual omitted mean, computed after subtracting the selected contributions. The refinement results above apply to \(P_S\). If \(S\subseteq T\), then \(P_T\) refines \(P_S\). If \(S=I\), the result is the dense distribution and dense output, with no omitted tail.

Let keys \(k_i\), values \(v_i\), query \(q\), and real coefficient \(\beta\) satisfy
\(s_i=\beta\langle q,k_i\rangle\). The key and value carriers may have different dimensions. Maintain, for each cell,

\[
 N_C=|C|,\qquad K_C=\sum_{i\in C}k_i,\qquad V_C=\sum_{i\in C}v_i.
\]

For the omitted part \(T_C=C\setminus S\), subtract selected contributions to obtain

\[
 n_C=N_C-|C\cap S|,\quad
 K_C^{\rm tail}=K_C-\sum_{i\in C\cap S}k_i,\quad
 V_C^{\rm tail}=V_C-\sum_{i\in C\cap S}v_i.
\]

When \(n_C>0\), linearity gives the exact mean-logit identity

\[
 \ell_C=\beta\left\langle q,\frac{K_C^{\rm tail}}{n_C}\right\rangle
 =\frac1{n_C}\sum_{i\in T_C}s_i.
\]

The refined coarse output is exactly \(N_S/D_S\), where

\[
 D_S=\sum_{i\in S}e^{s_i}
       +\sum_{C:n_C>0}n_Ce^{\ell_C},
\]

\[
 N_S=\sum_{i\in S}e^{s_i}v_i
       +\sum_{C:n_C>0}e^{\ell_C}V_C^{\rm tail}.
\]

Indeed, each omitted item has probability \(e^{\ell_C}/D_S\), so its entire cell contributes the exponential times the value sum. Equivalently this contribution is cell mass times the mean value. The denominator is positive because the visible carrier is nonempty. The formulas also allow \(S=\varnothing\), and never divide by an empty-cell count.

```text
coarse_read(query, beta, selected, cell_summaries):
    numerator = zero_value_vector
    denominator = 0
    removed_count, removed_keys, removed_values = zero_per_cell()

    for i in selected:
        weight = exp(beta * dot(query, key[i]))
        denominator += weight
        numerator += weight * value[i]
        c = cell_of(i)
        removed_count[c] += 1
        removed_keys[c] += key[i]
        removed_values[c] += value[i]

    for c in cells:
        count = cell_summaries[c].count - removed_count[c]
        if count == 0:
            continue
        key_sum = cell_summaries[c].key_sum - removed_keys[c]
        value_sum = cell_summaries[c].value_sum - removed_values[c]
        mean_logit = beta * dot(query, key_sum / count)
        weight_per_item = exp(mean_logit)
        denominator += count * weight_per_item
        numerator += weight_per_item * value_sum

    return numerator / denominator
```

The selected set contains each selected item once. The summaries and selected items must refer to the same visible carrier and assignments. The pseudocode specifies exact real arithmetic, not a floating-point error bound or an implementation-performance claim.

For causal use at time \(t\), apply the construction to the finite visible carrier \(I_t\). Its assignments, summaries and selection must not depend on future keys or values. A new visible item updates its cell's count, key sum and value sum. Removing selected contributions is query-specific arithmetic on those visible summaries. No claim of subquadratic computation, cheap selection, or a persistent-state lower bound follows from these identities.

## Upper certificates have a different direction

Suppose a separately justified number \(U\) satisfies \(U\ge Z\). Then

\[
 \operatorname{KL}(q_P\Vert p)=\log(Z/Z_P)
 \le\log(U/Z_P).
\]

With total variation defined as half the \(\ell^1\) distance, finite Pinsker gives

\[
 \operatorname{TV}(q_P,p)\le\sqrt{\tfrac12\log(U/Z_P)}.
\]

If \(\|v_i\|\le V\) for every value in a normed real vector space, the triangle inequality gives

\[
 \left\|\sum_i q_P(i)v_i-\sum_i p_iv_i\right\|
 \le V\sum_i|q_P(i)-p_i|
 \le 2V\sqrt{\tfrac12\log(U/Z_P)}.
\]

The same statements apply to the selected-singleton refinement. Their usefulness requires \(U/Z_P\) close to one. The construction does not supply a favorable \(U\), a cheap bound computation, or a small frontier. In particular, \(Z_P\) is a **lower** total-mass bound: it is not an upper bound on total or omitted mass and cannot serve as an upper omitted-mass certificate.

## Finite cell gains and the exact price of changing the optimum

For supplied finite real gains \(g_C\), define

\[
 Z_g=\sum_C m_Ce^{a_C+g_C},\qquad
 q_g(i)=e^{a_C+g_C}/Z_g\quad(i\in C).
\]

The modified distribution remains cell-uniform. The same cancellation of mean logits proves

\[
 \boxed{\operatorname{KL}(q_g\Vert p)
 =\log(Z/Z_g)+\sum_C q_g(C)g_C
 =\operatorname{KL}(q_g\Vert q_P)+\operatorname{KL}(q_P\Vert p).}
\]

No gain sign is needed. If every gain is nonpositive, then \(Z_g\le Z_P\), by monotonicity of the exponential. This does not improve the KL objective: the second identity gives
\(\operatorname{KL}(q_g\Vert p)\ge\operatorname{KL}(q_P\Vert p)\).

For selected-singleton replacement, assign gain zero to selected cells and optional nonpositive gains to omitted cells. The output formula multiplies each omitted cell's exponential contribution in both numerator and denominator by \(e^{g_C}\). A downstream objective may favor suppression of an inaccurate value contribution, but that is a different objective from the variational KL projection.

## A movement cap and the error criterion

Let \(o_S\) be a supplied base output and \(u\) a proposed positive normalized tail read. A selected-only output naturally supplies \(o_S\) when the selected set is nonempty; the streaming formula above itself also handles an empty selected set. For \(0\le\alpha\le\rho\le1\), put

\[
 o_\alpha=o_S+\alpha(u-o_S).
\]

Norm homogeneity gives

\[
 \|o_\alpha-o_S\|=\alpha\|u-o_S\|\le\rho\|u-o_S\|.
\]

In a real inner-product space, write \(e=o_{\rm dense}-o_S\) and \(d=u-o_S\). Expanding the squared norm gives the exact identity

\[
 \|e-\alpha d\|^2-\|e\|^2
 =\alpha^2\|d\|^2-2\alpha\langle e,d\rangle.
\]

For \(\alpha>0\), **non-increase** is equivalent to

\[
 \alpha\|d\|^2\le2\langle e,d\rangle,
\]

and **strict improvement** is equivalent to the strict inequality. Positivity and normalization of the proposed tail weights do not supply either alignment condition.

A counterexample with positive weights and values makes this limitation explicit. Use three values
\((2,\tfrac12,\tfrac72)\) with positive dense weights \((6,5,1)\), and select the first value. Then \(o_S=2\) and \(o_{\rm dense}=\tfrac32\). On the two omitted values, approximate weights \((1,5)\), normalized by their sum, give \(u=3\). For every \(\rho>0\), choose \(\alpha=\min(\rho,1)>0\). The error increases from \(\tfrac12\) to \(\tfrac12+\alpha\), despite \(\alpha\le\rho\) and \(\alpha\le1\). All weights and all values are positive, and the approximate tail uses the same omitted values.

These are finite variational and interpolation statements. They do not establish learned favorable geometry, retrieval preservation, training convergence, held-out length generalization, a useful fixed-budget certificate, cross-entropy monotonicity, or numerical soundness on CUDA.
