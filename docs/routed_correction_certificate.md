# A conditional certificate for routed corrections

Public specification imported from Substrate commit `68968fc7aae0a024172288f090f25c4d61451324`.
The finite mathematical results are machine-checked there. This public copy requires no access to the
private repository. SSA now has a float64 path checker and a complete small causal-transformer
instantiation with analytic uniform bounds in `ssa/routed_correction_certificate.py` and
`ssa/routed_certificate_experiment.py`; measurements are in `runs/routed_certificate_reference.json`.
These are numerical checks, not formally verified floating-point execution. Useful Qwen uniform path
bounds remain unimplemented. The separate `ssa/endpoint_acceptance.py` Qwen experiment compares two
computed endpoints directly and must not be cited as an instantiation of the nonlinear path theorem.

This specification gives exact real arithmetic conditions under which a corrected finite computation
preserves a supplied reference prediction. It connects changed discrete routes, signed downstream
effects, and nonlinear approximation error. It does not establish that the reference answer is
correct, that a correction improves prediction loss, or that the conditions are inexpensive to check.
No private library, model implementation, or empirical result is required to read or use the formulas.

## Restricted reads at one common state

Let I be a finite visible key carrier, with finite real logits s_i and values v_i in a real normed
space. Let S and T be nonempty selected subsets. All quantities in this comparison use **the same
logits and values at the same state**. Define

\[
 Z_A=\sum_{i\in A}e^{s_i},\qquad
 p_A(i)=\begin{cases}e^{s_i}/Z_A&i\in A,\\0&i\notin A,\end{cases}
 \qquad o_A=\sum_i p_A(i)v_i.
\]

Writing total variation as half the L1 difference gives the exact identity

\[
 \operatorname{TV}(p_S,p_T)
 =\tfrac12\sum_i|p_S(i)-p_T(i)|
 =1-\frac{Z_{S\cap T}}{\max(Z_S,Z_T)}.
\]

To see the overlap formula, on an intersection key the smaller of the two probabilities is
\(e^{s_i}/\max(Z_S,Z_T)\); outside the intersection it is zero. Total variation is one minus the
sum of these minima. If \(\|v_i\|\le V\), the triangle inequality therefore gives

\[
 \|o_S-o_T\|\le 2V\operatorname{TV}(p_S,p_T).
\]

Identical sets have zero variation; disjoint sets have variation one. If \(S\subseteq T\), the
value is \(1-Z_S/Z_T\). If their masses are equal to Z, it is \(1-Z_{S\cap T}/Z\).
No arbitrary-support finite KL bound is asserted. These formulas compare two restricted reads;
they do not assert that either approximates a dense read.

Only keys in \(S\cup T\) are needed, but their reads are not free:
\(|S\cup T|\le |S|+|T|\). Union attention logits are evaluated as actual attention logits;
a routing metric need not bound them. For multiple heads, retain the vector of actual head
output differences and apply the output projection to that vector. Replacing independent head
changes by a purported shared transform through the projection can lose cross-head cancellation.

## Finite ordered comparison traces

Represent a finite router by a binary comparison tree. A node evaluates a real gap as a function
of the original input; nonnegative values select its left child, including equality. The orientation
of a comparison and its labels encode the required deterministic tie order. A leaf supplies the
selected route or set. Finite unrolling is an assumption about the represented program, not a
termination proof for an arbitrary loop.

Every decision affecting the executed path must occur in this tree: queue order, pruning, skips,
cell assignments and final selection. Each node's gap function incorporates the preceding
computations on that path. Checking just a final top-k margin is insufficient. A nonsmooth clamp
inside a differentiated map also needs branch treatment; an actual injected correction can instead
remain outside that map.

For an input x, sign each executed gap g toward the selected branch, so its nominal value is
\(m=g(x)\). If every such gap satisfies

\[
 |g(y)-g(x)|<g(x),
\]

the perturbed execution has the same complete trace and leaf label. The proof follows the tree:
the inequality prevents the first comparison from changing sign, after which the same argument
applies in the shared child. Thus no first differing decision can occur.

A uniform sufficient condition on the closed ball \(\|z-x\|\le E\) is a supplied Lipschitz
constant K for each executed signed gap on that **same ball**, together with

\[
 K E<m.
\]

A zero nominal gap cannot pass this strict test. A failed test proves neither a changed route nor
an error. Some ties have coincident outputs. Nevertheless, global continuity of hard selection
cannot be assumed: give key true score x and value 1, and key false score 0 and value 0, with true
winning a tie. Inputs \(-\varepsilon/4\) and \(\varepsilon/4\) are less than \(\varepsilon\)
apart yet have output difference one. This read is discontinuous at zero.

## Nonlinear state paths and one shared tube system

At stage j let the state space be a real finite-dimensional Hilbert space; its dimension may vary
with j. The state includes all relevant cache entries and recurrent summaries. For a fixed branch a,
let \(F_{j,a}\) be the smooth stage map. Reference and corrected executions satisfy

\[
 x_{j+1}=F_{j,a_j}(x_j),\qquad
 y_{j+1}=F_{j,b_j}(y_j)+c_j,\qquad e_j=y_j-x_j.
\]

Both branch labels come from the same stage router evaluated at their respective states. The
increment \(c_j\) is the actual correction executed at \(y_j\), including any state dependence.
It is not silently replaced by a correction evaluated at \(x_j\).

Set

\[
 J_j=DF_{j,a_j}(x_j),\quad
 h_j=F_{j,b_j}(y_j)-F_{j,a_j}(y_j),\quad
 r_j=F_{j,a_j}(y_j)-F_{j,a_j}(x_j)-J_je_j.
\]

Adding and subtracting the fixed-branch terms proves the exact recurrence

\[
 e_{j+1}=J_je_j+c_j+h_j+r_j.
\]

Supply radii \(E_j\ge0\), correction bounds \(\|c_j\|\le C_j\), branch bounds B_j and
nonnegative curvature bounds H_j. The reference branch has its stated derivative throughout
\(\overline B(x_j,E_j)\), and uniformly for u,v in that ball require

\[
 \|DF_{j,a_j}(u)-DF_{j,a_j}(v)\|\le H_j\|u-v\|.
\]

This is a neighborhood hypothesis, not a derivative sampled at one point. Integrating the derivative
along the line segment from x_j to y_j gives
\(\|r_j\|\le (H_j/2)\|e_j\|^2\).

At each stage either certify all executed comparisons on this same ball, which gives h_j=0,
or supply the actual executed-state branch-difference bound

\[
 \|F_{j,b_j}(y_j)-F_{j,a_j}(y_j)\|\le B_j.
\]

The second alternative permits an observed-state union-read certificate. Its comparison uses both
masks at y_j with the same actual attention logits and values. No uniform branch-jump bound over
the ball is required by this alternative: the actual bound is already available before the radius
induction uses it. Derivative and comparison Lipschitz bounds remain uniform on the stated balls.
Operations after the read need their own transport bounds, or the attention output must be its own
stage. An attention-output bound is not automatically a whole-layer bound.

Starting from \(\|e_0\|\le E_0\), require

\[
 \|J_j\|E_j+C_j+B_j+\tfrac12H_jE_j^2\le E_{j+1}.
\]

Induction proves every \(y_j\) belongs to its stated ball. The derivative remainder estimate and
the guard certificates therefore apply on the very same radii used to propagate the error.
If these radii do not fit domains with the required bounds, the certificate has not been supplied.

## Signed terminal reads

Use continuous linear functionals consistently. Given a terminal read \(\ell\), pull it back by
composition:

\[
 \lambda_m=\ell,\qquad \lambda_j=\lambda_{j+1}\circ J_j.
\]

Applying these functionals to the recurrence and telescoping gives

\[
 \ell(e_m)=\lambda_0(e_0)+
 \sum_{j<m}\lambda_{j+1}(c_j+h_j+r_j).
\]

If actual branch differences are retained, define

\[
 D=\lambda_0(e_0)+\sum_{j<m}\lambda_{j+1}(c_j+h_j),\qquad
 R=\sum_{j<m}\|\lambda_{j+1}\|\,\tfrac12H_jE_j^2.
\]

Then \(|\ell(e_m)-D|\le R\). When only bounds on h_j are retained, omit h_j from D and
add \(\sum_j\|\lambda_{j+1}\|B_j\) to R. Affine reference branches have zero curvature
remainder, so the signed identity is exact even with nonzero route jumps. Signs matter: two scalar
corrections +1 and -1 transported by identity maps have signed terminal change zero, while the sum
of their norms is two. This statement is about one supplied terminal read, not just a global norm
product.

## Concrete complete certificates

A nonlinear two-stage example uses a square followed by an affine reflection:
\(F_0(t)=t^2\) and \(F_1(t)=1-t\). Both comparison trees have a single leaf. Take reference
states \((0,0,1)\), corrected states \((1/2,3/8,5/8)\), and actual corrections \((1/8,0)\).
The radii \((1/2,3/8,3/8)\) close the recurrence with curvature constants \((2,0)\).
For the identity terminal read, the backwards functionals have coefficients \((0,-1,1)\).
The signed retained effect is \(-1/8\), the curvature bound is \(1/4\), and the actual
terminal difference is \(-3/8\), attaining the interval endpoint. Comparing terminal reads
\(t\) and \(0\) gives nominal margin 1 and certified lower margin \(5/8>0\).

An affine route-change example selects \(F_{\mathrm{left}}(t)=t\) for \(t\ge0\) and
\(F_{\mathrm{right}}(t)=t+1\) otherwise. The reference starts at the ordered tie 0; the
corrected input is \(-1/4\). The selected branches differ, and their difference at that same
corrected input is 1. With no injected correction, the terminal discrepancy is \(3/4\),
exactly the initial discrepancy \(-1/4\) plus the branch jump 1. The radius recurrence
charges \(1/4+1=5/4\), curvature is zero, and the signed interval is exact. This example uses
the actual-jump alternative; it does not certify stability at a zero nominal margin.

## Acceptance semantics and implementation costs

Include final normalization among the smooth stages so the terminal reads are the actual candidate
logits. For reference choice a and competitor b, use the pairwise logit read and reference margin
\(m_{ab}=\mathrm{logit}_a(x)-\mathrm{logit}_b(x)\). With its signed term D_ab and remainder
R_ab, require

\[
 m_{ab}+D_{ab}-R_{ab}>0\quad\text{for every tested competitor }b\ne a.
\]

The lower endpoint then proves a remains the strict winner on that supplied candidate set. A tie
is not a strict win. To claim full-vocabulary preservation, the candidate set must contain the full
vocabulary; four candidates certify only those four. Preservation of a reference choice gives no
claim about its correctness, recovery of a reference failure, or cross-entropy improvement.

An application may execute its reference and candidate computations, evaluate a valid certificate,
and accept the candidate only if every required condition is verified; otherwise it can return its
computed reference output. This is a conditional acceptance policy. The implementation must account
for reference execution, union reads, derivative and bound evaluations, comparison tracing, and all
certificate work. Exact-real theorems do not verify floating-point rounding, device kernels, or
runtime cost. No assertion is made that these bounds are useful for any particular model, that routes
stay fixed on observed data, that training finds favorable corrections, or that the work is
unconditionally subquadratic. A causal implementation must also use only keys and state available at
the query time; finite-carrier mathematics alone does not establish causality.
