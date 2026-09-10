# Punctured-theta CCY component

This file records the implemented component for integration into the separate
LaTeX notes. It is not a claim that the end-to-end physical block has passed
the requested checks. No independent numerical comparison is included here.

The four internal edges, in order, have weights
\((h_1,h_L,h_R,h_3)\). The ordered three-point forms at the three spheres are
\(\rho(h_1,h_L,h_3)\), \(\rho(h_1,h_R,h_3)\), and
\(\rho(h_R,h_\psi,h_L)\), where the slots are \((\infty,1,0)\).
The external weight is held fixed throughout every central-charge recursion.
In particular, its degenerate value is not recomputed at a recursive pole.

## Large-central-charge term

The full block is the product of the universal large-central-charge vacuum
factor and a reduced block. The latter has the global SL(2) block as its
regular term. At four-edge levels \((a,b,c,d)\), the global coefficient is

\[
\frac{
 \rho(L_{-1}^{a}h_1,L_{-1}^{b}h_L,L_{-1}^{d}h_3)
 \rho(L_{-1}^{a}h_1,L_{-1}^{c}h_R,L_{-1}^{d}h_3)
 \rho(L_{-1}^{c}h_R,h_\psi,L_{-1}^{b}h_L)}
 {a!(2h_1)_a\,b!(2h_L)_b\,c!(2h_R)_c\,d!(2h_3)_d}.
\]

The global three-point coefficient is evaluated using CCY's finite
Pochhammer sum. There are no finite-central-charge descendant states or Gram
matrices in this calculation. The global seed is meromorphic at degenerate
internal weights; a prescribed limit is required there.

## Polar part

For each edge and each \(r\geq2,s\geq1\), with \(rs\) no greater than that
edge's descendant level, the coefficient receives the lower-level reduced
block at \(c_{r,s}(h_e)\), with \(h_e\) shifted by \(rs\), multiplied by

\[
\frac{-\partial_{h_e}c_{r,s}(h_e)\, A_{r,s}}
 {c-c_{r,s}(h_e)}\,P_{r,s}[\cdot,\cdot]P_{r,s}[\cdot,\cdot].
\]

The two fusion pairs are, in edge order,

\[
\begin{array}{c|cc}
h_1 &(h_3,h_L)&(h_3,h_R)\\
h_L &(h_3,h_1)&(h_R,h_\psi)\\
h_R &(h_3,h_1)&(h_L,h_\psi)\\
h_3 &(h_1,h_L)&(h_1,h_R).
\end{array}
\]

These orderings follow directly from the null-vector three-point
factorization. A null at infinity uses spectators (zero,one); a null at one
uses (zero,infinity); a null at zero uses (infinity,one). The order matters
when the null level is odd.

The implemented pole equation uses

\[
x=\frac{rs-1+2h_e\pm\sqrt{(r-s)^2+4(rs-1)h_e+4h_e^2}}{1-r^2},
\qquad c_{r,s}=13+6(x+x^{-1}).
\]

The noncancelling root (larger absolute value) is selected, consistently for
the pair \((r,s),(s,r)\). This avoids selecting the zero root at \(s=1\)
after continuation to negative weights. The universal factors \((x-1)(x+1)\)
in \(-\partial_h c A_{r,s}\) are cancelled algebraically before numerical
evaluation. Other exact confluent denominators raise an error. No residue is
discarded because a denominator is small or singular. A residue that is
exactly zero at otherwise nonsingular input contributes zero.

## Exact vacuum factor

Forgetting the inserted puncture composes the plumbing parameters exactly:
\(q_2=\hat q_{2,1}\hat q_{2,2}\). The vacuum factor therefore depends only
on \((q_1,q_2,q_3)\).

Its coefficients are evaluated by the Gaussian stress-tensor algebra that
survives at large central charge. Use vacuum states
\(\prod_{m\geq2}(L_{-m}/\sqrt c)^{k_m}|0\rangle\). Their leading diagonal
norm is

\[
\prod_{m\geq2} k_m!\left[\frac{m(m^2-1)}{12}\right]^{k_m}.
\]

Every vertex is the Wick contraction obtained from
\(\langle T(z)T(w)\rangle=c/[2(z-w)^4]\). The pair contractions between
ordered slots are

\[
\begin{aligned}
C_{\infty,0}(m,n)&=\delta_{mn}\frac{m(m^2-1)}{12},\\
C_{\infty,1}(m,n)&=\frac{m(m^2-1)}{12}\binom{m-2}{n-2}
 \quad(m\geq n),\\
C_{1,0}(m,n)&=\frac{(-1)^m(m+n-1)!}{12(m-2)!(n-2)!}.
\end{aligned}
\]

The second contraction is zero for \(m<n\); same-slot contractions are zero.
The first two expressions follow by expanding the stress tensor at infinity;
the third follows by differentiating its two-point function at one and zero.
At fixed three-edge level, sum the square of the three-leg Wick contraction
over the vacuum oscillator partitions and divide by their three diagonal
norms. The implementation uses exact rational arithmetic. It uses neither a
finite-central-charge vacuum Gram inverse nor extrapolation in \(1/c\).

## Interface and limitations

`PuncturedCCY(c, weights, external_weight, dps=80)` owns an independent mpmath
context. `reduced_series(indices=...)` returns coefficients for supplied
integer four-edge levels. The weighted truncation used by the physical
pipeline is `downward_indices(2*N, degree_weights=(2,1,1,2))`.
`theta_vacuum_seed(N)` returns rational coefficients keyed by three integer
theta levels; its four-edge pullback is \((a,b,c)\mapsto(a,b,b,c)\).

The engine computes generic Virasoro Verma blocks. An irreducible Ising block
requires a separately justified quotient prescription. A generic-weight or
central-charge limit alone can retain closed null-submodule loops. For the
Ramond Ising weight \(1/16\), the first singular vector is at level two.
Its propagation around all three split Ramond segments first affects
balanced physical level four. Thus a low-level coincidence cannot justify
the irreducible limit needed through level five or ten.

Source: Cho, Collier, and Yin, arXiv:1703.09805, especially Sections 4 and 5,
<https://arxiv.org/html/1703.09805#S5>.

## What remains to turn the Ising candidate into a full quotient algorithm

The obstruction is already distinguished from a mere numerical collision
in Cho--Collier--Yin, *Genus Two Modular Bootstrap*, Section 2.2: the
simultaneous three-weight vacuum limit retains identity null descendants.
See <https://arxiv.org/html/1705.05865#S2.SS2>. Likewise,
Javerzat--Santachiara--Foda, Sections 5.2--5.3,
<https://arxiv.org/html/1806.02790>, demonstrate that even the torus character
requires correlated regulators to obtain an irreducible module. Their
result concerns sphere four-point blocks and torus characters; it does not
establish the required four-edge higher-genus prescription.

A finite exact inclusion-exclusion target can be stated without guessing
null-cycle coefficients. At a fixed level, split an analytic Verma Gram
matrix into a quotient complement and the spans of the null generators and
their descendants:

\[
G=\begin{pmatrix}A&B\\B^{\mathsf T}&D\end{pmatrix},\qquad
S=D-B^{\mathsf T}A^{-1}B.
\]

Then

\[
G^{-1}-
\begin{pmatrix}-A^{-1}B\\1\end{pmatrix}
S^{-1}
\begin{pmatrix}-B^{\mathsf T}A^{-1}&1\end{pmatrix}
=\begin{pmatrix}A^{-1}&0\\0&0\end{pmatrix}.
\]

Replacing each of the four propagators by the right-hand side and expanding
the four differences gives an exact finite inclusion-exclusion, including
interacting null networks. This identity also specifies which terms a
resolution-based implementation must reproduce. It is not used as a
finite-level Gram substitute in the current production code.

The unresolved reduction is essential: away from the minimal-model point,
the subtracted matrix does not commute with Virasoro. Consequently its
sewing contribution cannot simply be declared an ordinary shifted Verma
block. The confluent corrections must be derived before ordinary CCY blocks
can be used to evaluate the full inclusion-exclusion. The documented
level-two simple-cycle approximation is not this complete algorithm.

The proposed generic-family detour has an additional nontrivial quotient:
at generic \(c\), \(h_\sigma=h_{2,1}\) has its level-two null, but
\(h_\psi=h_{3,1}\) already has its level-three null. Mixed cycles involving
the latter first enter balanced physical level \(11/2\). Quotienting only
the Ramond level-two loop at generic \(c\) therefore does not justify a
level-ten result after specialization to the Ising point.
