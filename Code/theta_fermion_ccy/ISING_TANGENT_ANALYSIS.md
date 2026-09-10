# Analytic regulator constraints for the irreducible Ising block

The calculations below are exact first variations of Virasoro Ward
polynomials. They are symbolic algebra, not additional numerical conformal
block cross-checks. Their executable derivation is
`ising_tangent_derivation.py`, and the triple-vertex gradients are retained in
`results/ising_primitive_triple_slopes.json`.

The task is not only to avoid coincident numerical poles. A regularized
Verma block can propagate a closed network of states which become null at
the Ising point. Such networks must either vanish in the chosen limit or be
subtracted with a derived prescription. The sphere and torus-character
regulators of Javerzat, Santachiara and Foda do not by themselves establish
that prescription on the present graph; see their Sections 3 and 5,
https://arxiv.org/html/1806.02790.

## Primitive null vectors and their first variations

At (c=\tfrac12), the relevant primitive singular vectors, in fixed raw
normalization, are

\[
\begin{aligned}
\chi_{\sigma,2}&=(L_{-2}-\tfrac43L_{-1}^2)\nu_{1/16},\\
\chi_{\psi,2}&=(L_{-2}-\tfrac34L_{-1}^2)\nu_{1/2},\\
\chi_{\psi,3}&=(\tfrac34L_{-3}-3L_{-2}L_{-1}+L_{-1}^3)\nu_{1/2},\\
\chi_{\sigma,4}&=(-\tfrac14L_{-4}+\tfrac{11}{6}L_{-3}L_{-1}
 +\tfrac{49}{144}L_{-2}^2-\tfrac{25}{6}L_{-2}L_{-1}^2
 +L_{-1}^4)\nu_{1/16}.
\end{aligned}
\]

A prime below denotes the coefficient of a common small parameter, evaluated
at the Ising point. For the level-two null with arbitrary weight,

\[
\| (L_{-2}-\tfrac{3}{2(2h+1)}L_{-1}^2)\nu_h\|^2
=\frac c2+\frac{h(8h-5)}{2h+1}.
\]

Thus the two level-two norm slopes are

\[
\|\chi_{\sigma,2}\|^{2\prime}=\frac{c'}2-\frac{28}{9}h_\sigma',
\qquad
\|\chi_{\psi,2}\|^{2\prime}=\frac{c'}2+\frac74h_\psi'.
\]

For the higher primitive vectors in the normalization above,

\[
\|\chi_{\psi,3}\|^{2\prime}=\frac{45}{8}c'-\frac{105}{8}h_\psi',
\qquad
\|\chi_{\sigma,4}\|^{2\prime}
=\frac{13475}{5184}c'+\frac{13475}{648}h_\sigma'.
\]

In a ((\psi,\sigma_1,\sigma_2)) three-point vertex, inserting the
level-two nulls on any two legs gives the same first variation,

\[
\frac{c'}2+\frac43h_\psi'-\frac{10}{3}(h_{\sigma_1}'+h_{\sigma_2}').
\tag{1}
\]

Putting all three level-two nulls into the vertex gives twice (1). Therefore
(c'=1\), (h_\sigma'=0\), (h_\psi'=-\tfrac38\) removes both the first
simple-null-cycle terms and the first interacting four-null divergence.
This assertion concerns the first primitive nulls only.

At the middle vertex the first variations with two Ramond nulls are

| Null levels on the Ramond legs | Coefficients of ((c',h_\psi',h_{\sigma_1}',h_{\sigma_2}')) |
|---|---|
| ((2,2)) | ((\tfrac12,\tfrac43,-\tfrac{10}3,-\tfrac{10}3)) |
| ((2,4)) | ((-\tfrac{11}9,\tfrac{77}{27},\tfrac{385}{54},-\tfrac{385}{54})) |
| ((4,2)) | ((-\tfrac{11}9,\tfrac{77}{27},-\tfrac{385}{54},\tfrac{385}{54})) |
| ((4,4)) | ((-\tfrac{3575}{1728},-\tfrac{2275}{216},-\tfrac{11375}{432},-\tfrac{11375}{432})) |

For reference, the first variation of a vacuum–Ramond–Ramond vertex with
both Ramond level-two nulls is

\[
\frac{c'}2-\frac{28}{9}h_0'
-\frac{14}{9}(h_{\sigma_1}'+h_{\sigma_2}').
\]

These formulas explain why independently perturbing all weights is not a
controlled quotient prescription.

## Primitive three-null vertices

The coefficients of
((c',h_\psi',h_{\sigma_1}',h_{\sigma_2}')) are:

| Primitive null levels ((\psi,\sigma_1,\sigma_2)) | First variation |
|---|---|
| ((2,2,2)) | ((1,\tfrac83,-\tfrac{20}3,-\tfrac{20}3)) |
| ((2,2,4)), ((2,4,2)) | identically zero |
| ((2,4,4)) | ((-\tfrac{3575}{432},-\tfrac{2275}{54},-\tfrac{11375}{108},-\tfrac{11375}{108})) |
| ((3,2,2)), ((3,4,4)) | identically zero |
| ((3,2,4)) | ((-\tfrac{275}{24},\tfrac{1925}{72},\tfrac{9625}{144},-\tfrac{9625}{144})) |
| ((3,4,2)) | ((\tfrac{275}{24},-\tfrac{1925}{72},\tfrac{9625}{144},-\tfrac{9625}{144})) |

The first proposed tangent leaves an interacting primitive-null graph with
levels ((3,4,4,2)) on ((\mathrm{NS},L,R,3)). Its physical degree is
(3+(4+4)/2+2+\tfrac12=\tfrac{19}{2}\), within level ten. The two outer
three-null slopes and the middle two-null slope are nonzero, so three
powers of the regulator cannot cancel the four inverse null norms.

A better tangent kills the middle ((2,2)) and ((4,4)) entries together:

\[
c'=1,\qquad h_L'=h_R'=h_3'=\frac1{56},\qquad
h_{\psi,\mathrm{external}}'=-\frac27.
\tag{2}
\]

The internal NS fermion slope can initially be left independent. The
Ramond slopes must agree if the vacuum weight is removed at a higher order:
otherwise the vacuum (L_{-1}) contribution contains
((h_L-h_3)^2/h_0), which can itself diverge.

With (2), every primitive four-edge null graph has sufficient regulator
powers to be finite. If the two split Ramond null levels agree, the middle
vertex loses its first-order term. If they differ, one of the outer
three-null vertices has identically zero first variation in the table.
This power counting does not address intersections of null submodules.
It also does not show that the finite unwanted contribution vanishes.

## The exact degenerate-field family behind (2)

Set the Virasoro parameter near (b^2=-\tfrac43\), and choose

\[
c=1+6(b+b^{-1})^2,\qquad
h_\sigma=\frac{(b+b^{-1})^2}{4}-\frac{b^2}{16},\qquad
h_{\psi,\mathrm{external}}=h_{2,1}=-\frac12-\frac{3b^2}{4}.
\tag{3}
\]

The Ramond momentum is (P_\sigma=b/4). The degenerate external field shifts
this momentum to (P_\sigma-b/2=-P_\sigma), giving the same conformal weight.
Thus the external degenerate fusion holds identically, rather than only to
first order. Differentiating (3) gives (2).

At the Ising point the level-two and level-four Ramond null primaries have
momenta (5b/4) and (7b/4), respectively. The external ((2,1)) field
connects these two null families to one another, but does not connect either
to itself. This gives a useful explanation of the two vanishing middle
slopes. It is insufficient to prove the full quotient when the new NS null
module also propagates.

## An explicit surviving secondary null cycle

Suppose the internal NS fermion is also continued as (h_{2,1}(c)), and its
always-null level-two submodule is quotiented first at generic (c). This
quotient is safe as an independent preliminary limit: only the single NS
edge is then degenerate, and that one edge does not form a closed loop.
At the Ising point the remaining NS module develops its second primitive
null at level three.

Along (3), with (h_{\psi,\mathrm{internal}}=h_{2,1}), the relevant slopes
are

\[
\begin{aligned}
\|\chi_{\psi,3}\|^{2\prime}&=\frac{75}{8},&
\|\chi_{\sigma,2}\|^{2\prime}&=\frac49,&
\|\chi_{\sigma,4}\|^{2\prime}&=\frac{1925}{648},\\
\rho(\chi_{\psi,3},\chi_{\sigma,2},\nu_\sigma)'&=\frac54,&
\rho(\chi_{\psi,3},\chi_{\sigma,4},\nu_\sigma)'&=\frac{275}{48}.
\end{aligned}
\]

Permuting the two Ramond punctures can change the signs of the last two
entries, but the two original vertices have the same ordering and their
product is unchanged. The simple null cycle using the NS edge and edge
three therefore has nonzero leading coefficients

\[
\frac{(5/4)^2}{(75/8)(4/9)}=\frac38
\quad\text{at degree }\frac{11}{2},
\qquad
\frac{(275/48)^2}{(75/8)(1925/648)}=\frac{33}{28}
\quad\text{at degree }\frac{15}{2}.
\tag{4}
\]

The preliminary quotient of the generic level-two NS null does not alter
(4). Its descendant insertion vanishes identically on the fusion family
(3), so adding such a descendant to the analytic representative of the
level-three null leaves these first variations unchanged.

Thus (3) alone does not give the irreducible Ising block through level ten.
The first counterexample is beyond level five, so success of the requested
level-five comparison would not resolve it.

## Why finiteness alone does not justify a shifted-block subtraction

For a simple null cycle, each relevant vertex starts at first order. To
that order the null vector transforms as a highest-weight vector, because
its off-critical positive-mode action is first order and its contraction
with another null leg supplies another zero. The leading cycle can
therefore be reduced to shifted Virasoro three-point forms.

An interacting four-null graph can instead require the second variation of
a vertex whose first variation vanishes. At that order the positive-mode
anomaly contributes at the same order as the retained vertex. It cannot be
discarded merely because the complete graph is finite. One must derive its
cancellation in the full quotient, or express the remaining terms through
appropriate derivatives or additional insertions of CCY blocks. Null-submodule
intersections provide another independent issue, since their norms can
vanish to higher order.

The exact family (3) and the gradients above are useful constraints for
that derivation; they are not a completed all-order quotient algorithm.
