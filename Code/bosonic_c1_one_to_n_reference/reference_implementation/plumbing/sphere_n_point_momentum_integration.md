# Liouville-momentum integration for sphere n-point amplitudes

## Purpose

For a trivalent sphere channel \(T\), an \(n\)-point Liouville correlator has
\(n-3\) internal momenta.  Schematically,

\[
G_T(q,\bar q)=
\int_{\mathcal C_T^{n-3}}
\prod_{e\in E(T)}\frac{dP_e}{\pi}
\prod_{v\in V(T)}C(P_{v,1},P_{v,2},P_{v,3})
\,\mathcal F_T(P;q)\mathcal F_T(P;\bar q).
\]

The numerical problem should be separated into two layers:

1. analytic continuation of the momentum contours, including every crossed
   DOZZ pole;
2. numerical quadrature on each continuous or residue stratum.

A single tensor-product Gauss rule does not make this separation and becomes
both inaccurate near a contour wall and exponentially expensive as \(n\)
grows.

## The first equal-energy five-point wall

On the convergent ray \(\omega=it\), the incoming and outgoing Liouville
momenta are \(2it\) and \(it/2\).  The incoming--outgoing cherry coefficient
\(C(2it,it/2,P)\) has the nearest pair of denominator poles

\[
P_\pm=\pm\left(\frac52\omega-i\right).
\]

They pinch the real quotient-contour endpoint at \(t=2/5\).  For
\(2/5<t<1/2\), the analytically continued channel is

\[
\int_0^\infty\frac{dP}{\pi}\,I(P)
-2i\mathop{\rm Res}_{P=(5/2)\omega-i}I(P).
\]

The residue removes one momentum integration.  It is therefore cheaper than
the continuous double integral.  A second DOZZ pole family meets this
residue at \(t=1/2\); the present first-wall formula must not be used beyond
that point without adding the next residue stratum.

The implementation in `sphere_five_point_equal_energy.py` now applies this
term whenever the selected OPE channel pairs the incoming operator with an
outgoing operator.  A channel in which the incoming operator is at the middle
vertex remains on its real double contour below \(t=1/2\).

## Threshold-adapted quadrature

The continued contribution is strongly concentrated near the Liouville
threshold \(P=0\).  Increasing a global Gauss--Legendre order converges slowly.
At the same time, an aggressive map such as \(P=P_{\max}u^2\) places nodes so
close to \(h=1+P^2=1\) that the fixed-weight \(c\)-recursion becomes
ill-conditioned.

The current practical rule uses

\[
P=P_{\max}u^{5/4},\qquad 0<u<1,
\]

with the Jacobian included in the weights.  In the \(t=0.46\) channel-overlap
test, the ordinary order-24 rule differed by about \(6\%\), whereas the
power-\(5/4\), order-20 rule differed by about \(0.17\%\).  This is an interim
rule, not the final general solution.

The preferred production rule is a three-part quadrature:

1. **Threshold panel.** Determine the exact small-\(P\) power of the complete
   integrand, factor it out, and apply Gauss--Jacobi quadrature.  Evaluate the
   block with a threshold expansion or direct Verma-module contraction rather
   than \(c\)-recursion at nearly degenerate weights.
2. **Bulk panels.** Use composite Gauss--Legendre rules split at the real
   projections and width scales of nearby DOZZ poles.
3. **Tail panel.** Use a Gaussian-tail or rational map with a cutoff chosen
   from the plumbing damping \(\exp[-2P_e^2|\log|q_e||]\).

The continuous integral and every residue integral should receive their own
quadrature design.  A residue stratum often has different threshold behavior
from its parent integral.

## Momentum forest for general n

For every trivalent vertex, form a pole ledger from the zeros of the four
denominator Upsilon functions.  These give affine hyperplanes in the adjacent
internal momenta and the external energies.  As the external energies move:

1. detect which pole hyperplanes cross the declared contours;
2. add their oriented residues;
3. recursively inspect the residue integrands for further crossings;
4. organize simultaneous crossings as lower-dimensional strata.

This is a momentum-space analogue of a forest formula.  It is local to the
trivalent graph and can be reused for every sphere multiplicity.  The output
in each kinematic chamber is a sum of integrals of dimensions
\(n-3,n-4,\ldots,0\).

## Avoiding exponential tensor grids

At fixed plumbing coordinates and block order, the DOZZ factors, propagators,
and descendant three-point tensors are local on the trivalent tree.  The
quadrature should therefore be contracted as a tensor network:

- momentum nodes are edge indices;
- DOZZ and descendant couplings are vertex tensors;
- plumbing powers are diagonal edge tensors;
- residue strata fix or remove selected edge indices.

Sum-product contraction on the tree replaces an explicit
\(M^{n-3}\) tensor grid by local contractions, generically of order
\(O(nM^3)\) before low-rank compression.  Singular-value compression or
tensor-train interpolation can reduce the vertex ranks further.  The same
contraction engine works on all momentum-forest strata.

## Required validation

Every production point should carry four independent diagnostics:

1. adjacent momentum rules and threshold-panel refinements;
2. adjacent Virasoro block orders;
3. agreement of overlapping OPE channels, including their residue terms;
4. distance to the next contour wall.

The equal-energy five-point scan is clean through \(t=0.48\) with the first
residue included.  The \(t=0.49\) point is useful as a near-wall diagnostic,
but should not be assigned the same systematic status until the second-wall
analysis at \(t=1/2\) is implemented.
