# BRST sewing and a possible reduction to ordinary CCY blocks

Status: bounded mathematical investigation, paused on user steering to the
level-ten direct-definition auxiliary benchmark. No numerical comparisons
were performed. This is a conditional construction and a precise limitation
of one continuation, not a completed production algorithm.

## What an alternating resolution must contain

Let a module be represented by a complex with cohomology only in degree
zero. An Euler-character identity determines its character, but does not
specify a trilinear vertex. To sew the resolution, every pair of pants must
carry a chain-compatible trilinear form, and the sewn pairing must be
compatible with the differential. Cutting a spanning tree expresses a
connected graph as a tree with one pair of open ends for each independent
cycle. The irreducible sewn amplitude is then a supertrace of the lifted
operator over the complexes on these cycle cuts, provided the chain-map
identities hold. Exact pairs cancel in that supertrace. The surviving
cohomology gives the desired irreducible block.

This is a graph construction, independent of genus. It does **not** assign
independent character signs to every edge: resolution degrees on the cut
edges propagate through the lifted vertices, altering screening numbers,
charges and phases on the tree.

Konno's screened-vertex construction supplies the relevant structure:
[hep-th/9212118](https://arxiv.org/pdf/hep-th/9212118), Eqs. (3.29),
(4.10)–(4.11), and (5.19). In particular, transporting a screening
BRST charge through a trilinear vertex produces another screened vertex
with changed screening number and an explicit phase. The alternating
trace formula retains that information. Its terms are screened Fock-space
sewings; identifying them with ordinary Virasoro CCY blocks requires an
additional continuation argument.

## The conditional CCY bridge

At generic momenta, a Heisenberg Fock module is isomorphic to its Virasoro
Verma module. A screened three-point form satisfying the Virasoro Ward
identities is then its highest-state three-point constant times the
standard Virasoro descendant form. Consequently, a sewn Fock-space term
at such generic parameters would equal

\[
 \frac{\prod_v C_v}{\prod_e N_e}\,
 \mathcal F_{\Gamma}(c,\{h_e\},\{h_a\};\{q_e\}),
\]

where the block on the right is an ordinary CCY block and the constants
are screened highest-state couplings and pairings. This observation avoids
computing high descendant vertex tensors individually.

To obtain a usable algorithm, one still must construct a simultaneous
parameter deformation for every contributing resolution term that:

1. preserves the local screening charge conditions and the specified
   contours;
2. makes the internal Fock modules generic Verma modules;
3. has a controlled limit to the chain-compatible critical vertices and
   pairings; and
4. retains the normalization and phase in each term of the supertrace.

The null-embedding weights alone do not establish these conditions. Nor
does declaring a generic shifted primary at each null position determine
the screened coupling constants.

## A concrete restriction on the previously proposed exact family

Use the standard Coulomb-gas convention
\(h=\alpha(Q-\alpha)\), \(Q=b+b^{-1}\), with screening charges
\(b,b^{-1}\). On a closed genus-\(g\) surface, global charge neutrality
requires

\[
 \sum_a\alpha_a+S_+b+S_-b^{-1}=Q(1-g).
\tag{1}
\]

It follows by adding the local neutrality conditions: the two charges on
each sewn pair sum to \(Q\). Hence no internal-momentum deformation can
repair a violation of (1).

For the punctured genus-two graph and the continued external
\((2,1)\) field, \(\alpha_\psi=-b/2\), equation (1) becomes

\[
 S_+b+S_-b^{-1}=-\frac b2-b^{-1}.
\tag{2}
\]

For generic \(b\), this would require \(S_+=-1/2\) and \(S_-=-1\).
In particular, it cannot be implemented by a fixed integer number of
ordinary screenings. At the Ising value \(b^2=-4/3\), however, the
integer choice \(S_+=S_-=1\) satisfies (2), because
\(3b/2+2b^{-1}=0\).

Thus the exact generic \((2,1)\) continuation used in the tangent
analysis is not itself a generic screened realization of the closed
one-point graph. This does not exclude a different continuation. For
example, retaining \(S_+=S_-=1\) and setting

\[
 \alpha_\psi=-2Q,\qquad h_\psi=-6Q^2=1-c
\tag{3}
\]

makes (1) hold identically. Equation (3) has the correct external weight
\(1/2\) at Ising, but differs from the \((2,1)\) Kac family away from
that point. It is a candidate for the conditional CCY bridge, not a proved
quotient regulator. The full local screening contours and complex lifts
have not yet been constructed along it.

## Finite state-counting data through physical level ten

The following signed offsets are the Euler-character data for the Ising
modules. They are necessary bookkeeping for a resolution, not vertex
couplings:

| Module | Base weight | Offsets and signs needed here |
|---|---:|---|
| vacuum | 0 | \(+0,-1,-6\) |
| fermion | \(1/2\) | \(+0,-2,-3,+7\) |
| spin, unsplit edge | \(1/16\) | \(+0,-2,-4,+10\) |
| spin, either split edge | \(1/16\) | \(+0,-2,-4,+10,+14\) |

The split edges carry half the physical degree of their individual
Virasoro level. Thus an individual split spin module can require its
level-fourteen resolution term even though the physical cutoff is ten.
For the spin module, the first intersection of its level-two and
level-four submodules is at level ten. For the fermion module, the first
intersection is at level seven; there is no extra level-eight character
term.

Along the momentum convention of the exact-family analysis, the spin
weights corresponding to the first five offsets are

\[
 \frac1{16},\quad\frac{33}{16},\quad\frac{65}{16},\quad
 \frac{161}{16},\quad\frac{225}{16};
\]

at Ising they have momenta \(b/4,5b/4,7b/4,11b/4,13b/4\).
The fermion weights are \(1/2,5/2,7/2,15/2\).
The possible shifted Kac continuation \(h_{4,1}(c)\) of the fermion's
secondary null obeys

\[
 h_{4,1}-h_{2,1}=-1-3b^2,
\]

which equals three only at Ising. Consequently it is not a generic
Virasoro submodule of the \((2,1)\) module; the null embedding and
its chain relation exist at the critical point. This is another reason
to specify the continuation of the entire screened tensor, rather than
only its highest weights.

## What remains

A complete CCY implementation would need the explicit lifted screened
three-point constants, their contour phases, and the generic-to-critical
identification for each resolution term. Once those are proved, the
truncation is finite and the same construction applies to any plumbing
graph. The present work has identified that route and one useful global
neutrality constraint, but has not established the requisite constants or
the limiting identity. The separately verified level-five numerical
method must not be described as this general BRST construction.
