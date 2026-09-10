# Adversarial review of the initial full notes

Reviewed `Machine Notes/theta_fermion_ramond_recovery.tex` and its Ising
component after the initial writeup. This review used algebra and source
inspection. No numerical component comparison was added or run. Discussion
with the CCY agent is incorporated below.

## Findings that need changes or explicit qualification

0. **Critical final finding: the hatted vertex needs contour transport.**
   The literal human-note tensor-product vertex does not by itself cancel
   the additional physical NS fermion-count sign in enlarged sewing.
   Enlarged sewing has (-1)^(A+mathsf A), the auxiliary factor has
   (-1)^mathsf A, and the stated physical sewing has no (-1)^A.
   A quadratic parity cocycle cannot remove this remaining linear sign.
   I raised this gap independently; the CCY agent derived the native
   three-point contour-transport phase whose square is (-1)^A and thus
   resolves it. See REVIEW_MATH_CCY.md for the complete phase, including
   eta_native=(-1)^f eta and the ground-frame conversion. The main notes
   must define the hatted vertex using that transported convention rather
   than claiming that the literal human tensor-product vertex already
   gives the displayed convolution. This is a correction to the
   derivation, not a numerical tolerance issue.

1. **The final numerical-status paragraph is stale.** The root reports
   that the authorized integrated level-five calculation with high-precision
   outer anchors and the first-null-cycle correction has now passed both
   comparisons. The notes must cite that particular report and distinguish
   its bounded quotient prescription from the uncompleted level-ten
   algorithm. A successful generic-regulator run is not interchangeable
   with a successful null-cycle run.

2. **The action-level bound must say “middle recurrence.”** The bound six
   is correct for middle pairs at physical level ten: the largest label
   is 9/4, so the larger endpoint's L1 image has descendant level six.
   The outer implementation currently also requests L−1 on the largest
   Ramond endpoint, which can require descendant level eight. The CCY
   agent independently found this scope discrepancy.

3. **Give the missing sewing-sign calculation.** The sentence that an even
   tube insertion leaves the original cocycle is correct only after the
   two split parities are identified and the physical middle metric has
   canceled the extra physical inverse metric. Write the quadratic theta
   sign as p1*p2+p1*p3+p2*p3, substitute p_i=epsilon_i+delta_i,
   and explicitly remove the physical and auxiliary quadratic terms.
   The remaining cross terms are exactly the displayed cocycle. This
   explains why there is one edge-2 parity bit despite two propagators.

4. **Make the first-cycle completeness argument precise.** It is false to
   say that every spin null of level four first affects physical level
   eight. A mixed loop with spin-null levels (4,2,2) on the split-left,
   split-right, and third edges has balanced degree five. In the vacuum
   branch, however, the NS edge must then remain the identity primary;
   its vertex between distinct shifted spin highest weights vanishes.
   In the NS fermion branch the same configuration starts at degree
   11/2. Thus it does not invalidate the bounded level-five prescription,
   but a selection argument is essential.

5. **Distinguish a fixed-weight divergence from impossibility of every
   regulated limit.** The explicit 8/(c−1/2) pole at physical level 13/2
   is a genuine counterexample to the fixed-weight limit. It does not
   prove that no correlated path works. A newly derived tangent cancels
   all first-order level-two null vertices (see below). Higher primitive
   nulls still prevent a claim that a complete path has been obtained.

6. **The full irreducible Ising algorithm remains missing.** The notes
   correctly say so. Neither the first-cycle result nor the passing
   level-five comparison proves the level-ten prescription. In particular,
   a shifted-primary block subtraction valid at leading order is not
   automatically sufficient when a larger null graph has a pole: its
   subleading positive-mode and Gram-mixing corrections can leave finite
   terms. A universal character multiplier is not justified.

## Algebra that survived the challenge

- The auxiliary definition gives Theta^2=−1 and anticommutation with every
  auxiliary fermion. With the auxiliary-first graded tensor product it
  also anticommutes with the physical supercurrent, so it commutes with
  the product psi*G and hence with both Virasoro algebras. Theta*psi is
  even and has the physical identity as its tensor factor.

- The raw and normalized middle matrix elements are consistent. The norm
  ratio is negative, so norm square roots can be chosen to make Theta
  act as −i on both normalized parities. Anticommuting Theta past the
  odd field then gives the stated +i times the normalized B coefficient.
  The raw squared-norm formula avoids reliance on this choice.

- The four-edge raw formula has one squared norm per internal edge,
  including both split Ramond edges. The external field has no inverse
  norm. Translating it to three normalized branching coefficients leaves
  precisely one factor of the external primary norm.

- The auxiliary parity polynomial is correct. In the NS vacuum branch,
  ground parities (0,0) and (1,1) give +Q/sqrt(2) and −Q/sqrt(2).
  In the NS fermion branch, (0,1) and (1,0) both give +Q/(2*sqrt(2))
  after the outer rho phases, four ground norms, and theta/NS signs are
  included. The CCY agent reconstructed these factors independently from
  the raw middle pairing.

- Multiplication by eta2*eta3 sends (1−eta2*eta3) to its negative and
  sends (eta1*eta2+eta1*eta3) to its negative in the star algebra.
  Consequently the complete auxiliary factor lies in the minus ideal.
  Its constant acts as sqrt(2)*Q on that ideal. The triangular quotient
  and the refusal to project away an out-of-sector remainder are correct.

- The physical propagators multiply because the middle physical insertion
  is the identity. This is independent of auxiliary incoming/outgoing
  levels and does not license projecting the full block onto equal split
  levels before division.

- The balanced truncation has positive additive degree and is downward
  closed. It therefore contains every term required in the triangular
  quotient. Different split parameters are tracked by a finite Laurent
  polynomial in u. The condition l=r is an output, not an imposed input.

- Forgetting the added vacuum puncture leaves the original theta sewing
  with q2=hat(q21)*hat(q22). Hence the universal large-c vacuum factor
  is a pullback from three parameters. Two numerator factors divided by
  one denominator factor leave one factor to restore after reduced
  convolution division.

## Why the first shifted null blocks are legitimate through level five

At fixed highest weights, a level-two null vector has positive-mode
images of order epsilon=c−1/2. A two-null three-point form is itself of
order epsilon. In its Ward identity the unwanted positive-mode image of
one null is coupled to the other null; the latter vanishes at epsilon=0.
That anomalous term is therefore order epsilon^2. Dividing by epsilon
and taking the limit gives homogeneous Ward identities for primaries
of the shifted weights. Thus the first derivative of the two-null
vertex generates the ordinary shifted Virasoro block, not an extra
descendant block. Its primary coefficient is one after division by the
null norm. This argument was independently challenged and accepted by
the CCY agent.

The next primitive nulls cannot form an allowed nonzero closed graph
within the remaining level-five budget, including the mixed spin-null
case described above. The always-null vacuum L−1 module is removed by
taking its weight deformation asymptotically smaller than c−1/2.
This supplies a bounded derivation; it does not address the later
interacting null networks.

## Additional exact symbolic result from the debate

Using fixed level-two null states, the first variation of every two-null
vertex with primary weights (1/2,1/16,1/16) is

    c'/2 + (4/3) h_psi' − (10/3)(h_sigma1' + h_sigma2').

The first variation of the triple-null vertex is exactly twice this.
For the vacuum spectator, the two-spin-null variation is

    c'/2 − (28/9) h_vac' − (14/9)(h_sigma1' + h_sigma2').

The exact derivative coefficients are recorded in
`results/ising_null2_symbolic_slopes.json`. In particular,

    c'=1, h_sigma'=0, h_psi'=−3/8

kills both the two-null and triple-null first variations in the fermion
vertices. Vacuum-weight variation can remain of higher order: a Ramond
cycle necessarily passes the inserted fermion vertex, while the
vacuum-null L−1 graphs are suppressed by the separate hierarchy.
Thus the fixed-weight pole is not an obstruction to every correlated
path. The middle-branching agent has derived additional level-three and
level-four null constraints; their compatibility, not the level-two
condition alone, determines whether this can complete the algorithm.

## Review conclusion

The block identity and its parity-valued convolution are internally
consistent only after the hatted-vertex contour transport and the missing
quadratic-sign explanation are supplied. The bounded
level-five Ising subtraction has a mathematical rationale in addition
to the authorized integrated numerical evidence. The notes must continue
to label the full Ising quotient and level-ten algorithm as unfinished.

## Changes made following the debate

- Added the raw auxiliary norm/sign table to the exact decomposition.
- Moved the long null analysis to `theta_fermion_ising_nulls.tex`, so the
  exact decomposition can be followed immediately by sector inversion.
- Added both explicit first-cycle shifted-block formulas and their
  homogeneous-Ward argument, including the mixed-null degree-five
  selection rule.
- Qualified the fixed-weight pole as a counterexample to that path.
- Added the exact degenerate-field family's surviving secondary cycles:
  coefficients 3/8 and 33/28 at physical levels 11/2 and 15/2. Their
  norm and vertex slopes make the counterexample independently readable.
- Promoted the bounded production API `IsingFermion`, with
  `established_through_level=5` and an enforced error outside that range.

The presentation debate with the CCY agent selected the order
block decomposition, convolution, sector inverse, then implementation
and null-state limitations. This keeps the exact identity separate from
the unresolved computational prescription without interrupting its proof.
