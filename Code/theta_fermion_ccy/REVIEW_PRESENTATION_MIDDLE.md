# Presentation and normalization review of the main working notes

Reviewed `Machine Notes/theta_fermion_ramond_recovery.tex` after the initial
writeup and the first presentation revisions. This review is independent
of the main author and focuses on the middle recurrence, coefficient
normalizations, label scope, and the claims supported by the current result.
No further numerical comparison was performed.

## Corrections that affect the reader's understanding

**Distinguish the original auxiliary block from its vanishing product.**
After `zero-mode-action`, the phrase “the original, vanishing auxiliary
factor in the opposite sector” is misleading. The ordinary auxiliary block
is nonzero; it belongs to the plus ideal and annihilates the desired physical
minus sector under convolution. Suggested replacement:

> This ground-state sign places the inserted auxiliary block in the minus
> ideal. The original auxiliary block lies in the plus ideal and annihilates
> the desired physical sector under convolution.

**Show the ground primaries used for the recurrence anchors.** The sentence
“obtained from ... the explicit ground primaries” currently refers to a
calculation the reader cannot reproduce from that section. In the human
notes' (w^\pm) basis with ground metric (operatorname{diag}(1,i)), no new
basis symbol is needed:

\[
\begin{aligned}
v_{\pm1/4}^0&=u^0\otimes w^+
 \mp e^{i\pi/4}u^1\otimes w^-,\\
v_{\pm1/4}^1&=\frac{u^1\otimes w^+
 \pm e^{i\pi/4}u^0\otimes w^-}{\sqrt2}.
\end{aligned}
\]

Their squared norms are (2) and (-1). The ordered zero-mode insertion
then gives

\[
Q\Theta\psi_0 v_{\pm1/4}^0=\frac{Q}{\sqrt2}v_{\mp1/4}^0,
\qquad
Q\Theta\psi_0 v_{\pm1/4}^1=-\frac{Q}{\sqrt2}v_{\mp1/4}^1.
\]

Taking the BPZ pairing immediately yields the two anchors (sqrt2Q) and
(Q/\sqrt2). This also makes clear why the parity-one anchor is positive
despite its negative squared norm. These equations should precede the
anchor formula, or replace the unexplained reference to explicit primaries.

**Keep the label (1/2) and the physical vacuum condition separate.** The
middle coefficient has branch label (n_{\rm external}=1/2) at physical
momentum (P=Q/2). Its physical factor is the irreducible identity module,
while its auxiliary factor is the fermion. It is not a physical NS primary
of conformal weight (1/2). The current first section largely establishes
this correctly; retain that distinction when shortening the discussion.
The unsuperscripted (h_\psi=1/2) used later for the Ising block also differs
from the two double-Virasoro weights (h_\psi^{(1,2)}).

## Strengths to preserve in revision

The main notes now put the radial puncture order directly beside the
middle branching coefficient. This avoids the earlier ambiguity between
an ((\NS,\R,\R)) coefficient and a radial ((\R,\NS,\R)) matrix element.
Do not abbreviate away that sentence.

The raw primary contraction with four squared norms is the clearest
primary formula. The normalized-(mathbb B) form should remain a
consequence. Its compatibility with a chosen norm square root is explicit
and does not enter the implementation.

The two physical Ward identities have the correct orientations. The text
explains why the distant branch is removed by fusion before presenting
the scalar recurrence. This is substantially clearer than describing an
opaque finite linear solve for the middle coefficient.

The recurrence's reusable action coefficients are explicitly identified
as inputs from a free-field descendant-span calculation. Retain this
statement: only the middle primary matrix elements are recursive here;
there is no proved independent recursion for the action coefficients.

The convolution proof now gives the metric cancellation and the
polarization of the parity sign. Both are necessary logical steps; saying
only that the inserted operator is even would not establish the formula.

## Remaining improvements in ordering and scope

1. Keep the sector inverse directly after the convolution, using its
   ground coefficient from the zero-mode action. The detailed Ising
   null-network analysis belongs after the generic CCY construction or
   in a clearly labeled later section. It should not interrupt the
   argument that explains how the desired physical block is recovered.

2. Define the spin-frame convention as an adopted lifted-coordinate
   convention when stating
   (eta_2=\hat\eta_{2,1}\hat\eta_{2,2}). A coordinate product alone does
   not fix a square root of its derivative. The ongoing frame review by
   the other author should supply the precise statement.

3. The detailed human-note primary norm formulas are currently assumed
   rather than repeated. If this note is intended to stand alone, include
   them with the existing (ell) notation in a short appendix. Otherwise
   state clearly at the beginning that those primary definitions and norm
   formulas are inputs from `Human Notes/SCblock.tex`, while the inserted
   block derivation itself is fully supplied here.

4. Qualify the finite action bound: descendant level six concerns the
   new middle recurrence at physical level ten. It is not the largest
   action system in the outer branching grid, which reaches level eight.

5. Keep the numerical scope in one table: the successful result is the
   stated (p_1=f=0), ((\eta,\eta')=(+,-)) sample through total level
   five. Algebraic frame corrections for other labels are evidence of a
   different kind and must not be described as additional numerical
   coverage. The measured 99.3412 seconds belongs to level five only.

## The mathematical gap that must remain explicit

The successful level-five result does not establish the Ising quotient
needed through level ten. The first-null-cycle prescription has a stated
cutoff. A more carefully correlated central-charge/weight family improves
the limit but also does not automatically give the desired irreducible
block.

The separate `ISING_TANGENT_ANALYSIS.md` supplies a concrete example, not a
general impossibility claim. Continuing the Ramond weights with momentum
(b/4), and the external and internal fermion as ((2,1)), permits the
generic NS level-two quotient first. Nevertheless the new NS level-three
null and the Ramond level-two null form a closed cycle with coefficient
(3/8), first at degree (11/2). Another cycle involving the Ramond
level-four null appears at degree (15/2). These contributions require
additional subtraction, or a different fully justified prescription.

The notes should say exactly what remains: derive the contributions of
interacting and higher null submodules, including their intersections,
in a form that can be evaluated through CCY recursion. They should not
claim that no regulator could work, or that the entire convolution
identity is obstructed. The established identity and middle recurrence
remain useful; the unfinished step is the all-required-level irreducible
Ising evaluation and hence the conditional level-ten timing.
