# Presentation review and proposed revisions

The initial notes contain the right mathematical distinction between the
all-order block identity and the unfinished irreducible Ising evaluation.
The revisions below aim to make that distinction easier to follow while
preserving the human notes' notation. They do not assert that the full
level-ten objective is complete.

## Opening

Replace the administrative opening and repeated validation disclaimers by
an immediate statement of the problem and current result:

> We derive a fermion-inserted double-Virasoro representation of the
> NS--R--R theta block with \(\eta\eta'=-1\), using the state and BPZ
> conventions of \texttt{Human Notes/SCblock.tex}. The resulting
> convolution uses irreducible Ising blocks. Its implementation is
> verified at the benchmark specified below through total level five;
> the complete Ising quotient required at level ten remains unresolved.

This states the practical limitation once without making the reader begin
with the history of the draft or computation.

## Put the inverse immediately after the convolution

The current long Ising subsection separates the full convolution identity
from the inverse that solves the original problem. This interrupts the
logical chain. Preferred order:

1. Geometry and the ordered insertion.
2. Double-Virasoro decomposition, then the middle Ward recurrence.
3. Tensor-product convolution, the short exact Ising decomposition, and
   the restricted inverse.
4. Generic CCY recursion and the balanced truncation.
5. Irreducible Ising evaluation: the first null cycles, their bounded
   correction, and the interacting-null obstruction at higher levels.
6. Implementation, precisely scoped numerical comparisons, and runtime.

The alternative is to move the whole Ising subsection after the inverse:
its constant coefficient already follows directly from the ground action
of \(Q\Theta\psi_0\). Either order is better than placing several pages
of null-network analysis between the convolution and its inverse.

## Define existing physical parameters before the first branch formula

No new notation is needed. Add one compact paragraph specifying the
selected conversion \(\beta_j=P_j/\sqrt2\) on the Ramond edges,
\(h_1=(Q^2/4-P_1^2)/2\), and physical central charge
\(3/2+3Q^2\). State that \(p_1\) is the intrinsic NS-primary parity,
\(\alpha,\gamma\) are total Ramond branch parities, and \(f\) is the
relative outer three-point label. If the notes are meant to be fully
self-contained, give the four nonzero physical ground three-point
components. This also makes the native odd-eta transport explicit and
reviewable.

The temporary \(L,R\) indices are justified by the four-edge Virasoro
block. Keep their definition adjacent to that block and use the same
ordering in the residue table and code interface.

## Expose the two calculations currently hidden in prose

For the tensor-product sign, replace “because the insertion is even” as a
complete explanation by the short sequence
\(G^{-1}GG^{-1}=G^{-1}\), equality of the split-cut parities, then the
expansion of the original quadratic theta sign. This provides a concrete
derivation with the existing parity symbols.

For the auxiliary Ising coefficient, show a four-row ground-state table
with NS primary, Ramond ground pair, norm/sign contribution, and resulting
parity monomial. It is substantially easier to audit than deriving the
factor \(1/2\) and signs from an unexplained final polynomial.

For the universal vacuum seed, retain the existing \(L_{-m}\) notation and
give its multiplicity norm and the three pair contractions. Explain that
forgetting the light insertion gives the exact pullback by
\(q_2=\hat q_{2,1}\hat q_{2,2}\). The term “Gaussian contractions” alone
does not enable the reader to reproduce the implementation.

## Keep computational distinctions where they matter

Several sentences repeat that a defining Gram contraction is not the
production algorithm. Say this once immediately after the block's
definition, then show the actual CCY recurrence explicitly. Likewise,
explain the full-field versus zero-mode distinction once after the mode
action. Later sections can refer back to those equations instead of
repeating warnings.

Suggested local rewrite for the middle recurrence:

> The physical stress tensor commutes with the auxiliary insertion.
> Applying BPZ conjugation gives the two identities below. We insert the
> sparse \(L_{\pm1}\) actions from the human notes and use the degenerate
> fusion rule to discard the distant branch. The remaining terms give a
> scalar recurrence between adjacent primary matrix elements.

Then display the recurrence, followed by its ground anchors and the
generic-momentum qualification. This makes the reason for each step
visible before its large equation.

Suggested rewrite introducing the rational-model difficulty:

> The denominator requires the irreducible Ising modules. A complete
> generic-Verma limit can retain states in null submodules when they
> propagate around a closed cycle. The following level-two calculation
> identifies the first such contribution and its normalization.

This is more direct than beginning with several prohibitions on possible
implementations.

## Scope every numerical statement

Use one short table containing the chosen \(b,P_j,p_1,f,\eta,\eta'\),
cutoff, arithmetic precision, PBW comparison residual, split-parameter
residual, and measured runtime. Refer to the saved output paths after the
table. State that the reported residuals are numerical differences rather
than interval bounds.

The tested \(f=0\) benchmark does not establish \(f=1\) by numerical
comparison. The native odd-form eta relabeling can be justified
algebraically, but these are different forms of evidence. Likewise, the
six-level action bound belongs only to the new middle recurrence; the
current outer grid reaches descendant level eight.

The first failed binary64-anchor run is useful in a development log, but
it should not dominate the final scientific status section after the
corrected authorized checks pass. One sentence describing the precision
correction is enough unless the failure explains a remaining limitation.

End with the precise outstanding mathematical task: remove interacting
Ising null submodules using a complete recursive prescription. Avoid an
open-ended list of hypothetical future checks, because the user specified
exactly the two comparisons and the conditional level-ten timing.
