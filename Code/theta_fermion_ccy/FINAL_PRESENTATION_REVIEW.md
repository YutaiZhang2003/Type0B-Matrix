# Final presentation review

This review concerns the current parent derivation and its direct-fermion
and Ising inclusions. It follows the completed mathematics reviews. No
production calculation, numerical comparison, or parent-TeX edit was made.
The suggestions below are deliberately small and use the existing notation.

## Highest-priority patches

**1. Keep convolution and inversion adjacent.** Currently the reader reaches
the full convolution formula, then passes through both auxiliary subsections
before learning how the physical block is recovered. Move the two input lines
from immediately after `full-convolution` to immediately after the paragraph
ending the inverse subsection. The sector proof already supplies its own
ground coefficient and therefore does not require the long auxiliary
construction to precede it. This gives a continuous argument:
convolution, sector identity, nonzero constant on that sector, triangular
division. The detailed denominator calculation then explains an input to an
already-understood recovery formula.

If the inputs are reordered at the same time, put the selected direct Fock
construction first and the compact Ising interpretation second. The Ising
formula remains useful as a representation-theoretic interpretation; it
need not look like the production denominator algorithm.

**2. Use one symbol for the auxiliary factor.** In
`theta_fermion_ising_component.tex`, replace
`\\mathbf F_{\\mathsf F}` by `\\mathbb F_{\\mathsf F}`. The parent and
direct subsection already use the latter. The present font change suggests
a different quantity without defining one.

**3. Clarify the two uses of Ramond parity labels locally.** After
`outer-frame`, replace the broad sentence about sign indices by:

> In sign exponents, descendant indices count fermionic generators;
> their absolute values denote levels. Here \(|\alpha|\) is the
> parity of the physical Ramond ground state. The labels
> \(\alpha,\gamma\in\{0,1\}\) in the branching sum below instead
> denote the parities of the enlarged primaries.

The current wording is also literally too broad: not every letter in a
sign exponent counts generators, since \(f,p_1\) and the ground labels are
parity labels. This patch supplies the missing local convention without
introducing another symbol.

**4. Repair the Ising table heading and subsection title.** Replace
`\\rho_{\\mathsf F}^{,2}` by `\\rho_{\\mathsf F}^{2}` in the sign
ledger. The entry is the square of the common outer ground form, as the
preceding derivation states. Rename “The Ising factor and its null states”
to “The Ising decomposition”: the subsection identifies irreducible
representations and does not derive null states. These two changes remove
a malformed expression and a misleading promise.

## Small changes that improve the exposition

**5. Keep implementation labels out of the first explanation of the contour
convention where possible.** The mathematical content to retain is the
explicit phase, the ratios under Ward steps, and its compatibility with
the integer-spin field \(:G\psi:\). “Native oracle” is an implementation
term that interrupts that explanation. The later implementation section
already gives the native-to-physical eta transport. Either move the native
oracle sentence and its example there, or call it “the implementation's
three-point label” when it first appears. Do not remove the explicit
statement that the adopted phase corrects the current human notes.

**6. Reduce repeated file-history statements.** The opening statement that
the original manuscript is unchanged belongs in a delivery note or the
implementation section, rather than the derivation's main paragraph. The
outer-frame section also ends with several sentences repeating that the
correction is explicit. One sentence after its mathematical justification
is enough:

> We use this phase convention throughout; it supplies the contour
> factor omitted from the current human-note definition of
> \(\hat\rho\).

Retain the preceding explanation of what sign would otherwise remain.

**7. Remove the unused alternative from the vacuum-factor paragraph.** The
CCY section correctly says that the selected numerator is missing two
universal factors and that both must be restored after division by the
full direct denominator. Its following sentence about restoring only one
factor with a reduced Ising denominator describes a different backend and
interrupts the selected construction. Delete that sentence here; the
optional backend's documentation can retain it.

**8. Compress the auxiliary timing record without removing the requested
measurements.** Retain the exact construction time, process-through-save
time, peak memory, nonzero coefficient counts, platform and saved path,
which the parent explicitly asked this subsection to document. These can
be displayed in a compact table or short paragraph. Cache counts,
contraction counts and the final metadata-write exclusion are
reproducibility details for the benchmark report. They need not interrupt
the physics derivation. Keep the explicit distinction from the complete
physical-block runtime. The initial suggestion to round the timings was
withdrawn after the subsection author identified the explicit request for
the recorded values.

**9. Correct the mode-action sentence.** Replace:

> Each nonzero mode thus connects just one pair of Fock states.

by:

> For each incoming state, each nonzero mode connects to at most one
> outgoing Fock state.

The formulas already say this correctly. This correction was coordinated
with the owner of the direct subsection.

**10. Define the universal factor before explaining how it is restored.**
The direct subsection currently contains a full explanation of restoring
two copies before the CCY section has defined the factor. The latter
section already provides the complete derivation. Replace the former
paragraph by a short forward reference to `ccy-and-truncation` stating
that the direct sum is the full auxiliary denominator. Keep the detailed
two-factor explanation after the universal-factor construction.

**11. Avoid introducing ideals before the parity algebra.** Early in
section 1, “minus ideal” and “plus ideal” appear before either is defined.
A direct replacement is:

> This ground-state sign makes recovery of the \(\eta\eta'=-1\)
> sector possible; the sector identity and inverse are derived below.
> The uninserted auxiliary block annihilates that sector under convolution.

The subsequent warning to retain the full field is still needed.

## Presentation debate and agreed outcome

The independent Ising author agreed with moving the inverse directly after
the convolution, placing the selected direct construction before the
compact Ising interpretation, and all three Ising notation/title fixes.
They raised two improvements adopted above: define the universal factor
before explaining its restoration, and postpone the ideal terminology
until the algebra is introduced. They also pointed out the explicit
request to retain exact standalone benchmark measurements; the review
therefore preserves those values and suggests compressing their format
instead of rounding them.

The parent accepted the Ising fixes and the local physical-versus-enlarged
parity clarification, and will perform the parent-file reordering. The
Ising author owns component-only edits. No mathematical formula or
production source needs to change for these presentation improvements.

## Material to preserve

The physical identity explanation before the field insertion prevents
confusion between the auxiliary fermion and a physical superconformal
operator. Keep it before the double-Virasoro weights.

The raw four-edge formula with squared norms must remain the primary
definition. Its normalized branching-coefficient version is a useful
consequence, with the compatible-root condition stated next to it. This
order prevents square-root conventions from obscuring the actual algorithm.

The middle coefficient's two Ward orientations, fusion argument, and
explicit ground anchors form a complete short derivation. Do not replace
them by a description of a numerical linear solve. The reusable action
coefficients still come from the existing finite descendant-span method;
the text correctly distinguishes these inputs from the recursive middle
matrix elements and the genuine CCY conformal blocks.

The direct auxiliary action and four-norm cancellation are central to
reproducibility. The rationalized representation introduces only one
temporary coefficient notation and then explains how it cancels back to
the physical normalization; shortening it by suppressing the ground
rescaling would make the exact-arithmetic claim harder to verify.

The arbitrary-genus paragraph must preserve the distinction between local
auxiliary sewing and the additional graph-dependent insertion, cocycle,
and sector data needed for recovery. Retain the explicit scope of the
implemented theta graph and the measured benchmark.

Finally, retain one clear validation paragraph naming the two authorized
comparisons and their tested parameters. The complete level-ten runtime
must come from the active production run's saved result. The standalone
auxiliary benchmark cannot substitute for it.

## Applied changes and closure

The parent and both included TeX files were reread after the coordinated
patches. The presentation review is closed for the current selected
theta-channel derivation.

| Review item | Verified final state |
| --- | --- |
| 1: argument order | The full convolution is followed immediately by the sector-inverse subsection; direct construction and then Ising interpretation follow the recovery formula. |
| 2: auxiliary notation | The Ising formula now uses the same \(\mathbb F_{\mathsf F}\) as the parent and direct subsection. |
| 3: parity scope | Immediately after `outer-frame`, the text explicitly distinguishes the physical-ground parity \(\lvert\alpha\rvert\) from the enlarged-primary parity labels in `full-branching`. The descendant-sign convention is narrowed accordingly. |
| 4: Ising title/table | The title is “The Ising decomposition”; the table heading is \(\rho_{\mathsf F}^{2}\). |
| 5–6: contour/history prose | “Native oracle” is replaced by “the implementation's three-point label”; redundant unchanged-file statements are removed, while the substantive correction to the Human-Note contour convention remains explicit. |
| 7: selected normalization | The unused reduced-Ising alternative is removed from the CCY vacuum-factor paragraph. The selected two-factor restoration remains. |
| 8: benchmark formatting | The requested exact figures are retained. Further compression of cache/write bookkeeping remains optional and is not required to close the review. No claimed runtime was changed by a presentation edit. |
| 9: sparse-action quantifier | The direct subsection now says each mode connects a given incoming state to at most one outgoing state. |
| 10: definition order | The direct subsection contains a short forward reference; the full universal-factor explanation occurs after its construction in the CCY section. |
| 11: early terminology | The opening now describes recovery of the opposite sector in plain terms; the ideals are introduced only with the convolution algebra and inverse. |

The relocated inverse argument is self-contained: it now states the two
auxiliary ground forms before deriving its constant coefficient. The
earlier mathematical-review fixes, \(Q\ne0\) and the explicit transported
spin-frame convention, are also present. No new undefined physical symbol
was introduced by these patches, and no mathematical formula was altered
by the reordering.

This closure is a static content review. TeX compilation and the active
level-ten production computation are owned and reported separately by
the parent; no numerical work was performed for this closure.
