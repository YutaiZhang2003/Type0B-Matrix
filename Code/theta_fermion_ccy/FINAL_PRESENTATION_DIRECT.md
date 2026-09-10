# Final presentation review: direct auxiliary construction

Static presentation review only. No numerical work or source-code edits.
The mathematical review of the new direct subsection found no error in
the action, norm cancellation, rational normalization, truncation or
restoration of the two universal vacuum factors.

Applied to the two component TeX files after the reviewers agreed:
the three Ising notation/title fixes, the per-incoming-state quantifier,
and replacement of the direct subsection's duplicate universal-factor
paragraph by a forward reference. Parent-file patches below remain
the parent agent's responsibility. Exact benchmark figures are retained.

## Agreed narrative order

The middle-coefficient reviewer and direct-fermion reviewer agree that
the full convolution identity should be followed immediately by the
sector inverse. The current two auxiliary inputs insert several pages
between the identity and its use. We also prefer the selected direct
construction before the compact Ising interpretation.

Small parent-file patch:

1. Move the two `\input` lines currently just before
   `\subsection{The inverse in the required sector}` to just after that
   subsection, before `\section{Central-charge recursion and truncation}`.
2. At their new location, put `theta_fermion_direct_definition.tex` first
   and `theta_fermion_ising_component.tex` second.
3. In the inverse subsection replace

   > The auxiliary formula gives the constant coefficient

   with

   > At zero level, the outer ground forms
   > $\rho_{\mathsf F}(\mathbf1,u^0,u^0)=1$ and
   > $\rho_{\mathsf F}(\mathbf1,u^1,u^1)=i$, together with
   > \eqref{zero-mode-action}, give the auxiliary constant

   This makes the inverse argument self-contained without relying on
   the newly postponed auxiliary formulas.

## Introduce the universal factor before discussing its restoration

The direct subsection's paragraph beginning
`\paragraph{Combining it with the reduced central-charge recursion.}`
currently discusses a universal factor that is not defined until the
following CCY section. Both reviewers agree that its detailed explanation
belongs at the end of the universal-factor derivation in that section.

Suggested patch: replace that paragraph in the direct subsection by

> Equation \eqref{direct-fermion-sum} gives the full auxiliary factor.
> Its normalization relative to the reduced central-charge recursion is
> explained in \S\ref{ccy-and-truncation}.

Use the moved paragraph to replace, rather than duplicate, the parent's
existing explanation of restoring two universal factors. No symbol
such as $U$ is needed: the existing wording is precise once the factor
has been defined.

A larger alternative would move the auxiliary construction after the
whole CCY section. We do not recommend this larger rearrangement: the
small paragraph move fixes the definition order while keeping the
denominator construction next to the convolution.

## Three Ising-notation corrections

In `Machine Notes/theta_fermion_ising_component.tex`:

| Current | Replace with | Reason |
| --- | --- | --- |
| `\subsection{The Ising factor and its null states}` | `\subsection{The Ising decomposition}` | The short subsection contains the decomposition, not the removed null-network analysis. |
| `\mathbf F_{\mathsf F}` | `\mathbb F_{\mathsf F}` | Matches the parent and direct-definition notation for the same auxiliary block. |
| `\rho_{\mathsf F}^{,2}` | `\rho_{\mathsf F}^{2}` | The comma has no meaning; the column is the product of the two equal outer ground forms. |

The notation for the state norm should remain distinct from the outer
three-point ground form: the Ramond norm is $(-1)^{\mathsf b}$, whereas
the outer form has values $1,i$. The existing derivation correctly
distinguishes them; preserve that explanation.

## Sparse action: correct the quantifier

In `Machine Notes/theta_fermion_direct_definition.tex`, replace

> Each nonzero mode thus connects just one pair of Fock states.

with

> For each incoming state, each nonzero mode connects to at most one
> outgoing Fock state.

This is the only wording issue identified by the independent mathematical
review. The old wording could be read as claiming one global pair per
mode.

## Avoid an undefined ideal in the opening geometry section

In the parent, the paragraph immediately after the diagonal zero-mode
formula mentions the minus and plus ideals before the parity algebra or
convolution has been defined. Replace

> This ground-state sign places the inserted auxiliary block in the
> minus ideal. The original auxiliary block is nonzero but lies in the
> plus ideal and annihilates the desired physical sector under convolution.

by

> This ground-state sign enables the sector division for
> $\eta\eta'=-1$ derived below.

The later sector subsection supplies the exact algebraic meaning.
If the reason the original block fails is needed in this opening, use
one plain sentence about its vanishing convolution, followed by the
forward reference, rather than introducing the two ideals here.

## Make the nonzero normalization assumption explicit once

At the first displayed definition $Q=b+b^{-1}$, add

> We assume $Q\ne0$, as required by the sector division below.

Keep the explanation at the inverse formula, but it can then say
`By the assumed $Q\ne0$, this scalar is invertible.` This is a scope
clarification, not an additional physical assumption invented for the
direct backend: the current inverse already requires it.

Also replace `It has no inverse in the entire parity algebra` there by
`This constant has no inverse in the entire parity algebra`, and specify
`multiplication by this constant` in the next sentence. The full
auxiliary series is not itself a scalar; its constant acts as a scalar
on the required sector.

## Benchmark detail: preserve the requested figures, separate bookkeeping

The parent requested the exact standalone timings, memory and output
counts in the new subsection. Retain those figures. The middle reviewer
prefers rounded prose, while the direct reviewer prefers preserving the
requested exact values in a compact table or short paragraph. This is
a presentation choice, not a mathematical disagreement.

Both reviewers agree that the additional cache/contraction counts and
the final metadata-rewrite exclusion can live in
`DIRECT_FERMION_BENCHMARK.md` and the saved JSON. A concise paragraph can
retain: construction 0.186281209 s, process through initial save
0.290214667 s, peak 46.765625 MiB, 4793 exponent tuples and 9586 nonzero
rational coefficients. Keep the explicit distinction between this
auxiliary-only measurement and the full physical-block runtime.

## Preserve the steps that carry the argument

Do not shorten the action-coefficient versus BPZ-matrix-element
distinction, the four-norm cancellation, or the rational rescaling
derivation. These explain otherwise easy-to-miss signs and factors.
The current $\rho_{\mathsf F}^{\mathrm{rat}}$ is the only added
stored-value notation and serves a specific purpose; replacing it by
another abstract kernel would make the argument less direct.

Keep the last paragraph's distinction between genus-independent local
sewing operations and the current four-edge assembler. Neither the
benchmark nor the notation should suggest an implemented arbitrary-graph
frontend or a runtime independent of genus.
