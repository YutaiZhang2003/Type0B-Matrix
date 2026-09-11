# Sparse outer Ward solve and reusable eta factorization

This change modifies only `outer_branching.py`. No block or component
comparison was performed for this review. Python syntax compilation passed;
the root agent will run the two authorized final physical-block checks.

## What was expensive

The old outer adapter converted its sparse Ward matrix into column maps and
called `multiprecision_span_fit`. In that generic span solver, every selected
row's multiprecision residual traversed every column, including structural
zeros. For a Ward matrix with only a few neighboring-label entries in each
row, this turned sparse matrix products into dense multiprecision products.
The same matrix was then rebuilt and refactored independently for the two
eta values. The old Cartesian support also included many unqueried triples.

This identifies avoidable work from the source. It is not a claim about
measured attribution of the previous runtime or a promised speedup factor.

## Required support and closure

The requested triples are obtained from exactly the primary quadruples used
in `pipeline.reduced_numerator`: retain each ordered middle pair with
`outgoing = incoming +/- 1/2` whenever

\[
4n_1^2+(2n_{2,1}^2-1/8)+(2n_{2,2}^2-1/8)
       +2(2n_3^2-1/8)\leq 2N.
\]

Add both outer triples `(n1, incoming, n3)` and `(n1, outgoing, n3)`.
`close_requested` repeatedly applies the label changes of every stored
physical-L1 action, for both Ramond parities. Thus every nonzero entry of a
retained Ward row has its column in the retained set. Same-label terms are
retained. The recursion is then partitioned by its original NS parity.

No descendant cutoff is reduced, no out-of-range coefficient is set to zero,
and no high-primary direct anchor is introduced. The existing low anchors
are retained whenever their triple lies in the closed support. Full column
rank is required after restriction; closure alone does not assert uniqueness.
At level 10, the exact support previously enumerated in
`LEVEL10_NUMERATOR_PREP.md` has 464 requested and 484 closed triples, with
260 even and 224 odd NS labels, versus 500 and 400 Cartesian unknowns.
The current closure is computed from actual action-term labels, rather than
hard-coding those counts or the analytic moves.

## Reusing the factorization

For fixed `(alpha2, alpha3, form_parity)`, the normalized Ward matrix contains
only physical action coefficients, ordinary Virasoro descendant forms, and
unit rows for the allowed low anchors. None depends on eta. Eta occurs only
in the low-anchor right-hand side. This independence is structural in the
new implementation: `_prepare_system` takes no eta argument; `solve` supplies
eta only to `direct.raw` when constructing its right-hand side.

Accordingly, the cache key includes both Ramond parities and the form parity.
Neither momenta nor precision are omitted from a cross-parameter cache:
the cache belongs to one grid instance with fixed rational-derived parameters
and fixed multiprecision setting. Each solve restores that grid's precision.
Changing the support after constructing a factorization raises an error.
Different parity systems never share their factorization. Each eta solution
independently undergoes the complete residual guard.

## Identical solver guarantees with sparse MP products

`_SparseWardFactorization` uses the same numerical strategy as the original
`multiprecision_span_fit`:

1. Normalize the original rows, then normalize columns in multiprecision.
2. Use pivoted QR of the binary64 transposed matrix to select independent rows.
3. Require full rank with the original `RANK_TOLERANCE` and factor the selected
   binary64 square matrix with LU.
4. Accumulate residuals and coefficient updates at the requested MP precision;
   use double LU only as a preconditioner for each correction.
5. Require the selected-row tolerance `10^(-max(20, dps-15))`, allowing at most
   the original 30 refinement iterations.
6. Evaluate the residual on **every original Ward and anchor row**, requiring
   `10^(-max(15, dps-20))`. No projector changes the result to pass this guard.

For a normalized row, the new product is the finite sum over its nonzero
stored entries. Omitting structural zeros does not alter the linear system.
The returned coefficients undo exactly the same column scaling. Column norms
and all residuals remain multiprecision numbers; converting a double initial
guess or correction does not replace the MP equations by binary64 equations.

The QR/LU and normalized sparse rows are reused for the second eta right-hand
side. The result records rank, condition estimate, both residuals, refinement
iterations, matrix nonzero count, factorization reuse, and timings for system
assembly, factorization and solve. `action_seconds` remains separate.

## Ordinary descendant forms

The original `ordinary_factor` constructs two `VirasoroThreePoint` evaluators
for every action term. The new grid-local equivalent reuses an evaluator for
each `(changed triple, Virasoro copy)`. Its weights and central charge are
identical; its existing descendant-value and Virasoro-action caches can now
serve multiple terms and multiple parity/eta systems. The raw action
coefficient is still multiplied separately for every term, so sharing a
descendant evaluator cannot erase an action-parity dependence.

## Exact transport of the Ramond action parity

The raw primary convention already derived in the notes gives

\[
\Theta v_n^0=-2^{(-1)^{2|n|-1/2}/2}v_n^1,
\qquad
[\Theta,L_k]=[\Theta,L_k^{(1)}]=[\Theta,L_k^{(2)}]=0.
\]

Apply Theta to a parity-zero descendant identity for physical `L_-1`, and
divide by the displayed coefficient of its incoming primary. For each term
whose primary label changes from `n` to `m`, the corresponding parity-one
coefficient equals its parity-zero coefficient times

\[
2^{\bigl((-1)^{2|m|-1/2}-(-1)^{2|n|-1/2}\bigr)/2}.
\]

The exponent is an integer in `{-1,0,1}`; no square-root choice is computed.
Same-primary terms have multiplier one. The ground moves
`+/-1/4 -> -/+3/4` have multiplier `1/2`, and the reverse moves have
multiplier `2`. All larger inward neighbor moves have multiplier one.
Both reflected charts use precisely the same formula because the raw Theta
coefficient depends on `|n|`; the notes' reflection preserves this primary
normalization. Equivalently, the reflection intertwiner used by
`RamondActions` acts on the physical factor and preserves the displayed
auxiliary Theta action.

Only parity zero now needs each Ramond descendant-span solve. Its full-rank
selection, iterative refinement and all-oscillator-row residual are retained.
Parity one is obtained by this exact intertwiner. On the oscillator basis,
Theta permutes the auxiliary ground parity with coefficients `+/-1`; after
division by the incoming raw coefficient it scales the residual and target
by the same absolute factor. Hence the relative Euclidean residual is
unchanged. The returned diagnostics explicitly distinguish a computed
parity-zero residual from its exact parity-one transport. They do not claim
that a second independent numerical fit was performed.

The low-anchor implementation, eta convention, branch normalization, middle
recursion, CCY recursion, auxiliary factor and restricted convolution are
unchanged by this file's optimization. This optimization is local to outer
action construction; the shared `middle_branching.RamondActions` source is
unchanged.
