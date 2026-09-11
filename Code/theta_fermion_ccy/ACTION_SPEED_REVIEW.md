# Final bounded action-construction optimization

This review describes `action_optimization.py` and its integration into the
outer and middle action providers. It introduces no block identity, branching
anchor, coefficient truncation, or saved-data dependency. The numerical
definition is the original physical-L1 descendant-span construction in
`compute_target.py`. That shared legacy source is unchanged.

No numerical computation or comparison was performed for this review. All
three edited Python files passed syntax compilation. The only numerical
validation planned after this change is the user's two prescribed final
physical-block checks through level 5, followed by the full level-10 timing.

## Exact reuse of descendant suffixes

For first-copy partition `(a, first_tail)` and second-copy partition `second`,
the descendant is computed by applying first-copy `L_-a` to the descendant
with partitions `(first_tail, second)`. If the first partition is empty, peel
the first element from the second partition instead. The recursion therefore
applies second-copy modes in reversed order, followed by first-copy modes in
reversed order. This is precisely the sequence in the old `descendant`
method, including the order within each noncommuting Virasoro algebra.

Many columns of one span reach the same shorter descendant in this recursion.
The new module caches those suffix expressions. It does not canonicalize
partitions differently, commute modes, or change any embedded-generator
formula. A cache hit returns the same sparse expression that the original
ordered applications would produce.

The cache is confined to a single action solve and is cleared even on an
exception. The key contains the primary object's identity and both ordered
partitions. Strong references to the primary objects prevent identity reuse
while an entry exists. Top-level requested columns are kept by the existing
span assembly, not inserted into the suffix cache; only intermediate
expressions shared by recursive construction are retained there. All returned
expressions are read-only in this computation. The module is bound to its
creation precision; reusing it at a different precision or changing precision
inside the cache raises an error.

This guard concerns a module entering a new action construction. The existing
`RamondActions.plus/minus` result caches can return an already-computed action
without entering that construction. Such values must not be treated as newly
computed at a higher precision. Production retains distinct providers for
the outer and middle precisions, so no action cache is promoted across them.

NS reflection constructs a new module at reflected momentum as before.
Ramond reflection remains in `RamondActions`: negative labels are evaluated
in the positive chart at reflected momentum and the resulting labels are
reversed. Both charts use the cached descendant builder. Their branch-state
construction and raw normalization are unchanged.

## Reusing embedded-generator images of oscillator basis vectors

Different descendant words can contain the same oscillator basis state even
when they do not share a whole suffix expression. In the same action-local
context, the module therefore also caches each embedded-generator image of
one basis vector, keyed by `(Virasoro copy, mode, oscillator basis state)`.
The image is computed by the unchanged superclass formula acting on a unit
basis vector. `br.apply_expression` then extends these images linearly to
every requested sparse expression. This avoids rebuilding the same physical-L,
auxiliary-L and mixed-U contributions for each occurrence of that basis state.

The embedded generators are linear operators, and their inherited oscillator
actions are pure at the fixed module parameters and precision. No generator
formula, coefficient or basis state is omitted. Applying the combined
generator to each basis vector first changes the order of finite-precision
summation and intermediate arithmetic-tolerance pruning relative to combining
its three pieces on the full expression; bitwise equality is not asserted.
The original arithmetic tolerance is used
throughout, and the complete span is still certified by the original-column
all-row residual guard. The prescribed final physical-block checks will use
this implementation.

These image dictionaries are read-only. They are cleared together with the
suffix cache after the action, including on failure. The existing creation
precision and context precision guards also cover this cache. Its request,
hit, retained-image and temporary coefficient counts are recorded inside the
same action-cache diagnostic object. They describe one temporary context,
not an indefinitely growing cache or a persistent saved-coefficient table.

## Precomputed normalized sparse rows

The original generic action fitter computes each selected-row residual by
traversing every column and repeatedly multiplying and dividing by the same
column norm. The new fitter computes column norms once, constructs normalized
sparse rows once, and reuses those entries during refinement. Structural
zeros require no multiprecision arithmetic.

The following original choices are retained:

* All original descendant columns and oscillator rows.
* Multiprecision column norms and the same requested precision.
* Pivoted QR of the binary64 transposed matrix for row selection.
* The original rank threshold and a mandatory full-rank square restriction.
* Binary64 LU only as a preconditioner for MP residual corrections.
* The selected-row stopping tolerance `10^(-max(20, dps-15))` and maximum
  30 refinement iterations.
* Unscaling the fitted coefficients with their original column norms.

Finally, the residual is recomputed with the **original unnormalized
columns**, on **every original oscillator row**. This retains the legacy
full-row residual measurement and additionally rejects a relative residual
above `10^(-max(15, dps-20))`, matching the explicit outer Ward guard.
The legacy generic action fitter reported this full-row residual without
rejecting it; the new implementation makes that requirement explicit.
No oscillator equation is dropped to make the guard pass.

The physical-L-minus-one inverse-identity diagnostic remains the same
algebraic check already inside the old action constructor. Its target state
is reused instead of applying physical L-minus-one a second time. This is
part of the existing solver diagnostics, not a new comparison of blocks.

## Integration and diagnostics

`_MPBranchingGrid` uses the cached NS module and the new NS action wrapper.
`RamondActions` uses the cached Ramond module and the new plus/minus wrappers.
The outer exact Theta transport from parity zero to parity one is unchanged.
The middle L1 recurrence itself is unchanged.

Each fit records suffix requests and cache hits, the temporary suffix count,
the normalized matrix nonzero count, matrix-assembly time, factorization time,
refinement time, rank and residuals. Temporary suffix counts describe one
action context and must not be interpreted as a persistent retained cache.
Outer diagnostics label parity-one action data as exact transports; its
factorization/refinement timing entries are zero, since no second fit ran.

The new module is automatically included by `pipeline.source_hashes`, which
hashes every Python source in this directory. Thus final validation and
timing artifacts record this implementation as well as the two integrations.
