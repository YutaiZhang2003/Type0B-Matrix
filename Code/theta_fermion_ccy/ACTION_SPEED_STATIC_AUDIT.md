# Independent review of physical-action reuse

This review inspected `action_optimization.py`, its integrations in
`middle_branching.py` and `outer_branching.py`, and the inherited free-field
generators and original span solver in `compute_target.py`. It performed no
numerical calculation or coefficient comparison. It is independent of the
implementation author's review in `ACTION_SPEED_REVIEW.md`.

## Descendant suffixes

The original descendant routine first applies the second Virasoro partition
in reverse order, then the first partition in reverse order. The recursive
suffix routine removes the first mode of the first partition, recursively
constructs the remaining descendant, and applies that mode last. When the
first partition is empty it does the same for the second partition. This
reproduces the original operator order exactly; it does not exchange modes
or assume that modes within one Virasoro copy commute.

Cache keys include both full partition tuples and the identity of the input
primary expression. Strong references retain those primaries throughout the
action, preventing object-identity reuse. Each action context creates fresh
caches and clears them in `finally`, including on failure. Full requested
columns are retained by the caller rather than duplicated in the suffix
cache. The inherited generator routines construct fresh dictionaries, so
returning a cached suffix cannot mutate an earlier column.

## Embedded generators on basis states

Each embedded Virasoro generator is the displayed linear combination of the
physical stress tensor, auxiliary stress tensor, and mixed generator. Its
action on an expression is therefore the linear combination of its actions
on individual Fock basis states. The new cache implements precisely this
identity. On a miss it calls the superclass formula on one unit basis state;
on a hit it reuses that image. The inherited `apply_expression` performs the
linear extension and constructs a fresh output dictionary.

The key contains the Virasoro-copy number, signed mode, and full oscillator
state, including ground-state labels. Sector, momentum, embedding parameter,
and realization belong to the enclosing module. The superclass calculation
calls the unchanged physical, auxiliary, and mixed generators, so it cannot
reenter this override recursively. Empty images are cached correctly. These
images share the action-local lifetime of the suffix cache and are cleared
on every exit.

The mathematical identity is exact. At finite precision it changes the order
of arithmetic and of the inherited intermediate coefficient pruning; it is
not a claim of bitwise agreement with the older loop ordering. The inherited
arithmetic tolerance is unchanged, no new coefficient cutoff is introduced,
and the independent physical stress-tensor target remains unchanged.

## Full span and residuals

The NS and Ramond wrappers retain every original partition pair. The Ramond
lowering action also retains both same-primary columns. Reflection still
uses the positive-label module at reflected momentum and reverses each
resulting branch label. No allowed action term, descendant degree, primary
label, or oscillator row is removed.

The solver uses the same Euclidean column norms as the previous numerical
span fit. Each column is divided by its norm once, and its nonzero entries
are placed in sparse rows. Pivoted QR chooses the same kind of square row
restriction; its double-precision LU supplies only the preconditioner.
Multiprecision refinement acts on the normalized matrix and retains the
previous selected-row tolerance. Dividing the resulting coefficients by the
column norms returns coefficients in the original, unnormalized basis.

The final residual is recomputed using every original column on the union
of all column and target oscillator rows. It is explicitly required to obey
the stated multiprecision bound. This is a stronger solver guard than the
legacy action solver, which computed and reported the full residual without
rejecting on it. It is not an additional block comparison. Numerical
acceptance of the optimized physical block remains the two requested checks:
independent physical PBW through level five and varying the split position
at fixed product.

## Precision and scope

An action module records its construction precision. Opening an action cache
at another precision, or changing precision while the cache is active, raises
an error. The caches cannot carry suffixes or basis-state images from one
action precision to another. The production pipeline creates separate
providers for its outer and middle precision settings.

The preexisting `RamondActions.plus/minus` result caches are a separate
interface: an already-cached result returns before an action-module guard is
entered. Consequently the new guard is not a claim that arbitrary reuse of
one provider at a higher precision upgrades or rejects all cached action
coefficients. Such providers must still be reused only at matching precision.
This limitation does not affect the current production pipeline.

## Conclusion

The implementation preserves the required action identities, span, grading,
and production precision scope. Static mathematical and source review is
closed. Runtime and the two authorized numerical comparisons must be read
from their own saved results, not inferred from this audit.
