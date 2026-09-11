# Static review of sparse-accumulation constant reuse

The actual diff in `Code/ramond_branching_recursion/compute_target.py` was
independently reviewed against
`profiling_baseline/compute_target_pre_constants.py`. It contains only a
shared multiprecision zero, a precision-aware cached tolerance, and use of
that zero as the default in `add_term`. No numerical conformal-block
calculation or comparison was performed for this review.

## Semantics

* `_MP_ZERO = mp.mpc(0)` is exact at every precision. Arithmetic reads this
  value without mutating it. The old MP default was also an mpmath complex
  zero; the binary64 default remains the built-in complex zero `0j`.
* `add_term` still retrieves the existing coefficient, adds the new
  coefficient in the same order, compares its absolute value with the same
  tolerance, and performs the same `pop` or assignment. No summation order,
  numeric type, arithmetic precision, or pruning rule is changed.
* The tolerance cache key is `(MP_DPS, mp.mp.prec, rounding)`, obtained from
  `(MP_DPS, *mp.mp._prec_rounding)`. The installed mpmath implementation uses
  that same context pair for arithmetic. Thus changing either the target
  decimal precision, the active binary precision, or rounding causes a
  cache miss. In particular, a `workprec`/`workdps` context is not confused
  with an older value merely because `MP_DPS` was unchanged.
* On a cache miss, the exact original expression
  `mp.power(10, -max(20, MP_DPS - 20))` is evaluated in the current context.
  On a hit, its immutable result is reused. The binary64 branch still reads
  the current `TOLERANCE` directly. The cache is bounded at 32 entries.
* Existing call sites use tolerance values for comparisons; no caller mutates
  a returned mpmath object. The private cached helper is called only by
  `arithmetic_tolerance`, which supplies the actual current context key.

The change therefore removes repeated constant construction while preserving
the operation and pruning sequence of the pre-change code. Rank checks,
descendant columns, physical-L targets and residual guards are untouched.

## Provenance and concurrent runs

The archived pre-change source hash is
`46ebb1a8bd10b356ee8697637f61ffb62b680258858e95aabbb69ad68b6568ca`.
It matches the `compute_target.py` hash stored in
`results/ccy_cached_actions_level5.json`. The reviewed new source hash is
`3368f697c713860f9c2a8619a29c8a5e1113da54240337dc1ebc65a846390c42`.

The already-running level-10 process imported the previous module before
this disk edit. Production does not reload `compute_target`; its in-memory
functions retain their original definitions. `pipeline.main` records its
source hashes before entering `reduced_numerator`, so that result belongs to
the previous implementation and must be labeled accordingly. The new
level-5 timing uses a fresh process and the new source hash. These runs do
not mix the two `add_term` implementations inside one computation.

Static review passed. Measured speedup and the two authorized final-output
checks must be taken from the fresh process's saved artifacts rather than
inferred from the profiler's cumulative times.
