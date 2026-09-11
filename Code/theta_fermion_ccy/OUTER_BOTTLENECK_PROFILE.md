# What dominates the outer branching runtime

The dominant cost is constructing the free-field descendant vectors needed
to determine the reusable physical-L1 action coefficients. Solving the
subsequent outer three-point Ward system is already inexpensive. The profile
identifies a particularly avoidable cost inside descendant construction:
rebuilding multiprecision zero and the same arithmetic tolerance for every
sparse coefficient update.

## Full level-5 production evidence

Source: `results/ccy_cached_actions_level5.json`, with the final selected
direct auxiliary backend, 70-digit outer Ward arithmetic, and 100-digit CCY
arithmetic. The following are saved stage times from that completed run:

| Work | Seconds |
| --- | ---: |
| Entire outer stage | 56.984985208 |
| Action setup and construction | 55.415058834 |
| Action fit: matrix assembly and normalization | 2.155897205 |
| Action fit: QR/SVD/LU | 0.061686168 |
| Action fit: MP refinement and original-column all-row residual | 2.236449621 |
| All action-fit work combined | 4.454032994 |
| Action-stage remainder, including descendant construction and initialization | 50.961025840 |
| Everything after action setup within the outer stage | 1.569926374 |

The three action-fit entries are summed over the computed NS and Ramond
parity-zero actions. Transported parity-one timing entries are zero and do
not contribute twice. The remainder is obtained by subtraction; the saved
level-5 timing alone does not resolve each suboperation inside that remainder.
The last row includes Ward-table assembly, low anchors, factorization,
refinement and other outer bookkeeping. It is not all time in a linear solve.

The same saved run records four computed NS actions and sixteen computed
Ramond actions. Its embedded basis-image caches receive 34,656 requests and
28,688 hits, including 34,488 requests and 28,616 hits for Ramond actions.
Those hits save generator-image reconstruction, but applying the images still
requires sparse multiprecision accumulation for every descendant column.

## Profile of one action, rather than another block check

Sources: `results/profile_outer_action_n7over4.json` and the underlying
`results/profile_outer_action_n7over4.prof`. Both were read independently for
this report. The profiled call is one `RamondActions.minus(7/4, 0)` at
`b=7/5`, `P=13/29`, 70 decimal digits. This is a timing profile, with no
component comparison or additional conformal-block cross-check.

Its measured duration under the profiler is **8.568980917 s**. A separate
level-10 process was active concurrently. Therefore these figures identify
where this sampled action spends time; they are not a full-outer benchmark
or a prediction of production speedup.

| Function | Calls | Cumulative seconds |
| --- | ---: | ---: |
| `descendant` | 67 | 8.047557456 |
| `action_span_fit` | 1 | 0.499574708 |
| `add_term` | 234,988 | 6.713887930 |
| `complex_number` | 236,604 | 3.302687923 |
| `arithmetic_tolerance` | 234,988 | 1.898532049 |
| `r_branch` | 2 | 0.002109833 |

These are cumulative call-graph times, so the rows are nested and must not
be summed as independent costs. Specifically, `add_term` contains almost all
the recorded `complex_number` work and all `arithmetic_tolerance` work.
The `.prof` caller data attributes 234,988 `complex_number` calls and
3.277212487 cumulative seconds directly to `add_term`; together with the
tolerance construction, those two child calls consume about 60% of this
sampled action's time.

## Why those calls occur

The Ramond action constructor builds two same-primary columns plus every
two-Virasoro descendant of the neighboring primary at the required degree
(`action_optimization.py:245-255`). At `n=7/4`, the neighboring primary is
`v_(3/4)` and the descendant degree is six, giving 65 neighboring columns
plus the two same-primary columns. This accounts for the 67 calls above.
Only after those vectors exist does `action_span_fit` determine their
coefficients.

Each embedded Virasoro action combines physical L, auxiliary L and the mixed
U operator (`compute_target.py:512-527`). The mixed term itself traverses
fermion modes and physical-supercurrent images (`compute_target.py:475-510`).
The current descendant builder uses the exact ordered suffix recursion and
cached unit-state images (`action_optimization.py:68-114`), but assembling
each output vector still enters `apply_expression` and `add_term`
(`compute_target.py:167-187`) repeatedly.

The concrete avoidable work is:

```python
value = expression.get(state, complex_number()) + coefficient
if abs(value) <= arithmetic_tolerance():
    ...
```

Python evaluates the default argument `complex_number()` before calling
`dict.get`, even when `state` is present. The helper converts zero through
Python complex, decimal strings, and a new mpmath complex object
(`compute_target.py:59-67`). The next line calls `mp.power` to reconstruct
the same tolerance (`compute_target.py:45-48`). Neither constant changes
during this fixed-precision action. The profiler counts and caller data show
that both operations occur about 235,000 times in this one action.

This is distinct from the final outer Ward solve, which occurs later in
`OuterBranching.prepare` (`outer_branching.py:460-475`). It is also distinct
from the CCY large-c seed: no CCY calculation is inside this profiled action.

## Reflection is not responsible here

For the sampled `n=7/4` action, both `v_(7/4)` and `v_(3/4)` use the native
positive chart. `r_branch` returns its raw primary directly
(`compute_target.py:740-744`), and the profile contains no `_level_transition`
call. Its two primary constructions take only 0.002109833 s in total.
The outer negative-label actions already use positive labels at reflected
momentum. The remaining chart-crossing ground action only needs a level-one
conversion, not a high-level reflection matrix.

## Most targeted next improvement

Hoist multiprecision zero and the pruning tolerance out of the innermost
sparse-accumulation loop, with the tolerance bound to the same precision
context. Reuse the zero object, or avoid constructing any default object
when the state is already present. Retain the original update order,
arithmetic precision, pruning threshold and all-row residual requirements.
This targets directly measured redundant work without changing a Ward
identity, a descendant span or a conformal-block recursion.

The profile establishes where to optimize; it does not establish the speedup
until an implementation is timed. No production source was edited for this
report while the existing level-10 process was running.
