# Pointwise double-Virasoro global resummation

`nsrr_resummed_backend.ResummedNSRR` evaluates the physical NSRR block at
specified plumbing coordinates. Both equal HJS signs (ordinary Virasoro
factors) and opposite HJS signs (punctured factors) use global-descendant
accuracy controls independent of the Virasoro pole-recursion order.

The reference is the current c=1 implementation in
`/Users/yutaizhang/Desktop/c=1 string/code/general/virasoro/ccy_genus2_block.py`,
particularly its resummed theta seed and partial-fraction recursion. The
implementation here uses the existing native pole and branching machinery,
with MPC arithmetic throughout at the requested decimal precision.

## Independent controls

| Control | Meaning |
|---|---|
| `branch_level` | Truncation of the double-Virasoro primary sum. |
| `branch_truncation` | `total` or `per-edge` primary domain. |
| `recursion_order` | Maximum physical total null level in each Virasoro copy. |
| `global_tolerance` | Local absolute/relative stopping criterion for every global seed. |
| `global_max_shell` | Safety ceiling; an unconverged seed raises an error. |
| `dps` | Native arithmetic precision; default 40 decimal digits. |

Global descendants consume no residue budget, including at the deepest
shifted-weight leaf. The global sum is retained after multiplication and
physical recovery. A resummed `branch_level=8, recursion_order=8` evaluation
is therefore a different approximation from the old finite q polynomial
through independent edge orders (8,8,8).

For punctured weights `(h0,h_left,h_right,h3)`, allowed null shifts obey
`s0+s3+max(s_left,s_right)<=recursion_order`: this is the downward closure
of the physical middle diagonal. Adding both middle levels would omit the
`(s_left,s_right)=(2,2)` residue at physical order two. The full PBW boundary
test checks this distinction.

## Seeds and the punctured projection

For tensor slots `(infinity, one, zero)`, the ordinary seed is

    sum_(i,k) q1^i q3^k rho(i,0,k)^2 / [i!(2h1)_i k!(2h3)_k]
        * 2F1(a,a;2h2;q2),   a=h2+h3-h1+k-i.

The middle chain is summed using 2F1. Endpoint shells are extended until
three consecutive sums of absolute term magnitudes satisfy the tolerance.
The normalization recurrence consistently uses the same steps
`sqrt(n*(2h+n-1))`. This is necessary for negative and complex weights and
does not change any square root of a plumbing parameter.

For the punctured seed, keep its two middle variables independent:

    x = sqrt(q_middle) z,     y = sqrt(q_middle) / z.

Each global seed is summed into a Laurent series in `z`, using adaptive
four-edge SL(2) descendant shells and their absolute term sums. The current
punctured implementation uses converged numerical summation, rather than a
closed hypergeometric expression. It retains the difference of the two
middle levels until after multiplying both Virasoro copies. The Laurent
coefficient that cancels the branching-level difference gives the physical
diagonal. No Fourier grid or aliasing approximation is used.

The punctured middle tensor, all shifted weights and all pole amplitudes
are included. Transposed incoming/outgoing branches reuse the same scalar
diagonal product. Global coefficient tables grow only to the shell actually
needed; the safety ceiling does not trigger a full precomputation.

## Physical recovery and conventions

The reduced enlarged numerator is recovered in the same supported parity
ideal as the coefficient implementation. The free auxiliary fermion is
evaluated with its own convergence check. Recovery uses the twisted Walsh
characters; physical spin projection uses ordinary lift sums afterward.
An excessive residual outside the recoverable ideal is rejected.

The two omitted universal Virasoro vacuum factors are restored by a direct
primitive Schottky-cycle product, with independent word-shell and oscillator
checks. These internal factors are not the external free-superfield
denominator. The saved free denominator, period matrix, Liouville measure,
and structure constants are supplied unchanged by the caller.

`nsrr_resummed_sewing.resummed_integrand` connects the new evaluator to the
existing two-lift projection and nonchiral contraction. All q values,
momenta and literal lifts are transported together through
`NSRRPlumbingInputs`. Primary q powers are applied once.

## Entry points

Build with `make -C C++ bin/ramond_resummed`. For a direct chiral call:

```python
from nsrr_resummed_backend import ResummedNSRR

block = ResummedNSRR(
    b, (p_ns, p_r_one, p_r_zero),
    branch_level=8, branch_truncation="per-edge", recursion_order=8,
    global_tolerance=1e-13, global_max_shell=64, dps=40,
)
parity_values = block.physical_values(q_slots, 0, +1, -1)
value = block.project(parity_values, literal_lifts_slots)
```

`NSRRDoubleVirasoroTheta.block()` and `chiral_block_in_geometry()` now select
resummation by default. Their legacy `cutoff` supplies a total primary cutoff;
`recursion_order` can be specified separately. An explicit
`global_method="polynomial"` retains the finite-polynomial diagnostic.
Coefficient-only APIs such as `physical_components()` remain polynomial
APIs: they have no q argument and cannot represent an infinite global sum.
Historical coefficient-bank scripts and frozen cluster bundles retain their
original approximation and must not be described as resummed results.

Native outputs and pointwise checkpoints record separate cutoffs and
tolerances, physical momenta, plumbing coordinates, HJS signs, precision,
convergence diagnostics and executable hashes. Failed convergence never
selects a PBW or polynomial fallback.

## Validation and limits

`make -C C++ check-resummed` compares ordinary and punctured resummed
recursions with an independent direct unnormalized global-coefficient
implementation at 65 digits, including residue order eight. It also checks
a complex-weight high-precision reference, shifted diagonal extraction and
rejection of an insufficient global ceiling.

`Data Set/nsrr_global_resummed_20260914/validate.py` checks full physical
parity vectors against stored independent 384-bit PBW data at small q,
tests all eight HJS/form channels, and varies only global tolerance in the
saved generic_04 sewn integrand. Its results and protected-note hashes are
retained alongside the script.

The stopping criteria are numerical convergence tests, not interval error
bounds. Pole and branching coefficients can amplify local seed errors, so
tolerance refinement must also be checked on the final sewn observable.
Agreement at selected momenta does not certify the full Liouville integral
or convergence of the separate primary and residue truncations.
