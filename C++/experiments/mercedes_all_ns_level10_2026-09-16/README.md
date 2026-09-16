# All-NS Mercedes, total level 10

This directory extends the independent double-Virasoro / NS c-recursion
comparison from `../mercedes_all_ns_c_recursion_2026-09-16/`. The level-six
sources and results there remain unchanged.

**Result: all 230,230 multidegrees pass at tolerance `1e-18`.** The maximum
scaled error is `2.05323406825940972e-25` (scale `max(1, abs(x), abs(y))`),
and the maximum absolute error is `4.00211708925139e-25`. There are zero
failed coefficients. The all-even vertex-form sector contains 29,029
multidegrees; the other seven sectors each contain 28,743.

| Fresh stage, 40 digits | Wall time (s) |
|---|---:|
| Double-Virasoro enlarged blocks | 681.135 |
| Direct auxiliary fermion factor | 0.183 |
| Collect enlarged coefficients | 0.071 |
| Recovery and coefficient output | 1.892 |
| Full double-Virasoro pipeline | 683.282 |
| NS Schottky seed | 0.178 |
| Independent NS recursion and coefficient output | 270.148 |
| Full NS c-recursion stage | 270.326 |

The two sequential computational stages total 953.608 seconds (15 min
54 s), excluding compilation and the subsequent saved-data audit. Both
start with empty numerical caches. The double-Virasoro calculation uses
45,405 branch tuples and 10,921,308 CCY residue transitions.

The parameters, ordered graph, and conventions are identical: `b=7/5`,
momenta `(11/23,13/29,17/31,19/37,23/41,29/43)`, all six edges NS, and even
intrinsic primary parities. Both methods use 40-digit MPC arithmetic.
The domain contains every nonnegative half-integer multidegree with total
level at most 10: **230,230 multidegrees**, distributed among all eight
allowed vertex-form assignments. The NS edge parity is fixed by its
half-integer level, so comparing these parity coefficients covers all 64
spin-lift choices.

## Complete Schottky seed

The level-six shortcut has been replaced by the full truncated primitive
product. There are 13 primitive unoriented cycles of lengths at most six and
20 allowed fermionic oscillator factors. Longer primitive cycles cannot
contribute through total level 10. For each word the code computes its
projective matrix, trace, determinant, exact formal multiplier, and lifted
half-multiplier. It retains all allowed higher oscillators and products of
primitive contributions, including repeated traversals through the
multiplier's Catalan series. These terms are computed with exact rational
coefficients before conversion to 40 digits.

Closed-cycle spin monomials commute under the graph convolution, but some
square to minus one. The half-multiplier therefore obeys `u star u = k`,
not an ordinary monomial square in the coefficient representation. The
implementation checks this identity exactly for every contributing word.
It transports the SL(2) lifts through the commuting cycle algebra; it does
not choose independent square roots for different primitive words.

The local ground supercurrent pairings fix the three independent lift
signs. In the directed-matrix convention used here, a closed fermion loop
contributes `-Tr(T^n)/(2n)` to the logarithm of the sewn Pfaffian. Accordingly
its oscillator factor is `1-u*k^(m-1)` with this choice of `u`. Converting
to the convention in which the factor is written `1+u*k^(m-1)` also changes
the definition of that loop lift. The explicit loop convention in the code
is fixed throughout; coefficients are never fitted to the double-Virasoro
answer. The extended seed reproduces all 18,564 saved level-six NS
coefficients exactly in the saved decimal output.

The projective matrix determinant also must follow the actual endpoint
coordinates. A directed edge map has determinant
`(-1)^(1 + [source slot is infinity] + [target slot is infinity]) * q_e`.
The old generic graph helper hard-coded `-q_e`. Those expressions agree on
theta's like-slot edges, but differ on some Mercedes edges. The first
resulting difference in the bosonic vacuum appears at total level 9,
beyond the previous test. The isolated `graph_ccy.hpp` here uses the actual
map determinant; the production theta helper and prior test files are
unchanged.

The NS regular seed combines the Schottky vacuum with the global
`osp(1|2)` block using the full graph-convolution sign, including the local
middle-global / ket-vacuum crossing verified by the level-six comparison.

## Computation and reuse

`dv_engine.hpp` is an isolated extension of the previous graph test engine.
Its dynamics are unchanged apart from the corrected Schottky determinant.
For an all-NS unsplit graph it generates the permitted residual Virasoro
simplex directly after subtracting each branch's starting levels. It
convolves the two Virasoro copies by iterating over contributing pairs,
instead of scanning pairs outside the cutoff. There are 45,405 eligible
branch tuples through physical level 10. Local NS branching data, CCY pole
geometry, fusion polynomials, and vertex values are reused.

The auxiliary factor is sewn directly as before. Physical recovery is a
triangular solve: each known coefficient updates only higher coefficients
that receive a nonconstant auxiliary contribution. The 64-by-64 parity sign
table is precomputed. No physical SCA PBW block is run.

The independent NS recursion caches poles, fusion polynomials, global
vertices, and norms. Its full recursive-state cache is cleared between
target multidegrees: every state satisfies
`accumulated shift + remaining level = target multidegree`, so no such state
can be shared between different targets. All reusable local caches remain
live. This bounds memory without discarding cross-target reusable data.

## Reproduce

Run the two stages sequentially so their timings do not include CPU
contention from the other method:

```sh
make
./compare ns 10 > ns_L10.log 2>&1
./compare dv 10 > dv_L10.log 2>&1
python3 audit_saved.py --level 10
```

Each process begins with empty numerical caches. Compilation is excluded.
The NS stage reports its Schottky construction separately; both stages
include coefficient output in the indicated output timings. The audit reads
saved coefficients only and checks complete domain coverage, parity and
sector assignments, and errors at every total half-level. It does not rerun
a block. The files prefixed `diagnostic_primitive_loop_sign_` preserve the
interrupted implementation diagnostic before fixing the universal closed
fermion-loop sign; they are not the final run.

Final coefficients are in `coefficients_L10.jsonl` and
`c_recursion_L10.jsonl`. Timings are in `timing_dv_L10.json` and
`timing_ns_L10.json`. The audit writes `results_L10.json`,
`discrepancies_L10.json`, and source/result hashes in `manifest_L10.json`.
