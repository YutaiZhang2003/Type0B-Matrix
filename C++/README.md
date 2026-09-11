# Native C++ Ramond block pipelines

The C++17 executable computes the **physical genus-two NS–R–R superconformal
block** through a specified total plumbing level. Both current production paths
are implemented:

| Mode | Vertices | Recovery |
| --- | --- | --- |
| `ordinary` | `(eta, eta)` | Ordinary theta double-Virasoro sum and restricted fermion inverse |
| `inserted` | `(eta, -eta)` | Theta v_(1/2) insertion, diagonal target sum, and fermion convolution |

Python is used only by the optional validation script to read existing reference
results. The Python implementations remain available under `Code/`.

## Build and run

Dependencies: a C++17 compiler, GMP (including `gmpxx`), MPFR, MPC, and LAPACK.
On this Mac, the numerical libraries are under `/opt/homebrew` and LAPACK is
supplied by Accelerate. From the repository root:

```sh
make -C 'C++' -j2

'C++/bin/ramond' --mode ordinary --level 5 --dps 0 \
  --json 'C++/results/ordinary_example.json'

'C++/bin/ramond' --mode inserted --level 5 --dps 40 \
  --json 'C++/results/inserted_example.json'
```

`--dps 0` selects `std::complex<double>`; `--dps D` selects MPC complex arithmetic
for `D >= 30`. Both instantiate the same mathematical code. At 40 decimal digits,
the working precision is 136 bits, matching the saved mpmath reference. Actual
working bits are recorded in the output. Fermion and Schottky computations use
exact GMP rationals in both modes.

The Makefile supplies a Linux LAPACK/BLAS link configuration as well; only the
macOS arm64 build has been tested. Compiler and include/library paths can be
overridden with Make variables: `CXX=... CPPFLAGS=... LDFLAGS=...`.

## Parameters and output

Defaults are `b=7/5`, `P1=11/23`, `P2=13/29`, `P3=17/31`, `p=f=0`, and `eta=1`.
Options `--b`, `--P1`, `--P2`, `--P3`, `--p`, `--f`, and `--eta` override them.
Real inputs accept decimals or rationals. Complex momenta use `real,imaginary`,
with decimal or rational parts, for example `--P2 '0,13/29'`.
The current free-field implementation requires real `b != 0, +/-1`.
Unsupported degenerate denominators and failed rank/residual checks raise errors;
the code does not infer a limiting prescription.

JSON `coefficients` contains the **recovered physical block**, with component
index `epsilon1 + 2*epsilon2 + 4*epsilon3`. An exponent record `[a,l,l,d]` means

\[
q_1^{a/2}q_2^lq_3^{d/2},\qquad a+2l+d\leq 2N.
\]

Here `d` is even. For the insertion, the middle monomial is `q_left^l q_right^l`
and `q2=q_left*q_right`. The current diagonal recovery algorithm retains the
unequal intermediate Virasoro levels needed to obtain these diagonal targets.
This executable is not a general plumbing-graph frontend.

The output also records the reduced numerator, full auxiliary factor, parameters,
stage times, branch/block counts, and action, Ward, and sector residuals. It is
written to a temporary file and renamed after serialization.

## Algorithm and reuse

1. **Branching actions:** free-field primaries and the needed L_(+/-1) actions
   in the two Virasoro embeddings. Interned oscillator states, physical images
   independent of auxiliary spectators, auxiliary images independent of physical
   spectators, and descendant suffixes are reused. Raising/lowering solves share
   construction data. Reflected charts and Theta transport supply other labels
   and parities.
2. **Outer and middle coefficients:** directly evaluate the complete allowed
   boundary box, NS n=0,+/-1/2 and R n=+/-1/4,+/-3/4, and store those values.
   Build higher coefficients from already stored lower-label values. The bulk
   update is a scalar division. Explicit boundary actions couple at most four
   unanchored coefficients at a time; these small relations are eliminated at
   the working precision. There is no global outer Ward matrix. If the first
   identity cannot determine the coefficient or boundary group, use the second
   conformal Ward identity with L1 on a Ramond leg. Needed NS L-1 and Ramond
   L0/L1 actions are computed lazily and cached; physical L0 acts with the
   physical oscillator level. Both vertex signs reuse the same Ward rows and
   local elimination. Embedded weights and closed single-leg descendant factors
   are cached. Every used Ward equation is checked after each update, and a
   failure identifies the affected labels and parities. The middle recurrence
   reuses the Ramond actions. Physical PBW is used only for the direct low
   boundary data, not a high-level physical block.
3. **Double-Virasoro sum:** forward CCY merges paths reaching the same null-state
   shift and central-charge pole. Pole geometry (including the universal A
   factor), fusion polynomials, and global three-point factors are cached within
   each block. Every branch uses its remaining level budget. Inserted products
   compute the required diagonal targets; opposite middle orientations reuse
   transposed products.
   In the inserted CCY factors, complete normalized vertex factors are cached
   by their three incident edge shifts and descendant levels. Spectator-edge
   choices reuse the same factor. The assembly sum reuses MPC scratch storage.
   Central-charge differences are inverted once per ordered pair of poles;
   each transition factors its common residue outside the incoming-amplitude
   sum. The pole-separation check is retained at the first inverse evaluation.
4. **Recovery:** direct rational fermion Ward identities and sewing give the
   auxiliary factor. Triangular star division takes place in the recoverable
   parity sector. The exact Schottky vacuum product is squared and restored
   after division, once for each reduced Virasoro factor.

The CCY seed comprises the weight-dependent sl(2) global factors and the
universal Schottky vacuum. No finite-c Virasoro Gram blocks are evaluated in CCY.
All reuse is within a fresh process; production runs load no numerical cache
from disk.
The directed performance and accuracy checks for complete vertex-factor and
denominator reuse are in
[the CCY reuse report](results/ccy_vertex_reuse_2026-09-11/README.md).

## Precision and residuals

Higher precision applies to branching entries, CCY, assembly, and recovery.
Outer branching uses the stored recurrence and small local elimination at the
working precision. Mode-action decomposition still uses LAPACK row selection
and a double-precision LU preconditioner, with residuals refined and checked
against the original multiprecision matrix. Increasing precision alone does
not remove a failure of that mode-action rank selection.

At 40 digits, mode-action refinement targets `1e-25`, and its full-row residual
and sparse-pruning thresholds are `1e-20`. The outer recursion checks each used
Ward identity using abs(sum of terms)/(1 + sum of absolute terms), with tolerance
`1e-20` at 40 digits or `1e-8` at machine precision. A local prefactor is tested
against the absolute contributions to that prefactor, independently of the
size of already-known lower-label terms. These thresholds tighten with the
requested precision; a precision setting does not certify that many digits.

Default `--sector-policy error` stops above a scaled sector residual of `1e-8`.
Ordinary recovery projects its accepted small residual into the sector;
inserted recovery retains its coefficients. Explicit `--sector-policy record`
retains unprojected coefficients in either mode and records the residual instead
of stopping. This option does not repair an inaccurate numerator.

Storage supports 64 positive fermion modes and boson modes 1 through 63.
Exceeding a representation bound raises an error rather than silently truncating.

## Validation and fresh level-5 timings

All 91 physical monomials (728 parity components) through level 5 were compared
against saved Python results at the default rational parameters:

| Pipeline | Machine time | 40-digit time | Machine max scaled difference | 40-digit max scaled difference |
| --- | ---: | ---: | ---: | ---: |
| Ordinary | 0.108 s | 0.838 s | 1.21e-11 | 1.98e-25 |
| Inserted | 0.386 s | 4.103 s | 1.59e-10 | 3.45e-25 |

Times are single fresh-process measurements on the current macOS arm64 machine.
They include all mathematical stages from empty numerical caches and exclude
compilation, process startup, and final JSON serialization.

| Stage, seconds | Ordinary machine | Ordinary 40 digits | Inserted machine | Inserted 40 digits |
| --- | ---: | ---: | ---: | ---: |
| Branching total | 0.0817 | 0.4934 | 0.3285 | 2.8794 |
| Actions (included above) | 0.0120 | 0.2059 | 0.2332 | 2.3766 |
| Outer Ward solves (included above) | 0.0697 | 0.2874 | 0.0953 | 0.5026 |
| Middle coefficients | 0 | 0 | 0.0004 | 0.0097 |
| CCY | 0.0152 | 0.2681 | 0.0441 | 1.1002 |
| Virasoro products | 0.0015 | 0.0322 | 0.0025 | 0.0255 |
| Branch assembly | 0.0011 | 0.0260 | 0.0019 | 0.0619 |
| Direct fermion | 0.0061 | 0.0043 | 0.0050 | 0.0043 |
| Convolution division | 0.0003 | 0.0089 | 0.0004 | 0.0084 |
| Schottky vacuum | 0.0005 | 0.0004 | 0.0005 | 0.0004 |
| Vacuum restoration | <0.0001 | 0.0005 | <0.0001 | 0.0005 |

Scaled difference means `abs(C++ - reference)/max(1,abs(C++),abs(reference))`.
The ordinary reference is the saved 384-bit restricted-inverse computation;
the inserted reference is the saved 40-digit forward CCY computation. Agreement
is numerical comparison, not a certified bound on common errors.
A level-2 inserted comparison with `p=f=1`, `eta=-1` agrees to `1.74e-13`.
Exact rational Schottky coefficients through level 6 and inserted fermion
coefficients through level 3 matched Python exactly.

The AddressSanitizer/UndefinedBehaviorSanitizer executable stalled at startup,
including a timed `--help` invocation. No sanitizer pass is claimed; the timeout
is recorded in `results/sanitizer_validation.json`. The optimized build and
numerical comparisons completed successfully.

See [pipeline_validation_level5.json](results/pipeline_validation_level5.json)
for reference hashes, errors, and all stage times. No full level-15 or new
physical PBW block was run for this migration.

To reproduce the bounded checks, install `mpmath` in the validation Python
environment and run:

```sh
make -C 'C++' check PYTHON=/path/to/python

# Repeat the level-5 comparison and timings from empty process caches:
/path/to/python 'C++/tests/verify_saved.py' --run --level 5
```

`make check` uses level 3. Omitting `--run` compares existing C++ outputs only.
The checks read saved references and never run PBW. Component drivers in `tests/`
provide branching-action, low-anchor, and exact-factor output.
See [MIGRATION.md](MIGRATION.md) for the source map.

## Current recursive branching timings

The fresh level-10 measurements, with separate action, outer-recursion, middle,
and process-wall times, are recorded in
[the recursion report](results/recursive_branching_2026-09-11/README.md).
The same report records the successful level-20 branching-only diagnostic.
It distinguishes the branching-only measurements from the full level-10 pipelines.

Reproduce the branching benchmark in separate fresh processes:

    make -C C++ all bin/outer_ward_driver
    python3 C++/tools/time_branching.py --level 10 --dps 0 40 --output /tmp/branching_level10

## Independent plumbing cutoffs

Use `--truncation per-edge --level 10` to retain every
`q1^(a/2) q2^l q3^m` with `0 <= a <= 20` and `0 <= l,m <= 10`.
This gives 2,541 monomials (20,328 parity components), including terms of
total level 30. The default `--truncation total` retains the original simplex.

The independent limits apply throughout branching support, CCY, products,
direct fermion sewing, convolution, and Schottky restoration. For each branch,
subtract its primary level separately on each edge. The ordinary Virasoro
domain is the resulting three-dimensional box. The inserted domain is a
four-dimensional box with separate remaining budgets on the two middle
segments; only their shifted diagonal is multiplied and assembled. Unequal
middle levels remain available inside each Virasoro factor.

Convolution is ordered by total degree but discards updates outside the box.
Schottky products are also truncated to the box at every multiplication.
Primitive walks need length at most half the sum of the three bounds; a walk
is discarded if twice its edge-visit counts exceed an individual bound.
The fermion factor enumerates occupation levels independently on the three
physical edges. No total-level-30 block is computed as an intermediate.

Fresh sequential timings at 40 digits, with empty numerical caches:

    python3 C++/tools/time_current_pipelines.py --levels 10 --truncation per-edge --dps 40 --output /tmp/ramond_per_edge10

Results and stage timings are saved in
[the independent-cutoff report](results/per_edge_level10_2026-09-11/README.md).
`make -C C++ bin/per_edge_cutoff && C++/bin/per_edge_cutoff` checks exact
Schottky and fermion projections and the shifted diagonal product at small
cutoffs. These checks do not compute physical PBW blocks.
