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
2. **Outer and middle coefficients:** low physical PBW anchors fix the outer L_1
   Ward systems. Matrix factorizations are reused for the vertex signs.
   Single-leg Virasoro descendant factors use closed Ward products. The middle
   recurrence reuses the Ramond actions. These small anchor systems do not
   compute a physical PBW block.
3. **Double-Virasoro sum:** forward CCY merges paths reaching the same null-state
   shift and central-charge pole. Pole geometry (including the universal A
   factor), fusion polynomials, and global three-point factors are cached within
   each block. Every branch uses its remaining level budget. Inserted products
   compute the required diagonal targets; opposite middle orientations reuse
   transposed products.
4. **Recovery:** direct rational fermion Ward identities and sewing give the
   auxiliary factor. Triangular star division takes place in the recoverable
   parity sector. The exact Schottky vacuum product is squared and restored
   after division, once for each reduced Virasoro factor.

The CCY seed comprises the weight-dependent sl(2) global factors and the
universal Schottky vacuum. No finite-c Virasoro Gram blocks are evaluated in CCY.
All reuse is within a fresh process; production runs load no numerical cache
from disk.

## Precision and residuals

Higher precision applies to branching entries, iterative-refinement residuals,
CCY, assembly, and recovery. LAPACK selects independent rows and supplies a
double-precision LU preconditioner. Refinement uses the original multiprecision
matrix and checks every row. Increasing `--dps` therefore does not remove a
failure of double-precision rank selection at an ill-conditioned parameter point.

Inherited thresholds are retained: at 40 digits, the selected-system residual
target is `1e-25`, the full-system residual guard is `1e-20`, and sparse arithmetic
prunes at `1e-20`. These tighten with `--dps`; a precision setting does not
guarantee that many correct output digits.

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
