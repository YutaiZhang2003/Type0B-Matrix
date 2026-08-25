# High-level evaluator profile

This directory isolates profiling prototypes.  It does not modify the stored
Ward evaluator.

## What makes the current evaluator slow

For a fixed free-field endpoint the current code first constructs a symbolic
free-field/PBW transition matrix and then solves for the complete PBW
expansion.  The largest systems required by the requested labels are

| branch | endpoint grade | current matrix | useful Ramond block |
|---|---:|---:|---:|
| `v_2` | `level2=16` | 70 | 70 |
| `v_5/2` | `level2=25` | 420 | 420 |
| `W_7/4` | `level=6` | 80 | 20 |
| `W_9/4` | `level=10` | 464 | 116 |

The Ramond matrix contains two identical ground blocks, and each ground block
splits by fermion parity.  The current code constructs all four blocks and
calls a dense inverse separately for several auxiliary sectors.  On the
profiling machine:

* the symbolic NS level-16 transition took 1.9 seconds, but
  `ns_components(2,...)` did not finish its polynomial-domain solves in 90
  seconds;
* symbolic construction of the NS level-25 transition exceeded 90 seconds;
* symbolic construction of the full Ramond level-6 transition took 10.8
  seconds, while `ramond_components(7/4,...)` did not finish its repeated
  dense inversions in 90 seconds;
* symbolic construction of the full Ramond level-10 transition exceeded 60
  seconds.

Thus the first bottleneck is symbolic expression growth in the transition and
its inverse, not the number of chi paths.

## Exact modular audit algorithm

`modular_transition.py` implements the same oscillator realization after
mapping all rational input into a prime field.  It groups endpoints of equal
grade into a single multi-column right-hand side and constructs only the
needed Ramond ground/parity block.  Its transition matrices were checked
entry by entry against the original symbolic matrices at NS `level2=9` and
Ramond `level=3`.

Run

```bash
python3 "python 2/ramond_screening_algorithm/profile/modular_transition.py"
```

Typical one-prime timings are

| complete branch | transition targets | systems | largest system | unique nonzero PBW coefficients | build | solve |
|---|---:|---:|---:|---:|---:|---:|
| `v_2` | 16 | 15 | 70 | 281 | 0.045 s | 0.003 s |
| `v_5/2` | 32 | 24 | 420 | 2311 | 2.17 s | 0.36 s |
| `W_7/4` | 8 | 8 | 20 | 51 | 0.006 s | <0.001 s |
| `W_9/4` | 16 | 15 | 116 | 397 | 0.27 s | 0.008 s |

These numbers include every distinct nonzero-mode physical endpoint of the
ordered chi string, not only the highest-grade endpoint.  A Ramond zero mode
doubles the endpoint records (to 16 for `W_7/4` and 32 for `W_9/4`) but uses
the identical transition solution in the other ground block; it therefore
does not require another matrix build or solve.  The fully expanded Ramond
component counts are consequently 102 and 794, respectively.

All arithmetic is exact.  For a rational input, repeat the calculation at
primes congruent to one modulo eight.  Such a field contains roots of `-1`
and `2`.  Evaluate the four choices

```text
(sqrt(-1),sqrt(2)), (sqrt(-1),-sqrt(2)),
(-sqrt(-1),sqrt(2)), (-sqrt(-1),-sqrt(2)).
```

The four-point Hadamard transform separates the coefficients of
`1,sqrt(2),i,i*sqrt(2)`.  Combine each rational residue by CRT and perform
rational reconstruction.  A fresh unused prime verifies the reconstructed
answer.  `crt_reconstruction.py` demonstrates this procedure for a complete
Gaussian-rational transition column and verifies the result over the
rationals; the example stabilizes after nine roughly twenty-bit primes.

## Exact three-point variable elimination

Finite fields remove coefficient swell, but explicitly expanding every leg
still gives 2311 physical components for `v_5/2` and 794 for each `W_9/4`
after the two Ramond ground blocks are restored.
A naive trilinear loop would inspect about

```text
2311 * 794 * 794 = 1,456,937,596 component triples,
```

so replacing symbolic numbers by residues alone is not sufficient.

`modular_three_point.py` avoids that component-triple loop by eliminating the
middle Ramond PBW variable first.  For each middle PBW state, the physical
NS--R--R Ward form is stored as one matrix between homogeneous bases on the
first and third legs.  Positive modes act by left or right multiplication
with cached representation matrices.  The 16 auxiliary path colors are
contracted only after this physical block has been formed.  The auxiliary
Ising factor is evaluated in the same prime field; its corrected Virasoro
expansion and the native spin-OPE Pfaffian agree on all 512 stored endpoints.

Run

```bash
python3 -m python.ramond_screening_algorithm.profile.modular_three_point --benchmark
```

For `(v_2,W_7/4,W_7/4)` at one prime, the input has `(281,202,202)` expanded
components, hence 11,465,924 candidate component triples.  The dense
middle-leg evaluator instead contracts 1,470 parity-allowed homogeneous
sector triples.  On the profiling machine it took 59.16 seconds in total
(0.21 seconds to build the three states and 58.95 seconds to contract), with
28,788 cached middle blocks, 5,835,170 stored block entries, and 245,378
scalar entries in the middle-primary two-leg tables.  The result is one exact
residue; CRT and rational reconstruction are applied only to that scalar.

If `d_1`, `d_2`, and `d_3` denote the relevant homogeneous PBW dimensions,
forming one middle block costs sparse mode actions plus dense products of at
most `d_1` by `d_3` matrices.  Across all required grades the cache size is
the sum of those block areas, rather than `d_1*d_2*d_3` component triples for
each auxiliary path triple.  This is an exact practical fallback, not a
polynomial all-level bound: the PBW dimensions still grow with the partition
numbers.

The Pfaffian--Selberg code in the parent directory is presently a partial
screening backend, not a completed production replacement.  The eight-chart
rank audit shows that the simple zero-plane reconstruction is underdetermined,
and the remaining positive and reflected nonzero callbacks and a uniform
Schur-width bound have not been proved.  It must therefore not be described
as a complete polynomial all-level algorithm.
