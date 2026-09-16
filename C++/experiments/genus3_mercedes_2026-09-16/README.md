# Genus-three Mercedes channel: directed low-level check

The graph is the tetrahedron K4, drawn as an outer triangle and three spokes
meeting at its center. It has four trivalent vertices, six internal edges,
and genus `6 - 4 + 1 = 3`. The outer triangle is Ramond; all three spokes
are NS. The central vertex is NS–NS–NS, and the outer vertices are NS–R–R.

The code uses zero-based edge and vertex labels. The table below uses
one-based edges for comparison with the notes. Each row lists the ordered
local punctures `(infinity, 1, 0)`.

| Vertex | Ordered edges | Vertex form |
|---|---|---|
| Center | (1, 2, 3) | rho_a |
| Outer 1 | (1, 4, 6) | rho_(f1)^(eta_v1) |
| Outer 2 | (2, 5, 4) | rho_(f2)^(eta_v2) |
| Outer 3 | (3, 6, 5) | rho_(f3)^(eta_v3) |

Edge pairs are ordered by edge number. The spokes run center to outer
vertex; the rim runs outer 1 to outer 2, outer 2 to outer 3, and outer 3
to outer 1. This order, including the orientation of the last pair, fixes
the graph permutation sign. In these conventions the loop element is
`J_C = i eta_4 eta_5 eta_6`, where these eta_e are the edge spin lifts.

All intrinsic primary parities are even. The central form label satisfies
`a = f1 + f2 + f3 mod 2`. The obstruction criterion is

```
(-1)^(f1+f2+f3) eta_v1 eta_v2 eta_v3 = -1.
```

This is the combined Ramond-loop sign of the arbitrary-plumbing note,
not the bare product of three vertex signs. The insertion is placed on
edge 4 (edge 3 in the code), with its ket end at outer 1 and its bra end
at outer 2. The implementation uses `Q Theta psi`; the same Q multiplies
the auxiliary divisor and cancels in physical recovery.

## Cutoff and parameters

- Every original edge is independently cut off at physical level 1.
  NS powers are 0, 1/2, 1; Ramond powers are 0, 1. Thus the box contains
  216 multidegrees per case and includes coefficients of total level 6.
- The inserted numerator also retains both unequal split powers `(0,1)`
  and `(1,0)`, as well as the equal pairs. Both segments have cutoff 1.
- `b=7/5`, with momenta `(11/23,13/29,17/31,19/37,23/41,29/43)` in edge order.
- Arithmetic is complex machine precision. The central NS branching
  polynomials are evaluated at 40 digits before conversion; this is not
  an end-to-end higher-precision run.
- All eight outer-form parity choices and all eight vertex-sign choices
  are tested: 32 ordinary cases and 32 obstructed cases. Every one of the
  64 formal edge-parity components is retained. All 64 evaluations of
  the edge spin lifts are compared as well.

## Independent computations

The physical reference contracts SCA PBW vertices and inverse SCA Gram
matrices on this graph. The Ramond vertices use `ScaWard`; the central
NS vertex uses the independent Python `NSDescendantThreeForm`. The native
Ramond basis conversion is applied consistently to vertices and metrics.
It does not use double-Virasoro branching data.

The enlarged numerator instead uses local branching coefficients, inverse
branch norms, and a product of two Virasoro graph blocks. At this cutoff
each Virasoro descendant level is at most one. The CCY answer therefore
equals its exact global term, evaluated with the closed L_-1 Ward formula.
No higher Kac residue or nonconstant Schottky vacuum correction can enter.
This test validates the genus-three graph assembly, branching conventions,
fermion signs, insertion and recovery; it does not test higher-level CCY
residues on the new graph.

The NS–R–R branching vertices use `LowAnchors`, the central NS–NS–NS
branching vertex uses `ns_fusion_data`, and the inserted vertex uses
`MiddleBranching`. No physical block is used to construct the enlarged
numerator. The graph contractions sum all allowed branches and all
distributions of the remaining level between the two Virasoro copies.
Intermediate vertex and branching values are reused within the run.

The free-fermion factor is independently sewn from auxiliary Fock states
and their Ward forms. The graph convolution kernel and physical parity
transport are exactly those in the arbitrary-plumbing note. Recovery
subtracts lower-level convolutions and divides by the constant action on
the matching loop sector, then undoes the physical parity transport.
It does not use physical PBW coefficients to determine the recovered block.

For unequal split powers the independently computed double-Virasoro
numerator is compared with the physical-PBW/free-fermion convolution using
the full fermion insertion. This includes nonzero fermion modes and does
not reduce the test to the zero mode.

During implementation, the new central-fermion adapter initially omitted
the bra-slot BPZ sign. The raw sphere Pfaffian must be multiplied by
`(-1)^delta_1`; in particular, the auxiliary `psi_-1/2` BPZ norm is -1.
This is the existing human-note convention, also implemented by
`current_fermion_three_point` in the earlier exact tests. Correcting the
adapter resolved the mismatch without changing any graph formula or
production source. The authoritative full run below uses that convention.

## Results

Errors are `abs(actual-expected)/max(1,abs(actual),abs(expected))`.
The acceptance tolerance was `1e-8`. All checks passed.

| Comparison | Ordinary cases: maximum scaled error | Inserted cases: maximum scaled error |
|---|---:|---:|
| Recovered physical block versus independent PBW | 1.55e-12 | 1.59e-12 |
| Diagonal enlarged numerator versus independent convolution | 1.26e-12 | 2.31e-12 |
| Physical loop-sector identity | 6.67e-16 | 6.67e-16 |
| All 64 edge-spin evaluations of recovered blocks | 2.14e-12 | 2.20e-12 |
| Unequal split powers versus independent convolution | not applicable | 2.03e-12 |

Each of the first four comparisons covers 442,368 components per sector
(32 cases × 216 multidegrees × 64 components/evaluations). The recovery
comparisons contain 6,912 nonzero formal components per sector; constrained
zeros are retained. The unequal-split comparison adds 442,368 components.
The uninserted enlarged blocks in all 32 obstructed cases vanish with
maximum absolute residual `9.14e-13`, comparable to the other roundoff
errors. Their physical blocks are generally nonzero.

The complete C++ test process took 1.93 seconds wall time, including saved
coefficient output; its internal timer recorded 1.59 seconds. Fresh central
data export, compilation, and the full test together took 5.32 seconds.
These are test-suite timings, not production high-level benchmarks.

## Reproduce and inspect

From this directory:

```sh
python3 run.py
```

`run.py` regenerates the central data, compiles the isolated C++ driver, runs
all cases, checks the verdict, and records source/result hashes and timings.

- `manifest.json`: graph definition, parameters, precision, commands, source
  hashes, timing and aggregate errors.
- `results.json`: per-case errors, component counts and worst coefficients.
- `coefficients.jsonl`: all 13,824 diagonal coefficient rows, with physical
  PBW, enlarged double-Virasoro, and recovered values. Parity arrays are sparse;
  omitted entries are exactly zero. The row's case number is the zero-based
  index into `results.json`'s `case_results` array.
- `central_manifest.json`: provenance for the independent central NS data.

The current test covers the triangular Ramond assignment. It does not claim
to cover the inequivalent four-edge Ramond cycle on K4, higher cutoffs,
intrinsic-odd NS primaries, or singular weights. No manuscript or production
header was modified.
