# Complete total-level-6 channel checks

These isolated tests extend every previously tested closed plumbing channel
to **all original-edge multidegrees with sum of physical levels at most 6**.
There is no individual-edge cutoff below 6. The manuscript and production
headers are unchanged.

| Channel | Cases | Original multidegrees per case |
|---|---:|---:|
| NS–R–R theta | 16 | 140 |
| All-NS theta | 2 | 455 |
| Glasses, R/R handles | 8 | 140 |
| Glasses, NS/R handles | 4 | 252 |
| Glasses, NS/NS handles | 2 | 455 |
| Genus-3 Mercedes, NS spokes and triangular R rim | 64 | 3,906 |

All constrained zero coefficients and all formal edge-parity components are
included in the comparisons. The theta NS–R–R cases include both intrinsic
NS primary parities, both form parities, and all endpoint signs. Other
channels use even intrinsic primary parities and the same complete form/sign
choices as the initial tests. R/NS glasses is the named-handle exchange of
NS/R, not a separate numerical run. The inequivalent four-edge Ramond cycle
on K4 is not included.

Parameters remain `b=7/5`, with momenta
`(11/23,13/29,17/31)` or, for Mercedes,
`(11/23,13/29,17/31,19/37,23/41,29/43)` in the documented edge order.
The primary calculations use 40-digit MPC arithmetic. The independent
existing Python NS PBW check uses machine precision and is reported separately.

## What is actually computed

`theta_validate.cpp` uses the existing optimized theta CCY pipeline with its
Schottky vacuum, recursive cached branching, direct auxiliary factor, and
independent physical and enlarged PBW references. Its full split-edge check
also retains nonzero fermion modes. The theta code is isolated from the
production source; its prior level-3 scope guard is extended to 6.

`graph_validate.cpp` handles all-NS theta, glasses, and Mercedes. Its enlarged
numerators use the **full CCY c-recursion**, not only the global term and not
a finite-c Virasoro Gram sum. `graph_ccy.hpp` implements the residue graph
using each ordered incident vertex, caches pole geometry and fusion
polynomials, accumulates repeated recursion states, and sews the exact
global L_-1 coefficients. Its universal vacuum factor is computed from
primitive Schottky cycles of the actual graph. The product of the two
Virasoro copies is then summed with local branching coefficients and inverse
branch norms. The uninserted numerator is computed for every case, including
those predicted to vanish.

NS–R–R branching uses the existing boundary-initialized recursive
`OuterBranching` implementation, reused by vertex, form parity and sign.
Inserted vertices use `MiddleBranching`. The original all-NS helper only
supported labels `n=0,+/-1/2`; the new `NSBranches` also constructs the
required `+/-1,+/-3/2` primaries in free fields, converts their components
to SCA PBW coordinates, and evaluates the local human-convention three-form.
These local branching data are cached. No final physical block is used to
construct a branching coefficient or a double-Virasoro numerator.

The physical reference directly sews SCA PBW vertices with **full inverse
Gram matrices**. In particular, the Mercedes spoke metrics act on all three
axes of its central tensor, so their off-diagonal entries are included.
The Ramond Ward engine is the existing `ScaWard`; `ns_local.hpp` ports the
existing human-convention NS Ward recurrence to the same 40-digit scalar type.
The resulting all-NS theta physical coefficients are additionally compared
with the independent existing Python `DirectThetaOracle` through level 6.

Auxiliary factors are sewn from fermion Fock states and Ward/Pfaffian forms,
with the bra-slot BPZ sign. On a split edge the complete mode matrix of
`Q Theta psi` is retained. Recovery uses only the enlarged numerator and
the auxiliary series, then undoes the known physical parity transport.
The recovered physical block is compared with the independent SCA PBW block.

## Split cutoffs and comparisons

For glasses and Mercedes, independent split powers obey

```
sum(levels on unsplit original edges)
  + sum(max(left level, right level) on each split original edge) <= 6.
```

This includes the complete physical diagonal through total level 6, as well
as every unequal split configuration in this downward-closed domain. It also
contains every expanded-graph multidegree of ordinary total level at most 6.
Full split numerators are compared with the physical-PBW/direct-fermion
convolution, including nonzero fermion modes. Theta retains its established
weighted split cutoff: `a+l+r+d<=12` in
`q1^(a/2) qL^l qR^r q3^(d/2)`, with even `d`.

Each graph driver checks forward convolution, physical recovery, all spin-lift
evaluations, and the vanishing uninserted numerator in every obstructed case.
The saved-data audit independently checks exact multidegree coverage and
the physical Ramond loop-sector identity. Theta additionally compares its
independently sewn enlarged PBW numerator and tests split-diagonal/zero-mode
identities. Errors use `abs(x-y)/max(1,abs(x),abs(y))`; vanishing errors are
absolute. The theta tolerance is `1e-20`; the general graph tolerance is
`1e-18`. The machine-precision NS port comparison uses `1e-9`.

## A higher-level issue exposed by these tests

The first extension to glasses failed first at physical levels `(0,0,3)`.
The new CCY self-loop residue lacked the factor `(-1)^(r*s)` for the two
sewn slots at local coordinates `(1,0)`. The sign is invisible at null
level 2, and is required at odd null levels by the middle-slot Virasoro
Ward normalization. Correcting this graph-driver residue resolves the
glasses mismatches. Diagnostic failures are retained under
`selfloop_phase_diagnostic/`; they are superseded by the corrected results.
The Mercedes graph has no self-loops, so the concurrent Mercedes process
never executes the corrected branch. Its initial source hashes are saved
in `mercedes_L6_initial_source_hashes.json`.

## Evidence and reproduction

Run `python3 run_all.py` from this directory for a fresh sequential build and
complete suite. It does not read old block coefficients. Numerical runs can
also be reproduced individually with `./graph_validate CHANNEL 6`; theta
NS–R–R uses `python3 theta_tests.py` after `make -f theta_makefile theta_validate`.

`theta_summary.json` and the individual theta files contain all 136 theta
comparisons. Each other channel has `CHANNEL_L6_results.json` with errors and
stage timings and `CHANNEL_L6_coefficients.jsonl` with all diagonal physical,
enlarged and recovered coefficients, saved as decimal strings. Case ordering
is defined by `Setup` in the source. Zero entries omitted from sparse parity
arrays are identically zero. `coverage_and_sector_results.json` records the
saved-data audit. `ns_physical_port_results.json` reports the separate check
of the NS PBW port. `manifest.json` summarizes the completed runs and hashes.

Reported timings include the independent references, comparisons and output;
they are validation-suite wall times, not production block benchmarks.
Some channels were run concurrently. A fresh sequential reproduction records
its own wall times in `fresh_run_manifest.json`.
