# Fresh five-point all-NS reference

Run status: all-NS integration completed locally in 1008.6 seconds
(16.8 minutes). All 125 momentum shards and the independent reduction passed.
The NSRR comparison remains unavailable.

This directory contains a **partial** recomputation of the requested
NSRR–NSNSNS comparison. Only the all-NS partition can currently be evaluated
with the supported adapters. There is no corrected NSRR partition or modular
ratio here. In `summary.json`, those source fields are explicitly null.

## Fresh results

| Original source t | Z_NSNSNS at transformed Omega | Q_NSNSNS |
|---:|---:|---:|
| 0.52 | 6.55342659918e-10 | 3.69931360235e-7 |
| 0.56 | 6.55797940267e-10 | 3.50593727067e-7 |
| 0.60 | 6.55686638701e-10 | 3.15384544256e-7 |
| 0.64 | 6.54397546966e-10 | 2.70639489636e-7 |
| 0.68 | 6.51377401164e-10 | 2.23190291095e-7 |

These numbers reproduce the previous all-NS R=16 run exactly. All 1,250
sector evaluations (125 nodes times five points times two sectors) also
agree exactly. This is reproducibility of the unchanged all-NS calculation,
not new evidence for NSRR modular equality or a certified integration error.

## Unchanged physical design

The original source period family is

```
Omega_source(t) = [[i, t+0.5i], [t+0.5i, i]],
t = 0.52, 0.56, 0.60, 0.64, 0.68.
```

The all-NS values are evaluated at its previously selected marked transform,

```
Omega_NS(t) = [[1-t+i*(0.75+(1-t)^2), 0.5+i*(1-t)],
               [0.5+i*(1-t),        -1+i]].
```

The horizontal plot coordinate remains the original source `t`, not the real
part of the transformed off-diagonal entry (which is constant at 0.5).

- Generic `b=1.4`; the common cosmological prefactor is omitted, as before.
- Same five target plumbing triples and common `N=5` generalized-Laguerre
  momentum grid: 125 momentum nodes, each evaluated at all five points.
- N=1 c-recursion at `R=16` accumulated null twice-level, with resummed global
  blocks. This is not a total plumbing-polynomial cutoff at level 8.
- Structure-constant precision 30 digits; collision-aware block precision
  40 digits; global tolerance `1e-7`, occupation cap 36.
- `C_HN=(C_BRY,i*tilde_C_BRY)` is squared, not absolute-squared, and the
  separate nonchiral sewing sign `(-1)^f` is retained.
- The free `X+psi` factor is freshly evaluated directly at the same all-NS
  q triple and lifts, with mode cutoffs 36 and 44.
- `Q=Z_SL/Z_free^kappa`, with `kappa=1+2*(b+1/b)^2=9.940408163265307`.

The new runner imports only the previous geometry and numerical design.
It does not import old source values, use the retired factor-four contraction,
relax old provenance checks, or edit the checked kernels.

## Geometric and source-side checks

The target forward period map was freshly evaluated at basis order 32 with
160 seam samples. The maximum period residual is `1.303e-9`, at `t=0.68`;
all points pass the `1e-8` threshold.

`source_geometry_audit.json` separately repeats the NS-at-infinity inverse
plumbing and forward checks for all five source charts. The largest source
forward residual is `5.524e-13`. The candidate theta-ratio free-spin conversion
still fails its necessary compatibility test on every re-marked source chart.
Those mismatches are **not** estimates of the error in Q.

The chiral adapter corrections therefore do not yet give a physical NSRR
partition. The nonchiral Ramond contraction and its spin/lift/free-field
translation remain unresolved. The checked PBW/double-Virasoro package is
unchanged, and its chiral agreement is not a certificate for these missing
nonchiral operations.

## Outputs and verification

- `config.json`: fresh forward-period and direct free-field evaluations,
  input-design digest, and current implementation fingerprint.
- `shards/`: fresh per-momentum results, with even/odd sector contributions
  and global-sum diagnostics; `logs/` contains their worker logs.
- `summary.json`: completed all-NS integration and null NSRR fields.
- `all_ns_reference.svg` / `.png`: Z and Q versus the original period coordinate.
- `verification.json`: independent reduction, target-only historical
  reproducibility, extra free-mode check, and protected-kernel hashes.

The regression suite passed 74 tests. Historical all-NS order changes were
small (`R=8 -> 16` at most `3.67e-7` relatively), but the previous all-NS
`N=4 -> 5` quadrature changes were `0.51%–0.73%`. This rerun does not refine
quadrature, and printed digits must not be read as a certified accuracy claim.

No global sum failed convergence; the largest occupation used was 28.
The free-field mode changes were at most `1.56e-15` from 36 to 44, and
`1.00e-15` from 44 to 48. All eight protected-kernel SHA-256 hashes still
match the pre-repair manifest. Only new orchestration, audit, test, and
output files were added in this recomputation.

## Reproduce

From the repository root, set `PYTHONDONTWRITEBYTECODE=1`,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`PYTHONPATH=Code:Code/genus_2:Code/c_Recursion:Code/full_ramond_block_runtime`.

```sh
python3 Code/genus_2/recompute_all_ns_reference.py run \
  --baseline-config 'Data Set/nsrr_nsnsns_fivepoint_L4_N5_20260830/config.json' \
  --output-dir 'Data Set/nsnsns_recompute_fivepoint_R16_N5_20260830' \
  --quadrature-order 5 --recursion-order 16 --workers 3

python3 Code/genus_2/nsrr_human_note_geometry.py \
  --baseline 'Data Set/nsrr_nsnsns_fivepoint_L4_N5_20260830/config.json' \
  --output 'Data Set/nsnsns_recompute_fivepoint_R16_N5_20260830/source_geometry_audit.json'

python3 Code/genus_2/audit_fresh_all_ns_reference.py \
  --run-dir 'Data Set/nsnsns_recompute_fivepoint_R16_N5_20260830' \
  --previous-dir 'Data Set/nsrr_nsnsns_target_R8_R12_R16_N5_20260830'
```

Existing fresh-run shards are resumed only if their configuration, momentum
node, complete evaluation design, convergence flags and implementation
fingerprint all validate.
