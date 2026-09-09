# Low-accuracy NSRR / NSNSNS toy comparison

Completed locally on 2026-08-30 in 159.3 seconds (380.0 summed node CPU
seconds), with 182 validated momentum-node shards and three concurrent fresh
processes. The larger cluster scan jobs 43016340, 43016341 and 43016342 were
cancelled at the user's request; their staged files and partial output remain.
No higher-accuracy run was started after this toy check.

## Setup

`b=1.4`, `Omega(t)=[[i,t+0.5i],[t+0.5i,i]]`, at `t=0.56,0.60,0.64`.
The target is the symplectically transformed period matrix, using the exact
matrix, matched plumbing parameters and spin lifts recorded in `config.json`.

- NSRR: branching-coefficient Ward recursion, then the product of ordinary
  genus-two Virasoro c-recursions. Physical levels 2 and 3 are compared.
- NSNSNS: collision-aware N=1 c-recursion at physical level 3, i.e. API
  `recursion_order=6` in twice-level units, with the resummed regular seed.
- Full Cartesian momentum quadrature orders N=3 and N=4. A common per-channel
  Gaussian envelope permits reuse of each source coefficient series at the
  three period matrices. No nodes were omitted or reweighted to fit equality.
- Same-frame physical scalar+Majorana normalization,
  `Q=Z_SL/(Z_free)^[1+2(b+1/b)^2]`, with exponent 9.940408163265307.
- The previous three-point convention ledger, including the all-NS top-form
  phase and closed-R-edge completeness factors, is retained explicitly.
- The Liouville cosmological prefactor is omitted consistently. Its common
  restoration factor would not change the channel ratio.

## Results

These are low-cutoff estimates, not precision-certified values. Both channels
use physical level 3 in this table, with N=4:

| t = Re Omega_12 | Q_NSRR | Q_NSNSNS at transformed Omega | Raw ratio |
|---|---|---|---|
| 0.56 | 3.78338556e-7 | 3.49724959e-7 | 1.08181743 |
| 0.60 | 3.06330962e-7 | 3.14735128e-7 | 0.97329765 |
| 0.64 | 2.47462554e-7 | 2.70200023e-7 | 0.91584949 |

| t | Ratio at N=3 | Relative ratio change, N=3 to 4 | NSRR Q change, level 2 to 3 at N=4 |
|---|---|---|---|
| 0.56 | 1.08611961 | -0.3961% | -0.03716% |
| 0.60 | 0.97775164 | -0.4555% | +0.05096% |
| 0.64 | 0.91960878 | -0.4088% | +0.07402% |

The separate Q values change by 0.81–0.82% (source) and 1.21–1.27% (target)
when N increases from 3 to 4. Maximum source Ward residual is
5.7365e-13, well below the toy acceptance threshold 1e-5. That residual is
not a bound on the partition-function error. An independent level-2 source
evaluation at one central-point momentum node agreed with the level-2
truncation of the level-3 series to approximately 1e-14 in both sectors.

## Assessment

The central value differs from unity by 2.67%, but nearby values differ by
+8.18% and -8.42%. These discrepancies are appreciably larger than the
observed small quadrature/NSRR-level refinements, and vary with the period
matrix; a single constant normalization cannot remove all three.

This is qualitative proximity, not a successful modular consistency check.
The limited convergence tests do not certify the omitted tails or establish
the cause. Before the expensive run is resumed, investigate spin transport,
the local-frame/free-field normalization and the sewing assembly, including
target truncation effects. Do not fit a normalization factor to these data.

## Files and reproduction

- `summary.json`: full results, conventions, configuration and fingerprint.
- `summary.svg`, `summary.png`: both Q curves and raw ratios versus Re Omega_12.
- `config.json`, `shards/`, `logs/`: complete toy inputs and node outputs.

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 Code/genus_2/run_nsrr_nsnsns_toy.py \
  --output-dir 'Data Set/nsrr_nsnsns_toy_20260830' --workers 3
```

Existing shards are reused only when both config and numerical implementation
fingerprints match. The command never submits or resumes a cluster job.
