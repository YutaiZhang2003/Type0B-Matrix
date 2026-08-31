# All-NS recursion-order diagnosis of the NSRR–NSNSNS mismatch

## Result

All 125 nodes completed locally in 1108.7 seconds (18.5 minutes), with 3243.7
summed node-seconds. Every saved `R=8` sector value was reproduced exactly.
An independent reduction check reproduced all 15 integrated results from the
node files. All 52 regression/diagnostic tests passed.

The all-NS order increase does not materially change the comparison:

| t | all-NS Q at R=16 | NSRR baseline / all-NS R=16 | Q16/Q8 - 1 | Q16/Q12 - 1 |
|---:|---:|---:|---:|---:|
| 0.52 | 3.69931360235e-7 | 1.26643691045 | -1.80717e-7 | -3.83602e-11 |
| 0.56 | 3.50593727067e-7 | 1.07985203465 | -1.37402e-8 | +5.53932e-11 |
| 0.60 | 3.15384544256e-7 | 0.97186419482 | +2.22199e-7 | -2.50929e-10 |
| 0.64 | 2.70639489636e-7 | 0.91497614340 | +3.66831e-7 | -1.96412e-9 |
| 0.68 | 2.23190291095e-7 | 0.89160493747 | +2.71065e-7 | -6.92139e-9 |

The largest `R=8 -> 16` relative change is `3.66831e-7` (0.0000367%). The
largest `R=12 -> 16` relative change is `6.92139e-9`. These are far below the
existing percent-level channel differences. No global sum failed convergence;
the largest occupation used was 28, below the unchanged cap of 36.

`summary.json` stores the full numerical reduction. `comparison.svg` and
`comparison.png` show Q, the channel ratio, and the all-NS order correction as
functions of the source period coordinate. The annotated figure explicitly
labels the source as the unmodified baseline, because the separate audits
below found unresolved spin-projection and edge-frame discrepancies.

The next useful experiment is a consistently defined NSRR physical spin
projection and geometric-to-trinion interface, followed by a low-order matched
comparison. Further increasing the all-NS recursion cutoff is not indicated
by these results. No NSRR repair or new cluster run was performed here.

## Controlled design

This experiment changes only the all-NS accumulated Kac-residue cutoff
`R = 8, 12, 16` on the existing five-point `N=5` momentum quadrature.
`R` is measured in twice-level units. The regular/global blocks are resummed;
these are not complete plumbing-polynomial truncations at levels 4, 6 and 8.

The baseline is `../nsrr_nsnsns_fivepoint_L4_N5_20260830/`.
The NSRR physical-level-four values are reused unchanged. All 125 target
momentum nodes are evaluated at all five surfaces and all three recursion
orders. Each node runs in a fresh process, with three concurrent workers.

## Fixed inputs

- `b = 1.4` and the same cosmological-prefactor convention.
- Source period family `Omega(t)=[[i,t+0.5i],[t+0.5i,i]]`,
  with `t=0.52,0.56,0.60,0.64,0.68` and the saved target symplectic transform.
- Full-precision saved plumbing triples, lifts, and physical free denominators.
- The exact same generalized-Laguerre nodes, weights, and envelope as baseline.
- `C_HN,odd=i*C_BRY,odd`, inserted once before squaring, and the separate
  odd-sector nonchiral sewing sign.
- `Q=Z_SL/(Z_free)^(1+2(b+1/b)^2)`, without fitted rescaling.
- Global tolerance `1e-7`, occupation cap 36, and block working precision 40.

The baseline `R=8` values are recomputed in every node. A shard is rejected if
its baseline sector reproduction error exceeds `1e-10`. The reduction refuses
missing or extra shards and checks the unchanged numerical-kernel fingerprint.

## Precision audit

`precision_audit.json` contains a separate audit of baseline task indices 283
and 284, the two largest-contributing `N=5` target nodes, at the two endpoints
and center. It compares `R=8,16` with the baseline controls and with global
tolerance `1e-11`, occupation cap 60, and working precision 60.

The largest relative change of an `R=16` node under this tightening is
`1.7591e-10`. The `R=8 -> 16` changes at those nodes range in absolute value
from `1.57e-7` to `2.69e-7`, and remain stable under tightening.
This is a selected-node audit, not a certified error bound on the integral.

## Ground-level spin-projection discrepancy: no repair applied

`spin_projection_audit.json` compares the NSRR runtime with the literal
fixed-lift evaluation of the independent Human-Note PBW ground components.
No positive-level direct PBW production calculation is used in this check.

Let `B_p` denote those parity components. They already include the theta
quadratic sign. The sewing formula labelled `Rblock` in `Human Notes/SCblock.tex`
therefore evaluates the fixed-lift block as

```
literal F(eta) = sum_p B_p product_e eta_e^p_e.
```

However, `star_spectrum(B)` inserts another factor
`(-1)^(p0*p1+p0*p2+p1*p2)` before its Walsh transform. The runtime returns one
such star character after scalar division by the auxiliary character. This
is an algebra homomorphism for the star product, but is not the literal spin
evaluation of the physical components.

For the actual runtime lift tuple `(1,1,-1)`, the even-form ground components
give the following test, independent of recursion accuracy:

| HJS labels (left,right) | Literal fixed-lift ground | Runtime ground |
|---|---:|---:|
| (+,+) | 0 | 2 |
| (+,-) | 2 | 0 |
| (-,+) | 2 | 0 |
| (-,-) | 0 | 2 |

The runtime agrees with the star character within `4.3e-15`, so the issue is
not numerical instability of the quotient. In the odd form, the literal and
star ground values agree; this is therefore not one common normalization
factor across all sectors. It changes which HJS coefficient products enter
the even sector.

This disproves the unqualified identification of the present fixed-star-
character quotient with the literal fixed-lift block. It does not yet give
the repaired partition function. In particular, blindly replacing the
transform inside scalar division is not justified: the auxiliary ground
character can vanish, and the physical spin projection must be derived with
the missing/singular spectral components treated correctly. The earlier
claim that selecting a nonsingular star character alone establishes the
physical spin sector was not sufficiently justified.

## Separate edge-order concern: no repair applied

The order audit identified a concrete interface inconsistency worth testing
before any further large production run:

- The period solver, all-NS public interface and physical free-field public
  interface use geometric order `(zero, one, infinity)`.
- The all-NS global tensor explicitly reverses its positional arguments into
  CCY order `(infinity, one, zero)`. The physical Majorana determinant similarly
  reverses the plumbing half-variables at its sphere-kernel boundary.
- The NSRR runtime constructs its double-Virasoro series in the literal
  `(NS,R,R)` trinion slots, using the CCY `(infinity,one,zero)` tensor. The scan
  driver evaluates this series on the geometric triple without an adapter.

Relevant code is `ns_genus2_partition.py::_theta_global_term`,
`physical_free_plumbing_resummation.py::theta_physical_fermion_fredholm`,
`compute_q_expansion.py::global_series`, and
`nsrr_nsnsns_theta_omega_scan.py::source_values`.

The distinction matters already in the ordinary global seed. For geometric
weights `(h0,h1,hInfinity)=(0.7,1.1,1.9)`, the `q_one` coefficient is

```
passing the triple unchanged to the CCY tensor: 2.404545454545454
converting geometric labels to CCY slots:      0.004545454545455
```

These equal `(h0-h1-hInfinity)^2/(2*h1)` and
`(hInfinity-h1-h0)^2/(2*h1)`, respectively. The result and scope are recorded
in `edge_order_audit.json`. This is an interface diagnostic, not a corrected
NSRR partition calculation or proof that it explains the entire discrepancy.
In particular, reversing `q` alone is not a justified repair: NS/R placement,
lifts, primary factors, period marking and the free denominator must all be
made consistent. No such change is included in this order-convergence run.

## Reproduction

From the repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 Code/genus_2/run_nsrr_nsnsns_target_order_scan.py run \
  --baseline-dir 'Data Set/nsrr_nsnsns_fivepoint_L4_N5_20260830' \
  --output-dir 'Data Set/nsrr_nsnsns_target_R8_R12_R16_N5_20260830' \
  --orders 8 12 16 --quadratures 5 --workers 3

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 Code/genus_2/audit_nsrr_nsnsns_target_precision.py \
  --baseline-dir 'Data Set/nsrr_nsnsns_fivepoint_L4_N5_20260830' \
  --task-indices 283 284 \
  --output 'Data Set/nsrr_nsnsns_target_R8_R12_R16_N5_20260830/precision_audit.json'

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 Code/genus_2/audit_nsrr_nsnsns_edge_order.py \
  --output 'Data Set/nsrr_nsnsns_target_R8_R12_R16_N5_20260830/edge_order_audit.json'

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python3 Code/genus_2/audit_nsrr_nsnsns_spin_projection.py \
  --output 'Data Set/nsrr_nsnsns_target_R8_R12_R16_N5_20260830/spin_projection_audit.json'

PYTHONDONTWRITEBYTECODE=1 \
python3 Code/genus_2/annotate_nsrr_target_order_plot.py \
  --summary 'Data Set/nsrr_nsnsns_target_R8_R12_R16_N5_20260830/summary.json' \
  --source-svg 'Data Set/nsrr_nsnsns_target_R8_R12_R16_N5_20260830/summary.svg' \
  --output-svg 'Data Set/nsrr_nsnsns_target_R8_R12_R16_N5_20260830/comparison.svg'
```

Cutoff differences and precision spot checks are numerical diagnostics, not
certified error bars. This experiment does not measure higher-order NSRR
convergence or improve momentum quadrature.
