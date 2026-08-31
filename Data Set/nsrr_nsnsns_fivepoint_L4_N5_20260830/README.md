# Local five-point NSRR / NSNSNS refinement

Completed 2026-08-30 in 426.1 seconds (7.1 minutes), with 378 validated
momentum-node shards, three fresh-process workers, and 1096.1 summed node
CPU seconds. No cluster job was submitted or resumed.

## Design

At b=1.4, the source period matrix is
`Omega(t)=[[i,t+0.5i],[t+0.5i,i]]` for t=0.52,0.56,0.60,0.64,0.68.
The target uses the stored symplectic transformation and its own matched
theta plumbing parameters. Both local free-field denominators and the
geometry at the three points shared with the toy are unchanged.

The requested one-order refinement is:

- NSRR physical levels 3 and 4 (previously 2 and 3).
- NSNSNS physical level 4, i.e. twice-level recursion order 8 (previously 6).
- Momentum quadrature orders 4 and 5 (previously 3 and 4).
- Regular-seed tolerance 1e-7 (previously 1e-6), occupation cap 36.
- Structure-constant precision 30 decimal digits; collision-aware block
  working precision 40 digits. Branching uses binary64 Ward solves with
  maximum accepted relative Ward residual 1e-5.

The NSRR method remains Ward-recursive branching coefficients followed by
the product of two ordinary Virasoro c-recursion blocks. No direct PBW
genus-two block contraction was substituted. The actual maximum Ward
residual was 9.5489e-13. This is not a partition-function error bound.

The observable is Q=Z_SL/(Z_free)^[1+2(b+1/b)^2], with exponent
9.940408163265307. The physical scalar+Majorana factors use their respective
local theta frames. The cosmological prefactor is omitted consistently and
no normalization factor was fitted.

## Results at physical level 4 and N=5

These are finite-cutoff numerical estimates, not precision-certified values.

| t | Q_NSRR | Q_NSNSNS at transformed Omega | Raw ratio | Relative mismatch |
|---|---|---|---|---|
| 0.52 | 4.68494729e-7 | 3.69931427e-7 | 1.26643668 | +26.6437% |
| 0.56 | 3.78589350e-7 | 3.50593732e-7 | 1.07985202 | +7.9852% |
| 0.60 | 3.06510946e-7 | 3.15384474e-7 | 0.97186441 | -2.8136% |
| 0.64 | 2.47628676e-7 | 2.70639390e-7 | 0.91497648 | -8.5024% |
| 0.68 | 1.98997566e-7 | 2.23190231e-7 | 0.89160518 | -10.8395% |

## Refinement diagnostics

| t | Ratio at N=4 | Ratio at N=5 | Relative ratio change N4 to N5 | NSRR change L3 to L4 at N5 |
|---|---|---|---|---|
| 0.52 | 1.27485888 | 1.26643668 | -0.66064% | +0.029274% |
| 0.56 | 1.08704035 | 1.07985202 | -0.66127% | +0.006838% |
| 0.60 | 0.97789600 | 0.97186441 | -0.61679% | -0.001709% |
| 0.64 | 0.91992254 | 0.91497648 | -0.53766% | +0.003788% |
| 0.68 | 0.89552425 | 0.89160518 | -0.43763% | +0.011950% |

The separate Q changes under N4 to N5 are 0.0619–0.0704% for NSRR and
0.5102–0.7284% for NSNSNS. The same-point ratios compared with the toy are:

| t | Toy L3/N4 ratio | Refined L4/N5 ratio | Relative change |
|---|---|---|---|
| 0.56 | 1.08181743 | 1.07985202 | -0.18168% |
| 0.60 | 0.97329765 | 0.97186441 | -0.14726% |
| 0.64 | 0.91584949 | 0.91497648 | -0.09532% |

Adding the outer period points changes the common Gaussian quadrature
envelope, so the toy-to-refined differences combine that change with block
cutoffs and tolerances. The within-run N4/N5 and source L3/L4 axes are the
controlled individual comparisons. These finite differences are not rigorous
tail bounds; target block-order convergence is not separately certified.

## Convention audit and interpretation

Both the all-NS odd coefficient phase C_HN^(1)=i*C_BRY_tilde and the separate
decomposition sign (-1)^(a+p1+p2+p3) were already present in the toy run.
For the even NS primaries used here, the two odd-sector minus signs cancel.
Coefficient squares are NOT replaced by absolute squares. The HJS Ramond
component phases and the literal double-Virasoro graded sewing sign were
checked separately. See [CONVENTIONS.md](CONVENTIONS.md).

All 46 selected regression tests passed, including tests that inspect both
signs independently. The full numerical-kernel fingerprint is unchanged from
the toy: `2d221e0ffb621270af992b836103cfc52e71b14b5fc739c0725501b29d46601d`.
The convention ledger is identical. A separate comparison identity check
also passed.

The discrepancy persists and is strongly period-dependent. This modest
refinement does not establish modular equality, and a single constant
rescaling cannot reconcile the five values. The audited all-NS i-factor and
decomposition sign are not missing. The full NSRR nonchiral assembly and
spin/frame transport remain candidates for investigation; the data alone
do not identify the cause. Further expensive accuracy increases should not
be assumed to resolve it.

## Files and reproduction

- `summary.json`: full values, sectors, config, geometry and fingerprints.
- `summary.svg`, `summary.png`: Q curves and ratios as functions of Re Omega_12.
- `comparison_with_toy.json`: same-point and within-run refinement diagnostics.
- `config.json`, `shards/`, `logs/`: complete inputs and node output.

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 Code/genus_2/run_nsrr_nsnsns_toy.py --design fivepoint-l4 \
  --output-dir 'Data Set/nsrr_nsnsns_fivepoint_L4_N5_20260830' --workers 3
python3 Code/genus_2/compare_nsrr_scan_refinement.py \
  --refined 'Data Set/nsrr_nsnsns_fivepoint_L4_N5_20260830/summary.json' \
  --reference 'Data Set/nsrr_nsnsns_toy_20260830/summary.json' \
  --output 'Data Set/nsrr_nsnsns_fivepoint_L4_N5_20260830/comparison_with_toy.json'
```

Existing node files are reused only after configuration and implementation
fingerprint validation. The default runner design remains the original toy;
the five-point refinement requires the explicit design flag.
