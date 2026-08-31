# Corrected NSRR toy — actual double-Virasoro computation

Completed locally in 23.8 seconds, using the corrected NSRR chiral adapter.
All 35 momentum nodes completed. No archived NSRR or all-NS partition values
were used as input to this calculation. The protected kernels were not edited.

**Scope:** this is a channel-resolved diagonal-norm diagnostic, not the
physical NSRR partition function Z and not the Weyl-normalized Q. Its
normalization is defined explicitly below. No physical Ramond projector,
ground-state multiplicity, free-spin conversion, or physical spin/lift
dictionary is inferred from it.

## Definition

For a channel `a=(f,eta)`, with `f=0,1` and `eta=+,-`, define

```
A_a(P;q,lifts) = C_eta(P) * primary(P;q) * F_f^(eta,eta)(P;q,lifts)
H_ab = integral_{P_e>=0} prod_e(dP_e/pi) A_a conjugate(A_b)
D = trace(H)
  = integral prod_e(dP_e/pi) sum_(f,eta) |C_eta * primary * F_f^(eta,eta)|^2.
```

`C_+` and `C_-` are the existing generic-b BRY `C_even` and `C_odd`.
`primary` contains the NS/R weights in the corrected local chart:
`h_NS=(b+1/b)^2/8+P_NS^2/2`,
`h_R=c_SL/24+P_R^2/2`.

The identity metric used in `trace(H)` is a **diagnostic choice**, not a
derivation of nonchiral Ramond sewing. Opposite-HJS-sign blocks are omitted
because the current auxiliary quotient does not determine them; they are
not asserted to vanish, nor is their omission a controlled higher-level
truncation. H is consequently only a matrix on the supported four-channel
subspace, not the full Ramond sewing tensor.

The corrected block retains its Human-Note odd-form phases and quadratic
sewing sign. However, a positive diagonal norm is phase-insensitive and
cannot certify the additional three-point phases and decomposition signs
of the still-missing physical nonchiral contraction. The full complex H
matrix and the individual chiral values are saved, rather than just D.

## Run design

- Generic `b=1.4`, common cosmological prefactor omitted.
- Original source family `Omega(t)=[[i,t+0.5i],[t+0.5i,i]]`, with
  `t=0.52,0.56,0.60,0.64,0.68`.
- Re-marked, re-plumbed NS-at-infinity charts from the corrected adapter.
  Their forward period map is freshly checked at basis order 32 with
  160 seam samples; maximum residual below `5.53e-13`.
- Geometry order `(zero,one,infinity)=(R,R,NS)` is converted together with
  momenta and lifts to package order `(infinity,one,zero)=(NS,R,R)`.
- Four literal lift representatives `(R0,R1,NSinf)=(+,+,+),(+,-,+),
  `(-,+,+),(-,-,+)`; none is declared to be the physical marked spin.
- Generalized-Laguerre continuum quadrature `N=2,3`, giving `8+27=35`
  momentum nodes on a common q envelope across the five points.
- Holomorphic total levels `L=0,1,2`; these are polynomial truncations of
  the same maximum-level-two multivariate series.
- Branching coefficients come from the canonical recursion. Both ordinary
  Virasoro factors come from c-recursion. Physical components use the
  equal-HJS Ward support relation followed by the ordinary literal lift sum.

During production the full genus-two PBW oracle is replaced by a raising
mock. Its recorded call count is zero at every node. The checked package's
normal low-state branching anchors are not changed.

## Fresh results

All four literal-lift traces coincide to within `1.12e-16` relatively on
this grid. The individual parity channels do not coincide: changing one
Ramond lift exchanges their contributions. Thus D has discarded information
needed to identify a physical spin structure. This equality is not a
spin-structure or modular-invariance check.

| Original t | D, N=3 L=1 | D, N=3 L=2 | L1 to L2 change | N2 to N3 change at L2 |
|---:|---:|---:|---:|---:|
| 0.52 | 1.66662696596e-9 | 1.69538205686e-9 | +1.7253% | +4.1147% |
| 0.56 | 1.63457704230e-9 | 1.66004246252e-9 | +1.5579% | +4.0886% |
| 0.60 | 1.57474085190e-9 | 1.59479126474e-9 | +1.2733% | +4.0367% |
| 0.64 | 1.49449253556e-9 | 1.50870114759e-9 | +0.9507% | +3.9607% |
| 0.68 | 1.40254646292e-9 | 1.41193928315e-9 | +0.6697% | +3.8638% |

These are finite-cutoff shifts, not error bounds. The numbers must not be
compared directly with the previous all-NS Z or Q as a physical modular test.

For example, at t=0.60, N=3, L=2 and literal lifts `(+,+,+)`, the four
diagonal entries in channel order `(0,+),(0,-),(1,+),(1,-)` are

```
1.034343752858692e-9
5.573246266275084e-10
1.3284553740879916e-13
2.9900397122988787e-12.
```

Changing to `(+,-,+)` exchanges the two f=0 entries with the two f=1
entries, leaving their sum unchanged. The overview figure shows this
alongside the numerical refinement curves.

## Verification

- 21 targeted regression tests passed.
- Maximum branching Ward residual: `1.924e-13`.
- Independent matrix reduction, scaled by trace: `2.594e-16` maximum error.
- Amplitudes reconstructed from saved C, primary and block: exact agreement.
- Separate validation-only PBW calculations checked five actual quadrature
  nodes, all five periods, all four lifts, all three levels and all four
  supported channels: 1,200 block values, maximum scaled error `1.454e-14`.
- Production PBW calls: zero. PBW validation did not supply missing terms.
- All eight protected-kernel hashes match their pre-repair manifest.

## Files

- `summary.json`: D and the full complex H matrices, with physical Z/Q null.
- `config.json`: exact geometry, channels, cutoffs, diagnostic definition and
  implementation fingerprint.
- `shards/`: new complex chiral blocks, primary factors, structure constants,
  amplitudes, quadrature weights and Ward residuals at each momentum node.
- `verification.json`: independent reduction, PBW validation and refinement.
- `nsrr_toy_overview.svg` / `.png`: concise result and parity-channel plot.
- `nsrr_diagonal_toy.svg`: separate level curves for every lift representative.

## Reproduce

From the repository root, set `PYTHONDONTWRITEBYTECODE=1`,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`PYTHONPATH=Code:Code/genus_2:Code/c_Recursion:Code/full_ramond_block_runtime`.

```sh
python3 Code/genus_2/run_corrected_nsrr_toy.py run \
  --geometry 'Data Set/nsnsns_recompute_fivepoint_R16_N5_20260830/source_geometry_audit.json' \
  --output-dir 'Data Set/corrected_nsrr_toy_L2_N3_20260830' --workers 3

python3 Code/genus_2/audit_corrected_nsrr_toy.py \
  --run-dir 'Data Set/corrected_nsrr_toy_L2_N3_20260830'
```

The physical NSRR partition remains a separate unfinished task: its
nonchiral contraction, opposite-HJS information, marked spin assignment
and compatible same-frame free denominator still have to be supplied and
checked. Increasing this toy's numerical accuracy cannot replace those data.
