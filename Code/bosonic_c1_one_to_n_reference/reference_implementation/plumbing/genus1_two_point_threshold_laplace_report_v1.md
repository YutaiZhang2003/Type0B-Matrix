# Genus-one two-point threshold versus Laplace cusp audit

## Setup

This comparison keeps the genus-one two-point necklace block orders fixed at
`(6,3)` and varies only the Liouville momentum integration.

- **Threshold audit:** local `alpha=1/2` generalized-Laguerre rules at
  orders 8 and 10, with order 12 available as a retry.  Every point stopped at
  order 10 with a last step below `2.3e-9`, and most were much more stable.
  These values are not used to select or accept a Laplace result.
- **Laplace result:** the exact `b=1` Barnes-G/BRY structure constant is
  expanded at `P1=P2=0` through degree fourteen in
  `(P1^2,P2^2)`.  Both Gaussian moments are done analytically.  The last
  three Watson truncations are combined with a Shanks transform.
- **Independent Laplace diagnostic:** choose the degree from 6 through 14
  having the smallest successive Shanks movement.  The relative uncertainty
  is twice that movement, plus the finite-grid descendant-block envelope and
  the 128-versus-192-node Cauchy-coefficient change.  No threshold value enters
  the degree selection, uncertainty estimate, or pass/fail decision.

The Laplace calculation sets the descendant block to one.  This is justified
only in the necklace cusp where both primary multipliers are small; the full
descendant block remains in the threshold reference.

## Results

The scan contains 16 points covering torus depth, asymmetric locations along
the necklace, and external energies `omega=i*x` with
`x=0.2,0.4,0.6,0.8`.

| point | minimum Gaussian decay | threshold | Laplace-Shanks | relative difference | Laplace self step |
|---|---:|---:|---:|---:|---:|
| `tau8_v025_x04` | 25.133 | -3.538092921186e-5 | -3.538112095143e-5 | 5.419e-6 | 4.675e-5 |
| `tau10_v025_x04` | 31.416 | -5.288304107563e-5 | -5.288308092593e-5 | 7.536e-7 | 6.220e-6 |
| `tau12_v025_x04` | 37.699 | -8.861381546080e-5 | -8.861382886643e-5 | 1.513e-7 | 1.179e-6 |
| `tau16_v025_x04` | 50.265 | -3.096097384094e-4 | -3.096097419621e-4 | 1.147e-8 | 8.318e-8 |
| `tau20_v025_x04` | 62.832 | -1.302243225283e-3 | -1.302243227150e-3 | 1.434e-9 | 1.039e-8 |
| `tau24_v025_x04` | 75.398 | -6.166268589465e-3 | -6.166268591006e-3 | 2.498e-10 | 1.871e-9 |
| `tau16_v010_x04` | 20.106 | -8.474904585745e-4 | -8.476120361505e-4 | **1.434e-4** | **7.227e-5** |
| `tau16_v015_x04` | 30.159 | -5.313628981903e-4 | -5.313642332103e-4 | 2.512e-6 | 4.355e-6 |
| `tau16_v040_x04` | 80.425 | -2.172022640461e-4 | -2.172022642238e-4 | 8.183e-10 | 1.699e-9 |
| `tau16_v050_x04` | 100.531 | -2.047357700204e-4 | -2.047357701654e-4 | 7.082e-10 | 2.258e-9 |
| `tau12_v025_x02` | 37.699 | -2.365503180895e-5 | -2.365503946362e-5 | 3.236e-7 | 3.017e-7 |
| `tau12_v025_x06` | 37.699 | -1.621150962797e-4 | -1.621150269495e-4 | 4.277e-7 | 3.463e-6 |
| `tau12_v025_x08` | 37.699 | -1.465001336083e-4 | -1.465354679769e-4 | **2.411e-4** | **4.730e-4** |
| `tau20_v025_x02` | 62.832 | -3.426050309945e-4 | -3.426050316926e-4 | 2.038e-9 | 2.528e-9 |
| `tau20_v025_x06` | 62.832 | -2.444566037849e-3 | -2.444566033784e-3 | 1.663e-9 | 3.204e-8 |
| `tau20_v025_x08` | 62.832 | -2.298222826615e-3 | -2.298226615088e-3 | 1.648e-6 | 4.205e-6 |

At fixed degree ten, fourteen of sixteen points pass the requested `5e-5`
comparison with threshold quadrature.  That statement is only an external
comparison and is not the criterion for trusting the Laplace calculation.

## Intrinsic Laplace classification

Reoptimizing the truncation through degree fourteen and applying the intrinsic
uncertainty prescription above certifies 13 of the 16 points at `5e-5`:

| point | selected degree | intrinsic relative uncertainty | intrinsic decision |
|---|---:|---:|---|
| `tau8_v025_x04` | 11 | 8.583e-5 | reject |
| `tau10_v025_x04` | 11 | 8.964e-6 | accept |
| `tau12_v025_x04` | 11 | 1.404e-6 | accept |
| `tau16_v025_x04` | 14 | 5.911e-8 | accept |
| `tau20_v025_x04` | 14 | 3.056e-9 | accept |
| `tau24_v025_x04` | 14 | 2.676e-10 | accept |
| `tau16_v010_x04` | 10 | 1.716e-4 | reject |
| `tau16_v015_x04` | 10 | 8.879e-6 | accept |
| `tau16_v040_x04` | 14 | 3.529e-10 | accept |
| `tau16_v050_x04` | 14 | 1.937e-10 | accept |
| `tau12_v025_x02` | 10 | 6.079e-7 | accept |
| `tau12_v025_x06` | 11 | 8.532e-7 | accept |
| `tau12_v025_x08` | 10 | 9.459e-4 | reject |
| `tau20_v025_x02` | 14 | 7.610e-10 | accept |
| `tau20_v025_x06` | 11 | 4.691e-9 | accept |
| `tau20_v025_x08` | 14 | 2.704e-6 | accept |

The subsequent threshold audit lies inside this intrinsic uncertainty estimate
at all 16 points.  This supports the estimator, but does not define it.  The
factor-of-two reserve is important: a single Shanks-to-Shanks movement is an
asymptotic scale estimate, not an alternating-series bound.

The three points not independently certified are:

- `tau8_v025_x04` agrees well with threshold quadrature, but its intrinsic
  estimate is `8.58e-5`; it therefore remains marginal at a `5e-5` standard.
- `tau16_v010_x04` is strongly anisotropic.  Although `tau2=16`, its broad
  edge has decay only `20.106` and the small-momentum expansion is too shallow.
- `tau12_v025_x08` has decay `37.699`, but `x=0.8` is closer to the nearest
  external-momentum singularity and its Watson coefficients grow earlier.
  The same `x=0.8` point passes at `tau2=20`.

Thus a gate based only on `tau2` or `max|q|` is insufficient.  The minimum
edge decay, external energy, and the accelerated-series self step must all be
used.

## Why sequence acceleration matters

The unaccelerated smallest-term Watson truncation passes only 7 of the 16
comparisons.  The expansion is alternating in the calibrated cusp, so the
three-point Shanks transform removes most of the leading terminant.  For the
standard `x=0.4` ray at `tau2=8`, the raw degree-ten truncation is off by
`5.51e-3`, whereas Laplace-Shanks is off by only `5.42e-6`.

The acceleration does not hide the bad regions: the intrinsic estimate rejects
them.  It also conservatively withholds certification at `tau2=8`, even though
the independent threshold audit is favorable there.

## Legacy saved threshold point

The old file `adaptive_momentum_point_audit_x04_cusp.json` stores

`-3.534372849771206e-5`

at `tau2=8`, while a fresh threshold sequence gives

`Q8  = -3.538092920722207e-5`,

`Q10 = -3.538092921185529e-5`,

`Q12 = -3.538092921189686e-5`.

Laplace-Shanks gives `-3.538112095142794e-5`.  It differs from the fresh
threshold endpoint by `5.42e-6`, but from the legacy saved value by
`1.058e-3` (0.1058%).

The legacy value is not reproducible with the current code.  Repeating the
threshold calculation with both the exact Barnes-G Upsilon implementation
and the older integral Upsilon backend gives the fresh value, so this is not
caused by the choice of special-function backend.  The legacy JSON lacks a
code hash, and the remaining likely source is an earlier two-point block or
coordinate implementation.  It should not be used as the threshold reference
for the new amplitude campaign.

## Recommended production use

1. Evaluate the Laplace sequence through degree fourteen and construct the
   Shanks sequence.
2. Select the smallest accelerated terminant from degrees 6--14 and require
   `2 * terminant + block envelope + coefficient stability < 5e-5`.
3. Also require both necklace multipliers to be small.  The empirical scan
   supports the standard `x<=0.6` ray from `tau2=8` onward, but the accepted
   region should be defined pointwise through the self step rather than a
   hard global `tau2` threshold.
4. At `x=0.8` or on an anisotropic edge with decay near 20, retain direct
   threshold/correlated quadrature unless the Laplace self test passes.
5. Use threshold quadrature afterward as a blind external audit, never as the
   rule for accepting the Laplace result.

This is an independently evaluable asymptotic uncertainty estimate, not yet a
rigorous inequality.  A proof-level bound would additionally require a Cauchy
supremum for the omitted Taylor series, an analytic Gaussian-tail/DOZZ bound,
and a global bound on the descendant-block remainder.

The machine-readable results are in
`results/genus1_two_point_worldsheet/threshold_vs_laplace_multi_point_v1/`.
