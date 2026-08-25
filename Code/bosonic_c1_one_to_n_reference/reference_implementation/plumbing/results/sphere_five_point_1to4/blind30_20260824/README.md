# Sphere 1->4 blind 30-point extension

This campaign preserves all 18 points in the frozen imaginary-ray table and
adds 12 direct worldsheet evaluations, giving exactly 30 stored data points.
The historical `t=0.49` point remains the declared near-second-wall
diagnostic, so the primary quadratic fit uses 29 points through `t=0.48`.

## Extension design

The new points are

```text
0.19, 0.21, 0.23, 0.27, 0.29, 0.31,
0.33, 0.35, 0.37, 0.39, 0.44, 0.47
```

They interlace the broad gaps in the historical design while retaining its
special probes at `t=0.249` and `t=0.251`.  The exact structure-constant zero
at `t=0.25`, the first contour wall at `t=0.4`, and the second wall at
`t=0.5` are not sampled.  The new randomizations use disjoint seed blocks
starting at `20277001`.

All points use momentum order 20, `Pmax=6`, the threshold-adapted
`P=6 u^(5/4)` rule, four scrambled Sobol replicates, radial power `0.15`, and
the independently checked c-recursion implementation.  Points through
`t=0.33` use block order 6 and Sobol power 10.  Points from `t=0.35` use block
order 8 and Sobol power 9; the crossed residue is active only at `t=0.44` and
`t=0.47`.

## Separation and frozen hashes

The extension, merge, and fit programs contain no matrix-model coefficient.
The downstream comparison was run only after these worldsheet artifacts were
written and hash-frozen.

- `worldsheet_extension_12point.json`:
  `be9af08cbd871f727ef6068167fb27467f18428013eaa6a9b9d2fee8e03c584c`
- `worldsheet_scan_30point.json`:
  `63e8df62c1301110f4131b846dc2908a18a62500564f5a0a891dd90d01f360d1`
- `worldsheet_points_30point_frozen.json`:
  `611d5f1f75d7fdd0240aaa134803b656f6a157c3928a91cca0e6f4cc85c00a12`
- `worldsheet_quadratic_fit_30point_frozen.json`:
  `5e165ec33cdbd721af8ce5550212315eb41a07323bad42b7bd83ae7f51e61907`

## Target-free fits

The primary unweighted 29-point fit is

```text
Q4(i t) = (2.004567422 +/- 0.001687358)
          + (-12.026247155 +/- 0.010762637) t
          + (16.032868550 +/- 0.016272573) t^2,
RMS residual = 6.03810e-4.
```

The 12 new points alone give

```text
Q4(i t) = (2.002855335 +/- 0.003580984)
          + (-12.017837392 +/- 0.022820237) t
          + (16.023397886 +/- 0.034654193) t^2,
RMS residual = 7.47249e-4.
```

The primary fit roots are `0.25001494` and `0.50008459`.

## Downstream matrix-model comparison

The separately evaluated target is

```text
Q4(i t) = 2 - 12 t + 16 t^2,
mu^3 A_tree(i t) = -4 t^5 Q4(i t).
```

For the 12 new points, the QMC-only pointwise statistic is `8.0124/12`, with
maximum absolute pull `1.8177`.  After the representative block-order audits
described below, the conservative statistic is `4.9972/12`, with maximum pull
`1.4166`.  The complete 29-point primary table gives `42.3489/29` using QMC
errors and `39.3337/29` conservatively.  Its largest pull, `4.8562`, is the
inherited historical `t=0.18` point rather than a new datum.

## New-point convergence audits

Independent block-order checks were run at a real-contour interior point
(`t=0.31`), the block-order transition region (`t=0.35`), and a
residue-corrected point (`t=0.44`).  Their production-to-audit spreads are
`6.46e-5`, `4.00e-3`, and `1.63e-3`, respectively.  The latter two dominate
the local QMC replicate errors and are therefore carried as conservative
uncertainties in the final comparison plot.  The audit was a post-comparison
convergence sensitivity and does not alter the frozen primary fit.

The frozen audit hash is
`3b3a966dcf215237baae09d5014f47dff735dc2e28c1bb67c68309e8b6a73ce5`.

The downstream numerical comparison is in
`matrix_comparison_30point.json`; the replacement plot is
`amplitude_comparison_30point.png`.
