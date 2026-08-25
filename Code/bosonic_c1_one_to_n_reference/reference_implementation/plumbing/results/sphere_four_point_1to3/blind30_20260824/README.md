# Sphere 1->3 blind 30-point extension

This run adds 14 new worldsheet points to the existing frozen 16-point scan,
for 30 distinct points in the residue-free chamber `0 < t < 1/2` on
`omega = i t`.

## Design and blinding

The known scan uses `t = 0.16, 0.18, ..., 0.46`.  The extension interlaces
the first fourteen gaps at

```text
0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29,
0.31, 0.33, 0.35, 0.37, 0.39, 0.41, 0.43
```

The unused midpoint nearest the first residue wall is `t = 0.45`.  Every
known point is retained.  The block order, momentum quadrature, cutoff,
Sobol depth, and replicate count are unchanged from the known scan.  The
extension starts at seed `20276968`, the disjoint continuation of the known
seed stream after 16 points.

The merger and fit programs contain no matrix-model coefficient.  They write
and hash the combined worldsheet table and target-free affine fit before the
separate comparison program evaluates the matrix model.

## Frozen artifacts

- `worldsheet_extension_14point.json`: 14 new direct worldsheet integrals,
  SHA-256 `eaa6a8f131922efed3cdd6029fa83ec98fc9c50ed43c5b7a68a139428e5fa87f`.
- `worldsheet_scan_30point.json`: merged 30-point worldsheet table,
  SHA-256 `ab785a09a9d57c68d1e84f12c7c738e83cfaae5fa9b4d7021afd2e4ac1e5a3b9`.
- `worldsheet_affine_fit_30point_frozen.json`: target-free affine fit,
  SHA-256 `c0acd1b43a9858c1187967b0ced75f6a383761b68a60fd9ff260e22be5fb274a`.
- `matrix_comparison_30point.json`: post-freeze pointwise and coefficient
  comparison.
- `amplitude_comparison_30point.png`: known/new/deep-audit comparison plot.

The adjacent manifest files carry the hashes and frozen statuses used by the
pipeline.

## Fits and post-freeze comparison

The primary unweighted fit to all 30 production points is

```text
Q3(i t) = (0.999773665 +/- 0.000087153)
          + (-2.999429293 +/- 0.000274480) t,
RMS residual = 1.26548e-4.
```

The 14 new points alone give

```text
Q3(i t) = (0.999958349 +/- 0.000075626)
          + (-2.999978571 +/- 0.000243450) t,
RMS residual = 6.79918e-5.
```

After replacing the five previously audited known points by their independent
deep worldsheet evaluations, the 30-point sensitivity fit is

```text
Q3(i t) = (0.999900263 +/- 0.000050097)
          + (-2.999776439 +/- 0.000157776) t,
RMS residual = 7.27419e-5.
```

The matrix-model result evaluated only after these fits were frozen is
`Q3(omega) = 1 + 3 i omega`, or coefficients `(1, -3)` on `omega = i t`.
The new cohort has QMC-only `chi^2 = 15.8519` for 14 points, maximum absolute
pull `2.3828`, and RMS Q3 difference `7.6593e-5`.  Combining the new points
with the inherited deep replacements and conservative errors gives
`chi^2 = 35.5552` for 30 points and the same maximum pull `2.3828`.

For transparency, the raw production-replicate statistic is `66.1818/30`
with maximum pull `5.1609`.  This is inherited from the known `t = 0.32`
replicate error, whose undercoverage was already established by the separate
deep audit; it is not caused by the 14-point extension.

## Reproduction

From the repository root, using the project virtual environment:

```bash
MPLCONFIGDIR=/private/tmp/stringmc-mpl .venv/bin/python \
  plumbing/sphere_four_point_worldsheet_scan.py \
  --t 0.17 0.19 0.21 0.23 0.25 0.27 0.29 0.31 0.33 0.35 0.37 0.39 0.41 0.43 \
  --seed 20276968 \
  --output plumbing/results/sphere_four_point_1to3/blind30_20260824/worldsheet_extension_14point.json \
  --freeze-manifest plumbing/results/sphere_four_point_1to3/blind30_20260824/worldsheet_extension_14point_frozen.json

.venv/bin/python plumbing/merge_sphere_four_point_worldsheet_scans.py
.venv/bin/python plumbing/sphere_four_point_imaginary_ray_fit.py
MPLCONFIGDIR=/private/tmp/stringmc-mpl .venv/bin/python \
  plumbing/sphere_four_point_30point_matrix_comparison.py
```
