# Sphere 1->5 imaginary-ray workflow

The production calculation uses only the residue-free chamber
`omega = i*t`, `0 < t < 1/3`. The worldsheet scan and its numerical audit
contain no matrix-model target.

The auditable order is:

1. `worldsheet_convergent_scan_16point_local.json` is checked against
   `worldsheet_freeze_manifest_16point_local.json`.
2. `sphere_six_point_imaginary_ray_fit.py` extracts the points-only table
   `worldsheet_imaginary_ray_points_frozen.json` and fits the explicit cubic
   ansatz without importing any target coefficient.
3. The resulting `worldsheet_imaginary_ray_cubic_fit_frozen.json` is frozen
   together with its source hash.
4. Only then does `sphere_six_point_imaginary_ray_fit_comparison.py` load the
   matrix-model polynomial, verify the source hash, and write
   `matrix_model_fit_comparison_16point_local.json` and the comparison figure.

The primary generalized least-squares fit treats scrambled-Sobol QMC errors
as independent and the single numerical-discretization estimate as one fully
correlated additive nuisance across all 16 values of `t`. The direct physical
`i-epsilon` problem is not part of this analytic-continuation workflow.
