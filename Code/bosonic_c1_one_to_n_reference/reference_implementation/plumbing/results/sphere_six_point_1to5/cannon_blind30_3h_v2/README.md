# Sphere 1->5 blind 30-point boundary-fix campaign

This is the corrected successor to the failed Cannon array `41514317`.  The
worldsheet numerical settings, 30 imaginary-energy points, normalization, and
matrix-model blinding are unchanged from `cannon_blind30_3h_v1`.

## Failure and correction

The first array aborted when floating-point reconstruction of an otherwise
interior moduli point produced an exactly zero plumbing coordinate in the
newly selected channel.  The original nonzero plumbing coordinates of the
Sobol proposal were still available and defined a convergent chart.

Code version `sphere_six_point_cannon_blind_v2_source_chart_recovery` therefore
evaluates such a point in its exact originating proposal chart.  It does not
drop, replace, jitter, or reweight the Sobol point, and the sample denominator
is unchanged.  Every shard records `source_chart_recovery_count` and
`source_chart_recovery_fraction`; the assembled point record aggregates the
same diagnostic.

Regression checks cover exact deep-collar proposal channels, forced recovery
with a synthetic kernel, synthetic assembly/freezing, and a real low-order
Liouville-kernel evaluation through the recovery path.

## Submission gate

Full-length validation job `41528350` runs production task zero with all
`2^15` Sobol samples.  Remaining array job `41528358` (tasks 1--449) has an
`afterok` dependency on that validation.  Assembly `41528369` and freeze
`41528381` are chained by `afterok` dependencies.  A validation failure thus
prevents the production campaign from starting.

No matrix-model comparison code or target formula is included in the staged
worker snapshot.  Comparison remains forbidden until the blind freeze gate
succeeds.
