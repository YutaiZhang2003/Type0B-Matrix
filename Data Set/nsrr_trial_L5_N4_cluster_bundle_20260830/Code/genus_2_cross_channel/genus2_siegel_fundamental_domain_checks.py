#!/usr/bin/env python3
"""Self-checks for the genus-two fundamental-domain sampler."""

from __future__ import annotations

import math

import numpy as np

try:
    from genus2_siegel_fundamental_domain import (
        GOTTSCHLING_SHIFTS,
        INVARIANT_WEIGHT_MAX,
        SIEGEL_VOLUME_G2,
        draw_minkowski_proposals,
        estimate_invariant_volume,
        gottschling_min_margin,
        in_gottschling_domain,
        minkowski_proposals_from_unit_cube,
        sample_invariant_domain,
    )
except ImportError:  # pragma: no cover
    from plumbing.genus2_siegel_fundamental_domain import (
        GOTTSCHLING_SHIFTS,
        INVARIANT_WEIGHT_MAX,
        SIEGEL_VOLUME_G2,
        draw_minkowski_proposals,
        estimate_invariant_volume,
        gottschling_min_margin,
        in_gottschling_domain,
        minkowski_proposals_from_unit_cube,
        sample_invariant_domain,
    )


def run_checks() -> None:
    if len(GOTTSCHLING_SHIFTS) != 15:
        raise AssertionError("Gottschling determinant list must contain 15 shifts")
    keys = {tuple(int(value) for value in shift.ravel()) for shift in GOTTSCHLING_SHIFTS}
    if len(keys) != 15:
        raise AssertionError("Gottschling determinant shifts are not unique")

    interior = 1.0j * np.asarray([[1.2, 0.2], [0.2, 1.5]])
    if not in_gottschling_domain(interior):
        raise AssertionError(f"known interior point was rejected: margin={gottschling_min_margin(interior)}")
    outside_real_box = interior.copy()
    outside_real_box[0, 1] += 0.51
    outside_real_box[1, 0] = outside_real_box[0, 1]
    if in_gottschling_domain(outside_real_box):
        raise AssertionError("real-box violation was accepted")
    outside_minkowski = 1.0j * np.asarray([[1.2, 0.7], [0.7, 1.5]])
    if in_gottschling_domain(outside_minkowski):
        raise AssertionError("Minkowski-cone violation was accepted")

    rng = np.random.default_rng(1138)
    omega, weight, _ = draw_minkowski_proposals(rng, 20_000)
    if np.max(weight) > INVARIANT_WEIGHT_MAX * (1.0 + 1.0e-13):
        raise AssertionError("analytic accept/reject envelope failed")
    vector = np.asarray(in_gottschling_domain(omega[:50]))
    scalar = np.asarray([in_gottschling_domain(value) for value in omega[:50]])
    if not np.array_equal(vector, scalar):
        raise AssertionError("scalar and vectorized domain tests disagree")

    unit = rng.random((20_000, 6))
    cube_omega, cube_weight, cube_coordinates = minkowski_proposals_from_unit_cube(unit)
    if cube_omega.shape != (20_000, 2, 2) or cube_coordinates.shape != (20_000, 3):
        raise AssertionError("unit-cube proposal map returned the wrong shape")
    if np.max(cube_weight) > INVARIANT_WEIGHT_MAX * (1.0 + 1.0e-13):
        raise AssertionError("unit-cube proposal map violates the analytic weight bound")
    expected_t1_mean = 1.0 / 3.0
    expected_t3_mean = 1.0 / 2.0
    if abs(float(np.mean(cube_coordinates[:, 0])) - expected_t1_mean) > 0.015:
        raise AssertionError("unit-cube t1 inverse-CDF map is inconsistent")
    if abs(float(np.mean(cube_coordinates[:, 1])) - expected_t3_mean) > 0.02:
        raise AssertionError("unit-cube t3 inverse-CDF map is inconsistent")

    sample = sample_invariant_domain(256, seed=9917, batch_size=512)
    if sample.omega.shape != (256, 2, 2):
        raise AssertionError("invariant sampler returned the wrong shape")
    if not np.all(in_gottschling_domain(sample.omega)):
        raise AssertionError("invariant sampler returned a point outside the domain")

    estimate = estimate_invariant_volume(proposal_count=250_000, seed=1701)
    if abs(estimate.importance_z_score) > 5.0:
        raise AssertionError(
            "importance estimate does not reproduce pi^3/270: "
            f"{estimate.importance_estimate} versus {SIEGEL_VOLUME_G2}"
        )
    if abs(estimate.rejection_z_score) > 5.0:
        raise AssertionError(
            "rejection estimate does not reproduce pi^3/270: "
            f"{estimate.rejection_estimate} versus {SIEGEL_VOLUME_G2}"
        )
    if not math.isclose(estimate.analytic_weight_bound, INVARIANT_WEIGHT_MAX):
        raise AssertionError("reported analytic envelope changed")

    print("genus2_siegel_fundamental_domain checks passed")
    print(
        f"  volume={estimate.importance_estimate:.9f} +/- "
        f"{estimate.importance_standard_error:.2g}; exact={SIEGEL_VOLUME_G2:.9f}"
    )


if __name__ == "__main__":
    run_checks()
