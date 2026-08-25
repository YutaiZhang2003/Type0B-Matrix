#!/usr/bin/env python3
"""Fast checks for the physical-measure genus-two RQMC mixture."""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import qmc

try:
    from genus2_integrand_normalization import GENUS2_GENERIC_STACK_WEIGHT
    from genus2_moduli_physical_mixture_rqmc import (
        DEFAULT_COMPONENTS,
        PhysicalProposalComponent,
        equal_mixture_log_density,
        estimate_physical_mixture_integral,
        generate_physical_mixture_design,
        physical_mixture_contribution_diagnostics,
        physical_mixture_proposals_from_unit_cube,
    )
    from genus2_siegel_fundamental_domain import (
        SIEGEL_VOLUME_G2,
        minkowski_proposals_from_unit_cube,
    )
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus2_integrand_normalization import GENUS2_GENERIC_STACK_WEIGHT
    from plumbing.genus2_moduli_physical_mixture_rqmc import (
        DEFAULT_COMPONENTS,
        PhysicalProposalComponent,
        equal_mixture_log_density,
        estimate_physical_mixture_integral,
        generate_physical_mixture_design,
        physical_mixture_contribution_diagnostics,
        physical_mixture_proposals_from_unit_cube,
    )
    from plumbing.genus2_siegel_fundamental_domain import (
        SIEGEL_VOLUME_G2,
        minkowski_proposals_from_unit_cube,
    )


def run_checks() -> None:
    points = qmc.Sobol(d=6, scramble=True, seed=8102).random_base2(m=8)
    bulk = PhysicalProposalComponent("bulk", 3.0, 2.0)
    old_omega, old_invariant_weight, old_coordinates = (
        minkowski_proposals_from_unit_cube(points)
    )
    new_omega, physical_weight, new_coordinates, log_mix, log_weight = (
        physical_mixture_proposals_from_unit_cube(
            points,
            component=bulk,
            components=(bulk,),
        )
    )
    if not np.allclose(new_omega, old_omega, rtol=0.0, atol=2.0e-15):
        raise AssertionError("bulk physical proposal changed the Minkowski map")
    if not np.allclose(new_coordinates, old_coordinates, rtol=0.0, atol=2.0e-15):
        raise AssertionError("bulk physical proposal changed the cusp coordinates")
    determinant = np.linalg.det(new_omega.imag)
    if not np.allclose(
        physical_weight / determinant**3,
        old_invariant_weight,
        rtol=2.0e-14,
        atol=0.0,
    ):
        raise AssertionError("physical and invariant Radon--Nikodym factors disagree")
    if not np.allclose(np.log(physical_weight), log_weight, rtol=0.0, atol=2.0e-14):
        raise AssertionError("saved physical log weights are inconsistent")
    expected_log_density = (
        math.log(6.0) - 3.0 * new_coordinates[:, 0] - 2.0 * new_coordinates[:, 1]
    )
    if not np.allclose(log_mix, expected_log_density, rtol=0.0, atol=2.0e-14):
        raise AssertionError("single-component mixture density is incorrect")

    t1 = np.asarray([0.0, 0.5, 3.0])
    t3 = np.asarray([0.0, 2.0, 5.0])
    observed = np.exp(equal_mixture_log_density(t1, t3, DEFAULT_COMPONENTS))
    expected = np.mean(
        [
            component.rate_t1
            * component.rate_t3
            * np.exp(-component.rate_t1 * t1 - component.rate_t3 * t3)
            for component in DEFAULT_COMPONENTS
        ],
        axis=0,
    )
    if not np.allclose(observed, expected, rtol=2.0e-15, atol=0.0):
        raise AssertionError("equal mixture density is not normalized componentwise")

    rows, summaries = generate_physical_mixture_design(
        replicate_count=8,
        power=7,
        base_seed=93011,
    )
    if any(int(row["rqmc_kernel_det_im_power"]) != 0 for row in rows):
        raise AssertionError("physical design requested a det(Y) kernel multiplier")
    if any(
        not math.isclose(
            float(row["rqmc_stack_integration_weight"]),
            GENUS2_GENERIC_STACK_WEIGHT
            * float(row["rqmc_physical_measure_weight"])
            / int(row["rqmc_proposal_count"]),
            rel_tol=2.0e-15,
        )
        for row in rows
    ):
        raise AssertionError("physical stack integration weights are inconsistent")
    control_values = [float(row["det_im_omega"]) ** -3 for row in rows]
    estimate = estimate_physical_mixture_integral(rows, control_values)
    expected_control = GENUS2_GENERIC_STACK_WEIGHT * SIEGEL_VOLUME_G2
    if abs(estimate.estimate - expected_control) > 6.0 * estimate.scramble_standard_error:
        raise AssertionError("physical mixture misses the invariant-volume control")
    summary_controls = [summary.invariant_volume_control for summary in summaries]
    if not math.isclose(
        2.0 * estimate.estimate,
        float(np.mean(summary_controls)),
        rel_tol=2.0e-15,
    ):
        raise AssertionError("row estimator and replicate normalization controls disagree")

    diagnostics = physical_mixture_contribution_diagnostics(rows, control_values)
    component_total = sum(
        float(entry["estimate"])
        for entry in diagnostics["component_contributions"]
    )
    if not math.isclose(component_total, estimate.estimate, rel_tol=2.0e-15):
        raise AssertionError("component diagnostics do not sum to the full estimate")
    if len(diagnostics["replicate_concentration"]) != 8:
        raise AssertionError("contribution diagnostics lost independent scrambles")

    print("genus2_moduli_physical_mixture_rqmc checks passed")


if __name__ == "__main__":
    run_checks()
