#!/usr/bin/env python3
"""Algebraic checks for the paired genus-two RQMC radius sweep."""

from __future__ import annotations

import math
import numpy as np

try:
    from genus2_integrand_normalization import GENUS2_GENERIC_STACK_WEIGHT
    from genus2_moduli_physical_mixture_rqmc import (
        PHYSICAL_MIXTURE_SAMPLING_SCHEME,
        generate_physical_mixture_design,
    )
    from genus2_siegel_fundamental_domain import SIEGEL_VOLUME_G2
    from reweight_genus2_c1_rqmc_radius import (
        _estimate_view,
        _period_certificate_follows_routing_policy,
        paired_shape_from_replicates,
    )
except ImportError:  # pragma: no cover
    from plumbing.genus2_integrand_normalization import GENUS2_GENERIC_STACK_WEIGHT
    from plumbing.genus2_moduli_physical_mixture_rqmc import (
        PHYSICAL_MIXTURE_SAMPLING_SCHEME,
        generate_physical_mixture_design,
    )
    from plumbing.genus2_siegel_fundamental_domain import SIEGEL_VOLUME_G2
    from plumbing.reweight_genus2_c1_rqmc_radius import (
        _estimate_view,
        _period_certificate_follows_routing_policy,
        paired_shape_from_replicates,
    )


def run_checks() -> None:
    common_period_fields = {
        "period_validation_tolerance": "1e-6",
        "period_final_residual": "2e-8",
        "period_final_map_step": "3e-9",
    }
    mixed_holomorphic_certificate = {
        **common_period_fields,
        "atlas_period_algorithm": "adaptive-schottky",
        "period_algorithm": "holomorphic-form-collocation",
        "period_map_region": "two-method-overlap",
        "period_certified_error_bound": "",
    }
    if not _period_certificate_follows_routing_policy(
        mixed_holomorphic_certificate
    ):
        raise AssertionError("a valid holomorphic certificate rejected its Schottky atlas seed")

    promoted_holomorphic_certificate = {
        **mixed_holomorphic_certificate,
        "period_final_map_step": "2.2e-6",
    }
    if not _period_certificate_follows_routing_policy(
        promoted_holomorphic_certificate
    ):
        raise AssertionError("a certificate inside the promoted 1e-5 production bar was rejected")

    mixed_schottky_certificate = {
        **common_period_fields,
        "atlas_period_algorithm": "holomorphic-form-collocation",
        "period_algorithm": "adaptive-schottky",
        "period_map_region": "two-method-overlap",
        "period_certified_error_bound": "4e-8",
    }
    if not _period_certificate_follows_routing_policy(mixed_schottky_certificate):
        raise AssertionError("a certified Schottky result rejected its holomorphic atlas seed")

    invalid_schottky_region = {
        **mixed_schottky_certificate,
        "period_map_region": "holomorphic-bulk",
    }
    if _period_certificate_follows_routing_policy(invalid_schottky_region):
        raise AssertionError("a Schottky certificate outside its routed region was accepted")

    loose_holomorphic_certificate = {
        **mixed_holomorphic_certificate,
        "period_final_residual": "2e-5",
    }
    if _period_certificate_follows_routing_policy(loose_holomorphic_certificate):
        raise AssertionError("a holomorphic certificate outside its numerical bar was accepted")

    radius_one = [1.0, 2.0, 3.5, 4.0, 6.0, 7.5, 8.0, 11.0]
    exact_shape = 1.75
    radius = [exact_shape * value for value in radius_one]
    result = paired_shape_from_replicates(
        radius,
        radius_one,
    )
    if not math.isclose(result[0], exact_shape, rel_tol=1.0e-15):
        raise AssertionError("paired RQMC ratio of means is incorrect")
    if result[1] > 2.0e-15:
        raise AssertionError("an exact paired shape acquired a jackknife error")

    physical_rows, _summaries = generate_physical_mixture_design(
        replicate_count=4,
        power=6,
        base_seed=99017,
    )
    control = np.asarray(
        [float(row["det_im_omega"]) ** -3 for row in physical_rows],
        dtype=float,
    )
    view = _estimate_view(
        physical_rows,
        control,
        PHYSICAL_MIXTURE_SAMPLING_SCHEME,
    )
    expected_control = GENUS2_GENERIC_STACK_WEIGHT * SIEGEL_VOLUME_G2
    if abs(float(view["estimate"]) - expected_control) > (
        6.0 * float(view["standard_error"])
    ):
        raise AssertionError("radius-sweep adapter misweighted physical rows")
    if not math.isnan(float(view["volume_calibrated_estimate"])):
        raise AssertionError("physical radius sweep invented a volume calibration")

    print("reweight_genus2_c1_rqmc_radius checks passed")


if __name__ == "__main__":
    run_checks()
