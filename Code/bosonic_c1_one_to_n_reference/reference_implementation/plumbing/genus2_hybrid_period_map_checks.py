#!/usr/bin/env python3
"""Focused checks for the adaptive holomorphic/Schottky region split."""

from __future__ import annotations

import cmath
import math

try:
    from genus2_hybrid_period_map import (
        HOLOMORPHIC_ALGORITHM,
        MULTIPRECISION_HOLOMORPHIC_ALGORITHM,
        SCHOTTKY_ALGORITHM,
        HybridPeriodMapConfig,
        InvalidPlumbingGeometry,
        classify_period_map_region,
        hybrid_period_matrix,
        plumbing_nonoverlap_slacks,
        plumbing_geometry,
        refine_multiprecision_holomorphic_inverse,
        refine_schottky_inverse,
    )
    from plumbing_algorithms import plumbing_genus2_period_matrix
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus2_hybrid_period_map import (
        HOLOMORPHIC_ALGORITHM,
        MULTIPRECISION_HOLOMORPHIC_ALGORITHM,
        SCHOTTKY_ALGORITHM,
        HybridPeriodMapConfig,
        InvalidPlumbingGeometry,
        classify_period_map_region,
        hybrid_period_matrix,
        plumbing_nonoverlap_slacks,
        plumbing_geometry,
        refine_multiprecision_holomorphic_inverse,
        refine_schottky_inverse,
    )
    from plumbing.plumbing_algorithms import plumbing_genus2_period_matrix


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_checks() -> None:
    config = HybridPeriodMapConfig(
        tolerance=1.0e-6,
        agreement_tolerance=1.0e-6,
        maximum_collocation_basis=72,
        maximum_schottky_word=9,
    )

    # The split has no scalar-q gap on valid charts.
    cases = (
        ("theta", (0.05 + 0.0j, 0.04 + 0.0j, 1.0e-13 + 0.0j), "schottky-all-small"),
        ("theta", (0.14 + 0.0j, 0.13 + 0.0j, 0.14 + 0.0j), "two-method-overlap"),
        ("glasses", (0.19 + 0.0j, 0.18 + 0.0j, 0.19 + 0.0j), "two-method-overlap"),
        ("theta", (0.20 + 0.0j, 0.18 + 0.0j, 0.12 + 0.0j), "holomorphic-bulk"),
    )
    for topology, q, expected_region in cases:
        region, geometry = classify_period_map_region(
            topology,
            q,
            config=config,
        )
        require(region == expected_region, f"{topology} region split changed: {region}")
        require(geometry.valid, f"{topology} representative has invalid geometry")
        result = hybrid_period_matrix(topology, q, config=config)
        require(math.isfinite(result.error_estimate), "hybrid error estimate is nonfinite")
        require(result.error_estimate <= config.tolerance, "hybrid method missed its bar")

    deep = hybrid_period_matrix("theta", cases[0][1], config=config)
    require(deep.algorithm == SCHOTTKY_ALGORITHM, "deep cusp did not select Schottky")
    require(
        deep.schottky is not None and deep.schottky.used_multiprecision,
        "deep cusp did not preserve its small coordinate with multiprecision",
    )
    require(
        deep.schottky.high_order >= config.minimum_schottky_word + 1,
        "Schottky convergence was accepted without two successive word steps",
    )

    # Regression for final-table glasses cusps: a nonseparating handle may
    # have Im(tau) well above the old 110 ceiling, so its q underflows in
    # binary64.  The inverse must retain the logarithm and must not clip the
    # seed or pass a literal zero multiplier to the Schottky map.
    deep_glasses_logs = (
        cmath.log(0.10 + 0.0j),
        -850.0 + 0.2j,
        cmath.log(0.08 + 0.0j),
    )
    deep_glasses_q = (
        cmath.exp(deep_glasses_logs[0]),
        cmath.exp(-690.0 + 0.2j),
        cmath.exp(deep_glasses_logs[2]),
    )
    deep_glasses_config = HybridPeriodMapConfig(
        tolerance=2.0e-8,
        agreement_tolerance=2.0e-8,
        minimum_schottky_word=5,
        maximum_schottky_word=8,
    )
    deep_glasses = hybrid_period_matrix(
        "glasses",
        deep_glasses_q,
        config=deep_glasses_config,
        log_q_values=deep_glasses_logs,
    )
    deep_glasses_inverse = refine_schottky_inverse(
        "glasses",
        deep_glasses.omega,
        deep_glasses_q,
        initial_log_q=deep_glasses_logs,
        config=deep_glasses_config,
        fit_word_length=6,
        max_nfev=2,
    )
    require(
        deep_glasses_inverse.success
        and deep_glasses_inverse.residual <= deep_glasses_config.tolerance,
        "deep glasses logarithmic inverse failed its final-table regression",
    )
    require(
        abs(deep_glasses_inverse.log_q[1] - deep_glasses_logs[1]) < 1.0e-10,
        "deep glasses logarithmic handle was clipped or underflowed",
    )

    # When every q is very small, binary64 word increments are dominated by
    # accumulated roundoff rather than the physical product tail.  This region
    # must promote to the short multiprecision sum even when no individual q
    # crosses the older one-small-q threshold.
    tiny_overlap = hybrid_period_matrix(
        "theta",
        (1.0e-5 + 0.0j, 2.0e-5 + 0.0j, 3.0e-5 + 0.0j),
        config=config,
    )
    require(
        tiny_overlap.schottky is not None
        and tiny_overlap.schottky.used_multiprecision
        and tiny_overlap.schottky.converged,
        "all-tiny Schottky sum was left at its binary64 roundoff floor",
    )
    require(
        tiny_overlap.overlap_residual is not None
        and tiny_overlap.overlap_residual <= config.agreement_tolerance,
        "all-tiny multiprecision Schottky sum missed the holomorphic overlap bar",
    )

    # Exhausting the preferred all-small word budget must promote to the
    # rescaled holomorphic solver, never to an uncertified mixed word sum.
    forced_fallback = hybrid_period_matrix(
        "theta",
        cases[0][1],
        config=HybridPeriodMapConfig(
            tolerance=1.0e-6,
            agreement_tolerance=1.0e-6,
            minimum_schottky_word=4,
            maximum_schottky_word=4,
        ),
    )
    require(
        forced_fallback.schottky is not None
        and not forced_fallback.schottky.converged
        and forced_fallback.holomorphic is not None
        and forced_fallback.holomorphic.converged
        and forced_fallback.algorithm == MULTIPRECISION_HOLOMORPHIC_ALGORITHM,
        "an all-small Schottky work-ceiling failure did not use the MP fallback",
    )

    # One-small-q mixed cusps are never treated as Schottky reference data.
    mixed_cases = (
        ("theta", (1.0e-13 + 0.0j, 0.10 + 0.0j, 0.16 + 0.0j)),
        ("glasses", (1.0e-13 + 0.0j, 0.10 + 0.0j, 0.21 + 0.0j)),
    )
    for topology, transition_q in mixed_cases:
        transition = hybrid_period_matrix(topology, transition_q, config=config)
        require(
            transition.region == "holomorphic-mixed-cusp",
            f"{topology} mixed cusp was not classified for holomorphic forms",
        )
        require(
            transition.holomorphic is not None
            and transition.holomorphic.used_multiprecision
            and transition.schottky is None
            and transition.algorithm == MULTIPRECISION_HOLOMORPHIC_ALGORITHM,
            f"{topology} mixed cusp used Schottky or skipped multiprecision",
        )
        require(
            transition.error_estimate <= config.tolerance,
            f"{topology} transition did not choose a method within the bar",
        )
        inverse = refine_multiprecision_holomorphic_inverse(
            topology,
            transition.omega,
            transition_q,
            config=config,
            max_nfev=2,
        )
        require(
            inverse.success and inverse.residual <= config.tolerance,
            f"{topology} mixed-cusp holomorphic inverse did not certify",
        )
        try:
            refine_schottky_inverse(
                topology, transition.omega, transition_q, config=config, max_nfev=2
            )
        except Exception as exc:
            require(
                "outside the all-small-q region" in str(exc),
                f"{topology} rejected mixed Schottky inverse for the wrong reason",
            )
        else:
            raise AssertionError(f"{topology} mixed cusp entered the Schottky inverse")

    # The central overlap explicitly evaluates both representations and checks
    # their period matrices modulo integral B-period shifts.
    for topology, q in ((cases[1][0], cases[1][1]), (cases[2][0], cases[2][1])):
        overlap = hybrid_period_matrix(topology, q, config=config)
        require(overlap.holomorphic is not None, "overlap omitted holomorphic forms")
        require(overlap.schottky is not None, "overlap omitted Schottky")
        require(overlap.holomorphic.converged, "overlap holomorphic solve did not converge")
        require(overlap.schottky.converged, "overlap Schottky sum did not converge")
        require(
            overlap.overlap_residual is not None
            and overlap.overlap_residual <= config.agreement_tolerance,
            "the two overlap period matrices exceed the agreement bar",
        )

    for topology, complex_overlap_q in (
        (
            "theta",
            (
                0.13 * cmath.exp(0.3j),
                0.14 * cmath.exp(-1.2j),
                0.14 * cmath.exp(2.0j),
            ),
        ),
        (
            "glasses",
            (
                0.18 * cmath.exp(0.3j),
                0.19 * cmath.exp(-1.2j),
                0.19 * cmath.exp(2.0j),
            ),
        ),
    ):
        overlap = hybrid_period_matrix(topology, complex_overlap_q, config=config)
        require(
            overlap.overlap_residual is not None
            and overlap.overlap_residual <= config.agreement_tolerance,
            f"{topology} complex-phase overlap exceeded the agreement bar",
        )

    bulk = hybrid_period_matrix("theta", cases[3][1], config=config)
    require(bulk.algorithm == HOLOMORPHIC_ALGORITHM, "bulk did not select holomorphic forms")

    # A large-q point is not accepted merely because a linear system can be
    # solved: the canonical coordinate disks must be disjoint.
    invalid_q = (0.30 + 0.0j, 0.30 + 0.0j, 0.30 + 0.0j)
    invalid = plumbing_geometry("theta", invalid_q)
    require(not invalid.valid, "overlapping equal-q theta disks were accepted")
    invalid_slacks = plumbing_nonoverlap_slacks("theta", invalid_q)
    require(
        len(invalid_slacks) == 3 and min(invalid_slacks) < 0.0,
        "theta q-only predicate omitted a pairwise disk overlap",
    )
    try:
        hybrid_period_matrix("theta", invalid_q, config=config)
    except InvalidPlumbingGeometry:
        pass
    else:
        raise AssertionError("invalid plumbing geometry did not fail loudly")

    # Optimizing the seam split strictly enlarges the glasses domain.  Here
    # sqrt(c) overlaps each handle disk, while the exact condition
    # c < ((1-a)/(1+a)) ((1-b)/(1+b)) is satisfied.
    asymmetric_glasses_q = (0.25 + 0.0j, 0.25 + 0.0j, 0.30 + 0.0j)
    asymmetric_glasses = plumbing_geometry("glasses", asymmetric_glasses_q)
    asymmetric_glasses_slacks = plumbing_nonoverlap_slacks(
        "glasses", asymmetric_glasses_q
    )
    require(
        min(asymmetric_glasses_slacks) > 0.0 and asymmetric_glasses.valid,
        "a glasses chart requiring asymmetric seam radii was rejected",
    )
    old_symmetric_radius = math.sqrt(0.25)
    old_bridge_radius = math.sqrt(0.30)
    require(
        1.0 - old_symmetric_radius - old_bridge_radius < 0.0,
        "the asymmetric glasses regression point no longer separates the tests",
    )
    for sphere_margins in asymmetric_glasses.sphere_margins:
        require(
            min(sphere_margins) > 0.0,
            "constructed glasses seam radii do not realize the analytic domain",
        )

    glasses_boundary = ((1.0 - 0.25) / (1.0 + 0.25)) ** 2
    require(
        plumbing_geometry(
            "glasses", (0.25 + 0.0j, 0.25 + 0.0j, (glasses_boundary - 1.0e-8) + 0.0j)
        ).valid,
        "glasses point immediately inside the exact boundary was rejected",
    )
    require(
        not plumbing_geometry(
            "glasses", (0.25 + 0.0j, 0.25 + 0.0j, (glasses_boundary + 1.0e-8) + 0.0j)
        ).valid,
        "glasses point immediately outside the exact boundary was accepted",
    )

    # The public auto entry point must use the same split.
    _, bulk_method = plumbing_genus2_period_matrix("theta", *cases[3][1], algorithm="auto")
    _, cusp_method = plumbing_genus2_period_matrix("theta", *cases[0][1], algorithm="auto")
    _, mixed_method = plumbing_genus2_period_matrix(
        "theta", *mixed_cases[0][1], algorithm="auto"
    )
    require(bulk_method == "holomorphic_form_collocation", "public bulk dispatch regressed")
    require(cusp_method == "schottky_series", "public cusp dispatch regressed")
    require(
        mixed_method == "holomorphic_form_collocation",
        "public mixed-cusp dispatch incorrectly reported Schottky",
    )

    print("genus2 hybrid period-map checks passed")


if __name__ == "__main__":
    run_checks()
