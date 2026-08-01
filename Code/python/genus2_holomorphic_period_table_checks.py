#!/usr/bin/env python3
"""Checks for the holomorphic reference table and calibrated hybrid policy."""

from __future__ import annotations

import cmath

import numpy as np

try:
    from genus2_holomorphic_period_table import (
        HOLOMORPHIC_PERIOD_ALGORITHM,
        HolomorphicPeriodMapTable,
        HolomorphicPeriodTableEntry,
        SchottkyValidityCell,
        SchottkyValidityEnvelope,
    )
    from genus2_calibrated_schottky import (
        UncertifiedSchottkyRegion,
        calibrated_schottky_period_from_q,
    )
    from genus2_hybrid_period_map import (
        SCHOTTKY_ALGORITHM,
        HybridPeriodMapConfig,
        hybrid_period_matrix,
    )
    from genus2_plumbing_atlas import LeadingMarking, build_plumbing_atlas, certify_marking
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus2_holomorphic_period_table import (
        HOLOMORPHIC_PERIOD_ALGORITHM,
        HolomorphicPeriodMapTable,
        HolomorphicPeriodTableEntry,
        SchottkyValidityCell,
        SchottkyValidityEnvelope,
    )
    from plumbing.genus2_calibrated_schottky import (
        UncertifiedSchottkyRegion,
        calibrated_schottky_period_from_q,
    )
    from plumbing.genus2_hybrid_period_map import (
        SCHOTTKY_ALGORITHM,
        HybridPeriodMapConfig,
        hybrid_period_matrix,
    )
    from plumbing.genus2_plumbing_atlas import (
        LeadingMarking,
        build_plumbing_atlas,
        certify_marking,
    )


SOURCE_OMEGA = np.asarray(
    [
        [
            -0.1012234827396521 + 1.061716378007721j,
            -0.3081905897275983 + 0.30474669363908585j,
        ],
        [
            -0.3081905897275983 + 0.30474669363908585j,
            0.464080342598051 + 1.310295156194303j,
        ],
    ],
    dtype=np.complex128,
)
CHART_OMEGA = SOURCE_OMEGA[::-1, ::-1].copy()
CHART_Q = (
    0.0002149836568291 - 0.001794679718100j,
    0.00007842302587057 + 0.009695814555419j,
    -0.05239034333940 - 0.1373822582533j,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _entry(*, row_id: str = "example", geometry_margin: float = 0.1):
    return HolomorphicPeriodTableEntry(
        row_id=row_id,
        topology="theta",
        q=CHART_Q,
        omega=CHART_OMEGA,
        basis_order=24,
        samples_per_seam=96,
        basis_stability=1.0e-10,
        seam_residual=1.0e-11,
        symmetry_error=1.0e-12,
        geometry_margin=geometry_margin,
        period_algorithm=HOLOMORPHIC_PERIOD_ALGORITHM,
        certified=True,
    )


def run_checks() -> None:
    table = HolomorphicPeriodMapTable((_entry(),))
    shifted = CHART_OMEGA + np.asarray([[2, -1], [-1, 3]], dtype=int)
    seeds = table.nearest_seeds("theta", shifted, count=2)
    _require(len(seeds) == 1, "the certified table row was not returned")
    _require(seeds[0].table_distance is not None and seeds[0].table_distance < 1.0e-14, "integral period branches were not removed in table lookup")

    try:
        HolomorphicPeriodTableEntry(
            **{
                **_entry().__dict__,
                "period_algorithm": "schottky-series",
            }
        )
    except ValueError:
        pass
    else:  # pragma: no cover - required failure path
        raise AssertionError("a Schottky period-map row entered the production table")

    cell = SchottkyValidityCell(
        cell_id="small-q-theta-example",
        topology="theta",
        center_q=CHART_Q,
        log_abs_radius=(0.1, 0.1, 0.1),
        phase_radius=(0.1, 0.1, 0.1),
        word_length=7,
        validation_point_count=64,
        boundary_point_count=48,
        interior_point_count=16,
        reference_table_sha256="0" * 64,
        max_reference_residual=2.0e-7,
        max_word_stability=1.0e-7,
        max_symmetry_error=1.0e-9,
        min_geometry_margin=0.02,
        safety_factor=2.0,
    )
    envelope = SchottkyValidityEnvelope((cell,), minimum_validation_points=32)
    certificate = envelope.certificate("theta", CHART_Q, tolerance=5.0e-7)
    _require(certificate is not None, "an in-cell calibrated Schottky query was rejected")
    _require(
        envelope.certificate("theta", CHART_Q, tolerance=3.0e-7) is None,
        "a Schottky cell was used below its calibrated error bound",
    )
    outside_q = (CHART_Q[0] * 2.0, CHART_Q[1], CHART_Q[2])
    _require(
        envelope.certificate("theta", outside_q, tolerance=5.0e-7) is None,
        "a query outside the calibrated cell was accepted",
    )
    try:
        calibrated_schottky_period_from_q(
            "theta",
            outside_q,
            envelope=envelope,
            tolerance=5.0e-7,
        )
    except UncertifiedSchottkyRegion:
        pass
    else:
        raise AssertionError("the guarded Schottky evaluator bypassed its envelope")

    atlas = build_plumbing_atlas(
        SOURCE_OMEGA,
        search_depth=3,
        prefilter_count=2,
        max_nfev=80,
        period_table=table,
        table_seed_count=1,
        include_leading_seed=False,
    )
    usable = [
        chart
        for chart in atlas.charts
        if chart.status in {"reference-q-envelope", "requires-recursion-order-study"}
    ]
    _require(bool(usable), "the table seed did not reach a certified inverse")
    _require(
        all(
            chart.period_algorithm
            in {HOLOMORPHIC_PERIOD_ALGORITHM, SCHOTTKY_ALGORITHM}
            for chart in usable
        ),
        "an uncertified period algorithm entered a usable chart",
    )
    _require(
        all(
            chart.period_map_region != "two-method-overlap"
            or (
                chart.period_overlap_residual is not None
                and chart.period_overlap_residual <= 2.0e-6
            )
            for chart in usable
        ),
        "an overlap chart entered the atlas without a two-method agreement bar",
    )
    _require(
        usable[0].inverse_seed_source.startswith("holomorphic-period-table:"),
        "the table seed provenance was not recorded",
    )

    deep_q = (0.03 + 0.0j, 1.0e-13 + 0.0j, 0.02 + 0.0j)
    deep_logs = tuple(cmath.log(value) for value in deep_q)
    deep_omega = hybrid_period_matrix(
        "theta",
        deep_q,
        config=HybridPeriodMapConfig(tolerance=2.0e-6),
    ).omega
    deep_marking = LeadingMarking(
        topology="theta",
        word="synthetic-deep-cusp",
        matrix=tuple(tuple(int(value) for value in row) for row in np.eye(4, dtype=int)),
        omega=deep_omega,
        leading_q=deep_q,
        leading_q_abs=tuple(abs(value) for value in deep_q),
        leading_q_max=max(abs(value) for value in deep_q),
        leading_q_spread=max(abs(value) for value in deep_q) / min(abs(value) for value in deep_q),
        leading_log_q=deep_logs,
    )
    covered = certify_marking(deep_marking.omega, deep_marking, max_nfev=30)
    _require(covered.inverse_success, "the adaptive hybrid map left a deep-cusp hole")
    _require(
        covered.period_algorithm == "adaptive-schottky",
        "deep-cusp atlas certification did not select adaptive Schottky",
    )
    _require(covered.period_max_residual < 2.0e-6, "deep-cusp period residual is too large")

    print("genus2 calibrated period-map table checks passed")


if __name__ == "__main__":
    run_checks()
