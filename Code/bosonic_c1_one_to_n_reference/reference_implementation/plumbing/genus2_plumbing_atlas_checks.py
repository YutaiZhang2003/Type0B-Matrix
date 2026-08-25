#!/usr/bin/env python3
"""Checks for the genus-two mapping-class plumbing atlas."""

from __future__ import annotations

import cmath
import json
from pathlib import Path

import numpy as np

try:
    import genus2_plumbing_atlas as atlas_module
    from bolza_ccy_recursion import bolza_period_matrix
    from genus2_plumbing_atlas import (
        LeadingMarking,
        best_leading_score,
        build_plumbing_atlas,
        leading_q_for_topology,
        shortlist_markings,
        symplectic_matrix_csv_fields,
        symplectic_matrix_from_csv_row,
    )
    from genus2_holomorphic_period_table import PeriodMapSeed
    from genus2_hybrid_period_map import HybridPeriodMapConfig, evaluate_schottky_period_map
    from liouville_genus2 import parse_complex
    from liouville_genus2_modular_check import named_transform
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing import genus2_plumbing_atlas as atlas_module
    from plumbing.bolza_ccy_recursion import bolza_period_matrix
    from plumbing.genus2_plumbing_atlas import (
        LeadingMarking,
        best_leading_score,
        build_plumbing_atlas,
        leading_q_for_topology,
        shortlist_markings,
        symplectic_matrix_csv_fields,
        symplectic_matrix_from_csv_row,
    )
    from plumbing.genus2_holomorphic_period_table import PeriodMapSeed
    from plumbing.genus2_hybrid_period_map import (
        HybridPeriodMapConfig,
        evaluate_schottky_period_map,
    )
    from plumbing.liouville_genus2 import parse_complex
    from plumbing.liouville_genus2_modular_check import named_transform


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _saved_overlap_omega() -> np.ndarray:
    path = Path("plumbing/results/theta_glasses_period_precision_w8_radial_q01500.json")
    payload = json.loads(path.read_text())
    return np.asarray(
        [[parse_complex(value) for value in row] for row in payload["omega_glasses"]],
        dtype=np.complex128,
    )


def check_leading_formulae() -> None:
    omega = np.asarray([[0.1 + 1.2j, 0.03 + 0.04j], [0.03 + 0.04j, -0.2 + 1.4j]])
    theta = leading_q_for_topology(omega, "theta")
    glasses = leading_q_for_topology(omega, "glasses")
    two_pi_i = 2.0j * np.pi
    expected_theta = (
        np.exp(two_pi_i * (omega[0, 0] - omega[0, 1])),
        np.exp(two_pi_i * (omega[1, 1] - omega[0, 1])),
        np.exp(two_pi_i * omega[0, 1]),
    )
    expected_glasses = (
        np.exp(two_pi_i * omega[0, 0]),
        np.exp(two_pi_i * omega[1, 1]),
        -two_pi_i * omega[0, 1],
    )
    _require(max(abs(a - b) for a, b in zip(theta, expected_theta)) < 1.0e-14, "theta leading map changed")
    _require(max(abs(a - b) for a, b in zip(glasses, expected_glasses)) < 1.0e-14, "glasses leading map changed")


def check_exact_marking_csv_roundtrip() -> None:
    matrix = np.asarray(
        [[1, 0, 2, -1], [0, 1, -1, 3], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.int64,
    )
    fields = symplectic_matrix_csv_fields(matrix)
    recovered = symplectic_matrix_from_csv_row(fields)
    _require(recovered is not None and np.array_equal(recovered, matrix), "exact marking CSV roundtrip failed")
    _require(symplectic_matrix_from_csv_row({"symplectic_word": "I"}) is None, "legacy row unexpectedly acquired a matrix")


def check_atlas_exposes_no_schottky_period_backend() -> None:
    _require(
        not hasattr(atlas_module, "_forward_period"),
        "the production atlas still exposes its former Schottky forward map",
    )
    _require(
        not hasattr(atlas_module, "schottky_period_matrix_cross_ratio"),
        "the production atlas still imports the Schottky period map",
    )


def check_theta_spanning_tree_conditioning_tiebreak() -> None:
    omega = np.asarray(
        [
            [0.4898584885522723 + 1.6440365350592538j, 0.09998014569282532 + 0.19149106251731626j],
            [0.09998014569282532 + 0.19149106251731626j, -0.28084141202270985 + 8.89806917387571j],
        ],
        dtype=np.complex128,
    )
    candidates = shortlist_markings(omega, "theta", search_depth=3, count=6)
    _require(bool(candidates), "conditioning tie-break produced no theta candidates")
    best = candidates[0]
    _require(
        best.leading_q_abs[2] > 0.29,
        "theta tie-break did not place the largest equal-score edge on the spanning tree",
    )


def check_leading_score_under_exact_marking_action() -> None:
    omega = _saved_overlap_omega()
    swapped = named_transform("swap-handles").transform_omega(omega)
    for topology in ("theta", "glasses"):
        original_score = best_leading_score(omega, topology, search_depth=2)
        swapped_score = best_leading_score(swapped, topology, search_depth=2)
        _require(
            abs(original_score - swapped_score) < 1.0e-12,
            f"{topology} finite-depth score changed under handle swap",
        )


def check_bolza_prefers_theta() -> None:
    result = build_plumbing_atlas(
        bolza_period_matrix(),
        search_depth=2,
        prefilter_count=2,
        word_length=4,
        period_tolerance=1.0e-5,
        stability_tolerance=1.0e-5,
    )
    _require(result.best_topology == "theta", "Bolza should select the theta topology")
    _require(result.best_q_max is not None and result.best_q_max < 0.06, "Bolza theta q score regressed")
    _require(
        result.coverage_status == "period-chart-inside-reference-q-envelope",
        "Bolza left the reference q envelope",
    )
    best = result.charts[0]
    _require(
        best.period_algorithm == "holomorphic-form-collocation"
        and best.period_map_region == "two-method-overlap"
        and best.period_overlap_residual is not None,
        "Bolza should carry the two-method all-small certificate",
    )
    _require(best.period_max_residual < 1.0e-8, "Bolza period inverse is inaccurate")
    _require(best.period_map_stability < 1.0e-5, "Bolza period map is unstable")
    _require(
        best.inverse_seed_source == "leading-plumbing-formula",
        "Bolza did not record its non-Schottky seed provenance",
    )


def check_table_first_deep_cusp_correction() -> None:
    q = (
        1.0e-8 * cmath.exp(0.1j),
        2.0e-9 * cmath.exp(-0.2j),
        1.0e-11 * cmath.exp(0.3j),
    )
    logs = tuple(cmath.log(value) for value in q)
    forward = evaluate_schottky_period_map(
        "theta",
        q,
        config=HybridPeriodMapConfig(
            tolerance=1.0e-9,
            agreement_tolerance=1.0e-9,
            minimum_schottky_word=2,
            maximum_schottky_word=5,
            crosscheck_overlap=False,
        ),
        log_q_values=logs,
    )
    perturbed_logs = (
        logs[0] + 2.0e-4 - 1.0e-4j,
        logs[1] - 1.0e-4 + 1.0e-4j,
        logs[2] + 1.0e-4,
    )
    seed = PeriodMapSeed(
        q=tuple(cmath.exp(value) for value in perturbed_logs),  # type: ignore[arg-type]
        log_q=perturbed_logs,
        source="synthetic-table-seed",
    )
    identity = tuple(tuple(int(row == column) for column in range(4)) for row in range(4))
    marking = LeadingMarking(
        topology="theta",
        word="I",
        matrix=identity,
        omega=forward.omega,
        leading_q=q,
        leading_q_abs=tuple(abs(value) for value in q),
        leading_q_max=max(abs(value) for value in q),
        leading_q_spread=max(abs(value) for value in q) / min(abs(value) for value in q),
        leading_log_q=logs,
        table_seed=seed,
    )
    candidate = atlas_module._table_first_schottky_candidate_from_seed(
        marking,
        seed,
        maximum_word=5,
        maximum_corrections=2,
        period_tolerance=1.0e-9,
        stability_tolerance=1.0e-9,
    )
    _require(candidate.success, "table-first deep-cusp correction did not certify")
    _require(candidate.nfev == 2, "table-first path did not use exactly one correction")
    _require(candidate.residual < 1.0e-9, "corrected table seed missed the target period")
    _require(candidate.high_order <= 5, "table-first certificate exceeded its word ceiling")


def check_fundamental_table_markings_skip_non_riemann_seed() -> None:
    """One numerically unusable nearby table marking is nonfatal."""

    class Item:
        def __init__(self, row_id: int, omega_marked: np.ndarray) -> None:
            self.row_id = row_id
            self.omega_marked = omega_marked
            self.matrix_fund_to_raw = tuple(
                tuple(int(row == column) for column in range(4))
                for row in range(4)
            )
            q = (1.0e-4 + 0.0j, 2.0e-4 + 0.0j, 3.0e-4 + 0.0j)
            self.seed = PeriodMapSeed(
                q=q,
                log_q=tuple(cmath.log(value) for value in q),
                source=f"synthetic-table-seed-{row_id}",
            )

    class Table:
        has_fundamental_index = True

        def nearest_fundamental_seeds(
            self,
            topology: str,
            omega: np.ndarray,
            *,
            count: int,
        ) -> list[Item]:
            del topology, omega, count
            invalid = np.asarray([[1.0j, 0.0j], [0.0j, -1.0j]])
            valid = np.asarray([[1.2j, 0.05j], [0.05j, 1.4j]])
            return [Item(1, invalid), Item(2, valid)]

    source = np.asarray([[1.1j, 0.02j], [0.02j, 1.3j]])
    markings = atlas_module.fundamental_table_markings(
        source,
        "glasses",
        period_table=Table(),
        count=2,
    )
    _require(len(markings) == 1, "invalid table marking was not skipped")
    _require(
        markings[0].word == "fundamental-table:2",
        "valid table marking did not survive after an invalid neighbour",
    )


def check_overlap_selects_glasses_and_retains_theta() -> None:
    result = build_plumbing_atlas(
        _saved_overlap_omega(),
        search_depth=3,
        prefilter_count=3,
        word_length=5,
        period_tolerance=3.0e-6,
        stability_tolerance=3.0e-6,
    )
    _require(result.best_topology == "glasses", "central overlap should select glasses by max |q|")
    _require(result.best_q_max is not None and abs(result.best_q_max - 0.15) < 1.0e-4, "glasses q inverse regressed")
    best_glasses = result.charts[0]
    _require(
        best_glasses.period_map_region == "two-method-overlap"
        and best_glasses.period_overlap_residual is not None
        and best_glasses.period_overlap_residual <= 3.0e-6,
        "the selected overlap chart lacks a holomorphic/Schottky agreement certificate",
    )
    theta = [chart for chart in result.charts if chart.topology == "theta" and chart.inverse_success]
    _require(bool(theta), "no finite-q theta image survived at the overlap point")
    best_theta = min(theta, key=lambda chart: chart.q_max)
    _require(
        best_theta.period_algorithm == "holomorphic-form-collocation",
        "bulk theta overlap should use the direct holomorphic-form period map",
    )
    _require(abs(best_theta.q_max - 0.1602337) < 2.0e-5, "theta overlap q inverse regressed")
    _require(best_theta.period_max_residual < 1.0e-6, "theta overlap period residual is too large")
    _require(
        best_theta.status == "requires-recursion-order-study",
        "q=0.16023 should not be silently placed inside the reference envelope",
    )


def check_hybrid_recertification_overrides_seed_backend_failure() -> None:
    """A passing final hybrid certificate controls the chart status."""

    omega = np.asarray(
        [
            [
                0.04008441526447393 + 1.5714702974786687j,
                -0.4449759751971737 + 0.37037908137377046j,
            ],
            [
                -0.4449759751971737 + 0.37037908137377046j,
                0.06618895617535536 + 2.029373442567302j,
            ],
        ],
        dtype=np.complex128,
    )
    result = build_plumbing_atlas(
        omega,
        search_depth=3,
        prefilter_count=2,
        word_length=4,
        max_nfev=120,
        q_reference_max=0.16,
        period_tolerance=5.0e-6,
        stability_tolerance=5.0e-6,
        stop_at_reference=True,
    )
    _require(
        result.coverage_status == "period-chart-inside-reference-q-envelope",
        "a passing final hybrid certificate inherited a failed seed-backend status",
    )
    best = result.charts[0]
    _require(best.inverse_success, "the final hybrid recertification did not promote success")
    _require(best.period_max_residual < 1.0e-8, "recertified period residual is too large")
    _require(best.period_map_stability < 1.0e-8, "recertified period map is unstable")


def check_saved_hard_band_refinement() -> None:
    path = Path(
        "plumbing/results/genus2_plumbing_atlas/"
        "symmetric_imaginary_hard_band_depth4_w6.json"
    )
    payload = json.loads(path.read_text())
    _require(len(payload["refinements"]) == 2, "hard-band refinement should contain two samples")
    expected_q = (0.2643279177, 0.2469205743)
    for item, expected in zip(payload["refinements"], expected_q):
        atlas = item["refined_atlas"]
        _require(
            atlas["coverage_status"] == "period-chart-found-but-block-order-unvalidated",
            "hard-band point did not resolve to a valid period chart",
        )
        _require(atlas["best_topology"] == "theta", "hard-band refinement changed topology")
        _require(abs(float(atlas["best_q_max"]) - expected) < 2.0e-8, "hard-band q score regressed")
        best = atlas["charts"][0]
        _require(best["period_max_residual"] < 1.0e-8, "hard-band inverse residual is too large")
        _require(best["forward_word_stability"] < 5.0e-6, "hard-band period series is unstable")


def run_checks() -> None:
    check_leading_formulae()
    check_exact_marking_csv_roundtrip()
    check_atlas_exposes_no_schottky_period_backend()
    check_theta_spanning_tree_conditioning_tiebreak()
    check_leading_score_under_exact_marking_action()
    check_table_first_deep_cusp_correction()
    check_fundamental_table_markings_skip_non_riemann_seed()
    check_bolza_prefers_theta()
    check_overlap_selects_glasses_and_retains_theta()
    check_hybrid_recertification_overrides_seed_backend_failure()
    check_saved_hard_band_refinement()
    print("genus2_plumbing_atlas checks passed")


if __name__ == "__main__":
    run_checks()
