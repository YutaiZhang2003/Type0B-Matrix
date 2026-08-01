#!/usr/bin/env python3
"""Pilot Monte Carlo integration of the genus-two c=1 string density.

The input period matrices are iid samples from the invariant Siegel measure
on the coarse Gottschling domain,

    p(Omega) d^6 Omega = d^6 Omega / (V_2 det(Im Omega)^3).

For the final ``g_s^2``-stripped c=1 kernel
``K_2=2 I_2^Xi/pi`` at ``alpha'=1``, the orbifold estimator is

    F_2/g_s^2 = (V_2 / 2) E_p[det(Im Omega)^3 K_2(Omega)].

The factor 1/2 is the generic genus-two stack weight.  The kernel includes the
string-note identity ``(-i)(-1)(-8i)/(8 pi)=1/pi`` and the physical c=1
sphere-topology factor two; it does not include ``g_s^2``.  Each node is evaluated
in an atlas-selected theta or glasses plumbing frame.  The full noncompact
scalar plumbing partition and the Liouville partition are paired in that same
frame, so their Weyl factor cancels in Z_L / Z_X^25.  The dimensionless
plumbing scalar is converted together with the critical 26-boson seed and the
compact target zero mode to Xi's ``dk_I/(2 pi)`` convention.  At
``alpha'=1`` this correlated conversion multiplies the former dimensionless
kernel by ``1/(2 pi)``.

This driver records low/high CFT and scalar truncations independently from the
Monte Carlo standard error.  A small run is a pilot, not a precision claim:
degeneration strata can generate heavy-tailed contributions.

The driver also accepts the direct-importance scrambled-Sobol designs emitted
by ``genus2_moduli_rqmc.py``.  In that case every in-domain node carries the
weight ``w/(2 M)`` and independent complete scrambles, rather than individual
nodes, determine the sampling error.

The physical-measure mixture emitted by
``genus2_moduli_physical_mixture_rqmc.py`` is also accepted.  Its node weight
already converts the mixture density to ``d^3 X d^3 Y``.  For that design the
local observable is ``K_2`` itself: no ``det(Im Omega)^3`` multiplier is used.

In particular, det(Im Omega)^3 is not part of the local ghost-matter density.
It is the Radon--Nikodym factor converting the invariant proposal measure back
to the period-coordinate volume d^3 X d^3 Y.  The local CFT constant is fixed
independently by lower-genus sewing; det(Im Omega)^3 has no role in it.
"""

from __future__ import annotations

import argparse
import cmath
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    from audit_q_to_omega_accuracy import PeriodMapValidation, validate_or_refine_period_map
    from bolza_torus_plumbing_reach import transform_omega
    from conformal_frame_labels import GLASSES_PLUMBING_FRAME, THETA_PLUMBING_FRAME
    from free_boson_plumbing import (
        glasses_free_boson_product,
        noncompact_scalar_zero_mode_factor,
        theta_free_boson_product,
        torus_raw_oscillator_abs,
    )
    from genus2_c1_string_integrand import (
        COMPACT_THETA_IMPLEMENTATION,
        DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION,
        SameFrameMatterPartitions,
        XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION,
        genus2_c1_string_integrand_density,
    )
    from genus2_integrand_normalization import (
        FULL_CFT_FACTORIZATION_CERTIFIED,
        GENUS2_GENERIC_STACK_WEIGHT,
        LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION,
        PRE_SPHERE_XI_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        c1_genus2_topology_correction,
        c1_sphere_normalized_genus2_kernel_multiplier,
        critical_prefactor_ratio_to_dhp,
        string_note_genus2_kernel_multiplier,
        xi_full_replacement_over_dimensionless,
        xi_genus2_scalar_over_dimensionless,
    )
    from genus2_plumbing_atlas import (
        MULTIPRECISION_FALLBACK_Q_MAX,
        PlumbingChartResult,
        best_leading_score,
        build_plumbing_atlas,
        symplectic_matrix_csv_fields,
        symplectic_matrix_from_csv_row,
        table_first_all_small_chart,
        table_first_mixed_cusp_chart,
    )
    from genus2_holomorphic_period_table import SchottkyValidityEnvelope
    from genus2_hybrid_period_map import MULTIPRECISION_HOLOMORPHIC_ALGORITHM
    from genus2_moduli_rqmc import estimate_rqmc_integral
    from genus2_moduli_physical_mixture_rqmc import (
        PHYSICAL_MIXTURE_SAMPLING_SCHEME,
        estimate_physical_mixture_integral,
        physical_mixture_contribution_diagnostics,
    )
    from genus2_period_table import Genus2PeriodMapTable
    from genus2_siegel_fundamental_domain import SIEGEL_VOLUME_G2, in_gottschling_domain
    from liouville_genus2_ccy import liouville_genus2_ccy_partition
    from liouville_genus2_glasses import liouville_genus2_glasses_partition
    from plumbing_algorithms import theta_cusp_surviving_multipliers
except ImportError:  # pragma: no cover
    from plumbing.audit_q_to_omega_accuracy import (
        PeriodMapValidation,
        validate_or_refine_period_map,
    )
    from plumbing.bolza_torus_plumbing_reach import transform_omega
    from plumbing.conformal_frame_labels import GLASSES_PLUMBING_FRAME, THETA_PLUMBING_FRAME
    from plumbing.free_boson_plumbing import (
        glasses_free_boson_product,
        noncompact_scalar_zero_mode_factor,
        theta_free_boson_product,
        torus_raw_oscillator_abs,
    )
    from plumbing.genus2_c1_string_integrand import (
        COMPACT_THETA_IMPLEMENTATION,
        DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION,
        SameFrameMatterPartitions,
        XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION,
        genus2_c1_string_integrand_density,
    )
    from plumbing.genus2_integrand_normalization import (
        FULL_CFT_FACTORIZATION_CERTIFIED,
        GENUS2_GENERIC_STACK_WEIGHT,
        LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION,
        PRE_SPHERE_XI_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        c1_genus2_topology_correction,
        c1_sphere_normalized_genus2_kernel_multiplier,
        critical_prefactor_ratio_to_dhp,
        string_note_genus2_kernel_multiplier,
        xi_full_replacement_over_dimensionless,
        xi_genus2_scalar_over_dimensionless,
    )
    from plumbing.genus2_plumbing_atlas import (
        MULTIPRECISION_FALLBACK_Q_MAX,
        PlumbingChartResult,
        best_leading_score,
        build_plumbing_atlas,
        symplectic_matrix_csv_fields,
        symplectic_matrix_from_csv_row,
        table_first_all_small_chart,
        table_first_mixed_cusp_chart,
    )
    from plumbing.genus2_holomorphic_period_table import SchottkyValidityEnvelope
    from plumbing.genus2_hybrid_period_map import MULTIPRECISION_HOLOMORPHIC_ALGORITHM
    from plumbing.genus2_moduli_rqmc import estimate_rqmc_integral
    from plumbing.genus2_moduli_physical_mixture_rqmc import (
        PHYSICAL_MIXTURE_SAMPLING_SCHEME,
        estimate_physical_mixture_integral,
        physical_mixture_contribution_diagnostics,
    )
    from plumbing.genus2_period_table import Genus2PeriodMapTable
    from plumbing.genus2_siegel_fundamental_domain import (
        SIEGEL_VOLUME_G2,
        in_gottschling_domain,
    )
    from plumbing.liouville_genus2_ccy import liouville_genus2_ccy_partition
    from plumbing.liouville_genus2_glasses import liouville_genus2_glasses_partition
    from plumbing.plumbing_algorithms import theta_cusp_surviving_multipliers


USABLE_CHART_STATUSES = {"reference-q-envelope", "requires-recursion-order-study"}
RQMC_SAMPLING_SCHEME = "scrambled_sobol_minkowski_importance"
RQMC_SAMPLING_SCHEMES = {
    RQMC_SAMPLING_SCHEME,
    PHYSICAL_MIXTURE_SAMPLING_SCHEME,
}
LEGACY_UNIT_MUMFORD_KERNEL_CONVENTION = "unit-mumford-residue-no-string-note-prefactor"


@dataclass(frozen=True)
class RescaledLiouvillePartition:
    """Liouville partition with the common ``|q|^(2 h_min)`` kept in logs."""

    scaled_partition: float
    log_threshold_factor: float

    @property
    def log_partition(self) -> float:
        return math.log(self.scaled_partition) + self.log_threshold_factor

    @property
    def raw_partition(self) -> float:
        try:
            return math.exp(self.log_partition)
        except OverflowError:
            return math.inf


def canonicalize_string_note_kernel_row(
    source_row: dict[str, object],
) -> dict[str, object]:
    """Return one saved row in the current string-note kernel convention.

    Rows written before the string-note convention was explicit stored the
    unit-Mumford transformed density.  The next generation used the
    dimensionless scalar loop measure, and the v2 generation used Xi's scalar
    measure but not the c=1 sphere-topology correction.  Their CFT content is
    unchanged, so every migration is an exact constant at ``alpha'=1`` and
    requires no CFT reevaluation.
    """

    row = dict(source_row)
    convention = str(row.get("integration_kernel_convention", "")).strip()
    if convention == STRING_NOTE_INTEGRATION_KERNEL_CONVENTION:
        return row
    if convention not in {
        "",
        LEGACY_UNIT_MUMFORD_KERNEL_CONVENTION,
        LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION,
        PRE_SPHERE_XI_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
    }:
        raise ValueError(f"unsupported integration-kernel convention {convention!r}")

    xi_scalar_multiplier = xi_full_replacement_over_dimensionless()
    pre_sphere_multiplier = string_note_genus2_kernel_multiplier()
    final_multiplier = c1_sphere_normalized_genus2_kernel_multiplier()
    topology_multiplier = c1_genus2_topology_correction()
    row_was_unit_mumford = convention in {"", LEGACY_UNIT_MUMFORD_KERNEL_CONVENTION}
    row_was_dimensionless = convention in {
        "",
        LEGACY_UNIT_MUMFORD_KERNEL_CONVENTION,
        LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION,
    }
    for order in ("low", "high"):
        key = f"transformed_integrand_{order}"
        if key not in row or str(row[key]).strip() == "":
            continue
        saved_value = float(row[key])
        if row_was_unit_mumford:
            unit_mumford_value = xi_scalar_multiplier * saved_value
        elif row_was_dimensionless:
            unit_mumford_value = (
                xi_scalar_multiplier * saved_value / pre_sphere_multiplier
            )
        else:
            unit_mumford_value = saved_value / pre_sphere_multiplier
        row[f"unit_mumford_transformed_integrand_{order}"] = unit_mumford_value
        row[key] = final_multiplier * unit_mumford_value
        log_key = f"factorized_log_density_{order}"
        if log_key in row and str(row[log_key]).strip() != "":
            if row_was_dimensionless:
                row[log_key] = float(row[log_key]) + math.log(xi_scalar_multiplier)
            row[f"string_note_kernel_log_density_{order}"] = float(
                row[log_key]
            ) + math.log(final_multiplier)
        density_key = f"factorized_density_{order}"
        if density_key in row and str(row[density_key]).strip() != "":
            if row_was_dimensionless:
                row[density_key] = xi_scalar_multiplier * float(row[density_key])
            row[f"string_note_kernel_density_{order}"] = (
                final_multiplier * float(row[density_key])
            )
    row["integration_kernel_convention"] = STRING_NOTE_INTEGRATION_KERNEL_CONVENTION
    row["string_note_kernel_multiplier"] = final_multiplier
    row["xi_scalar_over_dimensionless_full_integrand"] = xi_scalar_multiplier
    row["c1_sphere_topology_multiplier"] = topology_multiplier
    row["source_integration_kernel_convention"] = convention or LEGACY_UNIT_MUMFORD_KERNEL_CONVENTION
    row["legacy_kernel_converted"] = True
    return row


def omega_from_csv_row(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [
            [
                complex(float(row["x11"]), float(row["y11"])),
                complex(float(row["x12"]), float(row["y12"])),
            ],
            [
                complex(float(row["x12"]), float(row["y12"])),
                complex(float(row["x22"]), float(row["y22"])),
            ],
        ],
        dtype=np.complex128,
    )


def node_stack_integration_weight(source_row: dict[str, str]) -> float:
    """Return the coefficient multiplying one transformed node value."""

    scheme = source_row.get("sampling_scheme", "iid_invariant_domain")
    if scheme in RQMC_SAMPLING_SCHEMES:
        weight = float(source_row["rqmc_stack_integration_weight"])
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("RQMC stack integration weight must be positive and finite")
        return weight
    if scheme in {"", "iid", "iid_invariant_domain"}:
        return GENUS2_GENERIC_STACK_WEIGHT * SIEGEL_VOLUME_G2
    raise ValueError(f"unsupported sampling scheme {scheme!r}")


def kernel_det_im_power(source_row: dict[str, object]) -> int:
    """Return the proposal-dependent power multiplying the local kernel."""

    scheme = str(source_row.get("sampling_scheme", "iid_invariant_domain"))
    if scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME:
        power = int(source_row.get("rqmc_kernel_det_im_power", 0))
        if power != 0:
            raise ValueError("physical-mixture rows must use kernel det(Y) power zero")
        return 0
    if scheme in {"", "iid", "iid_invariant_domain", RQMC_SAMPLING_SCHEME}:
        return 3
    raise ValueError(f"unsupported sampling scheme {scheme!r}")


def sampling_scheme_for_rows(rows: Sequence[dict[str, object]]) -> str:
    """Return one validated sampling scheme for a collection of rows."""

    schemes = {
        str(row.get("sampling_scheme", "iid_invariant_domain"))
        for row in rows
    }
    schemes.discard("")
    if not schemes:
        return "iid_invariant_domain"
    if len(schemes) != 1:
        raise ValueError(f"mixed sampling schemes are not supported: {sorted(schemes)}")
    scheme = schemes.pop()
    if scheme == "iid":
        return "iid_invariant_domain"
    if scheme not in {"iid_invariant_domain", *RQMC_SAMPLING_SCHEMES}:
        raise ValueError(f"unsupported sampling scheme {scheme!r}")
    return scheme


def summarize_rqmc_rows(
    rows: Sequence[dict[str, object]],
    *,
    value_key: str,
) -> dict[str, object] | None:
    """Summarize only a complete collection of at least two scrambles."""

    if not rows:
        return None
    groups: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(int(row["rqmc_replicate"]), []).append(row)
    if len(groups) < 2:
        return None
    for replicate, group in groups.items():
        expected = int(group[0]["rqmc_domain_count"])
        if len(group) != expected or any(row.get("status") != "ok" for row in group):
            return None
        if any(int(row["rqmc_domain_count"]) != expected for row in group):
            raise ValueError(f"replicate {replicate} has inconsistent domain counts")

    scheme = sampling_scheme_for_rows(rows)
    values = [float(row[value_key]) for row in rows]
    if scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME:
        estimate = estimate_physical_mixture_integral(rows, values)
        return {
            "sampling_scheme": PHYSICAL_MIXTURE_SAMPLING_SCHEME,
            "replicate_count": estimate.replicate_count,
            "cft_node_count": estimate.cft_node_count,
            "estimate": estimate.estimate,
            "standard_error": estimate.scramble_standard_error,
            "replicate_estimates": list(estimate.replicate_estimates),
            "volume_calibrated_estimate": None,
            "volume_calibrated_standard_error": None,
            "volume_calibrated_replicate_estimates": None,
            "interpretation": (
                "The physical-measure balance-heuristic estimator is primary. "
                "Independent complete mixture scrambles determine its standard "
                "error; no det(Im Omega)^3 factor multiplies the local kernel."
            ),
        }
    estimate = estimate_rqmc_integral(rows, values)
    return {
        "sampling_scheme": RQMC_SAMPLING_SCHEME,
        "replicate_count": estimate.replicate_count,
        "cft_node_count": estimate.cft_node_count,
        "estimate": estimate.raw_estimate,
        "standard_error": estimate.raw_scramble_standard_error,
        "replicate_estimates": list(estimate.raw_replicate_estimates),
        "volume_calibrated_estimate": estimate.volume_calibrated_estimate,
        "volume_calibrated_standard_error": (
            estimate.volume_calibrated_scramble_standard_error
        ),
        "volume_calibrated_replicate_estimates": list(
            estimate.volume_calibrated_replicate_estimates
        ),
        "interpretation": (
            "The raw direct-importance estimator is primary. Independent complete "
            "scrambles determine its standard error. Exact-volume calibration is a "
            "finite-sample biased variance-reduction diagnostic."
        ),
    }


def _complex_string(value: complex) -> str:
    value = complex(value)
    return f"{value.real:+.16e}{value.imag:+.16e}j"


def _matrix_from_strings(values: Sequence[Sequence[str]]) -> np.ndarray:
    return np.asarray([[complex(value) for value in row] for row in values], dtype=np.complex128)


def validate_covariant_period_frames(
    omega_fundamental: np.ndarray,
    marking: Sequence[Sequence[int]],
    omega_plumbing: np.ndarray,
    *,
    tolerance: float = 2.0e-9,
) -> None:
    """Keep the modular-covariant kernel and invariant ratios in their frames."""

    fundamental = np.asarray(omega_fundamental, dtype=np.complex128)
    plumbing = np.asarray(omega_plumbing, dtype=np.complex128)
    if not bool(in_gottschling_domain(fundamental, tolerance=float(tolerance))):
        raise ValueError("the Igusa/measure period matrix is outside the Gottschling domain")
    expected_plumbing = transform_omega(np.asarray(marking, dtype=np.int64), fundamental)
    mismatch = float(np.max(np.abs(expected_plumbing - plumbing)))
    if mismatch > float(tolerance):
        raise ValueError(
            "the plumbing period matrix is not the saved Sp(4,Z) image of the "
            f"fundamental-domain matrix (mismatch={mismatch:.3e})"
        )


def _best_usable_chart(atlas: object) -> PlumbingChartResult | None:
    usable = [chart for chart in atlas.charts if chart.status in USABLE_CHART_STATUSES]
    return min(usable, key=lambda chart: (chart.q_max, chart.period_max_residual)) if usable else None


def select_plumbing_chart(
    omega: np.ndarray,
    *,
    q_reference_max: float,
    refine_above_q: float,
    base_search_depth: int,
    base_prefilter_count: int,
    base_word_length: int,
    base_period_tolerance: float,
    base_stability_tolerance: float,
    refined_search_depth: int,
    refined_prefilter_count: int,
    refined_word_length: int,
    refined_period_tolerance: float,
    refined_stability_tolerance: float,
    period_table: Genus2PeriodMapTable | None = None,
    table_first_period_tolerance: float | None = None,
    table_first_stability_tolerance: float | None = None,
    table_first_maximum_word: int = 5,
    table_first_maximum_corrections: int = 2,
) -> tuple[PlumbingChartResult, str]:
    fast_period_tolerance = (
        float(base_period_tolerance)
        if table_first_period_tolerance is None
        else float(table_first_period_tolerance)
    )
    fast_stability_tolerance = (
        float(base_stability_tolerance)
        if table_first_stability_tolerance is None
        else float(table_first_stability_tolerance)
    )
    table_chart = table_first_all_small_chart(
        omega,
        period_table=period_table,
        leading_search_depth=base_search_depth,
        leading_prefilter_count=max(2, base_prefilter_count),
        maximum_word=table_first_maximum_word,
        maximum_corrections=table_first_maximum_corrections,
        q_reference_max=q_reference_max,
        period_tolerance=fast_period_tolerance,
        stability_tolerance=fast_stability_tolerance,
    )
    if table_chart is not None:
        return table_chart, "direct-all-small"

    mixed_chart = table_first_mixed_cusp_chart(
        omega,
        period_table=period_table,
        leading_search_depth=base_search_depth,
        leading_prefilter_count=max(2, base_prefilter_count),
        maximum_candidates=max(8, 4 * base_prefilter_count),
        q_reference_max=q_reference_max,
        period_tolerance=fast_period_tolerance,
        stability_tolerance=fast_stability_tolerance,
    )
    if mixed_chart is not None:
        return mixed_chart, "direct-mixed-cusp"

    refined_first = False
    if refined_search_depth > base_search_depth:
        base_leading = min(
            best_leading_score(omega, topology, search_depth=base_search_depth)
            for topology in ("theta", "glasses")
        )
        refined_leading = min(
            best_leading_score(omega, topology, search_depth=refined_search_depth)
            for topology in ("theta", "glasses")
        )
        refined_first = bool(
            base_leading > MULTIPRECISION_FALLBACK_Q_MAX
            and refined_leading <= MULTIPRECISION_FALLBACK_Q_MAX
        )

    if refined_first:
        refined = build_plumbing_atlas(
            omega,
            search_depth=refined_search_depth,
            prefilter_count=refined_prefilter_count,
            word_length=refined_word_length,
            max_nfev=180,
            q_reference_max=q_reference_max,
            period_tolerance=refined_period_tolerance,
            stability_tolerance=refined_stability_tolerance,
            stop_at_reference=True,
            stop_at_usable_q_max=MULTIPRECISION_FALLBACK_Q_MAX,
            period_table=period_table,
        )
        refined_chart = _best_usable_chart(refined)
        if refined_chart is not None:
            return refined_chart, "refined"

    base = build_plumbing_atlas(
        omega,
        search_depth=base_search_depth,
        prefilter_count=base_prefilter_count,
        word_length=base_word_length,
        q_reference_max=q_reference_max,
        period_tolerance=base_period_tolerance,
        stability_tolerance=base_stability_tolerance,
        stop_at_reference=True,
        stop_at_usable_q_max=MULTIPRECISION_FALLBACK_Q_MAX,
        period_table=period_table,
    )
    candidates: list[tuple[str, PlumbingChartResult]] = []
    base_chart = _best_usable_chart(base)
    if base_chart is not None:
        candidates.append(("base", base_chart))

    base_needs_refinement = base_chart is None or base_chart.q_max > refine_above_q
    if base_needs_refinement:
        refined = build_plumbing_atlas(
            omega,
            search_depth=refined_search_depth,
            prefilter_count=refined_prefilter_count,
            word_length=refined_word_length,
            max_nfev=180,
            q_reference_max=q_reference_max,
            period_tolerance=refined_period_tolerance,
            stability_tolerance=refined_stability_tolerance,
            stop_at_reference=True,
            stop_at_usable_q_max=MULTIPRECISION_FALLBACK_Q_MAX,
            period_table=period_table,
        )
        refined_chart = _best_usable_chart(refined)
        if refined_chart is not None:
            candidates.append(("refined", refined_chart))

    if not candidates:
        raise RuntimeError("no finite-q plumbing chart passed the period-map checks")
    stage, chart = min(candidates, key=lambda item: (item[1].q_max, item[1].period_max_residual))
    return chart, stage


def table_first_period_validation(
    chart: PlumbingChartResult,
    q_values: Sequence[complex],
    log_q_values: Sequence[complex],
    *,
    tolerance: float,
) -> PeriodMapValidation:
    """Promote the table-first forward check to the final period certificate."""

    if chart.period_map_region not in {"schottky-all-small", "two-method-overlap"}:
        raise ValueError("table-first certificate is not in the all-small-q region")
    if chart.period_max_residual > float(tolerance):
        raise ValueError(
            "table-first period residual exceeds the final numerical bar: "
            f"{chart.period_max_residual:.3e} > {float(tolerance):.3e}"
        )
    if chart.period_map_stability > float(tolerance):
        raise ValueError(
            "table-first Schottky tail estimate exceeds the final numerical bar: "
            f"{chart.period_map_stability:.3e} > {float(tolerance):.3e}"
        )
    symmetry_error = (
        math.nan
        if chart.collocation_symmetry_error is None
        else float(chart.collocation_symmetry_error)
    )
    return PeriodMapValidation(
        q=tuple(complex(value) for value in q_values),  # type: ignore[arg-type]
        log_q=tuple(complex(value) for value in log_q_values),  # type: ignore[arg-type]
        period_algorithm=chart.period_algorithm,
        refined=bool(chart.inverse_nfev > 1),
        fixed_q_residual=float(chart.period_max_residual),
        fixed_q_stability=float(chart.period_map_stability),
        final_residual=float(chart.period_max_residual),
        final_stability=float(chart.period_map_stability),
        seam_residual=math.nan,
        symmetry_error=symmetry_error,
        low_order=int(chart.word_length),
        high_order=int(chart.stability_word_length),
        validation_order=int(chart.stability_word_length),
        reinverse_success=True,
        reinverse_message=(
            "the table/leading all-small seed passed the direct adaptive Schottky certificate; "
            "no nonlinear re-inversion was used"
        ),
        reinverse_nfev=0,
        max_tau_shift=0.0,
        certified_error_bound=float(chart.period_map_stability),
        period_map_region=chart.period_map_region,
        overlap_residual=chart.period_overlap_residual,
        agreement_tolerance=float(tolerance),
    )


def _positive_real_partition(value: complex, name: str) -> float:
    value = complex(value)
    scale = max(abs(value.real), 1.0e-300)
    if abs(value.imag) > 1.0e-8 * scale:
        raise ValueError(f"{name} has a significant imaginary part: {value!r}")
    if not math.isfinite(value.real) or value.real <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return float(value.real)


def evaluate_liouville_rescaled(
    topology: str,
    q_values: tuple[complex, complex, complex],
    *,
    log_q_values: Sequence[complex] | None = None,
    block_order: int,
    quadrature_order: int | Sequence[int],
    quadrature_scheme: str,
    dps: int,
    vacuum_word_length: int,
    vacuum_oscillator_level: int,
) -> RescaledLiouvillePartition:
    threshold = 1.0  # h_min=Q^2/4 at b=1.
    if any(abs(value) == 0.0 for value in q_values):
        raise ValueError("Liouville plumbing parameters must be nonzero")
    log_q_tuple = (
        tuple(complex(value) for value in log_q_values)
        if log_q_values is not None
        else None
    )
    if log_q_tuple is not None and len(log_q_tuple) != len(q_values):
        raise ValueError("log_q_values must match the plumbing edges")
    common = {
        "b": 1.0,
        "block_order": int(block_order),
        "quadrature_order": (
            int(quadrature_order)
            if isinstance(quadrature_order, int)
            else tuple(int(order) for order in quadrature_order)
        ),
        "quadrature_scheme": quadrature_scheme,
        "dps": int(dps),
        # Numerical factorization only: the exact factor is restored through
        # log_threshold_factor before the period-coordinate density is used.
        "propagator_shift": threshold,
        "include_vacuum_seed": True,
        "vacuum_word_len": int(vacuum_word_length),
        "vacuum_oscillator_level_max": int(vacuum_oscillator_level),
        "include_cosmological_prefactor": False,
        "store_samples": False,
        "log_q_values": log_q_tuple,
    }
    if topology == "theta":
        result = liouville_genus2_ccy_partition(
            q1=q_values[0],
            q2=q_values[1],
            q3=q_values[2],
            **common,
        )
    elif topology == "glasses":
        result = liouville_genus2_glasses_partition(
            q_left=q_values[0],
            q_right=q_values[1],
            q_bridge=q_values[2],
            **common,
        )
    else:
        raise ValueError(f"unsupported topology {topology!r}")
    scaled = _positive_real_partition(result.value, "rescaled Liouville partition")
    if log_q_tuple is None:
        log_threshold = 2.0 * threshold * sum(math.log(abs(value)) for value in q_values)
    else:
        log_threshold = 2.0 * threshold * sum(value.real for value in log_q_tuple)
    return RescaledLiouvillePartition(
        scaled_partition=scaled,
        log_threshold_factor=log_threshold,
    )


def evaluate_liouville(
    topology: str,
    q_values: tuple[complex, complex, complex],
    *,
    log_q_values: Sequence[complex] | None = None,
    block_order: int,
    quadrature_order: int,
    quadrature_scheme: str,
    dps: int,
    vacuum_word_length: int,
    vacuum_oscillator_level: int,
) -> float:
    """Return the raw partition when it is representable as a float."""

    result = evaluate_liouville_rescaled(
        topology,
        q_values,
        log_q_values=log_q_values,
        block_order=block_order,
        quadrature_order=quadrature_order,
        quadrature_scheme=quadrature_scheme,
        dps=dps,
        vacuum_word_length=vacuum_word_length,
        vacuum_oscillator_level=vacuum_oscillator_level,
    )
    raw = result.raw_partition
    if not math.isfinite(raw) or raw <= 0.0:
        raise ValueError(
            "raw Liouville partition is outside floating-point range; "
            "use evaluate_liouville_rescaled"
        )
    return raw


def evaluate_noncompact_scalar(
    topology: str,
    q_values: tuple[complex, complex, complex],
    omega_chart: np.ndarray,
    *,
    log_q_values: Sequence[complex] | None = None,
    word_length: int,
    max_mode: int,
    tolerance: float,
) -> tuple[float, float, int]:
    surviving_multipliers = (
        theta_cusp_surviving_multipliers(
            *q_values,
            log_q_values=log_q_values,
            threshold=1.0e-12,
        )
        if topology == "theta" and log_q_values is not None
        else None
    )
    if surviving_multipliers is not None:
        oscillator = math.prod(
            torus_raw_oscillator_abs(
                multiplier,
                max_mode=max_mode,
                tolerance=tolerance,
            )
            for multiplier in surviving_multipliers
        )
        full_partition = oscillator * noncompact_scalar_zero_mode_factor(omega_chart)
        largest_multiplier = max((abs(value) for value in surviving_multipliers), default=0.0)
        tail = float(
            max(tolerance, 1.0e-12)
            / max(1.0e-300, (1.0 - largest_multiplier) ** 2)
        )
        return float(full_partition), tail, len(surviving_multipliers)
    if topology == "theta":
        product = theta_free_boson_product(
            *q_values,
            max_word_length=word_length,
            max_mode=max_mode,
            tolerance=tolerance,
        )
        frame = THETA_PLUMBING_FRAME
    elif topology == "glasses":
        product = glasses_free_boson_product(
            *q_values,
            max_word_length=word_length,
            max_mode=max_mode,
            tolerance=tolerance,
        )
        frame = GLASSES_PLUMBING_FRAME
    else:
        raise ValueError(f"unsupported topology {topology!r}")
    full_partition = product.nonchiral_value * noncompact_scalar_zero_mode_factor(omega_chart)
    if not math.isfinite(full_partition) or full_partition <= 0.0:
        raise ValueError("noncompact scalar plumbing partition is not positive and finite")
    return float(full_partition), float(product.omitted_chiral_tail_estimate), int(product.primitive_count)


def estimate_from_transformed_values(values: Sequence[float]) -> dict[str, object]:
    """Summarize values g_i = det(Y_i)^3 I_i from invariant-domain draws."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("at least one transformed integrand value is required")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("transformed integrand values must be finite and nonnegative")
    prefactor = GENUS2_GENERIC_STACK_WEIGHT * SIEGEL_VOLUME_G2
    contributions = prefactor * array
    estimate = float(np.mean(contributions))
    standard_error = (
        0.0
        if len(contributions) == 1
        else float(np.std(contributions, ddof=1) / math.sqrt(len(contributions)))
    )
    total = float(np.sum(contributions))
    squared = float(np.sum(contributions**2))
    effective_sample_size = 0.0 if squared == 0.0 else total * total / squared
    ordered = np.sort(contributions)[::-1]
    running = [float(np.mean(contributions[: index + 1])) for index in range(len(contributions))]
    return {
        "sample_count": int(len(contributions)),
        "estimate": estimate,
        "standard_error": standard_error,
        "normal_95_low": estimate - 1.959963984540054 * standard_error,
        "normal_95_high": estimate + 1.959963984540054 * standard_error,
        "coarse_domain_estimate_before_stack_weight": estimate / GENUS2_GENERIC_STACK_WEIGHT,
        "effective_sample_size": effective_sample_size,
        "largest_sample_fraction": 0.0 if total == 0.0 else float(ordered[0] / total),
        "largest_four_sample_fraction": (
            0.0 if total == 0.0 else float(np.sum(ordered[: min(4, len(ordered))]) / total)
        ),
        "running_estimate": running,
    }


def _optional_float(row: Mapping[str, object], key: str) -> float | None:
    value = str(row.get(key, "")).strip()
    return None if value == "" else float(value)


def cached_period_solution_objects(
    source_row: Mapping[str, str],
    recovery_row: Mapping[str, object],
    *,
    tolerance: float,
) -> tuple[argparse.Namespace, PeriodMapValidation]:
    """Rehydrate a certified period chart without repeating Omega-to-q."""

    sample_index = int(source_row["sample_index"])
    if int(str(recovery_row.get("sample_index", "-1"))) != sample_index:
        raise ValueError("period-recovery cache sample index does not match the source row")
    source_node_id = str(source_row.get("rqmc_node_id", ""))
    recovered_node_id = str(recovery_row.get("rqmc_node_id", ""))
    if source_node_id and recovered_node_id and recovered_node_id != source_node_id:
        raise ValueError("period-recovery cache node id does not match the source row")
    if str(recovery_row.get("status", "")) != "ok":
        raise ValueError(
            "period-recovery cache has no accepted solution: "
            f"{recovery_row.get('error', 'unknown failure')}"
        )
    matrix = symplectic_matrix_from_csv_row(recovery_row)
    if matrix is None:
        raise ValueError("period-recovery cache lacks the exact Sp(4,Z) marking")
    topology = str(recovery_row.get("topology", ""))
    if topology not in {"theta", "glasses"}:
        raise ValueError(f"invalid cached plumbing topology {topology!r}")

    final_residual = float(recovery_row["final_period_residual"])
    final_stability = float(recovery_row["final_period_map_step"])
    if final_residual > float(tolerance) or final_stability > float(tolerance):
        raise ValueError(
            "cached period solution exceeds the requested numerical bar: "
            f"residual={final_residual:.3e}, stability={final_stability:.3e}, "
            f"bar={float(tolerance):.3e}"
        )
    q_values = tuple(
        complex(str(recovery_row[f"final_q{index}"])) for index in range(1, 4)
    )
    log_q_values = tuple(
        complex(str(recovery_row[f"final_log_q{index}"])) for index in range(1, 4)
    )
    atlas_q_values = tuple(
        complex(str(recovery_row.get(f"atlas_q{index}", q_values[index - 1])))
        for index in range(1, 4)
    )
    atlas_log_q_values = tuple(
        complex(
            str(recovery_row.get(f"atlas_log_q{index}", log_q_values[index - 1]))
        )
        for index in range(1, 4)
    )
    certificate_algorithm = str(recovery_row["period_algorithm"])
    inferred_region = (
        "schottky-all-small"
        if "schottky" in certificate_algorithm
        else (
            "holomorphic-mixed-cusp"
            if certificate_algorithm == MULTIPRECISION_HOLOMORPHIC_ALGORITHM
            else "holomorphic-bulk"
        )
    )
    cached_region = (
        str(recovery_row.get("period_map_region", ""))
        or str(recovery_row.get("atlas_period_map_region", ""))
        or inferred_region
    )
    chart = argparse.Namespace(
        topology=topology,
        word=str(recovery_row.get("symplectic_word", "cached-period-recovery")),
        matrix=tuple(tuple(int(value) for value in row) for row in matrix),
        status=str(recovery_row.get("chart_status", "requires-recursion-order-study")),
        period_algorithm=str(recovery_row["atlas_period_algorithm"]),
        period_map_region=(
            str(recovery_row.get("atlas_period_map_region", "")) or cached_region
        ),
        plumbing_geometry_margin=float(
            recovery_row.get("atlas_plumbing_geometry_margin", "nan") or "nan"
        ),
        period_overlap_residual=_optional_float(
            recovery_row, "atlas_period_overlap_residual"
        ),
        inverse_seed_source=str(recovery_row.get("atlas_inverse_seed_source", "")),
        q=tuple(_complex_string(value) for value in atlas_q_values),
        log_q=tuple(_complex_string(value) for value in atlas_log_q_values),
        q_max=float(recovery_row.get("atlas_q_max", max(abs(value) for value in q_values))),
        period_max_residual=float(recovery_row["atlas_period_residual"]),
        period_map_stability=float(recovery_row["atlas_period_map_stability"]),
    )
    certificate = PeriodMapValidation(
        q=q_values,  # type: ignore[arg-type]
        log_q=log_q_values,  # type: ignore[arg-type]
        period_algorithm=certificate_algorithm,
        refined=str(recovery_row.get("q_refined", "")).lower() in {"1", "true", "yes"},
        fixed_q_residual=float(recovery_row["fixed_q_period_residual"]),
        fixed_q_stability=float(recovery_row["fixed_q_period_map_step"]),
        final_residual=final_residual,
        final_stability=final_stability,
        seam_residual=float(recovery_row.get("period_seam_residual", "nan") or "nan"),
        symmetry_error=float(recovery_row.get("period_symmetry_error", "nan") or "nan"),
        low_order=int(float(recovery_row.get("period_map_low_order", 0) or 0)),
        high_order=int(float(recovery_row.get("period_map_high_order", 0) or 0)),
        validation_order=int(
            float(recovery_row.get("period_map_validation_order", 0) or 0)
        ),
        reinverse_success=True,
        reinverse_message=str(recovery_row.get("reinverse_message", "cached certificate")),
        reinverse_nfev=int(float(recovery_row.get("reinverse_nfev", 0) or 0)),
        max_tau_shift=float(recovery_row.get("max_tau_shift", 0.0) or 0.0),
        validity_cell_id=str(recovery_row.get("period_validity_cell_id", "")) or None,
        certified_error_bound=(
            _optional_float(recovery_row, "period_certified_error_bound")
            or final_stability
        ),
        validity_reference_table_sha256=(
            str(recovery_row.get("period_validity_reference_table_sha256", "")) or None
        ),
        period_map_region=cached_region,
        overlap_residual=_optional_float(recovery_row, "period_overlap_residual"),
        agreement_tolerance=(
            _optional_float(recovery_row, "period_agreement_tolerance")
            or float(tolerance)
        ),
    )
    return chart, certificate


def evaluate_node(
    source_row: dict[str, str],
    args: argparse.Namespace,
    schottky_envelope: SchottkyValidityEnvelope | None = None,
    period_table: Genus2PeriodMapTable | None = None,
    period_recovery_solution: Mapping[str, object] | None = None,
) -> dict[str, object]:
    started = time.time()
    stage = "initialize"
    omega_fundamental = omega_from_csv_row(source_row)
    det_y = float(np.linalg.det(omega_fundamental.imag))
    sample_index = int(source_row["sample_index"])
    integration_weight = node_stack_integration_weight(source_row)
    det_im_power = kernel_det_im_power(source_row)
    row: dict[str, object] = {
        "sample_index": sample_index,
        "status": "failed",
        "error": "",
        "det_im_omega": det_y,
        "x11": float(omega_fundamental[0, 0].real),
        "x12": float(omega_fundamental[0, 1].real),
        "x22": float(omega_fundamental[1, 1].real),
        "y11": float(omega_fundamental[0, 0].imag),
        "y12": float(omega_fundamental[0, 1].imag),
        "y22": float(omega_fundamental[1, 1].imag),
        "igusa_measure_frame": "siegel-fundamental-domain",
        "stack_integration_weight": integration_weight,
        "kernel_det_im_power": det_im_power,
    }
    for key, value in source_row.items():
        if key == "sampling_scheme" or key.startswith("rqmc_"):
            row[key] = value
    try:
        stage = "period-map-chart"
        if period_recovery_solution is None:
            chart, search_stage = select_plumbing_chart(
                omega_fundamental,
                q_reference_max=args.q_reference_max,
                refine_above_q=args.refine_above_q,
                base_search_depth=args.base_search_depth,
                base_prefilter_count=args.base_prefilter_count,
                base_word_length=args.base_word_length,
                base_period_tolerance=args.base_period_tolerance,
                base_stability_tolerance=args.base_stability_tolerance,
                refined_search_depth=args.refined_search_depth,
                refined_prefilter_count=args.refined_prefilter_count,
                refined_word_length=args.refined_word_length,
                refined_period_tolerance=args.refined_period_tolerance,
                refined_stability_tolerance=args.refined_stability_tolerance,
                period_table=period_table,
                table_first_period_tolerance=args.period_validation_tolerance,
                table_first_stability_tolerance=args.period_validation_tolerance,
                table_first_maximum_word=args.table_first_maximum_word,
                table_first_maximum_corrections=args.table_first_maximum_corrections,
            )
            period_certificate: PeriodMapValidation | None = None
        else:
            chart, period_certificate = cached_period_solution_objects(
                source_row,
                period_recovery_solution,
                tolerance=args.period_validation_tolerance,
            )
            search_stage = "cached-period-recovery"
        atlas_q_values = tuple(complex(value) for value in chart.q)
        atlas_log_q_values = tuple(complex(value) for value in chart.log_q)
        # The exact integer marking is the authoritative frame map.  Parsing
        # the atlas' human-readable decimal Omega loses an absolute amount of
        # precision proportional to a very long cusp entry and previously
        # caused false failures of the 2e-9 algebraic consistency check.
        omega_chart = transform_omega(
            np.asarray(chart.matrix, dtype=np.int64),
            omega_fundamental,
        )
        omega_chart = 0.5 * (omega_chart + omega_chart.T)
        validate_covariant_period_frames(
            omega_fundamental,
            chart.matrix,
            omega_chart,
        )
        if period_certificate is None:
            if search_stage == "direct-all-small":
                period_certificate = table_first_period_validation(
                    chart,
                    atlas_q_values,
                    atlas_log_q_values,
                    tolerance=args.period_validation_tolerance,
                )
            else:
                period_certificate = validate_or_refine_period_map(
                    chart.topology,
                    omega_chart,
                    atlas_q_values,
                    word_length=args.period_validation_word_length,
                    word_step=args.period_validation_word_step,
                    tolerance=args.period_validation_tolerance,
                    reinverse_validation_word_length=(
                        args.period_reinverse_validation_word_length
                    ),
                    reinverse_max_nfev=args.period_reinverse_max_nfev,
                    initial_log_q=atlas_log_q_values,
                    schottky_envelope=schottky_envelope,
                )
        certified_period_algorithms = {
            "holomorphic-form-collocation",
            MULTIPRECISION_HOLOMORPHIC_ALGORITHM,
            "adaptive-schottky",
            "calibrated-schottky",
        }
        if (
            chart.period_algorithm not in certified_period_algorithms
            or period_certificate.period_algorithm not in certified_period_algorithms
        ):
            raise RuntimeError(
                "production node used an uncertified period-map backend: "
                f"atlas={chart.period_algorithm!r}, "
                f"certificate={period_certificate.period_algorithm!r}"
            )
        q_values = period_certificate.q
        log_q_values = period_certificate.log_q or tuple(cmath.log(value) for value in q_values)
        frame = THETA_PLUMBING_FRAME if chart.topology == "theta" else GLASSES_PLUMBING_FRAME

        stage = "noncompact-scalar-low"
        scalar_low, scalar_tail_low, scalar_primitives_low = evaluate_noncompact_scalar(
            chart.topology,
            q_values,
            omega_chart,
            log_q_values=log_q_values,
            word_length=args.scalar_word_low,
            max_mode=args.scalar_max_mode,
            tolerance=args.scalar_tolerance,
        )
        if args.scalar_word_high == args.scalar_word_low:
            scalar_high = scalar_low
            scalar_tail_high = scalar_tail_low
            scalar_primitives_high = scalar_primitives_low
        else:
            stage = "noncompact-scalar-high"
            scalar_high, scalar_tail_high, scalar_primitives_high = evaluate_noncompact_scalar(
                chart.topology,
                q_values,
                omega_chart,
                log_q_values=log_q_values,
                word_length=args.scalar_word_high,
                max_mode=args.scalar_max_mode,
                tolerance=args.scalar_tolerance,
            )
        stage = "liouville-low"
        liouville_low = evaluate_liouville_rescaled(
            chart.topology,
            q_values,
            log_q_values=log_q_values,
            block_order=args.block_order_low,
            quadrature_order=args.quadrature_order_low,
            quadrature_scheme=args.quadrature_scheme,
            dps=args.dps,
            vacuum_word_length=args.vacuum_word_length,
            vacuum_oscillator_level=args.vacuum_oscillator_level,
        )
        if (
            args.block_order_high == args.block_order_low
            and args.quadrature_order_high == args.quadrature_order_low
        ):
            liouville_high = liouville_low
        else:
            stage = "liouville-high"
            liouville_high = evaluate_liouville_rescaled(
                chart.topology,
                q_values,
                log_q_values=log_q_values,
                block_order=args.block_order_high,
                quadrature_order=args.quadrature_order_high,
                quadrature_scheme=args.quadrature_scheme,
                dps=args.dps,
                vacuum_word_length=args.vacuum_word_length,
                vacuum_oscillator_level=args.vacuum_oscillator_level,
            )

        stage = "fundamental-frame-integrand-low"
        low = genus2_c1_string_integrand_density(
            omega_fundamental,
            args.radius,
            matter_partitions=SameFrameMatterPartitions(
                conformal_frame=frame,
                liouville_partition=liouville_low.scaled_partition,
                noncompact_scalar_partition=scalar_low,
            ),
            lattice_tolerance=args.lattice_tolerance,
            theta_tolerance=args.theta_tolerance,
            chi10_normalization="product",
        )
        stage = "fundamental-frame-integrand-high"
        high = genus2_c1_string_integrand_density(
            omega_fundamental,
            args.radius,
            matter_partitions=SameFrameMatterPartitions(
                conformal_frame=frame,
                liouville_partition=liouville_high.scaled_partition,
                noncompact_scalar_partition=scalar_high,
            ),
            lattice_tolerance=args.lattice_tolerance,
            theta_tolerance=args.theta_tolerance,
            chi10_normalization="product",
        )
        stage = "log-density-recombination"
        unit_mumford_log_low = (
            low.factorization_normalized_log_density
            + liouville_low.log_threshold_factor
        )
        unit_mumford_log_high = (
            high.factorization_normalized_log_density
            + liouville_high.log_threshold_factor
        )
        string_note_log_low = (
            low.string_note_kernel_log_density
            + liouville_low.log_threshold_factor
        )
        string_note_log_high = (
            high.string_note_kernel_log_density
            + liouville_high.log_threshold_factor
        )
        factorized_density_low = math.exp(unit_mumford_log_low)
        factorized_density_high = math.exp(unit_mumford_log_high)
        string_note_density_low = math.exp(string_note_log_low)
        string_note_density_high = math.exp(string_note_log_high)
        det_im_log_factor = float(det_im_power) * math.log(det_y)
        unit_mumford_transformed_low = math.exp(
            det_im_log_factor + unit_mumford_log_low
        )
        unit_mumford_transformed_high = math.exp(
            det_im_log_factor + unit_mumford_log_high
        )
        transformed_low = math.exp(det_im_log_factor + string_note_log_low)
        transformed_high = math.exp(det_im_log_factor + string_note_log_high)
        stage = "serialize-success"
        row.update(
            {
                "status": "ok",
                "search_stage": search_stage,
                "topology": chart.topology,
                "matter_frame": frame,
                "compact_theta_frame": "siegel-fundamental-domain",
                "liouville_scalar_quotient_frame": frame,
                "symplectic_word": chart.word,
                **symplectic_matrix_csv_fields(chart.matrix),
                "chart_status": chart.status,
                "atlas_period_algorithm": chart.period_algorithm,
                "atlas_period_map_region": chart.period_map_region,
                "atlas_plumbing_geometry_margin": chart.plumbing_geometry_margin,
                "atlas_period_overlap_residual": (
                    ""
                    if chart.period_overlap_residual is None
                    else chart.period_overlap_residual
                ),
                "atlas_inverse_seed_source": chart.inverse_seed_source,
                "atlas_q1": _complex_string(atlas_q_values[0]),
                "atlas_q2": _complex_string(atlas_q_values[1]),
                "atlas_q3": _complex_string(atlas_q_values[2]),
                "atlas_log_q1": _complex_string(atlas_log_q_values[0]),
                "atlas_log_q2": _complex_string(atlas_log_q_values[1]),
                "atlas_log_q3": _complex_string(atlas_log_q_values[2]),
                "q1": _complex_string(q_values[0]),
                "q2": _complex_string(q_values[1]),
                "q3": _complex_string(q_values[2]),
                "log_q1": _complex_string(log_q_values[0]),
                "log_q2": _complex_string(log_q_values[1]),
                "log_q3": _complex_string(log_q_values[2]),
                "q1_abs": math.exp(log_q_values[0].real) if log_q_values[0].real > -745.0 else 0.0,
                "q2_abs": math.exp(log_q_values[1].real) if log_q_values[1].real > -745.0 else 0.0,
                "q3_abs": math.exp(log_q_values[2].real) if log_q_values[2].real > -745.0 else 0.0,
                "atlas_q_max": chart.q_max,
                "q_max": math.exp(max(value.real for value in log_q_values)),
                "period_algorithm": period_certificate.period_algorithm,
                "period_map_region": period_certificate.period_map_region or "",
                "period_overlap_residual": (
                    ""
                    if period_certificate.overlap_residual is None
                    else period_certificate.overlap_residual
                ),
                "period_agreement_tolerance": (
                    ""
                    if period_certificate.agreement_tolerance is None
                    else period_certificate.agreement_tolerance
                ),
                "period_validity_cell_id": period_certificate.validity_cell_id or "",
                "period_validity_reference_table_sha256": (
                    period_certificate.validity_reference_table_sha256 or ""
                ),
                "period_certified_error_bound": (
                    ""
                    if period_certificate.certified_error_bound is None
                    else period_certificate.certified_error_bound
                ),
                "fixed_cft_order": (
                    args.block_order_low == args.block_order_high
                    and args.quadrature_order_low == args.quadrature_order_high
                ),
                "block_order_low": args.block_order_low,
                "block_order_high": args.block_order_high,
                "quadrature_order_low": args.quadrature_order_low,
                "quadrature_order_high": args.quadrature_order_high,
                "scalar_word_low": args.scalar_word_low,
                "scalar_word_high": args.scalar_word_high,
                "period_residual": chart.period_max_residual,
                "period_map_stability": chart.period_map_stability,
                "period_cusp_word_length": args.period_validation_word_length,
                "period_cusp_word_step": args.period_validation_word_step,
                "period_validation_tolerance": args.period_validation_tolerance,
                "period_fixed_q_residual": period_certificate.fixed_q_residual,
                "period_fixed_q_map_step": period_certificate.fixed_q_stability,
                "period_q_refined": period_certificate.refined,
                "period_final_residual": period_certificate.final_residual,
                "period_final_map_step": period_certificate.final_stability,
                "period_map_low_order": period_certificate.low_order,
                "period_map_high_order": period_certificate.high_order,
                "period_map_validation_order": period_certificate.validation_order,
                "period_seam_residual": period_certificate.seam_residual,
                "period_symmetry_error": period_certificate.symmetry_error,
                "period_reinverse_success": period_certificate.reinverse_success,
                "period_reinverse_message": period_certificate.reinverse_message,
                "period_reinverse_nfev": period_certificate.reinverse_nfev,
                "period_reinverse_max_tau_shift": period_certificate.max_tau_shift,
                "liouville_low": liouville_low.raw_partition,
                "liouville_high": liouville_high.raw_partition,
                "liouville_log_low": liouville_low.log_partition,
                "liouville_log_high": liouville_high.log_partition,
                "liouville_scaled_low": liouville_low.scaled_partition,
                "liouville_scaled_high": liouville_high.scaled_partition,
                "liouville_threshold_log": liouville_high.log_threshold_factor,
                "liouville_relative_change": (
                    liouville_high.scaled_partition
                    / liouville_low.scaled_partition
                    - 1.0
                ),
                "scalar_low": scalar_low,
                "scalar_high": scalar_high,
                "scalar_input_normalization": DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION,
                "scalar_integrand_normalization": XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION,
                "xi_scalar_over_dimensionless": xi_genus2_scalar_over_dimensionless(),
                "xi_full_replacement_over_dimensionless": (
                    xi_full_replacement_over_dimensionless()
                ),
                "scalar_relative_change": scalar_high / scalar_low - 1.0,
                "scalar_primitive_count_low": scalar_primitives_low,
                "scalar_primitive_count_high": scalar_primitives_high,
                "scalar_mode_tail_low": scalar_tail_low,
                "scalar_mode_tail_high": scalar_tail_high,
                "chi10_abs": abs(high.chi10),
                "chi10_log_abs": high.chi10_log_abs,
                "compact_winding_sum": high.compact_winding_sum,
                "lattice_nmax": high.lattice_nmax,
                "compact_theta_algorithm": high.compact_theta_algorithm,
                "compact_theta_implementation": COMPACT_THETA_IMPLEMENTATION,
                "compact_theta_momentum_nmax": high.compact_theta_momentum_nmax,
                "compact_theta_winding_nmax": high.compact_theta_winding_nmax,
                "factorized_density_low": factorized_density_low,
                "factorized_density_high": factorized_density_high,
                "factorized_log_density_low": unit_mumford_log_low,
                "factorized_log_density_high": unit_mumford_log_high,
                "integration_kernel_convention": high.string_note_kernel_convention,
                "string_note_kernel_multiplier": high.string_note_kernel_multiplier,
                "string_note_kernel_density_low": string_note_density_low,
                "string_note_kernel_density_high": string_note_density_high,
                "string_note_kernel_log_density_low": string_note_log_low,
                "string_note_kernel_log_density_high": string_note_log_high,
                "unit_mumford_transformed_integrand_low": unit_mumford_transformed_low,
                "unit_mumford_transformed_integrand_high": unit_mumford_transformed_high,
                "transformed_integrand_low": transformed_low,
                "transformed_integrand_high": transformed_high,
                "node_contribution_low": (
                    integration_weight * transformed_low
                ),
                "node_contribution_high": (
                    integration_weight * transformed_high
                ),
                "combined_relative_change": transformed_high / transformed_low - 1.0,
            }
        )
    except Exception as exc:  # noqa: BLE001 - every failed node is saved.
        if bool(getattr(args, "raise_on_error", False)):
            raise
        row["error"] = f"{type(exc).__name__} at {stage}: {exc}"
    row["runtime_seconds"] = time.time() - started
    return row


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resume_node_key(row: dict[str, object]) -> str:
    node_id = str(row.get("rqmc_node_id", "")).strip()
    if node_id:
        return f"rqmc:{node_id}"
    sample_index = str(row.get("sample_index", "")).strip()
    if sample_index:
        return f"sample:{sample_index}"
    raise ValueError("resume row lacks both rqmc_node_id and sample_index")


def _jsonable_argument(value: object) -> object:
    return str(value) if isinstance(value, Path) else value


def _resume_config(
    args: argparse.Namespace,
    selected: Sequence[dict[str, object]],
) -> dict[str, object]:
    return {
        "format_version": 1,
        "source_csv": str(args.source_csv.resolve()),
        "source_csv_sha256": hashlib.sha256(args.source_csv.read_bytes()).hexdigest(),
        "selected_node_keys": [_resume_node_key(row) for row in selected],
        "settings": {
            **{
                key: _jsonable_argument(value)
                for key, value in vars(args).items()
                if key not in {"source_csv", "out_dir", "resume"}
            },
            "compact_theta_implementation": COMPACT_THETA_IMPLEMENTATION,
        },
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _load_resume_rows(
    *,
    csv_path: Path,
    config_path: Path,
    expected_config: dict[str, object],
    resume: bool,
) -> dict[str, dict[str, object]]:
    if not resume:
        _write_json_atomic(config_path, expected_config)
        return {}
    if csv_path.exists() and not config_path.exists():
        raise ValueError("cannot resume samples.csv without run_config.json")
    if config_path.exists():
        observed_config = json.loads(config_path.read_text())
        if observed_config != expected_config:
            raise ValueError("resume settings or source manifest do not match run_config.json")
    else:
        _write_json_atomic(config_path, expected_config)
    if not csv_path.exists():
        return {}

    expected_keys = set(expected_config["selected_node_keys"])
    rows_by_key: dict[str, dict[str, object]] = {}
    for row in csv.DictReader(csv_path.open()):
        key = _resume_node_key(row)
        if key not in expected_keys:
            raise ValueError(f"resume CSV contains unexpected node {key}")
        if key in rows_by_key:
            raise ValueError(f"resume CSV contains duplicate node {key}")
        rows_by_key[key] = dict(row)
    return rows_by_key


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pilot genus-two c=1 string Monte Carlo integral.")
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=Path("plumbing/results/genus2_full_moduli_coverage/full_moduli_combined.csv"),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument(
        "--rqmc-replicate",
        type=int,
        help="evaluate every in-domain node in one RQMC scramble",
    )
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--q-reference-max", type=float, default=0.16)
    parser.add_argument("--refine-above-q", type=float, default=0.16)
    parser.add_argument("--base-search-depth", type=int, default=3)
    parser.add_argument("--base-prefilter-count", type=int, default=2)
    parser.add_argument("--base-word-length", type=int, default=4)
    parser.add_argument("--base-period-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--base-stability-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--refined-search-depth", type=int, default=4)
    parser.add_argument("--refined-prefilter-count", type=int, default=6)
    parser.add_argument("--refined-word-length", type=int, default=6)
    parser.add_argument("--refined-period-tolerance", type=float, default=5.0e-6)
    parser.add_argument("--refined-stability-tolerance", type=float, default=5.0e-6)
    parser.add_argument(
        "--table-first-maximum-word",
        type=int,
        default=5,
        help=(
            "maximum adaptive Schottky word used to certify a transported "
            "fundamental-table seed before falling back to the full atlas"
        ),
    )
    parser.add_argument(
        "--table-first-maximum-corrections",
        type=int,
        default=2,
        help="maximum analytic log-q corrections before full-atlas fallback",
    )
    parser.add_argument(
        "--period-validation-word-length",
        type=int,
        default=8,
        help="initial Schottky word order used by the hybrid period certificate",
    )
    parser.add_argument(
        "--period-validation-word-step",
        type=int,
        default=1,
        help="word-order step retained for period-certificate compatibility",
    )
    parser.add_argument("--period-validation-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--period-reinverse-validation-word-length",
        type=int,
        default=10,
        help="maximum Schottky validation order for cusp re-inversion",
    )
    parser.add_argument("--period-reinverse-max-nfev", type=int, default=80)
    parser.add_argument(
        "--schottky-validity-envelope",
        type=Path,
        help=(
            "optional CSV of Schottky cells calibrated against the holomorphic-form "
            "reference table; this strengthens the adaptive hybrid certificate"
        ),
    )
    parser.add_argument(
        "--period-table-index",
        type=Path,
        help=(
            "optional portable q-to-Omega table index used only to seed the "
            "certified Omega-to-q atlas inversion"
        ),
    )
    parser.add_argument(
        "--period-table-csv",
        type=Path,
        help="optional canonical table CSV used to verify the portable index hash",
    )
    parser.add_argument(
        "--period-recovery-cache",
        type=Path,
        help=(
            "assembled period_recovery.csv; accepted q and exact markings are "
            "reused without repeating Omega-to-q inversion"
        ),
    )
    parser.add_argument("--block-order-low", type=int, default=2)
    parser.add_argument("--block-order-high", type=int, default=4)
    parser.add_argument("--quadrature-order-low", type=int, default=3)
    parser.add_argument("--quadrature-order-high", type=int, default=4)
    parser.add_argument(
        "--quadrature-scheme",
        choices=("uniform", "edge-scaled", "primary-gaussian"),
        default="primary-gaussian",
    )
    parser.add_argument("--dps", type=int, default=24)
    parser.add_argument("--vacuum-word-length", type=int, default=6)
    parser.add_argument("--vacuum-oscillator-level", type=int, default=20)
    parser.add_argument("--scalar-word-low", type=int, default=4)
    parser.add_argument("--scalar-word-high", type=int, default=6)
    parser.add_argument("--scalar-max-mode", type=int, default=80)
    parser.add_argument("--scalar-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--lattice-tolerance", type=float, default=1.0e-12)
    parser.add_argument("--theta-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plumbing/results/genus2_c1_moduli_mc/pilot_R1_N8"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "reuse successful nodes from samples.csv after verifying the source "
            "manifest and every numerical setting against run_config.json"
        ),
    )
    parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="re-raise a node exception with a traceback for targeted diagnostics",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    schottky_envelope = (
        None
        if args.schottky_validity_envelope is None
        else SchottkyValidityEnvelope.from_csv(args.schottky_validity_envelope)
    )
    if args.period_table_csv is not None and args.period_table_index is None:
        raise ValueError("--period-table-csv requires --period-table-index")
    period_table = (
        None
        if args.period_table_index is None
        else Genus2PeriodMapTable.from_portable_index(
            args.period_table_index,
            verify_table_path=args.period_table_csv,
        )
    )
    if period_table is not None and not period_table.has_fundamental_index:
        raise ValueError(
            "--period-table-index must use schema v3 with fundamental-domain "
            "Omega coordinates and exact Sp(4,Z) markings"
        )
    period_recovery_by_index: dict[int, dict[str, str]] | None = None
    if args.period_recovery_cache is not None:
        recovery_rows = list(csv.DictReader(args.period_recovery_cache.open()))
        period_recovery_by_index = {}
        for recovery_row in recovery_rows:
            recovery_index = int(recovery_row["sample_index"])
            if recovery_index in period_recovery_by_index:
                raise ValueError(
                    f"period-recovery cache duplicates sample {recovery_index}"
                )
            period_recovery_by_index[recovery_index] = recovery_row

    source_rows = list(csv.DictReader(args.source_csv.open()))
    if args.rqmc_replicate is not None:
        selected = [
            row
            for row in source_rows
            if row.get("rqmc_replicate") != ""
            and int(row["rqmc_replicate"]) == args.rqmc_replicate
        ]
        if not selected:
            raise ValueError(f"RQMC replicate {args.rqmc_replicate} is absent")
        sample_range: list[int] | None = None
    else:
        if args.start_index < 0 or args.sample_count <= 0:
            raise ValueError("start-index must be nonnegative and sample-count positive")
        stop = args.start_index + args.sample_count
        if stop > len(source_rows):
            raise ValueError("requested sample range exceeds the source CSV")
        selected = source_rows[args.start_index:stop]
        sample_range = [args.start_index, stop]
    sampling_scheme = sampling_scheme_for_rows(selected)
    if args.rqmc_replicate is not None and sampling_scheme not in RQMC_SAMPLING_SCHEMES:
        raise ValueError("--rqmc-replicate requires scrambled-Sobol source rows")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "samples.csv"
    json_path = args.out_dir / "summary.json"
    config_path = args.out_dir / "run_config.json"
    config = _resume_config(args, selected)
    rows_by_key = _load_resume_rows(
        csv_path=csv_path,
        config_path=config_path,
        expected_config=config,
        resume=args.resume,
    )
    selected_keys = [_resume_node_key(row) for row in selected]
    completed_before = sum(
        str(row.get("status", "")) == "ok" for row in rows_by_key.values()
    )
    if completed_before:
        print(f"  resume: preserving {completed_before} successful nodes")
    for offset, source_row in enumerate(selected, start=1):
        key = _resume_node_key(source_row)
        previous = rows_by_key.get(key)
        if previous is not None and str(previous.get("status", "")) == "ok":
            print(f"  {offset}/{len(selected)} {key} status=resume")
            continue
        recovery_solution: Mapping[str, object] | None = None
        if period_recovery_by_index is not None:
            recovery_solution = period_recovery_by_index.get(
                int(source_row["sample_index"]),
                {
                    "sample_index": source_row["sample_index"],
                    "status": "missing",
                    "error": "sample is absent from the period-recovery cache",
                },
            )
        row = evaluate_node(
            source_row,
            args,
            schottky_envelope,
            period_table,
            recovery_solution,
        )
        rows_by_key[key] = row
        rows = [rows_by_key[item] for item in selected_keys if item in rows_by_key]
        _write_csv(csv_path, rows)
        if row["status"] == "ok":
            print(
                f"  {offset}/{len(selected)} index={row['sample_index']} "
                f"{row['topology']} q={float(row['q_max']):.4g} "
                f"g={float(row['transformed_integrand_high']):.4e} "
                f"delta={float(row['combined_relative_change']):+.2e} "
                f"time={float(row['runtime_seconds']):.1f}s"
            )
        else:
            print(
                f"  {offset}/{len(selected)} index={row['sample_index']} "
                f"FAILED: {row['error']}"
            )

    rows = [rows_by_key[key] for key in selected_keys]
    successful = [row for row in rows if row["status"] == "ok"]
    complete = len(successful) == len(rows)
    if sampling_scheme in RQMC_SAMPLING_SCHEMES:
        high_summary = summarize_rqmc_rows(rows, value_key="transformed_integrand_high")
        low_summary = summarize_rqmc_rows(rows, value_key="transformed_integrand_low")
    else:
        high_summary = (
            estimate_from_transformed_values(
                [float(row["transformed_integrand_high"]) for row in successful]
            )
            if successful
            else None
        )
        low_summary = (
            estimate_from_transformed_values(
                [float(row["transformed_integrand_low"]) for row in successful]
            )
            if successful
            else None
        )
    integration_complete = high_summary is not None and low_summary is not None
    if high_summary is not None:
        if sampling_scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME:
            high_summary["cusp_diagnostics"] = physical_mixture_contribution_diagnostics(
                rows,
                [float(row["transformed_integrand_high"]) for row in rows],
            )
        elif sampling_scheme != RQMC_SAMPLING_SCHEME:
            high_summary["interpretation"] = (
                "Unbiased for the requested iid sample only when complete=true. "
                "The standard error is Monte Carlo sampling error and excludes CFT truncation error."
            )
    discretization_change = (
        None
        if high_summary is None or low_summary is None
        else float(high_summary["estimate"]) / float(low_summary["estimate"]) - 1.0
    )
    node_discretization_changes = [
        float(row["combined_relative_change"])
        for row in successful
    ]
    node_discretization = (
        None
        if not node_discretization_changes
        else {
            "minimum_relative_change": min(node_discretization_changes),
            "maximum_relative_change": max(node_discretization_changes),
            "maximum_absolute_relative_change": max(
                abs(value) for value in node_discretization_changes
            ),
            "status": (
                "not established; the aggregate shift does not bound individual "
                "node truncation errors"
            ),
        }
    )
    payload = {
        "scope": (
            "Pilot integration of the string-note normalized, g_s^2-stripped "
            "c=1 genus-two kernel "
            "over the coarse Gottschling domain with the generic stack weight applied."
        ),
        "source_csv": str(args.source_csv),
        "sample_range": sample_range,
        "requested_rqmc_replicate": args.rqmc_replicate,
        "requested_sample_count": len(rows),
        "successful_sample_count": len(successful),
        "complete": complete,
        "integration_complete": integration_complete,
        "sampling_scheme": sampling_scheme,
        "radius": args.radius,
        "siegel_volume": SIEGEL_VOLUME_G2,
        "stack_weight": GENUS2_GENERIC_STACK_WEIGHT,
        "estimator": (
            "F2/g_s^2=(1/2)*mean[1_F2*(J_Y/p_mix)*K2_c1]"
            if sampling_scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME
            else (
                "F2/g_s^2=(1/2)*mean_over_all_proposals[1_F2*w*det(Im Omega)^3*K2_c1]"
                if sampling_scheme == RQMC_SAMPLING_SCHEME
                else "F2/g_s^2=(Vol(F2)/2)*mean[det(Im Omega)^3*K2_c1]"
            )
        ),
        "measure_identity": {
            "physical_volume_form": "|d^3 Omega|^2=d^3 Re(Omega) d^3 Im(Omega)",
            "local_period_form": "K2_c1 * d^3 Re(Omega) d^3 Im(Omega)",
            "unit_mumford_relation": "K2_c1=(2/pi)*I2_Xi_Mumford_residue",
            "alpha_prime": 1.0,
            "dimensionless_radius_definition": "r=R_phys/sqrt(alpha')",
            "scalar_input_normalization": DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION,
            "scalar_integrand_normalization": XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION,
            "xi_scalar_relation": "Z_X^Xi=Z_X^p/(4*pi^2*alpha') at genus two",
            "compact_zero_mode": "2*pi*R_phys",
            "xi_full_replacement_over_dimensionless": (
                xi_full_replacement_over_dimensionless()
            ),
            "g_s_squared_stripped": True,
            "proposal_measure": (
                "four-component physical-measure Minkowski mixture"
                if sampling_scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME
                else (
                    "six-dimensional Minkowski direct-importance proposal"
                    if sampling_scheme == RQMC_SAMPLING_SCHEME
                    else (
                        "p(Omega)|d^3 Omega|^2=|d^3 Omega|^2/"
                        "(Vol(F2)*det(Im Omega)^3)"
                    )
                )
            ),
            "importance_weight": (
                "1_F2*(J_Y/p_mix) per proposal; local observable is K2_c1"
                if sampling_scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME
                else (
                    "1_F2*w*det(Im Omega)^3 per proposal"
                    if sampling_scheme == RQMC_SAMPLING_SCHEME
                    else "Vol(F2)*det(Im Omega)^3"
                )
            ),
            "stack_weight_application": "multiply the coarse-domain estimate by 1/2 once",
            "warning": (
                "The physical-mixture design removes det(Im Omega)^3 from the local observable."
                if sampling_scheme == PHYSICAL_MIXTURE_SAMPLING_SCHEME
                else "det(Im Omega)^3 is proposal reweighting, not a ghost factor"
            ),
        },
        "integration_kernel_convention": STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        "string_note_kernel_multiplier": (
            c1_sphere_normalized_genus2_kernel_multiplier()
        ),
        "c1_sphere_topology_multiplier": c1_genus2_topology_correction(),
        "external_comparison_target": None,
        "modular_covariance_split": {
            "fundamental_domain": (
                "Igusa cusp form, det(Im Omega), all theta-function evaluations "
                "including the compact winding sum, and the coefficient of d^6 Omega"
            ),
            "plumbing_marking": (
                "the modular-invariant same-frame Z_L/Z_X^25 quotient"
            ),
            "guard": (
                "the saved exact Sp(4,Z) matrix must map the fundamental-domain "
                "period matrix to the plumbing period matrix"
            ),
        },
        "full_cft_factorization_certified": FULL_CFT_FACTORIZATION_CERTIFIED,
        "local_cft_normalization_crosscheck": {
            "repository_over_dhp_critical_prefactor": critical_prefactor_ratio_to_dhp(),
            "exact_prefactor_ratio": "(2*pi)^26",
            "scalar_loop_gaussian": (
                "coefficient one follows from delta-normalized momentum states "
                "and completeness measure dp"
            ),
            "interpretation": (
                "the DHP ratio is exactly the 26-scalar target-volume convention "
                "conversion already present at genus one"
            ),
        },
        "settings": {
            **{
                key: _jsonable_argument(value)
                for key, value in vars(args).items()
                if key not in {"source_csv", "out_dir"}
            },
            "compact_theta_implementation": COMPACT_THETA_IMPLEMENTATION,
        },
        "high_order_summary": high_summary,
        "low_order_summary": low_summary,
        "aggregate_discretization_relative_change": discretization_change,
        "node_discretization_diagnostic": node_discretization,
        "warnings": [
            "The tested q<=0.25 range is an empirical recursion benchmark, not a convergence theorem; larger-q nodes require a separate cutoff audit.",
            "A small design may have very large variance if degeneration strata dominate; inspect the shell and component diagnostics.",
            "The complete cusp is integrated with no IR subtraction; Liouville threshold stripping is an exactly restored numerical rescaling only.",
            "For RQMC input, at least two complete independent scrambles are required for a headline estimate.",
            "No failed node may be dropped from an unbiased estimate; complete=false invalidates the headline estimate.",
            "The normal confidence interval is diagnostic only for this small, potentially heavy-tailed pilot.",
            "The aggregate low/high shift can conceal much larger changes at individual nodes.",
            "No external matrix-model target or fitted normalization is used in this computation.",
        ],
    }

    _write_csv(csv_path, rows)
    _write_json_atomic(json_path, payload)
    print("Genus-two c=1 Monte Carlo pilot")
    print(
        f"  scheme={sampling_scheme}, node-complete={complete}, "
        f"integration-complete={integration_complete}, successful={len(successful)}/{len(rows)}"
    )
    if high_summary is not None:
        print(
            f"  F2/g_s^2 estimate={float(high_summary['estimate']):.12g} "
            f"+/- {float(high_summary['standard_error']):.3g}"
        )
        if sampling_scheme in RQMC_SAMPLING_SCHEMES:
            print(
                f"  independent complete scrambles={int(high_summary['replicate_count'])}"
            )
        if sampling_scheme == RQMC_SAMPLING_SCHEME:
            print(
                "  exact-volume-calibrated diagnostic="
                f"{float(high_summary['volume_calibrated_estimate']):.12g}"
            )
        elif sampling_scheme == "iid_invariant_domain":
            print(f"  effective sample size={float(high_summary['effective_sample_size']):.3f}")
            print(f"  largest sample fraction={float(high_summary['largest_sample_fraction']):.3f}")
        print(f"  low/high aggregate change={discretization_change:+.3e}")
    elif sampling_scheme in RQMC_SAMPLING_SCHEMES:
        print("  no headline estimate: selected rows do not contain two complete scrambles")
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")


if __name__ == "__main__":
    run()
