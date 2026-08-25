#!/usr/bin/env python3
"""Restartable blind refinement of the genus-one three-point imaginary scan.

The first four bulk scrambles reuse the frozen 256-point prefixes from
``equal_split_imaginary_t_scan10_p12_n256_v1``.  New Sobol points extend those
same scrambles, and four independent scrambles are added.  The cusp is no
longer completed by a fitted power law: it is integrated directly on
``tau2 in [8,infinity)`` with an exact importance map.  The old fixed-height
tail samples are retained and nestedly refined as a diagnostic only.

This module is worldsheet-only.  It contains no matrix-model expression or
target value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy.special import betaincinv, gammaln
from scipy.stats import qmc

try:
    from genus1_three_point_worldsheet import (
        LiouvilleTorusThreePointNecklace,
        reduced_worldsheet_integrand_three_point,
    )
    from genus1_two_point_worldsheet import MomentumRule
    from integrate_genus1_three_point_worldsheet import (
        fit_tail_power,
        integrated_power_tail,
    )
except ImportError:  # pragma: no cover
    from plumbing.genus1_three_point_worldsheet import (
        LiouvilleTorusThreePointNecklace,
        reduced_worldsheet_integrand_three_point,
    )
    from plumbing.genus1_two_point_worldsheet import MomentumRule
    from plumbing.integrate_genus1_three_point_worldsheet import (
        fit_tail_power,
        integrated_power_tail,
    )


DEFAULT_T_VALUES = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
DEFAULT_PREVIOUS_DIR = Path(
    "plumbing/results/genus1_three_point_worldsheet/"
    "equal_split_imaginary_t_scan10_p12_n256_v1"
)
DEFAULT_OUTPUT_DIR = Path(
    "plumbing/results/genus1_three_point_worldsheet/"
    "equal_split_imaginary_t_scan10_pade8_direct_cusp_n1024_r8_v1"
)
PREVIOUS_POWER = 8
PREVIOUS_REPLICATES = 4
TAIL_SEED_OFFSET = 20000


def _complex(record: Mapping[str, object]) -> complex:
    return complex(float(record["real"]), float(record["imag"]))


def _complex_record(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def point_tag(t: float) -> str:
    return f"t{int(round(100.0 * float(t))):03d}"


def previous_point_path(previous_dir: Path, t: float) -> Path:
    return previous_dir / point_tag(t) / "worldsheet_blind.json"


def output_point_path(output_dir: Path, t: float) -> Path:
    return output_dir / point_tag(t) / "refined_worldsheet_blind.json"


def parse_t_values(text: str) -> tuple[float, ...]:
    values = tuple(float(piece) for piece in text.split(",") if piece.strip())
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("all t values must lie strictly in 0<t<1")
    return values


def mean_and_standard_error(values: Sequence[complex]) -> tuple[complex, complex]:
    array = np.asarray(values, dtype=np.complex128)
    mean = complex(np.mean(array))
    if len(array) < 2:
        return mean, complex(float("nan"), float("nan"))
    standard_error = complex(
        float(np.std(array.real, ddof=1) / math.sqrt(len(array))),
        float(np.std(array.imag, ddof=1) / math.sqrt(len(array))),
    )
    return mean, standard_error


def validate_previous(record: Mapping[str, object], *, t: float) -> None:
    if record.get("blind_freeze") is not True:
        raise ValueError("previous point is not a blind freeze")
    if not math.isclose(float(record["kinematics"]["t"]), t, abs_tol=1.0e-14):
        raise ValueError("previous point has different kinematics")
    momentum = record["momentum_rule"]
    if momentum["kind"] != "power-legendre" or int(momentum["order_per_edge"]) != 12:
        raise ValueError("previous point has an incompatible momentum rule")
    if not math.isclose(float(momentum["p_max"]), 5.0, abs_tol=1.0e-14):
        raise ValueError("previous point has an incompatible momentum cutoff")
    blocks = record["block_design"]
    if blocks["backend"] != "exact-c25-descendants":
        raise ValueError("previous point has an incompatible block backend")
    if int(blocks["high_edge_max_order"]) != 4 or int(blocks["other_edge_order"]) != 2:
        raise ValueError("previous point has incompatible block orders")
    rqmc = record["rqmc"]
    if int(rqmc["sobol_power"]) != PREVIOUS_POWER:
        raise ValueError("previous bulk prefix is not 256 Sobol points")
    if int(rqmc["replicates"]) != PREVIOUS_REPLICATES:
        raise ValueError("previous bulk prefix does not have four scrambles")
    if [float(value) for value in record["cutoffs"]][-1] != 8.0:
        raise ValueError("previous bulk endpoint is not tau2=8")


def make_correlator(
    t: float,
    *,
    momentum_order: int,
    high_order: int = 8,
    cache_path: Path | None = None,
) -> LiouvilleTorusThreePointNecklace:
    rules = tuple(
        MomentumRule.power_legendre(5.0, int(momentum_order), 2.0 + 0.137 * edge)
        for edge in range(3)
    )
    correlator = LiouvilleTorusThreePointNecklace(
        float(t),
        momentum_rules=rules,
        high_order=int(high_order),
        low_order=2,
        adaptive_tolerance=5.0e-5,
        block_backend="exact-c25-descendants",
        special_dps=28,
    )
    if cache_path is not None and cache_path.is_file():
        correlator.load_banks(cache_path)
        print(f"loaded cached block banks from {cache_path}", flush=True)
    else:
        correlator.prepare()
        if cache_path is not None:
            correlator.save_banks(cache_path)
            print(f"cached block banks at {cache_path}", flush=True)
    return correlator


def sobol_points(dimension: int, power: int, seed: int) -> np.ndarray:
    return qmc.Sobol(d=int(dimension), scramble=True, seed=int(seed)).random_base2(
        int(power)
    )


def ordered_gap_importance(
    first_coordinate: float,
    second_coordinate: float,
    *,
    alpha: float,
) -> tuple[float, float, float]:
    r"""Return ordered puncture heights and the labeled-square measure weight.

    The three necklace gaps are sampled from a symmetric
    ``Dirichlet(alpha,alpha,alpha)`` distribution using stick breaking.  The
    returned weight is ``2/pdf``: the factor two restores the two label
    orderings in the original unit square.  At ``alpha=1`` the Dirichlet
    density is two and the weight is exactly one.
    """
    alpha = float(alpha)
    if alpha <= 0.0:
        raise ValueError("gap alpha must be positive")
    epsilon = np.finfo(float).eps
    first_coordinate = min(1.0 - epsilon, max(epsilon, float(first_coordinate)))
    second_coordinate = min(1.0 - epsilon, max(epsilon, float(second_coordinate)))
    first_gap = float(betaincinv(alpha, 2.0 * alpha, first_coordinate))
    split = float(betaincinv(alpha, alpha, second_coordinate))
    second_gap = (1.0 - first_gap) * split
    third_gap = (1.0 - first_gap) * (1.0 - split)
    # For alpha<1, betaincinv can round an exponentially small positive gap
    # to exactly zero even after the Sobol coordinate itself was clipped.
    # The endpoint has zero measure.  Replace only that binary64 underflow by
    # the smallest normal positive value and renormalize the simplex.
    minimum_gap = np.finfo(float).tiny
    gaps = np.maximum(
        np.asarray([first_gap, second_gap, third_gap], dtype=float),
        minimum_gap,
    )
    gaps /= np.sum(gaps)
    first_gap, second_gap, third_gap = (float(value) for value in gaps)
    log_density = float(gammaln(3.0 * alpha) - 3.0 * gammaln(alpha))
    log_density += (alpha - 1.0) * (
        math.log(first_gap) + math.log(second_gap) + math.log(third_gap)
    )
    weight = 2.0 * math.exp(-log_density)
    first_height = first_gap
    second_height = first_gap + second_gap
    return float(first_height), float(second_height), float(weight)


def evaluate_bulk_points(
    correlator: LiouvilleTorusThreePointNecklace,
    points: np.ndarray,
    *,
    cutoff: float = 8.0,
    order_cap: int | None = None,
    record_diagnostics: bool = True,
    position_alpha: float | None = None,
    pade_orders: tuple[int, int] | None = None,
) -> np.ndarray:
    values = np.zeros(len(points), dtype=np.complex128)
    for index, point in enumerate(points):
        tau1 = float(point[0]) - 0.5
        tau2_min = math.sqrt(1.0 - tau1 * tau1)
        tau2 = tau2_min + float(point[1]) * (float(cutoff) - tau2_min)
        tau = tau1 + 1.0j * tau2
        if position_alpha is None:
            position_pairs = [
                (
                    2.0 * math.pi * (float(point[2]) + float(point[3]) * tau),
                    2.0 * math.pi * (float(point[4]) + float(point[5]) * tau),
                )
            ]
            position_weight = 1.0
        else:
            first_horizontal, second_horizontal, horizontal_weight = ordered_gap_importance(
                float(point[2]),
                float(point[3]),
                alpha=position_alpha,
            )
            first_height, second_height, vertical_weight = ordered_gap_importance(
                float(point[4]),
                float(point[5]),
                alpha=position_alpha,
            )
            position_pairs = [
                (
                    2.0 * math.pi * (first_horizontal + first_height * tau),
                    2.0 * math.pi * (second_horizontal + second_height * tau),
                ),
                (
                    2.0 * math.pi * (second_horizontal + first_height * tau),
                    2.0 * math.pi * (first_horizontal + second_height * tau),
                ),
            ]
            position_weight = 0.5 * horizontal_weight * vertical_weight
        jacobian = (
            position_weight
            * (float(cutoff) - tau2_min)
            * (2.0 * math.pi) ** 4
            * tau2**2
        )
        values[index] = jacobian * sum(
            reduced_worldsheet_integrand_three_point(
                correlator,
                w1,
                w2,
                tau,
                order_cap=order_cap,
                record_diagnostics=record_diagnostics,
                pade_orders=pade_orders,
            )
            for w1, w2 in position_pairs
        )
    return values


def tail_height_and_jacobian(
    coordinate: float,
    *,
    tail_start: float,
    proposal_exponent: float,
) -> tuple[float, float]:
    r"""Map a uniform coordinate to an exact integral on ``[tail_start,infinity)``.

    The proposal survival function is proportional to
    ``tau2**(1-proposal_exponent)``.  It changes variance only; no asymptotic
    ansatz enters the estimator.
    """
    exponent = float(proposal_exponent)
    if exponent <= 1.0:
        raise ValueError("proposal exponent must exceed one")
    complement = 1.0 - float(coordinate)
    if not 0.0 < complement <= 1.0:
        raise ValueError("tail coordinate must lie in [0,1)")
    alpha = exponent - 1.0
    tau2 = float(tail_start) * complement ** (-1.0 / alpha)
    jacobian = (
        float(tail_start)
        / alpha
        * complement ** (-1.0 / alpha - 1.0)
    )
    return float(tau2), float(jacobian)


def evaluate_direct_tail_points(
    correlator: LiouvilleTorusThreePointNecklace,
    points: np.ndarray,
    *,
    tail_start: float = 8.0,
    proposal_exponent: float = 1.5,
    order_cap: int | None = None,
    record_diagnostics: bool = True,
    position_alpha: float | None = None,
    pade_orders: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(len(points), dtype=np.complex128)
    heights = np.zeros(len(points), dtype=float)
    for index, point in enumerate(points):
        tau2, tau2_jacobian = tail_height_and_jacobian(
            float(point[0]),
            tail_start=tail_start,
            proposal_exponent=proposal_exponent,
        )
        tau = (float(point[1]) - 0.5) + 1.0j * tau2
        if position_alpha is None:
            position_pairs = [
                (
                    2.0 * math.pi * (float(point[2]) + float(point[3]) * tau),
                    2.0 * math.pi * (float(point[4]) + float(point[5]) * tau),
                )
            ]
            position_weight = 1.0
        else:
            first_horizontal, second_horizontal, horizontal_weight = ordered_gap_importance(
                float(point[2]),
                float(point[3]),
                alpha=position_alpha,
            )
            first_height, second_height, vertical_weight = ordered_gap_importance(
                float(point[4]),
                float(point[5]),
                alpha=position_alpha,
            )
            position_pairs = [
                (
                    2.0 * math.pi * (first_horizontal + first_height * tau),
                    2.0 * math.pi * (second_horizontal + second_height * tau),
                ),
                (
                    2.0 * math.pi * (second_horizontal + first_height * tau),
                    2.0 * math.pi * (first_horizontal + second_height * tau),
                ),
            ]
            position_weight = 0.5 * horizontal_weight * vertical_weight
        position_jacobian = position_weight * (2.0 * math.pi) ** 4 * tau2**2
        reduced = sum(
            reduced_worldsheet_integrand_three_point(
                correlator,
                w1,
                w2,
                tau,
                order_cap=order_cap,
                record_diagnostics=record_diagnostics,
                pade_orders=pade_orders,
            )
            for w1, w2 in position_pairs
        )
        values[index] = tau2_jacobian * position_jacobian * reduced
        heights[index] = tau2
        if not (math.isfinite(values[index].real) and math.isfinite(values[index].imag)):
            raise FloatingPointError(f"non-finite direct-tail value at tau2={tau2:.8e}")
    return values, heights


def evaluate_fixed_slice_points(
    correlator: LiouvilleTorusThreePointNecklace,
    points: np.ndarray,
    *,
    tau2: float,
    order_cap: int | None = None,
) -> np.ndarray:
    values = np.zeros(len(points), dtype=np.complex128)
    for index, point in enumerate(points):
        tau = (float(point[0]) - 0.5) + 1.0j * float(tau2)
        w1 = 2.0 * math.pi * (float(point[1]) + float(point[2]) * tau)
        w2 = 2.0 * math.pi * (float(point[3]) + float(point[4]) * tau)
        values[index] = (2.0 * math.pi) ** 4 * float(tau2) ** 2
        values[index] *= reduced_worldsheet_integrand_three_point(
            correlator,
            w1,
            w2,
            tau,
            order_cap=order_cap,
            record_diagnostics=False,
        )
    return values


def combine_saved_prefix(
    saved_mean: complex,
    new_values: np.ndarray,
    *,
    previous_power: int,
    final_power: int,
) -> dict[int, complex]:
    previous_count = 2**int(previous_power)
    expected_new = 2**int(final_power) - previous_count
    if len(new_values) != expected_new:
        raise ValueError("nested continuation has the wrong number of new points")
    running_sum = complex(saved_mean) * previous_count
    result = {int(previous_power): complex(saved_mean)}
    consumed = 0
    for power in range(int(previous_power) + 1, int(final_power) + 1):
        target_new = 2**power - previous_count
        running_sum += complex(np.sum(new_values[consumed:target_new]))
        consumed = target_new
        result[power] = running_sum / (2**power)
    return result


def nested_means(values: np.ndarray, *, minimum_power: int, final_power: int) -> dict[int, complex]:
    return {
        power: complex(np.mean(values[: 2**power]))
        for power in range(int(minimum_power), int(final_power) + 1)
    }


def continue_previous_bulk(
    correlator: LiouvilleTorusThreePointNecklace,
    previous: Mapping[str, object],
    *,
    base_seed: int,
    target_power: int,
) -> dict[str, object]:
    replicate_stages: list[dict[int, complex]] = []
    reproduction_errors: list[float] = []
    for replicate in range(PREVIOUS_REPLICATES):
        points = sobol_points(6, target_power, base_seed + replicate)
        prefix_values = evaluate_bulk_points(
            correlator,
            points[: 2**PREVIOUS_POWER],
            order_cap=4,
            record_diagnostics=False,
        )
        saved_mean = _complex(previous["replicate_values"][replicate][-1])
        reproduction_errors.append(abs(complex(np.mean(prefix_values)) - saved_mean))
        new_values = evaluate_bulk_points(
            correlator,
            points[2**PREVIOUS_POWER :],
            order_cap=4,
            record_diagnostics=False,
        )
        replicate_stages.append(
            combine_saved_prefix(
                saved_mean,
                new_values,
                previous_power=PREVIOUS_POWER,
                final_power=target_power,
            )
        )
    stages: dict[str, object] = {}
    for power in range(PREVIOUS_POWER, target_power + 1):
        values = [row[power] for row in replicate_stages]
        mean, standard_error = mean_and_standard_error(values)
        stages[str(power)] = {
            "mean": _complex_record(mean),
            "rqmc_standard_error": _complex_record(standard_error),
            "replicate_values": [_complex_record(value) for value in values],
        }
    return {
        "status": "reused uniform-design diagnostic; excluded from refined central value",
        "reused_points": PREVIOUS_REPLICATES * 2**PREVIOUS_POWER,
        "added_points": PREVIOUS_REPLICATES
        * (2**target_power - 2**PREVIOUS_POWER),
        "sobol_power": int(target_power),
        "maximum_prefix_reproduction_error": float(max(reproduction_errors)),
        "nested_stages": stages,
    }


def continue_previous_slices(
    correlator: LiouvilleTorusThreePointNecklace,
    previous: Mapping[str, object],
    *,
    base_seed: int,
    target_power: int,
) -> dict[str, object]:
    tail = previous["tail_completion"]
    heights = [float(value) for value in tail["tau2_slices"]]
    saved = tail["replicate_slice_values"]
    replicate_profiles: list[list[complex]] = []
    for replicate in range(PREVIOUS_REPLICATES):
        profile: list[complex] = []
        for slice_index, height in enumerate(heights):
            points = sobol_points(
                5,
                target_power,
                base_seed + 10000 + 97 * replicate + slice_index,
            )
            new_values = evaluate_fixed_slice_points(
                correlator,
                points[2**PREVIOUS_POWER :],
                tau2=height,
                order_cap=4,
            )
            saved_mean = _complex(saved[replicate][slice_index])
            continued = combine_saved_prefix(
                saved_mean,
                new_values,
                previous_power=PREVIOUS_POWER,
                final_power=target_power,
            )
            profile.append(continued[target_power])
        replicate_profiles.append(profile)
    profile_array = np.asarray(replicate_profiles, dtype=np.complex128)
    slice_means = np.mean(profile_array, axis=0)
    exponent, first, second, residual = fit_tail_power(
        np.asarray(heights),
        slice_means,
    )
    fitted_tail = integrated_power_tail(8.0, exponent, first, second)
    return {
        "status": "refined diagnostic only; excluded from the central estimator",
        "reused_points": PREVIOUS_REPLICATES * len(heights) * 2**PREVIOUS_POWER,
        "added_points": PREVIOUS_REPLICATES
        * len(heights)
        * (2**target_power - 2**PREVIOUS_POWER),
        "sobol_power": int(target_power),
        "tau2_slices": heights,
        "replicate_slice_values": [
            [_complex_record(value) for value in row] for row in profile_array
        ],
        "slice_means": [_complex_record(value) for value in slice_means],
        "fitted_exponent": float(exponent),
        "relative_fit_residual": float(residual),
        "fitted_integrated_tail": _complex_record(fitted_tail),
    }


def shell_summary(
    replicate_values: Sequence[np.ndarray],
    replicate_heights: Sequence[np.ndarray],
) -> list[dict[str, object]]:
    edges = np.asarray(
        [8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 4096.0, 16384.0, np.inf]
    )
    count = len(replicate_values[0])
    rows: list[dict[str, object]] = []
    for left, right in zip(edges[:-1], edges[1:]):
        contributions = []
        sample_counts = []
        for values, heights in zip(replicate_values, replicate_heights):
            mask = (heights >= left) & (heights < right)
            contributions.append(complex(np.sum(values[mask]) / count))
            sample_counts.append(int(np.sum(mask)))
        mean, standard_error = mean_and_standard_error(contributions)
        rows.append(
            {
                "tau2_min": float(left),
                "tau2_max": None if math.isinf(right) else float(right),
                "mean_contribution": _complex_record(mean),
                "rqmc_standard_error": _complex_record(standard_error),
                "sample_counts_by_replicate": sample_counts,
            }
        )
    return rows


def conservative_component_bound(
    *,
    rqmc_se: complex,
    nested_shift: complex,
    momentum_shift: complex,
    momentum_se: complex,
    block_resummation_bound: complex,
) -> complex:
    return complex(
        2.0 * abs(rqmc_se.real)
        + abs(nested_shift.real)
        + abs(momentum_shift.real)
        + 2.0 * abs(momentum_se.real)
        + abs(block_resummation_bound.real),
        2.0 * abs(rqmc_se.imag)
        + abs(nested_shift.imag)
        + abs(momentum_shift.imag)
        + 2.0 * abs(momentum_se.imag)
        + abs(block_resummation_bound.imag),
    )


def refine_point(
    t: float,
    *,
    previous_path: Path,
    output_path: Path,
    bulk_power: int,
    tail_power: int,
    replicates: int,
    base_seed: int,
    proposal_exponent: float,
    slice_power: int,
    audit_power: int,
    momentum_audit_order: int,
    position_alpha: float,
    position_audit_alpha: float,
    position_audit_power: int,
    legacy_bulk_power: int,
) -> dict[str, object]:
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    validate_previous(previous, t=t)
    if replicates < PREVIOUS_REPLICATES:
        raise ValueError("refinement must keep all four previous scrambles")
    if bulk_power < audit_power or tail_power < audit_power:
        raise ValueError("refinement powers are inconsistent")
    if position_audit_power > min(bulk_power, tail_power):
        raise ValueError("position audit power exceeds the central sample power")
    if legacy_bulk_power < PREVIOUS_POWER:
        raise ValueError("legacy bulk continuation cannot discard saved points")

    correlator = make_correlator(
        t,
        momentum_order=12,
        high_order=8,
        cache_path=output_path.parent / "block_bank_p12_o8.npz",
    )
    bulk_nested: dict[int, list[complex]] = {
        power: [] for power in range(audit_power, bulk_power + 1)
    }
    tail_nested: dict[int, list[complex]] = {
        power: [] for power in range(audit_power, tail_power + 1)
    }
    bulk_audit_points: list[np.ndarray] = []
    bulk_audit_main: list[np.ndarray] = []
    bulk_full_values: list[np.ndarray] = []
    tail_audit_points: list[np.ndarray] = []
    tail_audit_main: list[np.ndarray] = []
    tail_full_values: list[np.ndarray] = []
    tail_full_heights: list[np.ndarray] = []

    for replicate in range(replicates):
        bulk_points = sobol_points(6, bulk_power, base_seed + replicate)
        audit_count = 2**audit_power
        all_values = evaluate_bulk_points(
            correlator,
            bulk_points,
            position_alpha=position_alpha,
            pade_orders=(4, 4),
        )
        audit_main = all_values[:audit_count]
        stages = nested_means(
            all_values,
            minimum_power=audit_power,
            final_power=bulk_power,
        )
        for power, value in stages.items():
            bulk_nested[power].append(value)
        bulk_full_values.append(all_values)
        if replicate < PREVIOUS_REPLICATES:
            bulk_audit_points.append(bulk_points[:audit_count])
            bulk_audit_main.append(audit_main)

        tail_points = sobol_points(6, tail_power, base_seed + TAIL_SEED_OFFSET + replicate)
        tail_values, tail_heights = evaluate_direct_tail_points(
            correlator,
            tail_points,
            proposal_exponent=proposal_exponent,
            position_alpha=position_alpha,
            pade_orders=(4, 4),
        )
        tail_stages = nested_means(
            tail_values,
            minimum_power=audit_power,
            final_power=tail_power,
        )
        for power, value in tail_stages.items():
            tail_nested[power].append(value)
        tail_full_values.append(tail_values)
        tail_full_heights.append(tail_heights)
        if replicate < PREVIOUS_REPLICATES:
            tail_audit_points.append(tail_points[:audit_count])
            tail_audit_main.append(tail_values[:audit_count])
        print(
            f"t={t:.2f} p12 replicate {replicate + 1}/{replicates}: "
            f"bulk={stages[bulk_power].imag:+.7e}j "
            f"tail={tail_stages[tail_power].imag:+.7e}j",
            flush=True,
        )

    common_powers = sorted(set(bulk_nested) & set(tail_nested))
    stage_records: dict[str, object] = {}
    final_replicates_by_power: dict[int, list[complex]] = {}
    for power in common_powers:
        finals = [
            bulk + tail
            for bulk, tail in zip(bulk_nested[power], tail_nested[power])
        ]
        final_replicates_by_power[power] = finals
        mean, standard_error = mean_and_standard_error(finals)
        bulk_mean, bulk_se = mean_and_standard_error(bulk_nested[power])
        tail_mean, tail_se = mean_and_standard_error(tail_nested[power])
        stage_records[str(power)] = {
            "points_per_replicate": 2**power,
            "bulk_mean": _complex_record(bulk_mean),
            "bulk_rqmc_standard_error": _complex_record(bulk_se),
            "direct_tail_mean": _complex_record(tail_mean),
            "direct_tail_rqmc_standard_error": _complex_record(tail_se),
            "final_mean": _complex_record(mean),
            "final_rqmc_standard_error": _complex_record(standard_error),
            "replicate_finals": [_complex_record(value) for value in finals],
        }

    final_power = min(bulk_power, tail_power)
    final_values = final_replicates_by_power[final_power]
    final_mean, final_se = mean_and_standard_error(final_values)
    previous_stage_mean, _ = mean_and_standard_error(
        final_replicates_by_power[final_power - 1]
    )
    nested_shift = final_mean - previous_stage_mean

    block_pade_ladder = ((3, 3), (4, 3), (3, 4))
    block_differences: dict[tuple[int, int], list[complex]] = {
        orders: [] for orders in block_pade_ladder
    }
    for bulk_points, bulk_main, tail_points, tail_main in zip(
        bulk_audit_points,
        bulk_audit_main,
        tail_audit_points,
        tail_audit_main,
    ):
        for pade_orders in block_pade_ladder:
            bulk_lower = evaluate_bulk_points(
                correlator,
                bulk_points,
                record_diagnostics=False,
                position_alpha=position_alpha,
                pade_orders=pade_orders,
            )
            tail_lower, _ = evaluate_direct_tail_points(
                correlator,
                tail_points,
                proposal_exponent=proposal_exponent,
                record_diagnostics=False,
                position_alpha=position_alpha,
                pade_orders=pade_orders,
            )
            block_differences[pade_orders].append(
                complex(
                    np.mean(bulk_main - bulk_lower)
                    + np.mean(tail_main - tail_lower)
                )
            )
    block_ladder_records: dict[str, object] = {}
    block_component_bounds: list[complex] = []
    for pade_orders, differences in block_differences.items():
        shift, standard_error = mean_and_standard_error(differences)
        block_ladder_records[f"{pade_orders[0]}/{pade_orders[1]}"] = {
            "paired_mean_shift_4/4_minus_neighbor": _complex_record(shift),
            "paired_rqmc_standard_error": _complex_record(standard_error),
        }
        block_component_bounds.append(
            complex(
                abs(shift.real) + 2.0 * abs(standard_error.real),
                abs(shift.imag) + 2.0 * abs(standard_error.imag),
            )
        )
    block_resummation_bound = complex(
        max(value.real for value in block_component_bounds),
        max(value.imag for value in block_component_bounds),
    )

    momentum_correlator = make_correlator(
        t,
        momentum_order=momentum_audit_order,
        high_order=8,
        cache_path=(
            output_path.parent
            / f"block_bank_p{int(momentum_audit_order)}_o8.npz"
        ),
    )
    momentum_differences: list[complex] = []
    for bulk_points, bulk_main, tail_points, tail_main in zip(
        bulk_audit_points,
        bulk_audit_main,
        tail_audit_points,
        tail_audit_main,
    ):
        bulk_high = evaluate_bulk_points(
            momentum_correlator,
            bulk_points,
            record_diagnostics=False,
            position_alpha=position_alpha,
            pade_orders=(4, 4),
        )
        tail_high, _ = evaluate_direct_tail_points(
            momentum_correlator,
            tail_points,
            proposal_exponent=proposal_exponent,
            record_diagnostics=False,
            position_alpha=position_alpha,
            pade_orders=(4, 4),
        )
        momentum_differences.append(
            complex(np.mean(bulk_high - bulk_main) + np.mean(tail_high - tail_main))
        )
    momentum_shift, momentum_se = mean_and_standard_error(momentum_differences)

    position_differences: list[complex] = []
    position_count = 2**position_audit_power
    for replicate in range(PREVIOUS_REPLICATES):
        bulk_points = sobol_points(6, bulk_power, base_seed + replicate)[:position_count]
        tail_points = sobol_points(
            6,
            tail_power,
            base_seed + TAIL_SEED_OFFSET + replicate,
        )[:position_count]
        alternate_bulk = evaluate_bulk_points(
            correlator,
            bulk_points,
            record_diagnostics=False,
            position_alpha=position_audit_alpha,
            pade_orders=(4, 4),
        )
        alternate_tail, _ = evaluate_direct_tail_points(
            correlator,
            tail_points,
            proposal_exponent=proposal_exponent,
            record_diagnostics=False,
            position_alpha=position_audit_alpha,
            pade_orders=(4, 4),
        )
        central_at_audit = complex(
            np.mean(bulk_full_values[replicate][:position_count])
            + np.mean(tail_full_values[replicate][:position_count])
        )
        position_differences.append(
            complex(np.mean(alternate_bulk) + np.mean(alternate_tail) - central_at_audit)
        )
    position_shift, position_se = mean_and_standard_error(position_differences)

    reused_bulk = continue_previous_bulk(
        correlator,
        previous,
        base_seed=base_seed,
        target_power=legacy_bulk_power,
    )

    refined_slices = continue_previous_slices(
        correlator,
        previous,
        base_seed=base_seed,
        target_power=slice_power,
    )
    conservative_half_width = conservative_component_bound(
        rqmc_se=final_se,
        nested_shift=nested_shift,
        momentum_shift=momentum_shift,
        momentum_se=momentum_se,
        block_resummation_bound=block_resummation_bound,
    )
    direct_tail_mean, direct_tail_se = mean_and_standard_error(tail_nested[final_power])
    bulk_mean, bulk_se = mean_and_standard_error(bulk_nested[final_power])

    result: dict[str, object] = {
        "calculation": "refined direct-cusp c=1 genus-one three-point worldsheet integral",
        "blind_worldsheet_freeze": True,
        "comparison_stage_present": False,
        "kinematics": previous["kinematics"],
        "normalization": previous["native_normalization"],
        "previous_data_reuse": {
            "source": str(previous_path),
            "source_sha256": _sha256(previous_path),
            "central_estimator_uses_previous_aggregate_values": False,
            "reason": (
                "the refined central estimator changes to exact two-cycle gap importance; "
                "the previous uniform samples are retained in an independent nested diagnostic"
            ),
            "bulk_uniform_diagnostic": reused_bulk,
            "fixed_slice_prefix_points_reused": int(refined_slices["reused_points"]),
        },
        "design": {
            "momentum_rule": "three distinct power-Legendre rules on [0,5]",
            "momentum_order": 12,
            "block_backend": "exact-c25-descendants",
            "block_orders": {"high_edge_cap": 8, "other_edges": 2},
            "central_block_resummation": "row-wise Pade [4/4] in the largest elliptic nome",
            "bulk_cutoff": 8.0,
            "bulk_sobol_power": int(bulk_power),
            "direct_tail_sobol_power": int(tail_power),
            "replicates": int(replicates),
            "base_seed": int(base_seed),
            "two_cycle_ordered_gap_importance_alpha": float(position_alpha),
            "both_relative_label_assignments_averaged": True,
            "direct_tail_map": (
                "tau2=8*(1-u)^(-1/(proposal_exponent-1)); exact Jacobian"
            ),
            "tail_proposal_exponent": float(proposal_exponent),
            "tail_asymptotic_fit_used_in_central_value": False,
        },
        "nested_convergence": stage_records,
        "bulk": {
            "mean": _complex_record(bulk_mean),
            "rqmc_standard_error": _complex_record(bulk_se),
            "replicate_values": [_complex_record(value) for value in bulk_nested[final_power]],
        },
        "direct_tail": {
            "mean": _complex_record(direct_tail_mean),
            "rqmc_standard_error": _complex_record(direct_tail_se),
            "replicate_values": [_complex_record(value) for value in tail_nested[final_power]],
            "maximum_tau2_sampled": float(max(np.max(row) for row in tail_full_heights)),
            "shells": shell_summary(tail_full_values, tail_full_heights),
        },
        "refined_legacy_slice_diagnostic": refined_slices,
        "systematic_audits": {
            "nested_last_stage_shift": _complex_record(nested_shift),
            "block_resummation_ladder": {
                "central": "4/4",
                "neighbors": block_ladder_records,
                "conservative_componentwise_bound": _complex_record(
                    block_resummation_bound
                ),
                "replicates": PREVIOUS_REPLICATES,
                "points_per_region_replicate": 2**audit_power,
            },
            f"momentum_order_{momentum_audit_order}_minus_12": {
                "paired_mean_shift": _complex_record(momentum_shift),
                "paired_rqmc_standard_error": _complex_record(momentum_se),
                "replicates": PREVIOUS_REPLICATES,
                "points_per_region_replicate": 2**audit_power,
            },
            "position_importance_exponent_audit": {
                "central_alpha": float(position_alpha),
                "alternate_alpha": float(position_audit_alpha),
                "paired_mean_shift_alternate_minus_central": _complex_record(position_shift),
                "paired_rqmc_standard_error": _complex_record(position_se),
                "replicates": PREVIOUS_REPLICATES,
                "points_per_region_replicate": 2**position_audit_power,
            },
        },
        "final_I_1,3": _complex_record(final_mean),
        "rqmc_standard_error": _complex_record(final_se),
        "conservative_componentwise_half_width": _complex_record(conservative_half_width),
        "conservative_interval_for_minus_imaginary_part": {
            "central": float(-final_mean.imag),
            "lower": float(max(0.0, -final_mean.imag - conservative_half_width.imag)),
            "upper": float(-final_mean.imag + conservative_half_width.imag),
        },
        "block_diagnostics_p12": correlator.diagnostics(),
        f"block_diagnostics_p{momentum_audit_order}": momentum_correlator.diagnostics(),
    }
    _atomic_json(output_path, result)
    print(
        f"t={t:.2f} frozen: -Im I={-final_mean.imag:.9g} "
        f"RQMC_SE={abs(final_se.imag):.3g} "
        f"conservative_half_width={conservative_half_width.imag:.3g}",
        flush=True,
    )
    return result


def assemble_summary(output_dir: Path, t_values: Sequence[float]) -> dict[str, object]:
    points: list[dict[str, object]] = []
    for t in t_values:
        path = output_point_path(output_dir, t)
        record = json.loads(path.read_text(encoding="utf-8"))
        points.append(
            {
                "t": float(t),
                "path": str(path),
                "sha256": _sha256(path),
                "minus_imaginary_I_1,3": float(-record["final_I_1,3"]["imag"]),
                "rqmc_standard_error": float(abs(record["rqmc_standard_error"]["imag"])),
                "conservative_interval": record[
                    "conservative_interval_for_minus_imaginary_part"
                ],
                "direct_tail_fraction": float(
                    abs(_complex(record["direct_tail"]["mean"]))
                    / max(abs(_complex(record["final_I_1,3"])), 1.0e-300)
                ),
                "maximum_tau2_sampled": float(record["direct_tail"]["maximum_tau2_sampled"]),
            }
        )
    payload = {
        "calculation": "blind refined ten-point torus three-point worldsheet scan",
        "comparison_stage_present": False,
        "central_tail_estimator": "direct importance-sampled integral to tau2=infinity",
        "t_values": [float(value) for value in t_values],
        "points": points,
    }
    _atomic_json(output_dir / "refined_worldsheet_scan_manifest.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser()
    out.add_argument("--t-values", default=",".join(str(value) for value in DEFAULT_T_VALUES))
    out.add_argument("--previous-dir", type=Path, default=DEFAULT_PREVIOUS_DIR)
    out.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    out.add_argument("--bulk-power", type=int, default=10)
    out.add_argument("--tail-power", type=int, default=10)
    out.add_argument("--replicates", type=int, default=8)
    out.add_argument("--base-seed", type=int, default=17051301)
    out.add_argument("--tail-proposal-exponent", type=float, default=1.5)
    out.add_argument("--slice-power", type=int, default=9)
    out.add_argument("--audit-power", type=int, default=7)
    out.add_argument("--momentum-audit-order", type=int, default=10)
    out.add_argument("--position-alpha", type=float, default=0.3)
    out.add_argument("--position-audit-alpha", type=float, default=0.4)
    out.add_argument("--position-audit-power", type=int, default=8)
    out.add_argument("--legacy-bulk-power", type=int, default=9)
    out.add_argument("--check-existing", action="store_true")
    return out


def main() -> None:
    args = parser().parse_args()
    t_values = parse_t_values(args.t_values)
    for index, t in enumerate(t_values):
        previous = previous_point_path(args.previous_dir, t)
        output = output_point_path(args.output_dir, t)
        if output.exists():
            if not args.check_existing:
                raise FileExistsError(f"refusing to overwrite {output}")
            print(f"keeping completed point {index + 1}/{len(t_values)} at t={t:.2f}")
            continue
        print(f"refining point {index + 1}/{len(t_values)} at t={t:.2f}", flush=True)
        refine_point(
            t,
            previous_path=previous,
            output_path=output,
            bulk_power=args.bulk_power,
            tail_power=args.tail_power,
            replicates=args.replicates,
            base_seed=args.base_seed,
            proposal_exponent=args.tail_proposal_exponent,
            slice_power=args.slice_power,
            audit_power=args.audit_power,
            momentum_audit_order=args.momentum_audit_order,
            position_alpha=args.position_alpha,
            position_audit_alpha=args.position_audit_alpha,
            position_audit_power=args.position_audit_power,
            legacy_bulk_power=args.legacy_bulk_power,
        )
    summary = assemble_summary(args.output_dir, t_values)
    print(
        f"wrote {args.output_dir / 'refined_worldsheet_scan_manifest.json'} "
        f"with {len(summary['points'])} points",
        flush=True,
    )


if __name__ == "__main__":
    main()
