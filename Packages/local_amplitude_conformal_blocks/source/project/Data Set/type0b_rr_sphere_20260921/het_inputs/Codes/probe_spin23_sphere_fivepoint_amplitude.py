#!/usr/bin/env python3
r"""Convergence probe for the physical diagonal SO(23) sphere five-point kernel.

The probe uses equal imaginary outgoing frequencies and records three distinct
errors: descendant cutoff, two-momentum quadrature, and (when requested) the
randomized-QMC error in one truncated linear-channel plumbing chart.  The
one-chart number is not a crossing-complete sphere amplitude.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

from spin23_sphere_fivepoint_amplitude import (
    SphereFivePointState,
    evaluate_fixed_fivepoint_node,
    integrate_fivepoint_linear_chamber,
    integrate_fivepoint_spectral_kernel,
)


def _complex_data(value: complex) -> dict[str, float]:
    number = complex(value)
    return {
        "real": float(number.real),
        "imag": float(number.imag),
        "abs": float(abs(number)),
    }


def _relative_change(current: complex, previous: complex) -> float:
    return float(
        abs(current - previous)
        / max(abs(current), abs(previous), 1.0e-300)
    )


def equal_imaginary_singlet_states(
    t: float,
) -> tuple[SphereFivePointState, ...]:
    """Return four outgoing ``it`` singlets and the neutral incoming leg."""

    if not math.isfinite(t) or not 0 < t < 0.2:
        raise ValueError("the conservative equal-imaginary probe uses 0<t<0.2")
    outgoing = tuple(
        SphereFivePointState.singlet(1j * t, 1j * t) for _ in range(4)
    )
    incoming = SphereFivePointState.singlet(4j * t, -4j * t)
    return outgoing + (incoming,)


def run_physical_probe(
    *,
    t: float = 0.12,
    q_1: complex = 0.18 + 0.05j,
    q_2: complex = 0.22 - 0.04j,
    internal_momenta: Sequence[float] = (0.37, 0.61),
    maximum_twice_levels: Sequence[int] = (2, 4, 6),
    spectral_orders: Sequence[int] = (9, 15),
    spectral_p_max: float = 3.0,
    structure_precision: int = 32,
    block_digits: int = 40,
    chamber_q_maxima: Sequence[float] = (0.25, 0.25),
    chamber_sample_power: int | None = None,
    chamber_replicates: int = 4,
    seed: int = 1729,
    include_point_profiles: bool = True,
) -> dict[str, Any]:
    """Run fixed-node, spectral, and optional one-chart convergence checks."""

    states = equal_imaginary_singlet_states(t)
    q_1, q_2 = complex(q_1), complex(q_2)
    if not 0 < abs(q_1) < 1 or not 0 < abs(q_2) < 1:
        raise ValueError("the probe requires 0<|q1|,|q2|<1")
    z_3, z_2 = q_2, q_1 * q_2
    internal = tuple(float(value) for value in internal_momenta)
    if len(internal) != 2:
        raise ValueError("two internal momenta are required")
    cutoffs = tuple(int(value) for value in maximum_twice_levels)
    if not cutoffs or any(value < 0 for value in cutoffs):
        raise ValueError("at least one nonnegative descendant cutoff is required")
    if tuple(sorted(set(cutoffs))) != cutoffs:
        raise ValueError("descendant cutoffs must be strictly increasing")
    orders = tuple(int(value) for value in spectral_orders)
    if len(orders) != 2 or not 0 < orders[0] < orders[1]:
        raise ValueError("spectral_orders must contain two increasing orders")

    fixed_rows: list[dict[str, Any]] | None = None
    spectral_row = None
    if include_point_profiles:
        fixed_rows = []
        previous_value: complex | None = None
        for cutoff in cutoffs:
            started = time.perf_counter()
            evaluation = evaluate_fixed_fivepoint_node(
                states,
                z_2,
                z_3,
                internal,
                maximum_twice_levels=(cutoff, cutoff),
                structure_precision=structure_precision,
                block_digits=block_digits,
            )
            row: dict[str, Any] = {
                "maximum_twice_levels": [cutoff, cutoff],
                "value": _complex_data(evaluation.value),
                "elapsed_seconds": time.perf_counter() - started,
                "maximum_gram_condition": evaluation.maximum_gram_condition,
                "pco_cancellation_ratio": evaluation.cancellation_ratio,
                "pco_components": [
                    {
                        "time_fermion_legs": list(
                            contribution.component.time_fermion_legs
                        ),
                        "value_before_common_factor": _complex_data(
                            contribution.value_before_common_factor
                        ),
                    }
                    for contribution in evaluation.pco_contributions
                ],
            }
            if previous_value is not None:
                row["relative_change_from_previous_cutoff"] = _relative_change(
                    evaluation.value, previous_value
                )
            fixed_rows.append(row)
            previous_value = evaluation.value

        started = time.perf_counter()
        spectral = integrate_fivepoint_spectral_kernel(
            states,
            z_2,
            z_3,
            maximum_twice_levels=(cutoffs[-1], cutoffs[-1]),
            order=orders[0],
            refined_order=orders[1],
            method="gauss_legendre",
            p_max=spectral_p_max,
            structure_precision=structure_precision,
            block_digits=block_digits,
        )
        quadrature = spectral.quadrature
        spectral_row = {
            "maximum_twice_levels": [cutoffs[-1], cutoffs[-1]],
            "orders": list(orders),
            "p_max": spectral_p_max,
            "coarse_value": _complex_data(quadrature.value),
            "refined_value": _complex_data(complex(quadrature.refined_value)),
            "relative_refinement_change": quadrature.relative_error,
            "elapsed_seconds": time.perf_counter() - started,
            "maximum_gram_condition": spectral.maximum_gram_condition,
            "maximum_node_pco_cancellation_ratio": (
                spectral.maximum_node_cancellation_ratio
            ),
            "coarse_evaluation_count": quadrature.evaluation_count,
            "refined_evaluation_count": quadrature.refined_evaluation_count,
        }

    chamber_row = None
    if chamber_sample_power is not None:
        started = time.perf_counter()
        chamber = integrate_fivepoint_linear_chamber(
            states,
            q_maxima=chamber_q_maxima,
            sample_power=chamber_sample_power,
            replicates=chamber_replicates,
            seed=seed,
            maximum_twice_levels=(cutoffs[-1], cutoffs[-1]),
            spectral_order=orders[-1],
            spectral_method="gauss_legendre",
            spectral_p_max=spectral_p_max,
            structure_precision=structure_precision,
            block_digits=block_digits,
        )
        chamber_row = {
            "scope": "one_truncated_linear_channel_chart_only",
            "value": _complex_data(chamber.value),
            "replicate_values": [
                _complex_data(value) for value in chamber.replicate_values
            ],
            "q_maxima": list(chamber.q_maxima),
            "sample_power": chamber.sample_power,
            "samples_per_replicate": chamber.samples_per_replicate,
            "replicates": chamber.replicates,
            "maximum_twice_levels": [cutoffs[-1], cutoffs[-1]],
            "spectral_order": orders[-1],
            "spectral_p_max": spectral_p_max,
            "seed": seed,
            "relative_standard_error": chamber.relative_standard_error,
            "elapsed_seconds": time.perf_counter() - started,
        }

    return {
        "scope": "physical_diagonal_super_liouville_reduced_kernel",
        "external_process": "four_equal_imaginary_outgoing_singlets",
        "equal_imaginary_t": t,
        "conservative_no_residue_margin": 1.0 - 5.0 * t,
        "q_1": _complex_data(q_1),
        "q_2": _complex_data(q_2),
        "z_2": _complex_data(z_2),
        "z_3": _complex_data(z_3),
        "internal_momenta_for_fixed_probe": list(internal),
        "fixed_node_descendant_profile": fixed_rows,
        "two_momentum_quadrature_profile": spectral_row,
        "linear_chamber_profile": chamber_row,
        "limitations": [
            "the chamber result covers one truncated comb chart, not a crossing atlas",
            "overall sphere normalization and external reflection phases are omitted",
            "analytic continuation to physical real frequencies is not performed",
        ],
    }


def _integer_list(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t", type=float, default=0.12)
    parser.add_argument("--q1-real", type=float, default=0.18)
    parser.add_argument("--q1-imag", type=float, default=0.05)
    parser.add_argument("--q2-real", type=float, default=0.22)
    parser.add_argument("--q2-imag", type=float, default=-0.04)
    parser.add_argument("--internal-p1", type=float, default=0.37)
    parser.add_argument("--internal-p2", type=float, default=0.61)
    parser.add_argument("--cutoffs", type=_integer_list, default=(2, 4, 6))
    parser.add_argument("--spectral-orders", type=_integer_list, default=(9, 15))
    parser.add_argument("--spectral-p-max", type=float, default=3.0)
    parser.add_argument("--structure-precision", type=int, default=32)
    parser.add_argument("--block-digits", type=int, default=40)
    parser.add_argument("--chamber-q-max", type=float, default=0.25)
    parser.add_argument("--chamber-sample-power", type=int)
    parser.add_argument("--chamber-replicates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--chamber-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_physical_probe(
        t=args.t,
        q_1=complex(args.q1_real, args.q1_imag),
        q_2=complex(args.q2_real, args.q2_imag),
        internal_momenta=(args.internal_p1, args.internal_p2),
        maximum_twice_levels=args.cutoffs,
        spectral_orders=args.spectral_orders,
        spectral_p_max=args.spectral_p_max,
        structure_precision=args.structure_precision,
        block_digits=args.block_digits,
        chamber_q_maxima=(args.chamber_q_max, args.chamber_q_max),
        chamber_sample_power=args.chamber_sample_power,
        chamber_replicates=args.chamber_replicates,
        seed=args.seed,
        include_point_profiles=not args.chamber_only,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
