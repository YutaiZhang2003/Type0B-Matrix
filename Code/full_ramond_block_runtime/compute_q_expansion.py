#!/usr/bin/env python3
"""Compute the formal q-expansion of the full enlarged Ramond theta block.

The output keeps the three tube-sign parities separate.  An exponent
``(e1,e2,e3)`` denotes ``q1**(e1/2) q2**(e2/2) q3**(e3/2)``.  Every ordinary
Virasoro block is truncated at the remaining descendant level after its
branch-primary shift has been subtracted from the requested total cutoff.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import os
import platform
import sys
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from compute_full_block import (  # noqa: E402
    BranchingGrid,
    base_twice_level,
    continuous_b_square_rs,
    encode,
    show,
)
from compute_target import norm_product  # noqa: E402
import plumbing.ccy_genus2_block as ccy  # noqa: E402
from plumbing.genus2_vacuum_blocks import (  # noqa: E402
    inverse_vacuum_gram_by_level,
    rho_vacuum_descendants,
)
from plumbing.virasoro_plumbing_graph import (  # noqa: E402
    direct_plumbing_graph_block,
    genus2_theta_graph,
)


ccy.b_square_rs_from_h = continuous_b_square_rs

Exponent = tuple[int, int, int]
Series = dict[Exponent, complex]
BlockKey = tuple[int, int, int, int, int, int]
BlockSeries = dict[BlockKey, complex]
ZERO: Exponent = (0, 0, 0)


def add_to(series: dict, key: tuple, value: complex, tolerance: float = 1.0e-13) -> None:
    value = series.get(key, 0.0j) + complex(value)
    if abs(value) <= tolerance:
        series.pop(key, None)
    else:
        series[key] = value


def series_scale(series: Series, factor: complex) -> Series:
    return {
        exponent: complex(factor) * coefficient
        for exponent, coefficient in series.items()
        if abs(factor * coefficient) > 1.0e-13
    }


def series_shift(series: Series, shift: Exponent, cutoff: int) -> Series:
    answer: Series = {}
    for exponent, coefficient in series.items():
        changed = tuple(exponent[index] + shift[index] for index in range(3))
        if sum(changed) <= cutoff:
            add_to(answer, changed, coefficient)
    return answer


def series_add(left: Series, right: Series) -> Series:
    answer = dict(left)
    for exponent, coefficient in right.items():
        add_to(answer, exponent, coefficient)
    return answer


def series_multiply(left: Series, right: Series, cutoff: int) -> Series:
    answer: Series = {}
    for first, coefficient1 in left.items():
        for second, coefficient2 in right.items():
            exponent = tuple(first[index] + second[index] for index in range(3))
            if sum(exponent) <= cutoff:
                add_to(answer, exponent, coefficient1 * coefficient2)
    return answer


@lru_cache(maxsize=None)
def global_series(weights: tuple[complex, complex, complex], cutoff: int) -> tuple[tuple[Exponent, complex], ...]:
    h1, h2, h3 = weights
    answer: Series = {}
    for first in range(cutoff + 1):
        norm1 = ccy.sl2_descendant_norm(h1, first)
        for second in range(cutoff + 1 - first):
            norm2 = ccy.sl2_descendant_norm(h2, second)
            for third in range(cutoff + 1 - first - second):
                rho = ccy.rho_lminus1_triple(first, second, third, h1, h2, h3)
                coefficient = (
                    rho * rho
                    / (norm1 * norm2 * ccy.sl2_descendant_norm(h3, third))
                )
                add_to(answer, (first, second, third), coefficient)
    return tuple(sorted(answer.items()))


@lru_cache(maxsize=None)
def reduced_virasoro_series_cached(
    central_charge: complex,
    weights: tuple[complex, complex, complex],
    cutoff: int,
) -> tuple[tuple[Exponent, complex], ...]:
    """CCY c-recursion after factoring out the universal vacuum seed."""

    total = dict(global_series(weights, cutoff))
    fusion_pairs = (
        (weights[2], weights[1]),
        (weights[2], weights[0]),
        (weights[0], weights[1]),
    )
    for edge in range(3):
        h_edge = weights[edge]
        for r in range(2, cutoff + 1):
            for s in range(1, cutoff // r + 1):
                level = r * s
                pole = ccy.c_rs_from_h(r, s, h_edge)
                denominator = central_charge - pole
                if abs(denominator) < 1.0e-11:
                    raise ZeroDivisionError(
                        f"c-recursion collision at edge={edge}, r={r}, s={s}, h={h_edge}"
                    )
                top, bottom = fusion_pairs[edge]
                residue = ccy.ccy_residue_prefactor_for_weights(
                    r, s, h_edge, top, bottom
                ) / denominator
                shifted_weights = list(weights)
                shifted_weights[edge] = h_edge + level
                subblock = dict(
                    reduced_virasoro_series_cached(
                        complex(pole),
                        tuple(complex(value) for value in shifted_weights),
                        cutoff - level,
                    )
                )
                exponent_shift = tuple(level if index == edge else 0 for index in range(3))
                total = series_add(
                    total,
                    series_scale(
                        series_shift(subblock, exponent_shift, cutoff),
                        residue,
                    ),
                )
    return tuple(sorted(total.items()))


def reduced_virasoro_series(
    central_charge: complex,
    weights: tuple[complex, complex, complex],
    cutoff: int,
) -> Series:
    return dict(
        reduced_virasoro_series_cached(
            complex(central_charge),
            tuple(complex(value) for value in weights),
            int(cutoff),
        )
    )


def finite_c_vacuum_series(c_value: float, cutoff: int) -> Series:
    """Direct vacuum-module coefficients with a total, not per-edge, cutoff."""

    c_complex = complex(c_value)
    gram = inverse_vacuum_gram_by_level(cutoff, c_complex)
    answer: Series = {}
    for level1 in range(cutoff + 1):
        basis1, inverse1 = gram[level1]
        for level2 in range(cutoff + 1 - level1):
            basis2, inverse2 = gram[level2]
            for level3 in range(cutoff + 1 - level1 - level2):
                basis3, inverse3 = gram[level3]
                coefficient = 0.0j
                for a, descendant_a in enumerate(basis1):
                    for b, descendant_b in enumerate(basis1):
                        metric1 = inverse1[a, b]
                        if metric1 == 0:
                            continue
                        for d, descendant_d in enumerate(basis2):
                            for e, descendant_e in enumerate(basis2):
                                metric2 = inverse2[d, e]
                                if metric2 == 0:
                                    continue
                                for g, descendant_g in enumerate(basis3):
                                    rho_left = rho_vacuum_descendants(
                                        descendant_a,
                                        descendant_d,
                                        descendant_g,
                                        c=c_complex,
                                    )
                                    if rho_left == 0:
                                        continue
                                    for h, descendant_h in enumerate(basis3):
                                        metric3 = inverse3[g, h]
                                        if metric3 == 0:
                                            continue
                                        rho_right = rho_vacuum_descendants(
                                            descendant_b,
                                            descendant_e,
                                            descendant_h,
                                            c=c_complex,
                                        )
                                        coefficient += (
                                            metric1
                                            * metric2
                                            * metric3
                                            * rho_left
                                            * rho_right
                                        )
                if abs(coefficient) > 1.0e-14:
                    answer[(level1, level2, level3)] = complex(coefficient)
    return answer


def large_c_vacuum_series(cutoff: int) -> tuple[Series, dict[str, float]]:
    """Extract the integer large-c vacuum coefficients by 1/c extrapolation."""

    first_c = 1.0e8
    second_c = 2.0e8
    first = finite_c_vacuum_series(first_c, cutoff)
    second = finite_c_vacuum_series(second_c, cutoff)
    answer: Series = {}
    maximum_rounding_error = 0.0
    maximum_two_point_change = 0.0
    for exponent in set(first) | set(second) | {ZERO}:
        first_value = first.get(exponent, 0.0j)
        second_value = second.get(exponent, 0.0j)
        extrapolated = (
            second_c * second_value - first_c * first_value
        ) / (second_c - first_c)
        rounded = round(extrapolated.real)
        rounding_error = abs(extrapolated - rounded)
        maximum_rounding_error = max(maximum_rounding_error, rounding_error)
        maximum_two_point_change = max(
            maximum_two_point_change, abs(second_value - first_value)
        )
        if rounding_error > 2.0e-5:
            raise AssertionError(
                f"large-c vacuum coefficient did not approach an integer at {exponent}: "
                f"{extrapolated}"
            )
        if rounded:
            answer[exponent] = complex(rounded)
    answer[ZERO] = 1.0 + 0.0j
    return answer, {
        "first_c": first_c,
        "second_c": second_c,
        "maximum_extrapolation_rounding_error": float(maximum_rounding_error),
        "maximum_change_between_samples": float(maximum_two_point_change),
    }


def evaluate_block(
    series: BlockSeries,
    q_values: tuple[float, float, float],
    eta_values: tuple[int, int, int],
) -> complex:
    answer = 0.0j
    for key, coefficient in series.items():
        exponents = key[:3]
        parities = key[3:]
        term = coefficient
        for edge in range(3):
            term *= q_values[edge] ** (exponents[edge] / 2.0)
            term *= eta_values[edge] ** parities[edge]
        answer += term
    return answer


def low_level_virasoro_check(branching: BranchingGrid) -> dict[str, object]:
    labels = (Fraction(0), Fraction(-1, 4), Fraction(1, 4))
    copy = 0
    weights = branching.weights.triple(labels, copy)
    recursion = reduced_virasoro_series(
        branching.weights.central_charges[copy], weights, 2
    )
    direct = direct_plumbing_graph_block(
        genus2_theta_graph(),
        central_charge=branching.weights.central_charges[copy],
        edge_weights=weights,
        q_values=(1.0, 1.0, 1.0),
        max_total_level=2,
    ).coefficient_by_levels
    keys = set(recursion) | set(direct)
    maximum = max(
        abs(recursion.get(key, 0.0j) - direct.get(key, 0.0j))
        for key in keys
    )
    return {
        "labels": [show(value) for value in labels],
        "copy": copy + 1,
        "maximum_absolute_error_through_level_2": float(maximum),
    }


def run(cutoff: int, output: Path) -> dict[str, object]:
    b = 7.0 / 5.0
    momenta = (11.0 / 23.0, 13.0 / 29.0, 17.0 / 31.0)
    evaluation_q = (0.019, 0.023, 0.029)
    cutoff_twice = 2 * int(cutoff)
    total_started = time.perf_counter()

    branching = BranchingGrid(b, momenta, cutoff)
    started = time.perf_counter()
    branching.build_actions()
    action_seconds = time.perf_counter() - started
    print(f"branch actions: {action_seconds:.6f} s", flush=True)

    started = time.perf_counter()
    raw_grids: dict[tuple[int, int], dict[tuple[Fraction, Fraction, Fraction], complex]] = {}
    ward_diagnostics: list[dict[str, object]] = []
    for alpha2 in (0, 1):
        for alpha3 in (0, 1):
            values, diagnostic = branching.solve(alpha2, alpha3)
            raw_grids[(alpha2, alpha3)] = values
            ward_diagnostics.append(diagnostic)
            print(
                f"Ward grid ({alpha2},{alpha3}): rank "
                f"{diagnostic['rank']}/{diagnostic['unknowns']}, residual "
                f"{diagnostic['relative_residual']:.3e}",
                flush=True,
            )
    ward_seconds = time.perf_counter() - started
    print(f"branching Ward systems: {ward_seconds:.6f} s", flush=True)

    started = time.perf_counter()
    vacuum, vacuum_diagnostic = large_c_vacuum_series(cutoff)
    vacuum_squared = series_multiply(vacuum, vacuum, cutoff)
    vacuum_seconds = time.perf_counter() - started
    print(
        f"formal large-c vacuum seed: {vacuum_seconds:.6f} s, "
        f"{len(vacuum)} coefficients",
        flush=True,
    )

    triples = tuple(
        labels
        for labels in (
            (first, second, third)
            for first in branching.ns
            for second in branching.r
            for third in branching.r
        )
        if base_twice_level(labels) <= cutoff_twice
    )
    started = time.perf_counter()
    reduced_products: dict[tuple[Fraction, Fraction, Fraction], Series] = {}
    virasoro_block_count = 0
    truncation_histogram: dict[int, int] = {}
    for count, labels in enumerate(triples, start=1):
        remaining = (cutoff_twice - base_twice_level(labels)) // 2
        truncation_histogram[remaining] = truncation_histogram.get(remaining, 0) + 2
        copies: list[Series] = []
        for copy in (0, 1):
            copies.append(
                reduced_virasoro_series(
                    branching.weights.central_charges[copy],
                    branching.weights.triple(labels, copy),
                    remaining,
                )
            )
            virasoro_block_count += 1
        reduced_products[labels] = series_multiply(
            copies[0], copies[1], remaining
        )
        if count % 50 == 0 or count == len(triples):
            print(f"formal Virasoro pairs: {count}/{len(triples)}", flush=True)
    virasoro_seconds = time.perf_counter() - started
    print(f"formal Virasoro c-recursions: {virasoro_seconds:.6f} s", flush=True)

    started = time.perf_counter()
    reduced_block: BlockSeries = {}
    for labels in triples:
        n1, _, _ = labels
        base = (
            int(4 * labels[0] * labels[0]),
            int(4 * labels[1] * labels[1] - Fraction(1, 4)),
            int(4 * labels[2] * labels[2] - Fraction(1, 4)),
        )
        alpha_pairs = (
            ((0, 0), (1, 1))
            if int(2 * n1) % 2 == 0
            else ((0, 1), (1, 0))
        )
        for alpha2, alpha3 in alpha_pairs:
            raw = raw_grids[(alpha2, alpha3)][labels]
            normalized = raw / norm_product(
                labels, alpha2, alpha3, b, momenta
            )
            sign = (-1) ** (
                int(2 * n1) * alpha2
                + int(2 * n1) * alpha3
                + alpha2 * alpha3
            )
            prefactor = sign * normalized * normalized
            parity = (int(2 * n1) % 2, alpha2, alpha3)
            for exponent, coefficient in reduced_products[labels].items():
                twice_exponent = tuple(
                    base[index] + 2 * exponent[index] for index in range(3)
                )
                if sum(twice_exponent) <= cutoff_twice:
                    add_to(
                        reduced_block,
                        twice_exponent + parity,
                        prefactor * coefficient,
                    )

    full_block: BlockSeries = {}
    for key, coefficient in reduced_block.items():
        exponent = key[:3]
        parity = key[3:]
        for vacuum_exponent, vacuum_coefficient in vacuum_squared.items():
            changed = tuple(
                exponent[index] + 2 * vacuum_exponent[index]
                for index in range(3)
            )
            if sum(changed) <= cutoff_twice:
                add_to(
                    full_block,
                    changed + parity,
                    coefficient * vacuum_coefficient,
                )
    assembly_seconds = time.perf_counter() - started
    print(f"formal assembly: {assembly_seconds:.6f} s", flush=True)

    started = time.perf_counter()
    validation = low_level_virasoro_check(branching)
    validation_seconds = time.perf_counter() - started
    evaluated = evaluate_block(full_block, evaluation_q, (1, 1, 1))
    total_seconds = time.perf_counter() - total_started

    coefficients = [
        {
            "twice_levels": list(key[:3]),
            "levels": [value / 2.0 for value in key[:3]],
            "eta_parity": list(key[3:]),
            "coefficient": encode(coefficient),
        }
        for key, coefficient in sorted(
            full_block.items(), key=lambda item: (sum(item[0][:3]), item[0])
        )
    ]
    result: dict[str, object] = {
        "calculation": "formal q-expansion of the full three-variable enlarged Ramond block",
        "block": "widehat F_0^(+,+)",
        "total_level_cutoff": cutoff,
        "exponent_convention": "twice_levels e represent q_i^(e_i/2)",
        "parameters": {
            "b": b,
            "P": list(momenta),
            "three_point_eta": [1, 1],
            "fermion_parity": 0,
        },
        "method": {
            "branching": "first L1 Ward identity with direct low-level anchors",
            "ordinary_blocks": "formal trivariate CCY central-charge recursion",
            "adaptive_virasoro_cutoff": True,
            "pbw_generic_block_sum_used": False,
            "vacuum_seed": "large-c vacuum-module limit, computed once",
        },
        "counts": {
            "branch_label_triples": len(triples),
            "ordinary_virasoro_blocks": virasoro_block_count,
            "full_block_coefficients_with_eta_parity": len(full_block),
            "vacuum_coefficients": len(vacuum),
        },
        "virasoro_cutoff_histogram": [
            {"descendant_cutoff": level, "block_count": truncation_histogram[level]}
            for level in sorted(truncation_histogram, reverse=True)
        ],
        "timing_seconds": {
            "branch_action_decompositions": action_seconds,
            "branching_ward_systems": ward_seconds,
            "formal_vacuum_seed": vacuum_seconds,
            "formal_virasoro_c_recursions": virasoro_seconds,
            "formal_assembly": assembly_seconds,
            "low_level_validation": validation_seconds,
            "total": total_seconds,
            "q_expansion_with_branching_coefficients_available": (
                vacuum_seconds + virasoro_seconds + assembly_seconds
            ),
        },
        "diagnostics": {
            "maximum_action_fit_relative_residual": max(
                item["relative_residual"] for item in branching.action_diagnostics
            ),
            "ward_systems": ward_diagnostics,
            "vacuum_extrapolation": vacuum_diagnostic,
            "low_level_virasoro_check": validation,
        },
        "evaluation_check": {
            "q": list(evaluation_q),
            "eta_tubes": [1, 1, 1],
            "value_from_formal_series": encode(evaluated),
        },
        "coefficients": coefficients,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"formal coefficients: {len(full_block)}", flush=True)
    print(
        f"evaluation check: {evaluated.real:+.15e}{evaluated.imag:+.15e}j",
        flush=True,
    )
    print(f"total runtime: {total_seconds:.6f} s", flush=True)
    print(f"wrote {output}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=10)
    parser.add_argument(
        "--json", type=Path, default=HERE / "level10_q_expansion.json"
    )
    arguments = parser.parse_args()
    if arguments.cutoff < 0:
        raise ValueError("cutoff must be nonnegative")
    run(arguments.cutoff, arguments.json)


if __name__ == "__main__":
    main()
