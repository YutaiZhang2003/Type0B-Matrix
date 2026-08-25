#!/usr/bin/env python3
"""Check the q1 -> 0 Ramond block using torus two-point c-recursion.

Left side:
    direct Ramond SCA torus two-point sewing, convolved with the direct
    auxiliary Ramond-fermion block.

Right side:
    Ramond branching coefficients times two ordinary Virasoro torus
    two-point blocks, each computed as a formal bivariate q-series by the
    central-charge recursion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD_CHECK = ROOT / "python" / "ramond_torus_limit_check"
FULL_Q = ROOT / "python" / "full_ramond_block_runtime"
for directory in (ROOT, OLD_CHECK, FULL_Q):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import plumbing.ccy_genus2_block as ccy  # noqa: E402
from check_torus_limit import (  # noqa: E402
    BranchingTorusLimit,
    DirectTorusTwoPoint,
    VirasoroTorusTwoPoint,
    comparison,
    encode,
    encode_series,
    ramond_star,
)
from compute_full_block import continuous_b_square_rs  # noqa: E402
from compute_q_expansion import large_c_vacuum_series  # noqa: E402


ccy.b_square_rs_from_h = continuous_b_square_rs

BivariateSeries = dict[tuple[int, int], complex]


def add(series: BivariateSeries, exponent: tuple[int, int], value: complex) -> None:
    value = series.get(exponent, 0.0j) + complex(value)
    if abs(value) <= 1.0e-12:
        series.pop(exponent, None)
    else:
        series[exponent] = value


def add_series(left: BivariateSeries, right: BivariateSeries) -> BivariateSeries:
    answer = dict(left)
    for exponent, coefficient in right.items():
        add(answer, exponent, coefficient)
    return answer


def scale_series(series: BivariateSeries, coefficient: complex) -> BivariateSeries:
    return {
        exponent: complex(coefficient) * value
        for exponent, value in series.items()
        if abs(coefficient * value) > 1.0e-12
    }


def shift_series(
    series: BivariateSeries,
    shift: tuple[int, int],
    cutoff: int,
) -> BivariateSeries:
    answer: BivariateSeries = {}
    for exponent, coefficient in series.items():
        changed = (exponent[0] + shift[0], exponent[1] + shift[1])
        if sum(changed) <= cutoff:
            add(answer, changed, coefficient)
    return answer


def multiply_series(
    left: BivariateSeries,
    right: BivariateSeries,
    cutoff: int,
) -> BivariateSeries:
    answer: BivariateSeries = {}
    for first, coefficient1 in left.items():
        for second, coefficient2 in right.items():
            exponent = (first[0] + second[0], first[1] + second[1])
            if sum(exponent) <= cutoff:
                add(answer, exponent, coefficient1 * coefficient2)
    return answer


@lru_cache(maxsize=None)
def global_torus_two_point_series(
    weights: tuple[complex, complex, complex],
    cutoff: int,
) -> tuple[tuple[tuple[int, int], complex], ...]:
    """Large-c global block with no descendants on the first leg."""

    h1, h2, h3 = weights
    answer: BivariateSeries = {}
    for level2 in range(cutoff + 1):
        norm2 = ccy.sl2_descendant_norm(h2, level2)
        for level3 in range(cutoff - level2 + 1):
            rho = ccy.rho_lminus1_triple(
                0, level2, level3, h1, h2, h3
            )
            coefficient = (
                rho
                * rho
                / (norm2 * ccy.sl2_descendant_norm(h3, level3))
            )
            add(answer, (level2, level3), coefficient)
    return tuple(sorted(answer.items()))


@lru_cache(maxsize=None)
def reduced_torus_two_point_series_cached(
    central_charge: complex,
    weights: tuple[complex, complex, complex],
    cutoff: int,
) -> tuple[tuple[tuple[int, int], complex], ...]:
    """Torus two-point c-recursion with the vacuum factor removed."""

    total = dict(global_torus_two_point_series(weights, cutoff))
    # The first weight labels the two external primaries created when q1=0.
    # Only the second and third internal modules can develop sewing poles.
    channels = (
        (1, 0, (weights[2], weights[0])),
        (2, 1, (weights[0], weights[1])),
    )
    for weight_slot, q_slot, fusion_pair in channels:
        internal_weight = weights[weight_slot]
        for r in range(2, cutoff + 1):
            for s in range(1, cutoff // r + 1):
                level = r * s
                pole = ccy.c_rs_from_h(r, s, internal_weight)
                denominator = central_charge - pole
                if abs(denominator) < 1.0e-11:
                    raise ZeroDivisionError(
                        f"torus two-point c-recursion collision at "
                        f"slot={weight_slot}, r={r}, s={s}"
                    )
                residue = ccy.ccy_residue_prefactor_for_weights(
                    r,
                    s,
                    internal_weight,
                    fusion_pair[0],
                    fusion_pair[1],
                ) / denominator
                shifted_weights = list(weights)
                shifted_weights[weight_slot] = internal_weight + level
                subblock = dict(
                    reduced_torus_two_point_series_cached(
                        complex(pole),
                        tuple(complex(value) for value in shifted_weights),
                        cutoff - level,
                    )
                )
                shift = (level, 0) if q_slot == 0 else (0, level)
                total = add_series(
                    total,
                    scale_series(
                        shift_series(subblock, shift, cutoff),
                        residue,
                    ),
                )
    return tuple(sorted(total.items()))


@lru_cache(maxsize=None)
def torus_vacuum_series(cutoff: int) -> tuple[tuple[tuple[int, int], complex], ...]:
    full, _ = large_c_vacuum_series(cutoff)
    restricted = {
        (level2, level3): coefficient
        for (level1, level2, level3), coefficient in full.items()
        if level1 == 0
    }
    return tuple(sorted(restricted.items()))


def torus_two_point_recursion_series(
    weights: tuple[complex, complex, complex],
    central_charge: complex,
    cutoff: int,
) -> BivariateSeries:
    reduced = dict(
        reduced_torus_two_point_series_cached(
            complex(central_charge),
            tuple(complex(value) for value in weights),
            int(cutoff),
        )
    )
    return multiply_series(dict(torus_vacuum_series(cutoff)), reduced, cutoff)


def compare_bivariate(left: BivariateSeries, right: BivariateSeries) -> dict[str, object]:
    keys = set(left) | set(right)
    maximum_absolute = 0.0
    maximum_relative = 0.0
    worst = (0, 0)
    for key in keys:
        first = left.get(key, 0.0j)
        second = right.get(key, 0.0j)
        absolute = abs(first - second)
        relative = absolute / max(1.0, abs(first), abs(second))
        if relative > maximum_relative:
            maximum_relative = relative
            worst = key
        maximum_absolute = max(maximum_absolute, absolute)
    return {
        "coefficient_count": len(keys),
        "maximum_absolute_error": float(maximum_absolute),
        "maximum_relative_error": float(maximum_relative),
        "worst_levels": list(worst),
    }


class BranchingTorusRecursion(BranchingTorusLimit):
    """Branching sum with c-recursive ordinary torus two-point blocks."""

    def __init__(self, b, momenta):
        super().__init__(b, momenta)
        self._recursive_virasoro_cache: dict[tuple, BivariateSeries] = {}
        self._direct_virasoro_cache: dict[tuple, BivariateSeries] = {}
        self._virasoro_checks: dict[tuple, dict[str, object]] = {}

    def virasoro_series(self, copy, label2, label3, cutoff):
        key = (copy, label2, label3, cutoff)
        if key not in self._recursive_virasoro_cache:
            labels = (0, label2, label3)
            weights = self.weights.triple(labels, copy)
            recursive = torus_two_point_recursion_series(
                weights,
                self.weights.central_charges[copy],
                cutoff,
            )
            direct = VirasoroTorusTwoPoint(
                weights,
                self.weights.central_charges[copy],
            ).series(cutoff)
            self._recursive_virasoro_cache[key] = recursive
            self._direct_virasoro_cache[key] = direct
            self._virasoro_checks[key] = compare_bivariate(direct, recursive)
        return self._recursive_virasoro_cache[key]

    def recursion_check_summary(self) -> dict[str, object]:
        worst_key = None
        maximum_absolute = 0.0
        maximum_relative = 0.0
        for key, check in self._virasoro_checks.items():
            maximum_absolute = max(
                maximum_absolute, check["maximum_absolute_error"]
            )
            if check["maximum_relative_error"] > maximum_relative:
                maximum_relative = check["maximum_relative_error"]
                worst_key = key
        return {
            "ordinary_block_count": len(self._virasoro_checks),
            "maximum_absolute_error": float(maximum_absolute),
            "maximum_relative_error": float(maximum_relative),
            "worst_block": None
            if worst_key is None
            else {
                "copy": worst_key[0] + 1,
                "n2": str(worst_key[1]),
                "n3": str(worst_key[2]),
                "cutoff": worst_key[3],
                "worst_levels": self._virasoro_checks[worst_key][
                    "worst_levels"
                ],
            },
        }


def parse_fraction(text: str) -> float:
    from fractions import Fraction

    return float(Fraction(text))


def run(cutoff: int, b: float, momenta: tuple[float, float, float], output: Path) -> dict[str, object]:
    started_total = time.perf_counter()
    timing: dict[str, float] = {}

    direct = DirectTorusTwoPoint(b, momenta)
    started = time.perf_counter()
    auxiliary = direct.auxiliary_series(cutoff)
    timing["direct_auxiliary_fermion"] = time.perf_counter() - started

    branching = BranchingTorusRecursion(b, momenta)
    sectors: list[dict[str, object]] = []
    for form_parity, eta in ((0, 1), (1, -1)):
        started = time.perf_counter()
        physical = direct.physical_series(cutoff, form_parity, eta)
        timing[f"direct_sca_f{form_parity}"] = time.perf_counter() - started

        started = time.perf_counter()
        left = ramond_star(auxiliary, physical, cutoff)
        timing[f"convolution_f{form_parity}"] = time.perf_counter() - started

        started = time.perf_counter()
        right = branching.series(cutoff, form_parity, eta)
        timing[f"branching_and_torus_recursion_f{form_parity}"] = (
            time.perf_counter() - started
        )
        sectors.append(
            {
                "form_parity": form_parity,
                "eta": eta,
                "direct_sca_torus_two_point": encode_series(physical),
                "left_direct_convolution": encode_series(left),
                "right_branching_torus_recursion": encode_series(right),
                "comparison": comparison(left, right, cutoff),
            }
        )

    timing["total"] = time.perf_counter() - started_total
    recursion_check = branching.recursion_check_summary()
    payload: dict[str, object] = {
        "description": "q1 -> 0 Ramond theta-block check with formal torus two-point c-recursion",
        "cutoff": cutoff,
        "parameters": {
            "b": b,
            "P": list(momenta),
        },
        "left_side": "direct auxiliary fermion star_R direct Ramond SCA",
        "right_side": "Ramond branching coefficients times two c-recursive Virasoro torus two-point blocks",
        "ordinary_torus_recursion_check": recursion_check,
        "torus_vacuum_coefficients": [
            {
                "levels": list(exponent),
                "coefficient": encode(coefficient),
            }
            for exponent, coefficient in torus_vacuum_series(cutoff)
        ],
        "sectors": sectors,
        "timing_seconds": timing,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for sector in sectors:
        check = sector["comparison"]
        print(
            f"f={sector['form_parity']}, eta={sector['eta']:+d}: "
            f"{check['coefficient_count']} coefficients, "
            f"max abs={check['maximum_absolute_error']:.3e}, "
            f"max rel={check['maximum_relative_error']:.3e}"
        )
    print(
        "ordinary torus recursion vs direct: "
        f"max rel={recursion_check['maximum_relative_error']:.3e}"
    )
    print(f"total runtime: {timing['total']:.6f} s")
    print(f"wrote {output}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--b", default="7/5")
    parser.add_argument("--p1", default="11/23")
    parser.add_argument("--p2", default="13/29")
    parser.add_argument("--p3", default="17/31")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    arguments = parser.parse_args()
    run(
        arguments.cutoff,
        parse_fraction(arguments.b),
        tuple(
            parse_fraction(value)
            for value in (arguments.p1, arguments.p2, arguments.p3)
        ),
        arguments.output,
    )


if __name__ == "__main__":
    main()
