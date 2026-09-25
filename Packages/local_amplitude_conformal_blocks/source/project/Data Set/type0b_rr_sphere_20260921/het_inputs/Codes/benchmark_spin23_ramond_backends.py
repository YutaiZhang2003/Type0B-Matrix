#!/usr/bin/env python3
"""Benchmark and compare certified genus-one Ramond block backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import sympy as sp

from ns_algebra.ns_sca import G
from spin23_genus1_blocks import b1_ramond_liouville_necklace_series
from spin23_genus1_recursive_blocks import (
    recursive_ramond_torus_two_point_series,
)
from spin23_ramond_fast import (
    direct_fast_b1_ramond_necklace_series,
    q_weighted_series_discrepancy,
)


PP = ((), ())
GG = (
    (G(sp.Rational(-1, 2)),),
    (G(sp.Rational(-1, 2)),),
)


def _timed(builder):
    started = time.perf_counter()
    value = builder()
    return value, time.perf_counter() - started


def benchmark_case(
    *,
    momenta: tuple[float, float],
    maximum_twice_levels: tuple[int, int],
    component: str,
    include_oracle: bool,
    include_recursion: bool,
) -> dict[str, object]:
    words = PP if component == "PP" else GG
    common = dict(
        internal_momenta=momenta,
        external_ns_momenta=(0.41, 0.41),
        structure_signs=(1, 1),
        maximum_twice_levels=maximum_twice_levels,
        edge_lift_signs=(1, 1),
        external_words=words,
    )
    fast, fast_cold = _timed(
        lambda: direct_fast_b1_ramond_necklace_series(**common)
    )
    _, fast_warm = _timed(
        lambda: direct_fast_b1_ramond_necklace_series(**common)
    )
    q_values = (0.05556, 0.01085)
    scale = max(
        1.0,
        sum(
            abs(coefficient)
            * q_values[0] ** (levels[0] / 2)
            * q_values[1] ** (levels[1] / 2)
            for levels, coefficient in fast.coefficients.items()
        ),
    )
    result: dict[str, object] = {
        "momenta": list(momenta),
        "maximum_twice_levels": list(maximum_twice_levels),
        "component": component,
        "fast_cold_seconds": fast_cold,
        "fast_warm_seconds": fast_warm,
        "maximum_fast_gram_condition": max(
            fast.gram_condition_numbers.values(),
            default=1.0,
        ),
    }
    if include_oracle:
        oracle, oracle_seconds = _timed(
            lambda: b1_ramond_liouville_necklace_series(**common)
        )
        result.update(
            {
                "oracle_seconds": oracle_seconds,
                "fast_over_oracle_q_weighted_error": (
                    q_weighted_series_discrepancy(fast, oracle, q_values)
                    / scale
                ),
            }
        )
    if include_recursion:
        recursive_result, recursive_seconds = _timed(
            lambda: recursive_ramond_torus_two_point_series(
                **common,
                radius=0.04,
                check_radius=0.05,
                samples=24,
            )
        )
        recursive = recursive_result.series
        result.update(
            {
                "recursive_seconds": recursive_seconds,
                "fast_over_recursion_q_weighted_error": (
                    q_weighted_series_discrepancy(fast, recursive, q_values)
                    / scale
                ),
                "recursive_continuation_scaled_error": (
                    recursive_result.scaled_finite_part_error
                ),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-twice-level", type=int, default=6)
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--include-recursion", action="store_true")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.maximum_twice_level < 0 or args.maximum_twice_level % 2:
        parser.error("--maximum-twice-level must be a nonnegative even integer")
    if args.case_limit is not None and args.case_limit <= 0:
        parser.error("--case-limit must be positive")

    cases = []
    for momenta in ((0.05, 0.37), (0.37, 0.52), (1.0, 2.0)):
        for component in ("PP", "GG"):
            if args.case_limit is not None and len(cases) >= args.case_limit:
                break
            cases.append(
                benchmark_case(
                    momenta=momenta,
                    maximum_twice_levels=(
                        args.maximum_twice_level,
                        args.maximum_twice_level,
                    ),
                    component=component,
                    include_oracle=not args.skip_oracle,
                    include_recursion=args.include_recursion,
                )
            )
        if args.case_limit is not None and len(cases) >= args.case_limit:
            break
    payload = {
        "schema": "spin23_ramond_backend_crossover_v1",
        "include_oracle": not args.skip_oracle,
        "include_recursion": args.include_recursion,
        "cases": cases,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
