#!/usr/bin/env python3
"""Fixed-seed generic-in-(h1,h2) sweep through total NS level six.

The central charge and external weights are held fixed while the two internal
weights are drawn independently from a rational grid.  Every allowed
coefficient is compared separately by ``check_level6.run_check``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

import sympy as sp

from check_level6 import HERE, run_check


SEED = 20260827


def generic_weight_pairs(sample_count: int) -> list[tuple[sp.Rational, sp.Rational]]:
    """Return reproducible, independently sampled rational internal weights."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    rng = random.Random(SEED)
    pairs: list[tuple[sp.Rational, sp.Rational]] = []
    used: set[tuple[int, int]] = set()
    while len(pairs) < sample_count:
        numerators = (rng.randint(43, 157), rng.randint(43, 157))
        if numerators[0] == numerators[1] or numerators in used:
            continue
        used.add(numerators)
        pairs.append(
            (sp.Rational(numerators[0], 100), sp.Rational(numerators[1], 100))
        )
    return pairs


def run_sweep(
    *,
    sample_count: int,
    max_total_level: int,
    b: sp.Rational,
    d1: sp.Rational,
    d2: sp.Rational,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    samples: list[dict[str, Any]] = []
    for index, (h1, h2) in enumerate(generic_weight_pairs(sample_count), start=1):
        sample = run_check(
            max_total_level=max_total_level,
            b=b,
            h1=h1,
            h2=h2,
            d1=d1,
            d2=d2,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            verbose=False,
        )
        samples.append(sample)
        print(
            f"sample {index}/{sample_count}: (h1,h2)=({h1},{h2}), "
            f"{sample['scope']['coefficient_count']} coefficients agree, "
            f"max relative error={sample['summary']['maximum_relative_error']:.3e}",
            flush=True,
        )

    coefficient_count = sum(
        sample["scope"]["coefficient_count"] for sample in samples
    )
    worst_sample_index, worst_sample = max(
        enumerate(samples),
        key=lambda item: item[1]["summary"]["maximum_relative_error"],
    )
    maximum_absolute_error = max(
        sample["summary"]["maximum_absolute_error"] for sample in samples
    )
    maximum_relative_error = worst_sample["summary"]["maximum_relative_error"]
    return {
        "description": (
            "Fixed-seed generic internal-weight sweep: exact NS PBW/Ward "
            "sewing versus fixed-difference h-recursion"
        ),
        "sampling": {
            "seed": SEED,
            "sample_count": sample_count,
            "h1_h2_numerator_range_over_100": [43, 157],
            "independent_draws": True,
            "equal_weight_pairs_excluded": True,
        },
        "fixed_parameters": {
            "b": str(b),
            "d1": str(d1),
            "d2": str(d2),
            "max_total_physical_level": max_total_level,
        },
        "tolerances": {
            "absolute": absolute_tolerance,
            "relative": relative_tolerance,
        },
        "summary": {
            "all_passed": all(
                sample["summary"]["all_passed"] for sample in samples
            ),
            "coefficient_comparison_count": coefficient_count,
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_error": maximum_relative_error,
            "worst_sample_index": worst_sample_index + 1,
            "worst_h1": worst_sample["parameters"]["h1"],
            "worst_h2": worst_sample["parameters"]["h2"],
            "worst_twice_levels": worst_sample["summary"]["worst_twice_levels"],
            "elapsed_seconds": time.perf_counter() - started,
        },
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--max-total-level", type=int, default=6)
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-9)
    parser.add_argument("--relative-tolerance", type=float, default=2.0e-10)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results_generic_h_sweep_level6.json",
    )
    args = parser.parse_args()
    result = run_sweep(
        sample_count=args.sample_count,
        max_total_level=args.max_total_level,
        b=sp.Rational(127, 100),
        d1=sp.Rational(27, 100),
        d2=sp.Rational(9, 25),
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = result["summary"]
    print(
        f"all {summary['coefficient_comparison_count']} coefficient comparisons "
        f"passed; max abs={summary['maximum_absolute_error']:.3e}, "
        f"max rel={summary['maximum_relative_error']:.3e}; "
        f"wrote {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
