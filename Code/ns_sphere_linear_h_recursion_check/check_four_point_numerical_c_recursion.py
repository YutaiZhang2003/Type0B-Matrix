#!/usr/bin/env python3
"""Numerical NS sphere four-point h-recursion versus c-recursion.

The h-recursion coefficients are produced by the fixed-difference recursion
in ``check_order3.py``.  The comparison side is the production BRY
central-charge recursion in ``Code/c_Recursion/superconformal_blocks.py``.
No PBW/Shapovalov data enter this check.

For each generic parameter set, the script checks every local coefficient
through physical level 16 in both the even and odd internal sectors.  It then
evaluates the two chiral blocks at real and complex cross ratios.  Finally it
compares the level-16 h-series with the independently resummed pointwise
c-recursion, whose leaves are exact global osp(1|2) blocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import mpmath
import sympy as sp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "Code" / "c_Recursion"))

from check_order3 import (  # noqa: E402
    ExactSphereLinearHRecursion,
    SAMPLES,
    ordinary_c,
)
from superconformal_blocks import (  # noqa: E402
    HighPrecisionNSSphereFourPointBlock,
)


WORKING_PRECISION = 80
MAX_PHYSICAL_LEVEL = 16
Z_POINT_STRINGS = (
    ("0.08", "0"),
    ("0.15", "0"),
    ("0.25", "0.05"),
)
COEFFICIENT_TOLERANCE = mpmath.mpf("1e-65")
POINTWISE_TOLERANCE = mpmath.mpf("2e-10")


def mp_from_sympy(value: sp.Expr):
    """Convert an exact real SymPy value without a binary64 round trip."""

    return mpmath.mpf(sp.N(value, WORKING_PRECISION + 10).__str__())


def encoded(value, digits: int = 30) -> dict[str, str]:
    value = mpmath.mpc(value)
    return {
        "real": mpmath.nstr(value.real, digits),
        "imag": mpmath.nstr(value.imag, digits),
    }


def finite_h_series(
    *,
    coefficients: dict[int, object],
    z,
    parity: str,
    leading_power,
    max_physical_level: int,
):
    parity_bit = 0 if parity == "even" else 1
    reduced = mpmath.mpc(0)
    for twice_level in range(parity_bit, 2 * max_physical_level + 1, 2):
        reduced += coefficients[twice_level] * z ** (
            mpmath.mpf(twice_level) / 2
        )
    return z**leading_power * reduced


def run_sample(
    sample: dict[str, Any],
    *,
    max_physical_level: int,
) -> dict[str, Any]:
    b_exact = sample["b"]
    external_exact = sample["four_external"]
    internal_exact = sample["four_internal"][0]
    c_exact = ordinary_c(b_exact)

    h_recursion = ExactSphereLinearHRecursion(
        b=b_exact,
        external_weights=external_exact,
        internal_weights=(internal_exact,),
    )
    with mpmath.workdps(WORKING_PRECISION):
        external = tuple(mp_from_sympy(value) for value in external_exact)
        internal = mp_from_sympy(internal_exact)
        c_value = mp_from_sympy(c_exact)
        c_recursion = HighPrecisionNSSphereFourPointBlock(
            c=c_value,
            h1=external[0],
            h2=external[1],
            h3=external[2],
            h4=external[3],
            internal_weight=internal,
            working_precision=WORKING_PRECISION,
        )
        z_points = tuple(
            mpmath.mpc(real, imag) for real, imag in Z_POINT_STRINGS
        )

        h_coefficients: dict[int, object] = {}
        coefficient_rows = []
        maximum_coefficient_error = mpmath.mpf(0)
        for twice_level in range(2 * max_physical_level + 1):
            routing = (0, 0) if twice_level % 2 == 0 else (1, 1)
            h_exact = h_recursion.coefficient((twice_level,), routing)
            h_value = mp_from_sympy(h_exact)
            c_coefficient = c_recursion.coefficient(twice_level)
            error = abs(h_value - c_coefficient)
            maximum_coefficient_error = max(maximum_coefficient_error, error)
            h_coefficients[twice_level] = h_value
            coefficient_rows.append(
                {
                    "twice_level": twice_level,
                    "level": str(sp.Rational(twice_level, 2)),
                    "sector": "even" if twice_level % 2 == 0 else "odd",
                    "h_recursion": encoded(h_value),
                    "c_recursion": encoded(c_coefficient),
                    "absolute_error": mpmath.nstr(error, 12),
                    "passed": bool(error < COEFFICIENT_TOLERANCE),
                }
            )

        leading_power = internal - external[0] - external[1]
        block_rows = []
        maximum_pointwise_error = mpmath.mpf(0)
        for parity in ("even", "odd"):
            c_pointwise_values = c_recursion.recursive_z_blocks(
                z_points,
                max_physical_level,
                parity,
            )
            for z, c_pointwise in zip(z_points, c_pointwise_values):
                h_series = finite_h_series(
                    coefficients=h_coefficients,
                    z=z,
                    parity=parity,
                    leading_power=leading_power,
                    max_physical_level=max_physical_level,
                )
                pointwise_error = abs(h_series - c_pointwise)
                maximum_pointwise_error = max(
                    maximum_pointwise_error, pointwise_error
                )
                block_rows.append(
                    {
                        "parity": parity,
                        "z": encoded(z),
                        "h_recursion_truncated_series": encoded(h_series),
                        "c_recursion_global_leaf_resummation": encoded(
                            c_pointwise
                        ),
                        "absolute_error": mpmath.nstr(pointwise_error, 12),
                        "passed": bool(pointwise_error < POINTWISE_TOLERANCE),
                    }
                )

    all_passed = all(row["passed"] for row in coefficient_rows + block_rows)
    return {
        "name": sample["name"],
        "parameters": {
            "b": str(b_exact),
            "c": str(c_exact),
            "external_weights": [str(value) for value in external_exact],
            "internal_weight": str(internal_exact),
        },
        "summary": {
            "all_passed": all_passed,
            "coefficient_count": len(coefficient_rows),
            "block_value_count": len(block_rows),
            "maximum_coefficient_absolute_error": mpmath.nstr(
                maximum_coefficient_error, 12
            ),
            "maximum_pointwise_absolute_error": mpmath.nstr(
                maximum_pointwise_error, 12
            ),
        },
        "coefficients": coefficient_rows,
        "block_values": block_rows,
    }


def run_check(max_physical_level: int = MAX_PHYSICAL_LEVEL) -> dict[str, Any]:
    started = time.perf_counter()
    samples = [
        run_sample(sample, max_physical_level=max_physical_level)
        for sample in SAMPLES
    ]
    return {
        "description": (
            "NS sphere four-point fixed-difference h-recursion versus "
            "production BRY c-recursion"
        ),
        "scope": {
            "working_precision_digits": WORKING_PRECISION,
            "maximum_physical_level": max_physical_level,
            "coefficient_tolerance": mpmath.nstr(COEFFICIENT_TOLERANCE, 8),
            "pointwise_tolerance": mpmath.nstr(POINTWISE_TOLERANCE, 8),
            "pointwise_c_recursion_leaf": "exact global osp(1|2) block",
            "sample_count": len(samples),
        },
        "summary": {
            "all_passed": all(sample["summary"]["all_passed"] for sample in samples),
            "coefficient_comparison_count": sum(
                sample["summary"]["coefficient_count"] for sample in samples
            ),
            "block_value_comparison_count": sum(
                sample["summary"]["block_value_count"] for sample in samples
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-level",
        type=int,
        default=MAX_PHYSICAL_LEVEL,
        help="maximum physical level in each parity sector",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results_four_point_numerical_c_recursion.json",
    )
    args = parser.parse_args()
    if args.max_level < 1:
        raise ValueError("--max-level must be positive")
    result = run_check(args.max_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = result["summary"]
    print(
        f"PASS={summary['all_passed']}: "
        f"{summary['coefficient_comparison_count']} coefficients, "
        f"{summary['block_value_comparison_count']} block values, "
        f"elapsed={summary['elapsed_seconds']:.3f} s; wrote {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
