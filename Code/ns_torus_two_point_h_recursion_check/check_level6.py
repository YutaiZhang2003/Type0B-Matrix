#!/usr/bin/env python3
"""Exact PBW versus h-recursion check for the NS torus two-point block.

The direct side uses only the NS algebra, its Shapovalov matrices, and the
fixed-parity three-point Ward identities.  The recursive side uses the
fixed-weight-difference necklace h-recursion.  The two implementations meet
only at the numerical input data and at the final coefficient comparison.

Levels are stored as twice-level integers.  ``--max-total-level 6`` checks
all monomials q1**n1 q2**n2 with n1+n2 <= 6, including half-integral n_i.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys
import time
from typing import Any

import sympy as sp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "Code" / "c_Recursion"))
sys.path.insert(0, str(ROOT / "Code" / "h_recursion"))

from ns_genus2_symbolic_low_order import (  # noqa: E402
    ExactNSDescendantThreeForm,
    ExactNSVermaModule,
)
from superconformal_torus_two_point import (  # noqa: E402
    NSTorusTwoPointHRecursionBlock,
)


def _rational(text: str) -> sp.Rational:
    return sp.Rational(text)


def _ordinary_c(b: sp.Expr) -> sp.Expr:
    return sp.Rational(3, 2) + 3 * (b + 1 / b) ** 2


def _decimal(value: complex, digits: int = 17) -> dict[str, str]:
    return {
        "real": format(value.real, f".{digits}g"),
        "imag": format(value.imag, f".{digits}g"),
    }


class ExactNSTorusTwoPointPBW:
    """Direct two-vertex NS necklace contraction in an exact PBW basis."""

    def __init__(
        self,
        *,
        c: sp.Expr,
        h1: sp.Expr,
        h2: sp.Expr,
        d1: sp.Expr,
        d2: sp.Expr,
    ) -> None:
        self.c = c
        self.h1 = h1
        self.h2 = h2
        self.d1 = d1
        self.d2 = d2
        self.modules = (
            ExactNSVermaModule(c=c, weight=h1),
            ExactNSVermaModule(c=c, weight=h2),
        )
        self.forms = (
            ExactNSDescendantThreeForm(c=c, weights=(h1, d1, h2)),
            ExactNSDescendantThreeForm(c=c, weights=(h2, d2, h1)),
        )

    @lru_cache(maxsize=None)
    def inverse_gram(self, edge: int, twice_level: int) -> sp.Matrix:
        return self.modules[edge].gram_matrix(twice_level).inv()

    @lru_cache(maxsize=None)
    def vertex(self, vertex: int, twice_left: int, twice_right: int) -> sp.Matrix:
        if vertex not in (0, 1):
            raise ValueError("vertex must be zero or one")
        left_edge, right_edge = ((0, 1), (1, 0))[vertex]
        left_basis = self.modules[left_edge].basis(twice_left)
        right_basis = self.modules[right_edge].basis(twice_right)
        form = self.forms[vertex]
        return sp.Matrix(
            [
                [form.value(left, (), right) for right in right_basis]
                for left in left_basis
            ]
        )

    @lru_cache(maxsize=None)
    def coefficient(self, twice_level_1: int, twice_level_2: int) -> sp.Expr:
        # A bottom-component external NS primary is even.  Hence both edge
        # descendants at either vertex must have the same fermion parity.
        if (twice_level_1 - twice_level_2) % 2:
            return sp.S.Zero
        inverse_1 = self.inverse_gram(0, twice_level_1)
        inverse_2 = self.inverse_gram(1, twice_level_2)
        vertex_1 = self.vertex(0, twice_level_1, twice_level_2)
        vertex_2 = self.vertex(1, twice_level_2, twice_level_1)
        contraction = inverse_1 * vertex_1 * inverse_2 * vertex_2
        return sp.cancel(sp.trace(contraction))


def run_check(
    *,
    max_total_level: int,
    b: sp.Rational,
    h1: sp.Rational,
    h2: sp.Rational,
    d1: sp.Rational,
    d2: sp.Rational,
    absolute_tolerance: float,
    relative_tolerance: float,
    verbose: bool = True,
) -> dict[str, Any]:
    if max_total_level < 0:
        raise ValueError("max_total_level must be nonnegative")
    max_total_twice = 2 * max_total_level
    c = sp.factor(_ordinary_c(b))
    direct = ExactNSTorusTwoPointPBW(
        c=c,
        h1=h1,
        h2=h2,
        d1=d1,
        d2=d2,
    )
    recursive = NSTorusTwoPointHRecursionBlock(
        b=complex(sp.N(b, 30)),
        internal_weight_1=complex(sp.N(h1, 30)),
        internal_weight_2=complex(sp.N(h2, 30)),
        external_weight_1=complex(sp.N(d1, 30)),
        external_weight_2=complex(sp.N(d2, 30)),
    )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for total_twice in range(max_total_twice + 1):
        level_started = time.perf_counter()
        checked_here = 0
        for twice_level_1 in range(total_twice + 1):
            twice_level_2 = total_twice - twice_level_1
            if (twice_level_1 - twice_level_2) % 2:
                continue
            exact = direct.coefficient(twice_level_1, twice_level_2)
            direct_value = complex(sp.N(exact, 50))
            recursive_value = recursive.raw_coefficient(
                twice_level_1, twice_level_2
            )
            absolute_error = abs(recursive_value - direct_value)
            scale = max(1.0, abs(direct_value), abs(recursive_value))
            relative_error = absolute_error / scale
            passed = (
                absolute_error <= absolute_tolerance
                or relative_error <= relative_tolerance
            )
            rows.append(
                {
                    "twice_levels": [twice_level_1, twice_level_2],
                    "levels": [twice_level_1 / 2, twice_level_2 / 2],
                    "direct_pbw": _decimal(direct_value),
                    "h_recursion": _decimal(recursive_value),
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "passed": passed,
                }
            )
            checked_here += 1
            if not passed:
                raise AssertionError(
                    "PBW/h-recursion mismatch at twice-levels "
                    f"({twice_level_1},{twice_level_2}): "
                    f"direct={direct_value!r}, recursive={recursive_value!r}, "
                    f"absolute_error={absolute_error:.3e}, "
                    f"relative_error={relative_error:.3e}"
                )
        elapsed = time.perf_counter() - level_started
        if verbose and checked_here:
            print(
                f"total level {total_twice / 2:g}: "
                f"{checked_here} coefficients agree ({elapsed:.3f} s)",
                flush=True,
            )

    nonzero_rows = [row for row in rows if row["passed"]]
    max_absolute = max((row["absolute_error"] for row in rows), default=0.0)
    max_relative = max((row["relative_error"] for row in rows), default=0.0)
    worst = max(rows, key=lambda row: row["relative_error"], default=None)
    return {
        "description": (
            "NS torus two-point necklace: exact PBW/Ward sewing versus "
            "fixed-difference internal-weight recursion"
        ),
        "scope": {
            "max_total_physical_level": max_total_level,
            "selection_rule": "n1+n2 <= max level and 2n1 = 2n2 mod 2",
            "coefficient_count": len(nonzero_rows),
        },
        "parameters": {
            "b": str(b),
            "c": str(c),
            "h1": str(h1),
            "h2": str(h2),
            "d1": str(d1),
            "d2": str(d2),
        },
        "tolerances": {
            "absolute": absolute_tolerance,
            "relative": relative_tolerance,
        },
        "summary": {
            "all_passed": all(row["passed"] for row in rows),
            "maximum_absolute_error": max_absolute,
            "maximum_relative_error": max_relative,
            "worst_twice_levels": None if worst is None else worst["twice_levels"],
            "elapsed_seconds": time.perf_counter() - started,
        },
        "coefficients": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-total-level", type=int, default=6)
    parser.add_argument("--b", default="127/100")
    parser.add_argument("--h1", default="71/100")
    parser.add_argument("--h2", default="83/100")
    parser.add_argument("--d1", default="27/100")
    parser.add_argument("--d2", default="9/25")
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-9)
    parser.add_argument("--relative-tolerance", type=float, default=2.0e-10)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results_level6.json",
    )
    args = parser.parse_args()

    result = run_check(
        max_total_level=args.max_total_level,
        b=_rational(args.b),
        h1=_rational(args.h1),
        h2=_rational(args.h2),
        d1=_rational(args.d1),
        d2=_rational(args.d2),
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = result["summary"]
    print(
        f"PASS: {result['scope']['coefficient_count']} coefficients; "
        f"max abs={summary['maximum_absolute_error']:.3e}; "
        f"max rel={summary['maximum_relative_error']:.3e}; "
        f"elapsed={summary['elapsed_seconds']:.3f} s"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
