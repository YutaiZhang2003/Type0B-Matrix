#!/usr/bin/env python3
"""Analytic fixed-c check of the NS two-point h-recursion through level 3.

Here ``c``, ``d1``, and ``d2`` are fixed rational numbers, while ``h1`` and
``h2`` remain algebraically independent symbols.  For every parity-allowed
pair with n1+n2 <= 3, the script constructs the direct PBW coefficient and
the fixed-difference h-recursion coefficient as rational functions of
``(h1,h2)``.  It clears their denominators and requires the residual
numerator to vanish identically.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import sympy as sp

from check_level6 import ExactNSTorusTwoPointPBW, HERE, _ordinary_c


def ns_degenerate_weight(b: sp.Expr, r: int, s: int) -> sp.Expr:
    if r < 1 or s < 1 or (r + s) % 2:
        raise ValueError("NS Kac labels require r,s >= 1 and r+s even")
    return sp.factor(
        -sp.Rational(r * s - 1, 4)
        + sp.Rational(1 - r * r, 8) * b**2
        + sp.Rational(1 - s * s, 8) / b**2
    )


@lru_cache(maxsize=None)
def ns_a_factor(b: sp.Expr, r: int, s: int) -> sp.Expr:
    result = sp.Rational(1, 2)
    for p in range(1 - r, r + 1):
        for q in range(1 - s, s + 1):
            if (p + q) % 2 or (p, q) in ((0, 0), (r, s)):
                continue
            result /= (p * b + q / b) / sp.sqrt(2)
    return sp.factor(result)


@lru_cache(maxsize=None)
def ns_ns_fusion_polynomial(
    b: sp.Expr,
    r: int,
    s: int,
    lower_weight: sp.Expr,
    upper_weight: sp.Expr,
    starred: bool,
) -> sp.Expr:
    """Exact fusion polynomial, paired so no square-root branch remains."""
    q_background = b + 1 / b
    lower_lambda_squared = q_background**2 - 8 * lower_weight
    upper_lambda_squared = q_background**2 - 8 * upper_weight
    wanted_parity = 1 if starred else 0
    linears: list[sp.Expr] = []
    for k in range(r):
        for ell in range(s):
            if (k + ell) % 2 != wanted_parity:
                continue
            p = 1 - r + 2 * k
            q = 1 - s + 2 * ell
            linears.append(sp.factor(p * b + q / b))

    # The selected Kac lattice is invariant under linear -> -linear.  Pairing
    # those terms evaluates the product of the two momentum factors without
    # introducing either square-root branch.
    remaining = list(linears)
    result = sp.S.One
    while remaining:
        linear = remaining.pop(0)
        if linear == 0:
            result *= (lower_lambda_squared - upper_lambda_squared) / 8
            continue
        partner = next(
            (index for index, candidate in enumerate(remaining)
             if sp.simplify(candidate + linear) == 0),
            None,
        )
        if partner is None:
            raise AssertionError("fusion-polynomial Kac lattice is not paired")
        remaining.pop(partner)
        common = lower_lambda_squared + linear**2 - upper_lambda_squared
        result *= (common**2 - 4 * lower_lambda_squared * linear**2) / 64
    return sp.factor(result)


@lru_cache(maxsize=None)
def ns_character_coefficient(twice_level: int) -> sp.Integer:
    coefficients = [sp.S.Zero] * (twice_level + 1)
    coefficients[0] = sp.S.One
    for mode in range(2, twice_level + 1, 2):
        for level in range(mode, twice_level + 1):
            coefficients[level] += coefficients[level - mode]
    for mode in range(1, twice_level + 1, 2):
        for level in range(twice_level, mode - 1, -1):
            coefficients[level] += coefficients[level - mode]
    return sp.Integer(coefficients[twice_level])


class SymbolicNSTorusTwoPointHRecursion:
    """Exact fixed-difference recursion with symbolic internal weights."""

    def __init__(
        self,
        *,
        b: sp.Expr,
        h1: sp.Symbol,
        h2: sp.Symbol,
        d1: sp.Expr,
        d2: sp.Expr,
    ) -> None:
        self.b = b
        self.h1 = h1
        self.h2 = h2
        self.d1 = d1
        self.d2 = d2

    @lru_cache(maxsize=None)
    def coefficient_on_line(
        self,
        twice_level_1: int,
        twice_level_2: int,
        base_weight: sp.Expr,
        weight_difference: sp.Expr,
        routing: int,
    ) -> sp.Expr:
        if twice_level_1 < 0 or twice_level_2 < 0:
            return sp.S.Zero
        weight_1 = base_weight
        weight_2 = base_weight + weight_difference
        result = sp.S.Zero
        if routing == 0 and twice_level_1 == twice_level_2:
            result = ns_character_coefficient(twice_level_1)

        for edge, available_level in (
            (1, twice_level_1),
            (2, twice_level_2),
        ):
            for r in range(1, available_level + 1):
                for s in range(1, available_level // r + 1):
                    product = r * s
                    if product > available_level or (r + s) % 2:
                        continue
                    degenerate = ns_degenerate_weight(self.b, r, s)
                    if edge == 1:
                        denominator = weight_1 - degenerate
                        adjacent_at_pole = degenerate + weight_difference
                    else:
                        denominator = weight_2 - degenerate
                        adjacent_at_pole = degenerate - weight_difference
                    residue = (
                        ns_a_factor(self.b, r, s)
                        * ns_ns_fusion_polynomial(
                            self.b,
                            r,
                            s,
                            self.d1,
                            adjacent_at_pole,
                            bool(routing),
                        )
                        * ns_ns_fusion_polynomial(
                            self.b,
                            r,
                            s,
                            self.d2,
                            adjacent_at_pole,
                            bool(routing),
                        )
                    )
                    next_routing = routing ^ (product % 2)
                    if edge == 1:
                        tail = self.coefficient_on_line(
                            twice_level_1 - product,
                            twice_level_2,
                            degenerate + sp.Rational(product, 2),
                            weight_difference - sp.Rational(product, 2),
                            next_routing,
                        )
                    else:
                        tail = self.coefficient_on_line(
                            twice_level_1,
                            twice_level_2 - product,
                            degenerate - weight_difference,
                            weight_difference + sp.Rational(product, 2),
                            next_routing,
                        )
                    result += residue * tail / denominator
        return sp.cancel(result)

    def coefficient(self, twice_level_1: int, twice_level_2: int) -> sp.Expr:
        return self.coefficient_on_line(
            twice_level_1,
            twice_level_2,
            self.h1,
            self.h2 - self.h1,
            0,
        )


def expression_record(expression: sp.Expr, h1: sp.Symbol, h2: sp.Symbol) -> dict[str, Any]:
    expression = sp.cancel(expression)
    numerator, denominator = sp.fraction(expression)
    text = sp.sstr(expression)
    return {
        "expression": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "operation_count": int(sp.count_ops(expression)),
        "numerator_total_degree": sp.Poly(numerator, h1, h2).total_degree(),
        "denominator_total_degree": sp.Poly(denominator, h1, h2).total_degree(),
    }


def run_check(max_total_level: int = 3) -> dict[str, Any]:
    if max_total_level < 0:
        raise ValueError("max_total_level must be nonnegative")
    b = sp.Rational(127, 100)
    c = sp.factor(_ordinary_c(b))
    d1 = sp.Rational(27, 100)
    d2 = sp.Rational(9, 25)
    h1, h2 = sp.symbols("h1 h2")
    direct = ExactNSTorusTwoPointPBW(c=c, h1=h1, h2=h2, d1=d1, d2=d2)
    recursive = SymbolicNSTorusTwoPointHRecursion(
        b=b, h1=h1, h2=h2, d1=d1, d2=d2
    )

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for total_twice in range(2 * max_total_level + 1):
        for twice_level_1 in range(total_twice + 1):
            twice_level_2 = total_twice - twice_level_1
            if (twice_level_1 - twice_level_2) % 2:
                continue
            coefficient_started = time.perf_counter()
            direct_coefficient = direct.coefficient(twice_level_1, twice_level_2)
            recursive_coefficient = recursive.coefficient(
                twice_level_1, twice_level_2
            )
            residual = sp.together(direct_coefficient - recursive_coefficient)
            residual_numerator, residual_denominator = sp.fraction(residual)
            expanded_numerator = sp.expand(residual_numerator)
            passed = expanded_numerator == 0
            row = {
                "twice_levels": [twice_level_1, twice_level_2],
                "levels": [str(sp.Rational(twice_level_1, 2)),
                           str(sp.Rational(twice_level_2, 2))],
                "passed_after_clearing_denominators": passed,
                "residual_numerator": sp.sstr(expanded_numerator),
                "residual_denominator_operation_count": int(
                    sp.count_ops(residual_denominator)
                ),
                "direct_pbw": expression_record(direct_coefficient, h1, h2),
                "h_recursion": expression_record(recursive_coefficient, h1, h2),
                "elapsed_seconds": time.perf_counter() - coefficient_started,
            }
            rows.append(row)
            print(
                f"levels ({row['levels'][0]},{row['levels'][1]}): "
                f"residual numerator = {row['residual_numerator']} "
                f"({row['elapsed_seconds']:.3f} s)",
                flush=True,
            )
            if not passed:
                raise AssertionError(
                    "symbolic PBW/h-recursion mismatch at twice-levels "
                    f"({twice_level_1},{twice_level_2}): "
                    f"numerator={expanded_numerator}"
                )

    return {
        "description": (
            "Analytic fixed-c NS torus two-point check: exact PBW/Ward "
            "sewing versus fixed-difference h-recursion"
        ),
        "scope": {
            "max_total_physical_level": max_total_level,
            "selection_rule": "n1+n2 <= max level and 2n1 = 2n2 mod 2",
            "coefficient_count": len(rows),
            "symbolic_variables": ["h1", "h2"],
        },
        "fixed_parameters": {
            "b": str(b),
            "c": str(c),
            "d1": str(d1),
            "d2": str(d2),
        },
        "criterion": (
            "expand(numerator(together(F_PBW-F_hrec))) == 0"
        ),
        "summary": {
            "all_passed": all(
                row["passed_after_clearing_denominators"] for row in rows
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "coefficients": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-total-level", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results_symbolic_level3.json",
    )
    args = parser.parse_args()
    result = run_check(args.max_total_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"PASS: {result['scope']['coefficient_count']} analytic coefficient "
        f"identities; elapsed={result['summary']['elapsed_seconds']:.3f} s; "
        f"wrote {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
