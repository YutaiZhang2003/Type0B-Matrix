#!/usr/bin/env python3
"""Exact PBW versus fixed-difference h-recursion for NS sphere blocks.

The four-point and five-point linear-channel blocks are checked through
physical level three on every internal edge.  All admissible fixed-parity
trinion-sector routings are included.  Both sides use exact SymPy arithmetic,
but the direct PBW side imports no Kac weights, null slopes, fusion
polynomials, or h-recursion routine.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import sympy as sp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "Code" / "c_Recursion"))
sys.path.insert(
    0, str(ROOT / "Code" / "ns_torus_two_point_h_recursion_check")
)

from ns_genus2_symbolic_low_order import (  # noqa: E402
    ExactNSDescendantThreeForm,
    ExactNSVermaModule,
)
from check_symbolic_level3 import (  # noqa: E402
    ns_a_factor,
    ns_degenerate_weight,
    ns_ns_fusion_polynomial,
)


def ordinary_c(b: sp.Expr) -> sp.Expr:
    return sp.factor(sp.Rational(3, 2) + 3 * (b + 1 / b) ** 2)


def compatible_edge_parities(vertex_sectors: Sequence[int]) -> tuple[int, ...]:
    """Solve the bottom-component tree parity constraints."""
    sectors = tuple(int(value) for value in vertex_sectors)
    if len(sectors) < 2 or any(value not in (0, 1) for value in sectors):
        raise ValueError("vertex sectors must be a zero/one tuple of length >= 2")
    parities = [sectors[0]]
    for sector in sectors[1:-1]:
        parities.append(parities[-1] ^ sector)
    if parities[-1] != sectors[-1]:
        raise ValueError("inconsistent bottom-component vertex routing")
    return tuple(parities)


class ExactDirectSphereLinearPBW:
    """Direct four- or five-point linear-channel descendant contraction."""

    def __init__(
        self,
        *,
        c: sp.Expr,
        external_weights: Sequence[sp.Expr],
        internal_weights: Sequence[sp.Expr],
    ) -> None:
        self.c = c
        self.external_weights = tuple(external_weights)
        self.internal_weights = tuple(internal_weights)
        if len(self.external_weights) not in (4, 5):
            raise ValueError("the direct checker supports four or five points")
        if len(self.internal_weights) != len(self.external_weights) - 3:
            raise ValueError("incorrect number of internal weights")
        self.modules = tuple(
            ExactNSVermaModule(c=c, weight=weight)
            for weight in self.internal_weights
        )
        if len(self.internal_weights) == 1:
            h1 = self.internal_weights[0]
            d1, d2, d3, d4 = self.external_weights
            self.forms = (
                ExactNSDescendantThreeForm(c=c, weights=(h1, d2, d1)),
                ExactNSDescendantThreeForm(c=c, weights=(d4, d3, h1)),
            )
        else:
            h1, h2 = self.internal_weights
            d1, d2, d3, d4, d5 = self.external_weights
            self.forms = (
                ExactNSDescendantThreeForm(c=c, weights=(h1, d2, d1)),
                ExactNSDescendantThreeForm(c=c, weights=(h2, d3, h1)),
                ExactNSDescendantThreeForm(c=c, weights=(d5, d4, h2)),
            )

    @lru_cache(maxsize=None)
    def inverse_gram(self, edge: int, twice_level: int) -> sp.Matrix:
        return self.modules[edge].gram_matrix(twice_level).inv()

    @lru_cache(maxsize=None)
    def coefficient(self, twice_levels: tuple[int, ...]) -> sp.Expr:
        if len(twice_levels) != len(self.internal_weights):
            raise ValueError("incorrect number of edge levels")
        if len(twice_levels) == 1:
            (level,) = twice_levels
            basis = self.modules[0].basis(level)
            left = sp.Matrix(
                [self.forms[0].value(state, (), ()) for state in basis]
            )
            right = sp.Matrix(
                [self.forms[1].value((), (), state) for state in basis]
            )
            return sp.cancel((right.T * self.inverse_gram(0, level) * left)[0])

        level_1, level_2 = twice_levels
        basis_1 = self.modules[0].basis(level_1)
        basis_2 = self.modules[1].basis(level_2)
        left = sp.Matrix(
            [self.forms[0].value(state, (), ()) for state in basis_1]
        )
        middle = sp.Matrix(
            [
                [self.forms[1].value(state_2, (), state_1) for state_1 in basis_1]
                for state_2 in basis_2
            ]
        )
        right = sp.Matrix(
            [self.forms[2].value((), (), state) for state in basis_2]
        )
        return sp.cancel(
            (
                right.T
                * self.inverse_gram(1, level_2)
                * middle
                * self.inverse_gram(0, level_1)
                * left
            )[0]
        )


class ExactSphereLinearHRecursion:
    """Exact coefficient recursion in the correlated sphere-linear family."""

    def __init__(
        self,
        *,
        b: sp.Expr,
        external_weights: Sequence[sp.Expr],
        internal_weights: Sequence[sp.Expr],
    ) -> None:
        self.b = b
        self.external_weights = tuple(external_weights)
        self.internal_weights = tuple(internal_weights)
        self.edge_count = len(self.internal_weights)
        if len(self.external_weights) != self.edge_count + 3:
            raise ValueError("sphere linear data have inconsistent lengths")
        if self.edge_count not in (1, 2):
            raise ValueError("the checker supports four or five points")
        self.fixed_middle_weights = self.external_weights[1:-1]
        self.initial_h = self.internal_weights[0]
        self.initial_differences = tuple(
            weight - self.initial_h for weight in self.internal_weights
        )
        self.initial_e_left = self.external_weights[0] - self.initial_h
        self.initial_e_right = self.external_weights[-1] - self.initial_h

    @lru_cache(maxsize=None)
    def coefficient_on_line(
        self,
        twice_levels: tuple[int, ...],
        vertex_sectors: tuple[int, ...],
        base_weight: sp.Expr,
        differences: tuple[sp.Expr, ...],
        e_left: sp.Expr,
        e_right: sp.Expr,
    ) -> sp.Expr:
        if any(level < 0 for level in twice_levels):
            return sp.S.Zero
        if len(vertex_sectors) != self.edge_count + 1:
            raise ValueError("incorrect vertex-sector count")
        result = sp.S.One if (
            all(level == 0 for level in twice_levels)
            and all(sector == 0 for sector in vertex_sectors)
        ) else sp.S.Zero

        for edge, available_level in enumerate(twice_levels):
            for r in range(1, available_level + 1):
                for s in range(1, available_level // r + 1):
                    product = r * s
                    if product > available_level or (r + s) % 2:
                        continue
                    level_shift = sp.Rational(product, 2)
                    degenerate = ns_degenerate_weight(self.b, r, s)
                    pole_base = degenerate - differences[edge]
                    pole_internal = tuple(
                        pole_base + difference for difference in differences
                    )
                    pole_external = (
                        pole_base + e_left,
                        *self.fixed_middle_weights,
                        pole_base + e_right,
                    )

                    if edge == 0:
                        left_pair = (pole_external[0], pole_external[1])
                    else:
                        left_pair = (
                            pole_internal[edge - 1],
                            pole_external[edge + 1],
                        )
                    if edge == self.edge_count - 1:
                        right_pair = (pole_external[-1], pole_external[-2])
                    else:
                        right_pair = (
                            pole_internal[edge + 1],
                            pole_external[edge + 2],
                        )

                    residue = (
                        # Converting the two slot-factorization identities
                        # to the human-note fixed-parity block leaves this
                        # ordinary-edge transport phase.  The (1,1) odd
                        # four-point coefficient detects its sign directly.
                        (-1) ** product
                        * ns_a_factor(self.b, r, s)
                        * ns_ns_fusion_polynomial(
                            self.b,
                            r,
                            s,
                            left_pair[0],
                            left_pair[1],
                            bool(vertex_sectors[edge]),
                        )
                        * ns_ns_fusion_polynomial(
                            self.b,
                            r,
                            s,
                            right_pair[0],
                            right_pair[1],
                            bool(vertex_sectors[edge + 1]),
                        )
                    )
                    child_levels = list(twice_levels)
                    child_levels[edge] -= product
                    child_sectors = list(vertex_sectors)
                    if product % 2:
                        child_sectors[edge] ^= 1
                        child_sectors[edge + 1] ^= 1

                    child_differences = list(differences)
                    if edge == 0:
                        child_base = degenerate + level_shift
                        for index in range(1, self.edge_count):
                            child_differences[index] -= level_shift
                        child_e_left = e_left - level_shift
                        child_e_right = e_right - level_shift
                    else:
                        child_base = pole_base
                        child_differences[edge] += level_shift
                        child_e_left = e_left
                        child_e_right = e_right

                    denominator = (
                        base_weight + differences[edge] - degenerate
                    )
                    tail = self.coefficient_on_line(
                        tuple(child_levels),
                        tuple(child_sectors),
                        child_base,
                        tuple(child_differences),
                        child_e_left,
                        child_e_right,
                    )
                    result += residue * tail / denominator
        return sp.cancel(result)

    def coefficient(
        self, twice_levels: tuple[int, ...], vertex_sectors: tuple[int, ...]
    ) -> sp.Expr:
        return self.coefficient_on_line(
            twice_levels,
            vertex_sectors,
            self.initial_h,
            self.initial_differences,
            self.initial_e_left,
            self.initial_e_right,
        )


SAMPLES = (
    {
        "name": "generic-rational-1",
        "b": sp.Rational(127, 100),
        "four_external": tuple(
            map(sp.Rational, ("41/100", "27/100", "9/25", "53/100"))
        ),
        "four_internal": (sp.Rational(71, 100),),
        "five_external": tuple(
            map(
                sp.Rational,
                ("41/100", "27/100", "9/25", "43/100", "57/100"),
            )
        ),
        "five_internal": (sp.Rational(71, 100), sp.Rational(83, 100)),
    },
    {
        "name": "generic-rational-2",
        "b": sp.Rational(137, 100),
        "four_external": tuple(
            map(sp.Rational, ("47/100", "31/100", "43/100", "61/100"))
        ),
        "four_internal": (sp.Rational(79, 100),),
        "five_external": tuple(
            map(
                sp.Rational,
                ("47/100", "31/100", "43/100", "37/100", "67/100"),
            )
        ),
        "five_internal": (sp.Rational(79, 100), sp.Rational(97, 100)),
    },
)


def sector_routings(edge_count: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        routing
        for routing in itertools.product((0, 1), repeat=edge_count + 1)
        if sum(routing) % 2 == 0
    )


def level_tuples(
    vertex_sectors: tuple[int, ...], max_physical_level: int
) -> tuple[tuple[int, ...], ...]:
    parities = compatible_edge_parities(vertex_sectors)
    values = [
        tuple(
            level
            for level in range(2 * max_physical_level + 1)
            if level % 2 == parity
        )
        for parity in parities
    ]
    return tuple(itertools.product(*values))


def run_block_check(
    *,
    b: sp.Expr,
    external_weights: tuple[sp.Expr, ...],
    internal_weights: tuple[sp.Expr, ...],
    max_physical_level: int,
) -> dict[str, Any]:
    c = ordinary_c(b)
    direct = ExactDirectSphereLinearPBW(
        c=c,
        external_weights=external_weights,
        internal_weights=internal_weights,
    )
    recursive = ExactSphereLinearHRecursion(
        b=b,
        external_weights=external_weights,
        internal_weights=internal_weights,
    )
    rows: list[dict[str, Any]] = []
    for routing in sector_routings(len(internal_weights)):
        for levels in level_tuples(routing, max_physical_level):
            started = time.perf_counter()
            direct_value = direct.coefficient(levels)
            recursive_value = recursive.coefficient(levels, routing)
            residual = sp.cancel(direct_value - recursive_value)
            passed = residual == 0
            row = {
                "vertex_sectors": list(routing),
                "twice_levels": list(levels),
                "levels": [str(sp.Rational(level, 2)) for level in levels],
                "direct_pbw": sp.sstr(direct_value),
                "h_recursion": sp.sstr(recursive_value),
                "residual": sp.sstr(residual),
                "passed": passed,
                "elapsed_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            print(
                f"N={len(external_weights)} sectors={routing} "
                f"levels={row['levels']}: residual={row['residual']} "
                f"({row['elapsed_seconds']:.3f} s)",
                flush=True,
            )
            if not passed:
                raise AssertionError(
                    f"sphere PBW/h-recursion mismatch: N={len(external_weights)}, "
                    f"routing={routing}, twice_levels={levels}, residual={residual}"
                )
    return {
        "point_count": len(external_weights),
        "max_physical_level_on_each_edge": max_physical_level,
        "cutoff": "rectangular on internal edges",
        "parameters": {
            "b": str(b),
            "c": str(c),
            "external_weights": [str(value) for value in external_weights],
            "internal_weights": [str(value) for value in internal_weights],
        },
        "summary": {
            "all_passed": all(row["passed"] for row in rows),
            "coefficient_count": len(rows),
        },
        "coefficients": rows,
    }


def run_check(max_physical_level: int = 3) -> dict[str, Any]:
    started = time.perf_counter()
    sample_results: list[dict[str, Any]] = []
    for sample in SAMPLES:
        print(f"starting {sample['name']}", flush=True)
        four = run_block_check(
            b=sample["b"],
            external_weights=sample["four_external"],
            internal_weights=sample["four_internal"],
            max_physical_level=max_physical_level,
        )
        five = run_block_check(
            b=sample["b"],
            external_weights=sample["five_external"],
            internal_weights=sample["five_internal"],
            max_physical_level=max_physical_level,
        )
        sample_results.append(
            {"name": sample["name"], "four_point": four, "five_point": five}
        )
    total = sum(
        result[key]["summary"]["coefficient_count"]
        for result in sample_results
        for key in ("four_point", "five_point")
    )
    return {
        "description": (
            "Exact NS sphere linear-channel PBW/Ward sewing versus "
            "correlated fixed-difference internal-weight recursion"
        ),
        "scope": {
            "max_physical_level_on_each_internal_edge": max_physical_level,
            "all_admissible_bottom-component_vertex_routings": True,
            "sample_count": len(sample_results),
        },
        "summary": {
            "all_passed": all(
                result[key]["summary"]["all_passed"]
                for result in sample_results
                for key in ("four_point", "five_point")
            ),
            "coefficient_comparison_count": total,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "samples": sample_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-level", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results_order3.json",
    )
    args = parser.parse_args()
    result = run_check(args.max_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"PASS: {result['summary']['coefficient_comparison_count']} exact "
        f"coefficient identities; elapsed={result['summary']['elapsed_seconds']:.3f} s; "
        f"wrote {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
