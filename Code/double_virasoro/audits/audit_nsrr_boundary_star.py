#!/usr/bin/env python3
"""Audit Yuchen's fundamental NS--R--R boundary against direct PBW sewing.

The Human Note is not modified. The production ``L_1`` boundary grid is
compared with the independently sewn auxiliary-Majorana star physical-PBW
product at both integral and half-integral NS boundaries.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from pathlib import Path
import sys

import sympy as sp


HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
for directory in (
    HERE,
    CODE / "double_virasoro" / "nsrr",
    CODE / "c_Recursion",
    CODE / "genus_2_cross_channel",
    CODE / "ramond_branching_recursion",
    CODE / "full_ramond_block_runtime",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from compute_full_block import BranchingGrid  # noqa: E402
from compute_target import norm_product  # noqa: E402
from nsrr_genus2_block import (  # noqa: E402
    ZERO_VECTOR,
    auxiliary_majorana_nsrr_series,
    direct_pbw_nsrr_series,
    star_convolve_series,
)
from theta_star_algebra import theta_quadratic_sign  # noqa: E402


B = sp.Rational(7, 5)
MOMENTA = (
    sp.Rational(11, 23),
    sp.Rational(13, 29),
    sp.Rational(17, 31),
)
RAMOND_SIGNS = (-1, 1)


def production_boundary_vectors(*, primary_parity: int):
    """Return the certified L1-grid vectors at NS twice-levels zero and one."""

    branching = BranchingGrid(
        float(B),
        tuple(float(value) for value in MOMENTA),
        1,
        primary_parity=primary_parity,
    )
    branching.build_actions()
    raw_grids = {
        (alpha2, alpha3): branching.solve(alpha2, alpha3)[0]
        for alpha2, alpha3 in product((0, 1), repeat=2)
    }
    answer = {
        (0, 0, 0): [0.0j] * 8,
        (1, 0, 0): [0.0j] * 8,
    }
    for twice_n1, sign1 in ((0, 0), (1, -1), (1, 1)):
        n1 = Fraction(0) if twice_n1 == 0 else Fraction(sign1, 2)
        alpha_pairs = (
            ((0, 0), (1, 1))
            if twice_n1 == 0
            else ((0, 1), (1, 0))
        )
        for sign2, sign3, (alpha2, alpha3) in product(
            RAMOND_SIGNS, RAMOND_SIGNS, alpha_pairs
        ):
            labels = (
                n1,
                Fraction(sign2, 4),
                Fraction(sign3, 4),
            )
            raw = raw_grids[(alpha2, alpha3)][labels]
            normalized = raw / norm_product(
                labels,
                alpha2,
                alpha3,
                float(B),
                tuple(float(value) for value in MOMENTA),
            )
            component = (
                (twice_n1 + primary_parity) % 2
                | (alpha2 << 1)
                | (alpha3 << 2)
            )
            human_enlarged_sign = (-1) ** twice_n1
            answer[(twice_n1, 0, 0)][component] += (
                human_enlarged_sign
                * theta_quadratic_sign(component)
                * normalized
                * normalized
            )
    return {levels: tuple(vector) for levels, vector in answer.items()}




def certify_star_product():
    """Compare the certified f=0 production boundaries with R-star PBW."""

    auxiliary = auxiliary_majorana_nsrr_series(
        maximum_total_twice_level=1
    )
    maximum_error = 0.0
    rows = []
    for primary_parity in (0, 1):
        production = production_boundary_vectors(
            primary_parity=primary_parity
        )
        physical = direct_pbw_nsrr_series(
            b=B,
            momenta=MOMENTA,
            form_parity=0,
            primary_parity=primary_parity,
            etas=(1, 1),
            maximum_total_twice_level=1,
        )
        factorized_series = star_convolve_series(
            auxiliary,
            physical,
            maximum_total_twice_level=1,
        )
        for levels, layer in (((0, 0, 0), "integer NS"), ((1, 0, 0), "half NS")):
            boundary = production.get(levels, ZERO_VECTOR)
            factorized = factorized_series.get(levels, ZERO_VECTOR)
            error = max(
                abs(left - right)
                for left, right in zip(boundary, factorized)
            )
            maximum_error = max(maximum_error, error)
            if error > 5.0e-13:
                raise AssertionError(
                    f"R-star mismatch for p1={primary_parity}, "
                    f"levels={levels}: {error}"
                )
            rows.append(
                (
                    layer,
                    primary_parity,
                    0,
                    1,
                    1,
                    tuple(
                        (index, value)
                        for index, value in enumerate(boundary)
                        if abs(value) > 1.0e-12
                    ),
                )
            )
    return len(rows), maximum_error, rows


def main() -> None:
    star_checks, maximum_error, rows = certify_star_product()
    print(f"R-star sectors: {star_checks}")
    print(f"maximum absolute error: {maximum_error:.3e}")
    for layer, p1, form_parity, eta, eta_prime, support in rows:
        print(
            f"  {layer}: p1={p1}, f={form_parity}, "
            f"etas=({eta:+d},{eta_prime:+d}): {support}"
        )


if __name__ == "__main__":
    main()
