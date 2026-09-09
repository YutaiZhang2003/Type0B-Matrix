#!/usr/bin/env python3
"""Preliminary NS five-point elliptic h-recursion check, not production code.

Compare all bottom-component parity channels with the existing independent
NS c-recursion after the exact pillow change of variables. The candidate
seed is theta_3(q^2) in the all-even channel and zero in the others.
Also check the pillow oscillator product and the equivalent unit-seed
normalization H_hat=H/theta_3(q^2).
Only generic nonconfluent parameters are tested here.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from math import isqrt
from pathlib import Path
import sys

import mpmath as mp
import sympy as sp


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "Code" / "c_Recursion"))

from ns_multipoint_c_recursion import NSSphereLinearCRecursion  # noqa: E402
from ns_multipoint_h_recursion import (  # noqa: E402
    ns_b_from_c,
    ns_degenerate_weight_mp,
)
from ns_recursion_recipe import (  # noqa: E402
    ns_fusion_polynomial_mp,
    ns_inverse_null_slope_mp,
)
from check_pillow_h_recursion_symbolic_order4 import (  # noqa: E402
    direct_reduced_pillow_coefficients,
)


CASES = (
    ("A", "14.7", ("0.31", "0.42", "0.53", "0.47", "0.28"), ("0.73", "1.10")),
    ("B", "21.3", ("0.22", "0.61", "0.39", "0.74", "0.45"), ("1.13", "0.85")),
)


def vertex_labels(edge_parities):
    left, right = edge_parities
    return left, left ^ right, right


def sympy_number(value, digits=65):
    return sp.Float(str(mp.re(value)), digits) + sp.I * sp.Float(
        str(mp.im(value)), digits
    )


def mpmath_number(value, digits=60):
    return mp.mpc(str(sp.N(sp.re(value), digits)), str(sp.N(sp.im(value), digits)))


def inverse_theta3_q_squared_coefficients(order):
    """Exact integer coefficients of 1/theta_3(q^2)."""

    powers = tuple(2 * j * j for j in range(1, isqrt(order // 2) + 1))
    inverse = [1]
    for degree in range(1, order + 1):
        inverse.append(-2 * sum(inverse[degree - k] for k in powers if k <= degree))
    return tuple(inverse)


def check_oscillator_product(order=10, precision=75):
    """Check the product identity and the geometric-prefactor conversion."""

    q = sp.Symbol("q")
    theta_product = sp.S.One
    denominator = sp.S.One
    for j in range(1, order // 2 + 1):
        theta_product = sp.series(
            theta_product * (1 - q ** (4 * j)) * (1 + q ** (4 * j - 2)) ** 2,
            q, 0, order + 1,
        ).removeO()
        denominator = sp.series(
            denominator * (1 - q ** (2 * j)) ** (-sp.Rational(3, 4)),
            q, 0, order + 1,
        ).removeO()
    theta_series = 1 + 2 * sum(q ** (2 * j * j) for j in range(1, isqrt(order // 2) + 1))
    assert sp.expand(theta_product - theta_series) == 0
    oscillator = sp.series(theta_product * denominator, q, 0, order + 1).removeO()
    assert oscillator.coeff(q, 2) == sp.Rational(11, 4)
    assert oscillator.coeff(q, 4) == sp.Rational(93, 32)

    with mp.workdps(precision):
        worst = mp.mpf(0)
        for q_text in ("0.01", "0.1", "0.3"):
            nome = mp.mpf(q_text)
            theta3 = mp.jtheta(3, 0, nome)
            z = (mp.jtheta(2, 0, nome) / theta3) ** 4
            conversion = ((16 * nome) / (z * (1 - z))) ** (mp.mpf(1) / 16)
            conversion *= theta3 ** (-mp.mpf(3) / 4)
            expected = mp.qp(nome ** 2) ** (-mp.mpf(3) / 4)
            worst = max(worst, abs(conversion / expected - 1))
        assert worst < mp.power(10, -(precision - 10))
        print(
            f"Pillow product: exact through q^{order}; geometric conversion "
            f"max relative error {mp.nstr(worst, 8)}", flush=True,
        )


def make_elliptic_coefficient(c, external_weights, *, unit_seed=False):
    """Return the candidate coefficient function indexed by twice-levels."""

    b = ns_b_from_c(c)
    d = tuple(external_weights)

    @lru_cache(maxsize=None)
    def coefficient(levels, internal_weights):
        parities = tuple(level % 2 for level in levels)
        labels = vertex_labels(parities)
        result = mp.mpc(0)

        # q=p1*p2 and theta_3(q^2)=1+2*sum_(j>=1) q^(2*j*j).
        if levels[0] == levels[1]:
            degree = levels[0]
            if degree == 0:
                result = mp.mpc(1)
            elif not unit_seed and degree % 4 == 0 and isqrt(degree // 4) ** 2 == degree // 4:
                result = mp.mpc(2)

        for edge in range(2):
            for r in range(1, levels[edge] + 1):
                for s in range(1, levels[edge] // r + 1):
                    if (r + s) % 2:
                        continue
                    product = r * s
                    pole = ns_degenerate_weight_mp(b, r, s)
                    pole_weights = tuple(
                        h - internal_weights[edge] + pole for h in internal_weights
                    )
                    left_pair = (
                        (d[0], d[1]) if edge == 0 else (pole_weights[0], d[2])
                    )
                    right_pair = (
                        (pole_weights[1], d[2]) if edge == 0 else (d[4], d[3])
                    )
                    residue = (-1) ** product * ns_inverse_null_slope_mp(r, s, b)
                    for label, pair in (
                        (labels[edge], left_pair),
                        (labels[edge + 1], right_pair),
                    ):
                        residue *= ns_fusion_polynomial_mp(
                            r=r,
                            s=s,
                            a=label,
                            first_weight=pair[0],
                            second_weight=pair[1],
                            b=b,
                        )
                    child_weights = list(pole_weights)
                    child_weights[edge] += mp.mpf(product) / 2
                    child_levels = list(levels)
                    child_levels[edge] -= product
                    # Both five-point edges touch a cap: (4*p_i)^(rs/2).
                    result += (
                        mp.mpf(2) ** product
                        * residue
                        / (internal_weights[edge] - pole)
                        * coefficient(tuple(child_levels), tuple(child_weights))
                    )
        return result

    return coefficient


def compare_case(case, degree, precision):
    name, c_text, external_text, internal_text = case
    with mp.workdps(precision):
        c = mp.mpf(c_text)
        external = tuple(map(mp.mpf, external_text))
        internal = tuple(map(mp.mpf, internal_text))
        recursive = make_elliptic_coefficient(c, external)
        unit_recursive = make_elliptic_coefficient(c, external, unit_seed=True)
        inverse_theta = inverse_theta3_q_squared_coefficients(degree)
        worst = mp.mpf(0)
        unit_worst = mp.mpf(0)
        worst_location = None
        count = 0

        for parities in ((0, 0), (0, 1), (1, 0), (1, 1)):
            residual_degree = (2 * degree - sum(parities)) // 2
            if residual_degree < 0:
                continue
            block = NSSphereLinearCRecursion(
                central_charge=c,
                external_weights=external,
                internal_weights=internal,
                vertex_sectors=vertex_labels(parities),
                working_precision=precision,
            )
            plane = {
                (i, j): sympy_number(
                    block.coefficient((2 * i + parities[0], 2 * j + parities[1])),
                    precision - 10,
                )
                for i in range(residual_degree + 1)
                for j in range(residual_degree + 1 - i)
            }
            # The existing bosonic kinematic helper strips Lambda^(c_arg-1).
            # c_arg=c_NS-1/2 therefore strips Lambda^(c_NS-3/2). The actual
            # central charge used in the NS c-recursion above is unchanged.
            # Shifting its h arguments by epsilon/2 supplies the fractional
            # powers of the coordinate units. Restore their leading monomial
            # and the factor 4^((epsilon1+epsilon2)/2) in the comparison below.
            pulled_back = direct_reduced_pillow_coefficients(
                residual_degree,
                plane_coefficients=plane,
                c=sp.Rational(c_text) - sp.Rational(1, 2),
                h1=sp.Rational(internal_text[0]) + sp.Rational(parities[0], 2),
                h2=sp.Rational(internal_text[1]) + sp.Rational(parities[1], 2),
                weights=tuple(map(sp.Rational, external_text)),
            )
            for i in range(residual_degree + 1):
                for j in range(residual_degree + 1 - i):
                    levels = 2 * i + parities[0], 2 * j + parities[1]
                    predicted = recursive(levels, internal)
                    direct = 2 ** sum(parities) * mpmath_number(
                        pulled_back.get((i, j), sp.S.Zero), precision - 15
                    )
                    error = abs(direct - predicted) / max(1, abs(direct), abs(predicted))
                    count += 1
                    if error > worst:
                        worst = error
                        worst_location = parities, levels
                    # q^k=(p1*p2)^k shifts both integer edge powers by k.
                    unit_direct = 2 ** sum(parities) * mpmath_number(
                        sum(
                            inverse_theta[k] * pulled_back.get((i - k, j - k), sp.S.Zero)
                            for k in range(min(i, j) + 1)
                        ),
                        precision - 15,
                    )
                    unit_predicted = unit_recursive(levels, internal)
                    unit_worst = max(
                        unit_worst,
                        abs(unit_direct - unit_predicted)
                        / max(1, abs(unit_direct), abs(unit_predicted)),
                    )

        print(
            f"case {name}: {count} coefficients; max relative error "
            f"{mp.nstr(worst, 8)} at {worst_location}; "
            f"unit-seed max relative error {mp.nstr(unit_worst, 8)}",
            flush=True,
        )
        tolerance = mp.power(10, -(precision - 30))
        if max(worst, unit_worst) > tolerance:
            raise AssertionError(f"case {name} exceeds tolerance {tolerance}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--precision", type=int, default=75)
    args = parser.parse_args()
    if args.degree < 0 or args.precision < 50:
        parser.error("degree must be nonnegative and precision at least 50")
    check_oscillator_product(precision=args.precision)
    for case in CASES:
        compare_case(case, args.degree, args.precision)
    print("Preliminary five-point checks passed; the general-n seed remains a proposal.")


if __name__ == "__main__":
    main()
