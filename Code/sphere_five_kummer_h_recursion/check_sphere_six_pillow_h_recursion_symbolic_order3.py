#!/usr/bin/env python3
"""Exact PBW check of the proposed sphere six-point pillow h-recursion.

The plane block is the comb block at

    (0, z, t1, t2, 1, infinity),

with nested OPE coordinates ``x1=z/t1``, ``x2=t1/t2``, ``x3=t2``.  The
aligned pillow segment variables obey

    rho1=4*p1,  rho2=p2,  rho3=4*p3,  p1*p2*p3=q.

The script computes the defining Virasoro PBW contraction, applies the exact
sphere-to-pillow conformal factor, strips the universal twisted character,
and compares the resulting H_6 coefficient-by-coefficient with the proposed
three-edge fixed-difference h-recursion.  By default all 20 coefficients of
total degree at most three are compared as exact SymPy expressions.
"""

from __future__ import annotations

import argparse
import functools
import time
from collections.abc import Mapping

import sympy as sp

from check_pillow_h_recursion_symbolic_order4 import (
    DegenerateData,
    inverse_gram_matrix,
    rho,
)


Index3 = tuple[int, int, int]
Series3 = dict[Index3, sp.Expr]


def _clean(series: Mapping[Index3, sp.Expr], order: int) -> Series3:
    return {
        key: value
        for key, value in series.items()
        if min(key) >= 0 and sum(key) <= order and value != 0
    }


def series_add(left: Series3, right: Series3, order: int) -> Series3:
    out = dict(left)
    for key, value in right.items():
        out[key] = out.get(key, sp.S.Zero) + value
    return _clean(out, order)


def series_scale(series: Series3, scale: sp.Expr, order: int) -> Series3:
    return _clean({key: scale * value for key, value in series.items()}, order)


def series_mul(left: Series3, right: Series3, order: int) -> Series3:
    out: Series3 = {}
    for key1, value1 in left.items():
        for key2, value2 in right.items():
            key = tuple(a + b for a, b in zip(key1, key2))
            if sum(key) <= order:
                out[key] = out.get(key, sp.S.Zero) + value1 * value2
    return _clean(out, order)


def monomial(key: Index3, coefficient: sp.Expr, order: int) -> Series3:
    return {} if sum(key) > order else {key: coefficient}


def one_plus(key: Index3, order: int) -> Series3:
    return series_add({(0, 0, 0): sp.S.One}, monomial(key, sp.S.One, order), order)


def unit_power(series: Series3, exponent: sp.Expr, order: int) -> Series3:
    constant = series.get((0, 0, 0), sp.S.Zero)
    if constant != 1:
        raise ValueError(f"unit series must have constant one, found {constant!r}")
    remainder = dict(series)
    remainder[(0, 0, 0)] = remainder.get((0, 0, 0), sp.S.Zero) - 1
    remainder = _clean(remainder, order)
    out: Series3 = {(0, 0, 0): sp.S.One}
    power: Series3 = {(0, 0, 0): sp.S.One}
    for k in range(1, order + 1):
        power = series_mul(power, remainder, order)
        if not power:
            break
        out = series_add(
            out,
            series_scale(power, sp.binomial(exponent, k), order),
            order,
        )
    return out


def z_unit_series(order: int) -> Series3:
    """Return z/(16q), with q=p1*p2*p3."""

    out: Series3 = {(0, 0, 0): sp.S.One}
    n = 1
    while 3 * (2 * n - 1) <= order:
        if 6 * n <= order:
            out = series_mul(
                out,
                unit_power(one_plus((2 * n, 2 * n, 2 * n), order), 8, order),
                order,
            )
        out = series_mul(
            out,
            unit_power(
                one_plus((2 * n - 1, 2 * n - 1, 2 * n - 1), order),
                -8,
                order,
            ),
            order,
        )
        n += 1
    return out


def mobile_y_unit(
    order: int,
    *,
    left_key: Index3,
    right_key: Index3,
) -> Series3:
    """Return t/(4*v) for aligned aggregate nomes u and v with uv=q."""

    q_key = tuple(a + b for a, b in zip(left_key, right_key))
    if q_key != (1, 1, 1):
        raise ValueError("aggregate left and right nomes must multiply to q")
    out = unit_power(one_plus(left_key, order), 2, order)
    n = 1
    while min(
        sum(right_key) + 3 * (2 * n - 2),
        sum(left_key) + 3 * (2 * n - 1),
        3 * (2 * n - 1),
    ) <= order:
        even_q = tuple(2 * n * value for value in q_key)
        odd_q = tuple((2 * n - 1) * value for value in q_key)
        if sum(even_q) <= order:
            out = series_mul(out, unit_power(one_plus(even_q, order), 4, order), order)
        if sum(odd_q) <= order:
            out = series_mul(out, unit_power(one_plus(odd_q, order), -4, order), order)

        factors = (
            (tuple(a + 2 * n * q for a, q in zip(left_key, q_key)), 2),
            (tuple(a + (2 * n - 1) * q for a, q in zip(right_key, q_key)), 2),
            (tuple(a + (2 * n - 1) * q for a, q in zip(left_key, q_key)), -2),
            (tuple(a + (2 * n - 2) * q for a, q in zip(right_key, q_key)), -2),
        )
        for key, exponent in factors:
            if sum(key) <= order:
                out = series_mul(
                    out,
                    unit_power(one_plus(key, order), exponent, order),
                    order,
                )
        n += 1
    return out


def pillow_map_units(
    order: int,
) -> tuple[Series3, Series3, Series3, Series3, Series3, Series3, Series3, Series3]:
    """Return Z,Y1,Y2,X1,X2,X3,theta3,chi^-1 as exact series."""

    one: Series3 = {(0, 0, 0): sp.S.One}
    z_unit = z_unit_series(order)
    # t1 separates p1 from p2*p3; t2 separates p1*p2 from p3.
    y1 = mobile_y_unit(order, left_key=(1, 0, 0), right_key=(0, 1, 1))
    y2 = mobile_y_unit(order, left_key=(1, 1, 0), right_key=(0, 0, 1))
    x1 = series_mul(z_unit, unit_power(y1, -1, order), order)
    x2 = series_mul(y1, unit_power(y2, -1, order), order)
    x3 = dict(y2)

    theta3 = dict(one)
    n = 1
    while 3 * n * n <= order:
        theta3 = series_add(
            theta3,
            monomial((n * n, n * n, n * n), sp.Integer(2), order),
            order,
        )
        n += 1

    chi_inverse = dict(one)
    n = 1
    while 6 * n <= order:
        factor = series_add(
            one,
            monomial((2 * n, 2 * n, 2 * n), -sp.S.One, order),
            order,
        )
        chi_inverse = series_mul(
            chi_inverse,
            unit_power(factor, sp.Rational(1, 2), order),
            order,
        )
        n += 1
    return z_unit, y1, y2, x1, x2, x3, theta3, chi_inverse


def direct_pbw_coefficients(
    order: int,
    *,
    c: sp.Expr,
    internal_weights: tuple[sp.Expr, sp.Expr, sp.Expr],
    external_weights: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr],
) -> Series3:
    """Return the defining plane descendant series in (x1,x2,x3)."""

    h1, h2, h3 = internal_weights
    d1, d2, d3, d4, d5, d6 = external_weights
    grams = tuple(
        {
            level: inverse_gram_matrix(level, weight, c)
            for level in range(order + 1)
        }
        for weight in internal_weights
    )
    out: Series3 = {}
    for n1 in range(order + 1):
        basis1, inverse1 = grams[0][n1]
        left = sp.Matrix([rho(desc, (), h1, d2, d1, c) for desc in basis1])
        propagated = inverse1 * left
        for n2 in range(order + 1 - n1):
            basis2, inverse2 = grams[1][n2]
            middle12 = sp.Matrix(
                [
                    [rho(desc2, desc1, h2, d3, h1, c) for desc1 in basis1]
                    for desc2 in basis2
                ]
            )
            propagated2 = inverse2 * middle12 * propagated
            for n3 in range(order + 1 - n1 - n2):
                basis3, inverse3 = grams[2][n3]
                middle23 = sp.Matrix(
                    [
                        [rho(desc3, desc2, h3, d4, h2, c) for desc2 in basis2]
                        for desc3 in basis3
                    ]
                )
                right = sp.Matrix(
                    [rho((), desc3, d6, d5, h3, c) for desc3 in basis3]
                )
                value = (right.T * inverse3 * middle23 * propagated2)[0]
                out[(n1, n2, n3)] = sp.cancel(value)
    return out


def direct_reduced_pillow_coefficients(
    order: int,
    *,
    plane_coefficients: Series3,
    c: sp.Expr,
    internal_weights: tuple[sp.Expr, sp.Expr, sp.Expr],
    external_weights: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr],
) -> Series3:
    """Apply the exact conformal factor and extract the proposed H_6."""

    h1, h2, h3 = internal_weights
    d1, d2, d3, d4, d5, d6 = external_weights
    a2 = h2 - h1
    a3 = h3 - h1
    z_unit, y1, y2, x1, x2, x3, theta3, chi_inverse = pillow_map_units(order)
    one: Series3 = {(0, 0, 0): sp.S.One}

    t1 = series_scale(series_mul(monomial((0, 1, 1), 1, order), y1, order), 4, order)
    t2 = series_scale(series_mul(monomial((0, 0, 1), 1, order), y2, order), 4, order)
    x1_series = series_scale(
        series_mul(monomial((1, 0, 0), 1, order), x1, order),
        4,
        order,
    )
    z_over_t2 = series_scale(
        series_mul(
            monomial((1, 1, 0), 1, order),
            series_mul(x1, x2, order),
            order,
        ),
        4,
        order,
    )
    mobile1 = series_mul(
        series_add(one, series_scale(t1, -1, order), order),
        series_add(one, series_scale(x1_series, -1, order), order),
        order,
    )
    mobile2 = series_mul(
        series_add(one, series_scale(t2, -1, order), order),
        series_add(one, series_scale(z_over_t2, -1, order), order),
        order,
    )
    z_series = series_scale(
        series_mul(monomial((1, 1, 1), 1, order), z_unit, order),
        16,
        order,
    )
    one_minus_z = series_add(one, series_scale(z_series, -1, order), order)

    theta_exponent = c / 2 - 4 * (d1 + d2 + d5 + d6) - 2 * (d3 + d4)
    prefactor = dict(one)
    for unit, exponent in (
        (z_unit, h1 - c / 24),
        (y1, a2),
        (y2, a3 - a2),
        (mobile1, d3 / 2),
        (mobile2, d4 / 2),
        (theta3, -theta_exponent),
        (one_minus_z, -c / 24 + d2 + d5),
    ):
        prefactor = series_mul(prefactor, unit_power(unit, exponent, order), order)
    prefactor = series_mul(prefactor, chi_inverse, order)

    descendant: Series3 = {}
    for (n1, n2, n3), coefficient in plane_coefficients.items():
        term = monomial(
            (n1, n2, n3),
            coefficient * sp.Integer(4) ** (n1 + n3),
            order,
        )
        for unit, level in ((x1, n1), (x2, n2), (x3, n3)):
            term = series_mul(term, unit_power(unit, sp.Integer(level), order), order)
        descendant = series_add(descendant, term, order)
    result = series_mul(prefactor, descendant, order)
    return {key: sp.cancel(value) for key, value in sorted(result.items())}


def recursion_coefficients(
    order: int,
    *,
    data: DegenerateData,
    internal_weights: tuple[sp.Expr, sp.Expr, sp.Expr],
    external_weights: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr],
) -> Series3:
    """Generate coefficients of the proposed three-edge pillow recursion."""

    h1, h2, h3 = internal_weights
    d1, d2, d3, d4, d5, d6 = external_weights
    initial_a2 = h2 - h1
    initial_a3 = h3 - h1

    @functools.lru_cache(maxsize=None)
    def coefficient(
        n1: int,
        n2: int,
        n3: int,
        current_h: sp.Expr,
        current_a2: sp.Expr,
        current_a3: sp.Expr,
    ) -> sp.Expr:
        total = sp.S.One if (n1, n2, n3) == (0, 0, 0) else sp.S.Zero
        levels = (n1, n2, n3)
        for edge, available in enumerate(levels):
            for r in range(1, available + 1):
                for s in range(1, available // r + 1):
                    level = r * s
                    pole = data.h(r, s)
                    if edge == 0:
                        denominator = current_h - pole
                        left = data.fusion(r, s, d1, d2)
                        right = data.fusion(r, s, pole + current_a2, d3)
                        shifted = (
                            pole + level,
                            current_a2 - level,
                            current_a3 - level,
                        )
                        plumbing_factor = sp.Integer(4) ** level
                    elif edge == 1:
                        denominator = current_h + current_a2 - pole
                        left = data.fusion(r, s, pole - current_a2, d3)
                        right = data.fusion(
                            r, s, pole + current_a3 - current_a2, d4
                        )
                        shifted = (
                            pole - current_a2,
                            current_a2 + level,
                            current_a3,
                        )
                        plumbing_factor = sp.S.One
                    else:
                        denominator = current_h + current_a3 - pole
                        left = data.fusion(r, s, pole + current_a2 - current_a3, d4)
                        right = data.fusion(r, s, d6, d5)
                        shifted = (
                            pole - current_a3,
                            current_a2,
                            current_a3 + level,
                        )
                        plumbing_factor = sp.Integer(4) ** level
                    remainder = list(levels)
                    remainder[edge] -= level
                    total += (
                        plumbing_factor
                        * data.a_factor(r, s)
                        * left
                        * right
                        / denominator
                        * coefficient(*remainder, *shifted)
                    )
        return sp.cancel(total)

    return {
        (n1, n2, n3): coefficient(
            n1, n2, n3, h1, initial_a2, initial_a3
        )
        for n1 in range(order + 1)
        for n2 in range(order + 1 - n1)
        for n3 in range(order + 1 - n1 - n2)
    }


def compare_coefficients(
    direct: Series3,
    proposed: Series3,
    *,
    substitutions: Mapping[sp.Expr, sp.Expr] | None = None,
) -> list[Index3]:
    def rational_functions_equal(left: sp.Expr, right: sp.Expr) -> bool:
        """Test an exact rational identity using sparse polynomial arithmetic."""

        left = sp.expand_func(left)
        right = sp.expand_func(right)
        left_num, left_den = sp.fraction(sp.together(left))
        right_num, right_den = sp.fraction(sp.together(right))
        generators = sorted(
            left.free_symbols | right.free_symbols,
            key=lambda symbol: symbol.name,
        )
        if not generators:
            return sp.cancel(left - right) == 0
        left_num_poly = sp.Poly(left_num, *generators, domain=sp.QQ)
        left_den_poly = sp.Poly(left_den, *generators, domain=sp.QQ)
        right_num_poly = sp.Poly(right_num, *generators, domain=sp.QQ)
        right_den_poly = sp.Poly(right_den, *generators, domain=sp.QQ)
        return left_num_poly * right_den_poly == right_num_poly * left_den_poly

    failures: list[Index3] = []
    substitutions = dict(substitutions or {})
    keys = sorted(set(direct) | set(proposed), key=lambda key: (sum(key), key))
    for key in keys:
        direct_value = direct.get(key, sp.S.Zero).subs(substitutions)
        proposed_value = proposed.get(key, sp.S.Zero)
        equal = rational_functions_equal(direct_value, proposed_value)
        status = "PASS" if equal else "FAIL"
        print(f"  {key}: {status}", flush=True)
        if not equal:
            failures.append(key)
            print(
                "    exact rational functions differ; "
                f"direct={direct_value}, proposed={proposed_value}"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=3)
    args = parser.parse_args()
    order = int(args.order)
    if order < 0:
        raise ValueError("order must be non-negative")

    b = sp.Symbol("b", nonzero=True)
    c_symbol = sp.Symbol("c")
    h = sp.Symbol("H")
    a2, a3 = sp.symbols("a2 a3")
    external = sp.symbols("d1 d2 d3 d4 d5 d6")
    internal = (h, h + a2, h + a3)
    data = DegenerateData(b)

    print(f"symbolic sphere six-point pillow check, total order <= {order}")
    print("variables: b,H,a2,a3,d1,d2,d3,d4,d5,d6")
    start = time.perf_counter()
    plane = direct_pbw_coefficients(
        order,
        c=c_symbol,
        internal_weights=internal,
        external_weights=external,
    )
    print(f"direct PBW coefficients: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    direct = direct_reduced_pillow_coefficients(
        order,
        plane_coefficients=plane,
        c=c_symbol,
        internal_weights=internal,
        external_weights=external,
    )
    print(f"exact pillow transformation: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    proposed = recursion_coefficients(
        order,
        data=data,
        internal_weights=internal,
        external_weights=external,
    )
    print(f"three-edge recursion: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    failures = compare_coefficients(
        direct,
        proposed,
        substitutions={c_symbol: data.c},
    )
    print(f"exact comparisons: {time.perf_counter() - start:.2f}s")
    if failures:
        raise AssertionError(f"symbolic six-point pillow recursion failed at {failures}")
    print(f"all {len(direct)} coefficients agree exactly")


if __name__ == "__main__":
    main()
