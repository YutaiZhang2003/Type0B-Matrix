#!/usr/bin/env python3
"""Analytic level-four PBW check of the four-point pillow formula E.103.

The external primaries (h1,h2,h3,h4) are placed at (0,z,1,infinity).
The script computes the plane block directly from PBW Verma Gram matrices,
re-expands it in the elliptic nome, and strips the two candidate E.103
prefactors:

  correct: (1-z)^(c/24-h2-h3),
  printed: (1-z)^(c/24-h3-h4).

The correctly stripped block is compared exactly, coefficient by
coefficient through q^4, with the independent Zamolodchikov h-recursion.
The comparison is analytic in h,h1,h2,h3,h4 on two exact rational-b slices;
the prefactor ratio itself is proved at generic c.  The printed candidate
differs from the correct result by the exact factor (1-z)^(h4-h2), so its
first obstruction is 16*(h2-h4)*q.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Mapping

import sympy as sp

from check_pillow_h_recursion_symbolic_order4 import (
    DegenerateData,
    gram_matrix,
    rho,
)


Series = dict[int, sp.Expr]


def clean(series: Mapping[int, sp.Expr], order: int) -> Series:
    return {
        power: value
        for power, value in series.items()
        if 0 <= power <= order and value != 0
    }


def add(left: Series, right: Series, order: int) -> Series:
    out = dict(left)
    for power, value in right.items():
        out[power] = out.get(power, sp.S.Zero) + value
    return clean(out, order)


def scale(series: Series, coefficient: sp.Expr, order: int) -> Series:
    return clean({power: coefficient * value for power, value in series.items()}, order)


def multiply(left: Series, right: Series, order: int) -> Series:
    out: Series = {}
    for power1, value1 in left.items():
        for power2, value2 in right.items():
            power = power1 + power2
            if power <= order:
                out[power] = out.get(power, sp.S.Zero) + value1 * value2
    return clean(out, order)


def unit_power(series: Series, exponent: sp.Expr, order: int) -> Series:
    if series.get(0, sp.S.Zero) != 1:
        raise ValueError("unit series must have constant coefficient one")
    remainder = dict(series)
    remainder[0] = remainder.get(0, sp.S.Zero) - 1
    remainder = clean(remainder, order)
    out: Series = {0: sp.S.One}
    power: Series = {0: sp.S.One}
    for integer in range(1, order + 1):
        power = multiply(power, remainder, order)
        if not power:
            break
        out = add(out, scale(power, sp.binomial(exponent, integer), order), order)
    return out


def direct_plane_pbw_coefficients(
    order: int,
    *,
    c: sp.Expr,
    internal: sp.Expr,
    external: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr],
) -> Series:
    """Return F_n in z^(h-h1-h2) sum_n F_n z^n."""

    h1, h2, h3, h4 = external
    out: Series = {}
    for level in range(order + 1):
        basis, gram = gram_matrix(level, internal, c)
        inverse_numerator_dm, inverse_denominator_domain = gram.to_DM().inv_den()
        inverse_numerator = inverse_numerator_dm.to_Matrix()
        inverse_denominator = inverse_numerator_dm.domain.to_sympy(
            inverse_denominator_domain
        )
        upper = sp.Matrix(
            [rho((), desc, h4, h3, internal, c) for desc in basis]
        )
        lower = sp.Matrix(
            [rho(desc, (), internal, h2, h1, c) for desc in basis]
        )
        numerator = sp.expand((upper.T * inverse_numerator * lower)[0])
        out[level] = sp.cancel(numerator / inverse_denominator)
    return out


def compose_plane_series(plane: Series, z: Series, order: int) -> Series:
    out: Series = {}
    power: Series = {0: sp.S.One}
    for level in range(order + 1):
        if level > 0:
            power = multiply(power, z, order)
        out = add(out, scale(power, plane.get(level, sp.S.Zero), order), order)
    return out


def direct_elliptic_h(
    order: int,
    *,
    plane: Series,
    c: sp.Expr,
    internal: sp.Expr,
    external: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr],
) -> tuple[Series, Series, Series]:
    """Return correctly stripped H, literally-E.103-stripped H, and z(q)."""

    h1, h2, h3, h4 = external
    # lambda(q) through q^5; the extra term is needed for lambda/(16q)
    # through q^4.
    z_full = {
        1: sp.Integer(16),
        2: sp.Integer(-128),
        3: sp.Integer(704),
        4: sp.Integer(-3072),
        5: sp.Integer(11488),
    }
    z = clean(z_full, order)
    z_over_16q = {
        power - 1: coefficient / 16
        for power, coefficient in z_full.items()
        if power - 1 <= order
    }
    one_minus_z = add({0: sp.S.One}, scale(z, -1, order), order)
    theta3 = {0: sp.S.One, 1: sp.Integer(2), 4: sp.Integer(2)}
    descendant = compose_plane_series(plane, z, order)
    delta = (c - 1) / 24

    correct: Series = {0: sp.S.One}
    for unit, exponent in (
        (z_over_16q, internal - delta),
        (one_minus_z, h2 + h3 - delta),
        (theta3, -(c - 1) / 2 + 4 * sum(external)),
    ):
        correct = multiply(correct, unit_power(unit, exponent, order), order)
    correct = multiply(correct, descendant, order)

    printed = multiply(
        correct,
        unit_power(one_minus_z, h4 - h2, order),
        order,
    )
    return correct, printed, z


def recursion_coefficients(
    order: int,
    *,
    data: DegenerateData,
    internal: sp.Expr,
    external: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr],
) -> Series:
    h1, h2, h3, h4 = external

    @functools.lru_cache(maxsize=None)
    def coefficient(level: int, current_h: sp.Expr) -> sp.Expr:
        total = sp.S.One if level == 0 else sp.S.Zero
        for r in range(1, level + 1):
            for s in range(1, level // r + 1):
                null_level = r * s
                pole = data.h(r, s)
                residue = (
                    16**null_level
                    * data.a_factor(r, s)
                    * data.fusion(r, s, h1, h2)
                    * data.fusion(r, s, h4, h3)
                    / (current_h - pole)
                )
                total += residue * coefficient(
                    level - null_level,
                    pole + null_level,
                )
        return sp.cancel(total)

    return {level: coefficient(level, internal) for level in range(order + 1)}


def normalized_difference(left: sp.Expr, right: sp.Expr) -> sp.Expr:
    return sp.cancel(sp.expand_func(left - right))


def main() -> None:
    order = 4
    internal = sp.Symbol("h")
    external = sp.symbols("h1 h2 h3 h4")
    h1, h2, h3, h4 = external

    print("four-point E.103 analytic PBW check")
    print("insertions: (h1,h2,h3,h4) at (0,z,1,infinity)")

    # Use two exact nonresonant c-slices while retaining symbolic
    # h,h1,h2,h3,h4.  This avoids a prohibitively large generic-b gcd at
    # q^4 without turning any conformal weight into a numerical value.
    first_correct: Series | None = None
    first_printed: Series | None = None
    for b_value in (sp.Integer(2), sp.Rational(3, 2)):
        data = DegenerateData(b_value)
        print(f"exact slice b={b_value}, c={sp.factor(data.c)}")
        start = time.perf_counter()
        plane = direct_plane_pbw_coefficients(
            order,
            c=data.c,
            internal=internal,
            external=external,
        )
        print(f"  direct PBW through level {order}: {time.perf_counter() - start:.2f}s")
        expected_level_one = (
            (internal + h2 - h1) * (internal + h3 - h4) / (2 * internal)
        )
        if normalized_difference(plane[1], expected_level_one) != 0:
            raise AssertionError("level-one PBW coefficient has the wrong convention")
        print(f"  F_1={sp.factor(plane[1])}")

        start = time.perf_counter()
        correct, printed, _z = direct_elliptic_h(
            order,
            plane=plane,
            c=data.c,
            internal=internal,
            external=external,
        )
        print(f"  analytic q re-expansion: {time.perf_counter() - start:.2f}s")
        if first_correct is None:
            first_correct = correct
            first_printed = printed

        start = time.perf_counter()
        recursive = recursion_coefficients(
            order,
            data=data,
            internal=internal,
            external=external,
        )
        print(f"  independent h-recursion: {time.perf_counter() - start:.2f}s")

        print("  corrected E.103 versus direct PBW and h-recursion")
        for level in range(order + 1):
            difference = normalized_difference(correct[level], recursive[level])
            status = "PASS" if difference == 0 else "FAIL"
            print(f"    q^{level}: {status}")
            if difference != 0:
                print(f"      difference={sp.factor(difference)}")
                raise AssertionError(f"corrected E.103 fails at q^{level}")

    if first_correct is None or first_printed is None:
        raise AssertionError("no exact comparison slice was evaluated")
    obstruction = normalized_difference(first_printed[1], first_correct[1])
    expected_obstruction = 16 * (h2 - h4)
    if normalized_difference(obstruction, expected_obstruction) != 0:
        raise AssertionError("literal E.103 obstruction has the wrong value")
    print("literal printed E.103")
    print(f"  [q](H_printed-H_correct)={sp.factor(obstruction)}")
    print("  expected nonzero obstruction=16*(h2-h4)")
    print("analytic level-four E.103 check passed")


if __name__ == "__main__":
    main()
