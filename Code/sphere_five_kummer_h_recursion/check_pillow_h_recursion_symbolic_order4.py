#!/usr/bin/env python3
"""Symbolic PBW check of the proposed sphere five-point pillow h-recursion.

The direct block is computed in the plane comb frame

    (0, z, t, 1, infinity),

with plumbing variables x=z/t and y=t.  It is then transformed to the
two pillow-segment variables p1,p2 using the exact branched covering map,
including the E.103 conformal factor and the pillow character.  The result
is compared coefficient by coefficient with the proposed fixed-difference
h-recursion for H_5.

The default truncation is total bidegree n1+n2 <= 3.  All PBW contractions,
Gram-matrix inversions, coordinate changes, and comparisons use exact SymPy
expressions.  The aligned local-coordinate convention is

    p1 = -exp(i w5),  p2 = -exp(i(pi tau-w5)),  p1*p2=q,

for which z/t=4*p1+... and t=4*p2+....  The two minus signs are the harmless
half-period convention needed to make the proposed (4*p_i) residues have
positive leading local coordinates.
"""

from __future__ import annotations

import argparse
import functools
import math
import time
from collections.abc import Mapping

import sympy as sp


Descendant = tuple[int, ...]
State = dict[Descendant, sp.Expr]
Series1 = dict[int, sp.Expr]
Series2 = dict[tuple[int, int], sp.Expr]


def integer_partitions(
    total: int,
    *,
    max_part: int | None = None,
) -> tuple[Descendant, ...]:
    """Descending integer partitions of ``total``."""

    if total < 0:
        return ()
    if total == 0:
        return ((),)
    if max_part is None:
        max_part = total
    out: list[Descendant] = []
    for part in range(min(max_part, total), 0, -1):
        for tail in integer_partitions(total - part, max_part=part):
            out.append((part,) + tail)
    return tuple(out)


def _state_add(*terms: tuple[sp.Expr, Mapping[Descendant, sp.Expr]]) -> State:
    out: State = {}
    for scale, state in terms:
        for desc, coefficient in state.items():
            out[desc] = out.get(desc, sp.S.Zero) + scale * coefficient
    return {desc: sp.expand(value) for desc, value in out.items() if value != 0}


@functools.lru_cache(maxsize=None)
def normal_order_negative_word(word: Descendant) -> tuple[tuple[Descendant, sp.Expr], ...]:
    """Put negative modes into descending PBW order."""

    for index in range(len(word) - 1):
        left = word[index]
        right = word[index + 1]
        if left < right:
            swapped = word[:index] + (right, left) + word[index + 2 :]
            commutator = word[:index] + (left + right,) + word[index + 2 :]
            result = _state_add(
                (sp.S.One, dict(normal_order_negative_word(swapped))),
                (sp.Integer(right - left), dict(normal_order_negative_word(commutator))),
            )
            return tuple(sorted(result.items()))
    return ((tuple(word), sp.S.One),)


def prepend_negative_mode(mode: int, state: State) -> State:
    out: State = {}
    for desc, coefficient in state.items():
        ordered = dict(normal_order_negative_word((int(mode),) + desc))
        out = _state_add((sp.S.One, out), (coefficient, ordered))
    return out


@functools.lru_cache(maxsize=None)
def act_mode_on_descendant(
    mode: int,
    desc: Descendant,
    h: sp.Expr,
    c: sp.Expr,
) -> tuple[tuple[Descendant, sp.Expr], ...]:
    """Act with L_mode on L_-desc |h> in a generic Verma module."""

    mode = int(mode)
    desc = tuple(desc)
    if mode < 0:
        return normal_order_negative_word((-mode,) + desc)
    if mode == 0:
        return ((desc, h + sum(desc)),)
    if not desc:
        return ()

    first = desc[0]
    rest = desc[1:]
    moved = prepend_negative_mode(
        first,
        dict(act_mode_on_descendant(mode, rest, h, c)),
    )
    pieces: list[tuple[sp.Expr, State]] = [(sp.S.One, moved)]

    commutator_mode = mode - first
    commutator_scale = mode + first
    if commutator_mode < 0:
        commutator_state = dict(
            normal_order_negative_word((-commutator_mode,) + rest)
        )
    elif commutator_mode == 0:
        commutator_state = {rest: h + sum(rest)}
    else:
        commutator_state = dict(
            act_mode_on_descendant(commutator_mode, rest, h, c)
        )
    pieces.append((sp.Integer(commutator_scale), commutator_state))

    if mode == first:
        central = c * sp.Rational(mode * (mode * mode - 1), 12)
        pieces.append((central, {rest: sp.S.One}))
    return tuple(sorted(_state_add(*pieces).items()))


def act_mode(mode: int, state: State, *, h: sp.Expr, c: sp.Expr) -> State:
    out: State = {}
    for desc, coefficient in state.items():
        out = _state_add(
            (sp.S.One, out),
            (coefficient, dict(act_mode_on_descendant(mode, desc, h, c))),
        )
    return out


@functools.lru_cache(maxsize=None)
def gram_matrix(level: int, h: sp.Expr, c: sp.Expr) -> tuple[tuple[Descendant, ...], sp.Matrix]:
    basis = integer_partitions(level)
    entries: list[list[sp.Expr]] = []
    for bra in basis:
        row: list[sp.Expr] = []
        for ket in basis:
            state: State = {ket: sp.S.One}
            for mode in bra:
                state = act_mode(mode, state, h=h, c=c)
            row.append(sp.expand(state.get((), sp.S.Zero)))
        entries.append(row)
    return basis, sp.Matrix(entries)


@functools.lru_cache(maxsize=None)
def inverse_gram_matrix(
    level: int,
    h: sp.Expr,
    c: sp.Expr,
) -> tuple[tuple[Descendant, ...], sp.Matrix]:
    basis, matrix = gram_matrix(level, h, c)
    return basis, matrix.inv(method="DM")


def _series1_add(*terms: tuple[sp.Expr, Mapping[int, sp.Expr]]) -> Series1:
    out: Series1 = {}
    for scale, series in terms:
        for power, coefficient in series.items():
            out[power] = out.get(power, sp.S.Zero) + scale * coefficient
    return {power: sp.expand(value) for power, value in out.items() if value != 0}


def _primary_differential(
    series: Series1,
    *,
    mode: int,
    middle_weight: sp.Expr,
    base_exponent: sp.Expr,
) -> Series1:
    out: Series1 = {}
    for shift, coefficient in series.items():
        factor = base_exponent + shift + (mode + 1) * middle_weight
        out[shift + mode] = out.get(shift + mode, sp.S.Zero) + coefficient * factor
    return {power: sp.expand(value) for power, value in out.items() if value != 0}


@functools.lru_cache(maxsize=None)
def two_leg_series(
    desc_infinity: Descendant,
    desc_zero: Descendant,
    h_infinity: sp.Expr,
    h_one: sp.Expr,
    h_zero: sp.Expr,
    c: sp.Expr,
) -> tuple[tuple[int, sp.Expr], ...]:
    """Ward-recursive three-point tensor with a primary in the one-slot."""

    exponent = h_infinity - h_one - h_zero
    if desc_infinity:
        mode = desc_infinity[0]
        rest = desc_infinity[1:]
        commutator = _primary_differential(
            dict(two_leg_series(rest, desc_zero, h_infinity, h_one, h_zero, c)),
            mode=mode,
            middle_weight=h_one,
            base_exponent=exponent,
        )
        acted_zero = act_mode(mode, {desc_zero: sp.S.One}, h=h_zero, c=c)
        terms: list[tuple[sp.Expr, Mapping[int, sp.Expr]]] = [
            (sp.S.One, commutator)
        ]
        for resulting_desc, coefficient in acted_zero.items():
            terms.append(
                (
                    coefficient,
                    dict(
                        two_leg_series(
                            rest,
                            resulting_desc,
                            h_infinity,
                            h_one,
                            h_zero,
                            c,
                        )
                    ),
                )
            )
        return tuple(sorted(_series1_add(*terms).items()))

    if desc_zero:
        mode = desc_zero[0]
        rest = desc_zero[1:]
        result = _series1_add(
            (
                -sp.S.One,
                _primary_differential(
                    dict(two_leg_series((), rest, h_infinity, h_one, h_zero, c)),
                    mode=-mode,
                    middle_weight=h_one,
                    base_exponent=exponent,
                ),
            )
        )
        return tuple(sorted(result.items()))
    return ((0, sp.S.One),)


def rho(
    desc_infinity: Descendant,
    desc_zero: Descendant,
    h_infinity: sp.Expr,
    h_one: sp.Expr,
    h_zero: sp.Expr,
    c: sp.Expr,
) -> sp.Expr:
    return sp.expand(
        sum(
            coefficient
            for _, coefficient in two_leg_series(
                desc_infinity,
                desc_zero,
                h_infinity,
                h_one,
                h_zero,
                c,
            )
        )
    )


def direct_pbw_coefficients(
    order: int,
    *,
    c: sp.Expr,
    h1: sp.Expr,
    h2: sp.Expr,
    weights: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr],
) -> Series2:
    """Plane descendant coefficients in x=z/t and y=t."""

    d1, d2, d3, d4, d5 = weights
    gram1 = {
        level: inverse_gram_matrix(level, h1, c)
        for level in range(order + 1)
    }
    gram2 = {
        level: inverse_gram_matrix(level, h2, c)
        for level in range(order + 1)
    }
    out: Series2 = {}
    for n1 in range(order + 1):
        basis1, inverse1 = gram1[n1]
        left = sp.Matrix(
            [rho(desc, (), h1, d2, d1, c) for desc in basis1]
        )
        propagated_left = inverse1 * left
        for n2 in range(order + 1 - n1):
            basis2, inverse2 = gram2[n2]
            middle = sp.Matrix(
                [
                    [rho(desc2, desc1, h2, d3, h1, c) for desc1 in basis1]
                    for desc2 in basis2
                ]
            )
            right = sp.Matrix(
                [rho((), desc, d5, d4, h2, c) for desc in basis2]
            )
            value = (right.T * inverse2 * middle * propagated_left)[0]
            out[(n1, n2)] = sp.cancel(value)
    return out


def _series2_clean(series: Mapping[tuple[int, int], sp.Expr], order: int) -> Series2:
    return {
        (i, j): value
        for (i, j), value in series.items()
        if i >= 0 and j >= 0 and i + j <= order and value != 0
    }


def series2_add(left: Series2, right: Series2, order: int) -> Series2:
    out = dict(left)
    for key, value in right.items():
        out[key] = out.get(key, sp.S.Zero) + value
    return _series2_clean(out, order)


def series2_scale(series: Series2, scale: sp.Expr, order: int) -> Series2:
    return _series2_clean({key: scale * value for key, value in series.items()}, order)


def series2_mul(left: Series2, right: Series2, order: int) -> Series2:
    out: Series2 = {}
    for (i1, j1), value1 in left.items():
        for (i2, j2), value2 in right.items():
            key = (i1 + i2, j1 + j2)
            if sum(key) <= order:
                out[key] = out.get(key, sp.S.Zero) + value1 * value2
    return _series2_clean(out, order)


def series2_monomial(i: int, j: int, coefficient: sp.Expr, order: int) -> Series2:
    return {} if i + j > order else {(i, j): coefficient}


def series2_unit_power(series: Series2, exponent: sp.Expr, order: int) -> Series2:
    constant = series.get((0, 0), sp.S.Zero)
    if constant != 1:
        raise ValueError(f"unit series must have constant one, found {constant!r}")
    remainder = dict(series)
    remainder[(0, 0)] = remainder.get((0, 0), sp.S.Zero) - 1
    remainder = _series2_clean(remainder, order)
    out: Series2 = {(0, 0): sp.S.One}
    power: Series2 = {(0, 0): sp.S.One}
    for k in range(1, order + 1):
        power = series2_mul(power, remainder, order)
        if not power:
            break
        out = series2_add(
            out,
            series2_scale(power, sp.binomial(exponent, k), order),
            order,
        )
    return out


def pillow_map_units(order: int) -> tuple[Series2, Series2, Series2, Series2, Series2]:
    """Return Z=z/(16q), Y=t/(4p2), X=(z/t)/(4p1), theta3, chi^-1."""

    one: Series2 = {(0, 0): sp.S.One}
    p1 = series2_add(one, {(1, 0): sp.S.One}, order)
    p2 = series2_add(one, {(0, 1): sp.S.One}, order)
    q = {(1, 1): sp.S.One}
    q2 = {(2, 2): sp.S.One}

    z_unit = series2_add(one, series2_scale(q, -8, order), order)
    z_unit = series2_add(z_unit, series2_scale(q2, 44, order), order)

    # Product formula for t=z sn^2(Kw/pi|z), with exp(iw)=-p1.
    y_unit = series2_unit_power(p1, sp.Integer(2), order)
    y_unit = series2_mul(
        y_unit,
        series2_unit_power(series2_add(one, q2, order), sp.Integer(4), order),
        order,
    )
    y_unit = series2_mul(
        y_unit,
        series2_unit_power(series2_add(one, q, order), sp.Integer(-4), order),
        order,
    )
    y_unit = series2_mul(
        y_unit,
        series2_unit_power(
            series2_add(one, {(1, 2): sp.S.One}, order),
            sp.Integer(2),
            order,
        ),
        order,
    )
    y_unit = series2_mul(
        y_unit,
        series2_unit_power(
            series2_add(one, {(2, 1): sp.S.One}, order),
            sp.Integer(-2),
            order,
        ),
        order,
    )
    y_unit = series2_mul(
        y_unit,
        series2_unit_power(p2, sp.Integer(-2), order),
        order,
    )

    x_unit = series2_mul(
        z_unit,
        series2_unit_power(y_unit, sp.Integer(-1), order),
        order,
    )
    theta3 = series2_add(one, series2_scale(q, 2, order), order)
    chi_inverse = series2_add(one, series2_scale(q2, sp.Rational(-1, 2), order), order)
    return z_unit, y_unit, x_unit, theta3, chi_inverse


def direct_reduced_pillow_coefficients(
    order: int,
    *,
    plane_coefficients: Series2,
    c: sp.Expr,
    h1: sp.Expr,
    h2: sp.Expr,
    weights: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr],
) -> Series2:
    """Transform the direct plane PBW block to the proposed H_5."""

    d1, d2, d3, d4, d5 = weights
    a = h2 - h1
    z_unit, y_unit, x_unit, theta3, chi_inverse = pillow_map_units(order)
    one: Series2 = {(0, 0): sp.S.One}

    t_series = series2_scale(
        series2_mul({(0, 1): sp.S.One}, y_unit, order),
        4,
        order,
    )
    x_series = series2_scale(
        series2_mul({(1, 0): sp.S.One}, x_unit, order),
        4,
        order,
    )
    one_minus_t = series2_add(one, series2_scale(t_series, -1, order), order)
    one_minus_x = series2_add(one, series2_scale(x_series, -1, order), order)
    mobile_unit = series2_mul(one_minus_t, one_minus_x, order)

    q = {(1, 1): sp.S.One}
    z_series = series2_scale(series2_mul(q, z_unit, order), 16, order)
    one_minus_z = series2_add(one, series2_scale(z_series, -1, order), order)

    theta_exponent = c / 2 - 4 * (d1 + d2 + d4 + d5) - 2 * d3
    prefactor: Series2 = {(0, 0): sp.S.One}
    for unit, exponent in (
        (z_unit, h1 - c / 24),
        (y_unit, a),
        (mobile_unit, d3 / 2),
        (theta3, -theta_exponent),
        # The standard four-point pillow factor is
        # (1-z)^(c/24-h_at_z-h_at_1).  In the present five-point ordering
        # these are d2 and d4.  Using the literal h3+h4 printed in E.103 of
        # the string note leaves a spurious 16*(d2-d5) p1*p2 term.
        (one_minus_z, -c / 24 + d2 + d4),
    ):
        prefactor = series2_mul(
            prefactor,
            series2_unit_power(unit, exponent, order),
            order,
        )
    prefactor = series2_mul(prefactor, chi_inverse, order)

    descendant: Series2 = {}
    for (n1, n2), coefficient in plane_coefficients.items():
        term = series2_mul(
            series2_monomial(n1, n2, coefficient * 4 ** (n1 + n2), order),
            series2_unit_power(x_unit, sp.Integer(n1), order),
            order,
        )
        term = series2_mul(
            term,
            series2_unit_power(y_unit, sp.Integer(n2), order),
            order,
        )
        descendant = series2_add(descendant, term, order)
    result = series2_mul(prefactor, descendant, order)
    return {
        key: sp.cancel(value)
        for key, value in sorted(result.items())
    }


class DegenerateData:
    """Virasoro Kac data in the c=1+6(b+b^-1)^2 convention."""

    def __init__(self, b: sp.Symbol):
        self.b = b
        self.q_background = b + 1 / b
        self.c = 1 + 6 * self.q_background**2
        self._fusion_cache: dict[tuple[int, int, sp.Expr, sp.Expr], sp.Expr] = {}

    def h(self, r: int, s: int) -> sp.Expr:
        return sp.cancel(
            (self.q_background**2 - (r * self.b + s / self.b) ** 2) / 4
        )

    def a_factor(self, r: int, s: int) -> sp.Expr:
        value = sp.Rational(1, 2)
        for m in range(1 - r, r + 1):
            for ell in range(1 - s, s + 1):
                if (m, ell) in {(0, 0), (r, s)}:
                    continue
                value /= m * self.b + ell / self.b
        return sp.cancel(value)

    @functools.lru_cache(maxsize=None)
    def _fusion_template(self, r: int, s: int) -> sp.Expr:
        lambda_top, lambda_bottom = sp.symbols("lambda_top lambda_bottom")
        expression = sp.S.One
        for p in range(1 - r, r, 2):
            for ell in range(1 - s, s, 2):
                shift = p * self.b + ell / self.b
                expression *= (lambda_top + lambda_bottom + shift) / 2
                expression *= (lambda_top - lambda_bottom + shift) / 2
        polynomial = sp.Poly(sp.expand(expression), lambda_top, lambda_bottom)
        top_square, bottom_square = sp.symbols("top_square bottom_square")
        reduced = sp.S.Zero
        for (top_degree, bottom_degree), coefficient in polynomial.terms():
            if top_degree % 2 or bottom_degree % 2:
                raise AssertionError(
                    f"fusion polynomial ({r},{s}) is not even in its momenta"
                )
            reduced += (
                coefficient
                * top_square ** (top_degree // 2)
                * bottom_square ** (bottom_degree // 2)
            )
        return sp.factor(reduced)

    def fusion(self, r: int, s: int, top: sp.Expr, bottom: sp.Expr) -> sp.Expr:
        key = (r, s, top, bottom)
        if key not in self._fusion_cache:
            top_square, bottom_square = sp.symbols("top_square bottom_square")
            value = self._fusion_template(r, s).subs(
                {
                    top_square: self.q_background**2 - 4 * top,
                    bottom_square: self.q_background**2 - 4 * bottom,
                }
            )
            self._fusion_cache[key] = sp.cancel(value)
        return self._fusion_cache[key]


def recursion_coefficients(
    order: int,
    *,
    data: DegenerateData,
    h1: sp.Expr,
    h2: sp.Expr,
    weights: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr],
) -> Series2:
    """Coefficients generated by the proposed fixed-difference recursion."""

    d1, d2, d3, d4, d5 = weights
    initial_a = h2 - h1

    @functools.lru_cache(maxsize=None)
    def coefficient(n1: int, n2: int, current_h: sp.Expr, current_a: sp.Expr) -> sp.Expr:
        total = sp.S.One if (n1, n2) == (0, 0) else sp.S.Zero
        for r in range(1, n1 + 1):
            for s in range(1, n1 // r + 1):
                level = r * s
                pole = data.h(r, s)
                residue = (
                    4**level
                    * data.a_factor(r, s)
                    * data.fusion(r, s, d1, d2)
                    * data.fusion(r, s, pole + current_a, d3)
                    / (current_h - pole)
                )
                total += residue * coefficient(
                    n1 - level,
                    n2,
                    pole + level,
                    current_a - level,
                )
        for r in range(1, n2 + 1):
            for s in range(1, n2 // r + 1):
                level = r * s
                pole = data.h(r, s)
                residue = (
                    4**level
                    * data.a_factor(r, s)
                    * data.fusion(r, s, d5, d4)
                    * data.fusion(r, s, pole - current_a, d3)
                    / (current_h + current_a - pole)
                )
                total += residue * coefficient(
                    n1,
                    n2 - level,
                    pole - current_a,
                    current_a + level,
                )
        return sp.cancel(total)

    return {
        (n1, n2): coefficient(n1, n2, h1, initial_a)
        for n1 in range(order + 1)
        for n2 in range(order + 1 - n1)
    }


def compare_coefficients(
    direct: Series2,
    proposed: Series2,
    *,
    direct_substitutions: Mapping[sp.Expr, sp.Expr] | None = None,
) -> list[tuple[int, int]]:
    failures: list[tuple[int, int]] = []
    substitutions = dict(direct_substitutions or {})
    for key in sorted(set(direct) | set(proposed), key=lambda item: (sum(item), item)):
        direct_value = direct.get(key, sp.S.Zero).subs(substitutions)
        difference = sp.cancel(
            sp.expand_func(direct_value - proposed.get(key, sp.S.Zero))
        )
        if difference != 0:
            difference = sp.factor(difference)
        status = "PASS" if difference == 0 else "FAIL"
        print(f"  {key}: {status}")
        if difference != 0:
            failures.append(key)
            print(f"    difference={difference}")
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
    a = sp.Symbol("a")
    d1, d2, d3, d4, d5 = sp.symbols("d1 d2 d3 d4 d5")
    weights = (d1, d2, d3, d4, d5)
    data = DegenerateData(b)

    print(f"symbolic sphere five-point pillow check, total order <= {order}")
    print("variables: b,H,a,d1,d2,d3,d4,d5")
    start = time.perf_counter()
    plane = direct_pbw_coefficients(
        order,
        # Keeping c independent makes the level-four PBW Gram inversion much
        # smaller.  The exact c=1+6(b+b^-1)^2 substitution is made only in
        # the final coefficient comparison.
        c=c_symbol,
        h1=h,
        h2=h + a,
        weights=weights,
    )
    print(f"direct PBW coefficients: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    direct = direct_reduced_pillow_coefficients(
        order,
        plane_coefficients=plane,
        c=c_symbol,
        h1=h,
        h2=h + a,
        weights=weights,
    )
    print(f"pillow change of variables: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    proposed = recursion_coefficients(
        order,
        data=data,
        h1=h,
        h2=h + a,
        weights=weights,
    )
    print(f"proposed recursion coefficients: {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    failures = compare_coefficients(
        direct,
        proposed,
        direct_substitutions={c_symbol: data.c},
    )
    print(f"exact comparisons: {time.perf_counter() - start:.2f}s")
    if failures:
        raise AssertionError(f"symbolic pillow recursion failed at {failures}")
    print(f"all {len(direct)} coefficients agree exactly")


if __name__ == "__main__":
    main()
