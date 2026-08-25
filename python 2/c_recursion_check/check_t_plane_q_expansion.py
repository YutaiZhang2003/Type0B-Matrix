#!/usr/bin/env python3
"""Symbolic certificate for the first fixed-weight t-plane coefficients."""

from itertools import product

import sympy as sp


I = sp.I
h1, h2, h3, t = sp.symbols("h1 h2 h3 t", nonzero=True)
eta, eta_prime, lift1, lift2, lift3 = sp.symbols(
    "eta eta_prime lift1 lift2 lift3"
)

beta2 = (t + (h3 - h2) / t) / 2
beta3 = (t - (h3 - h2) / t) / 2
c = 6 * t**2 + 12 * (h2 + h3) + 6 * (h3 - h2) ** 2 / t**2


def ground_rho(form_parity, structure, second, third):
    if form_parity == 0:
        return {(0, 0): sp.Integer(1), (1, 1): structure}.get(
            (second, third), sp.Integer(0)
        )
    return {(0, 1): sp.Integer(1), (1, 0): I * structure}.get(
        (second, third), sp.Integer(0)
    )


def inverse_ground_metric(parity):
    return sp.Integer(1) if parity == 0 else -I


def g0_coefficient(beta, parity):
    eighth = sp.exp(I * sp.pi / 4)
    return beta * eighth if parity == 0 else I * beta * eighth


def second_g_rho(form_parity, structure, second, third):
    return sp.simplify(
        -sp.Rational(1, 2)
        * g0_coefficient(beta2, second)
        * ground_rho(form_parity, structure, 1 - second, third)
        - I
        * (-1) ** third
        * g0_coefficient(beta3, third)
        * ground_rho(form_parity, structure, second, 1 - third)
    )


def third_g_rho(form_parity, structure, second, third):
    return sp.simplify(
        -I
        * (-1) ** third
        * g0_coefficient(beta2, second)
        * ground_rho(form_parity, structure, 1 - second, third)
        + sp.Rational(1, 2)
        * g0_coefficient(beta3, third)
        * ground_rho(form_parity, structure, second, 1 - third)
    )


def ramond_level_one_inverse(weight, beta, total_parity):
    mixed = -sp.Rational(3, 2) * beta * sp.exp(-I * sp.pi / 4)
    if total_parity == 0:
        gram = sp.Matrix([[2 * weight, mixed], [mixed, I * (c / 4 + 2 * weight)]])
    else:
        gram = sp.Matrix([[2 * I * weight, mixed], [mixed, c / 4 + 2 * weight]])
    return sp.simplify(gram.inv())


def q2_coefficient(form_parity, first_structure, second_structure):
    result = 0
    for total_parity in (0, 1):
        third = (form_parity - total_parity) % 2
        second_l = total_parity
        second_g = 1 - total_parity
        left = sp.Matrix(
            [
                (h1 - h2 - h3)
                * ground_rho(form_parity, first_structure, second_l, third),
                second_g_rho(form_parity, first_structure, second_g, third),
            ]
        )
        right = sp.Matrix(
            [
                (h1 - h2 - h3)
                * ground_rho(form_parity, second_structure, second_l, third),
                second_g_rho(form_parity, second_structure, second_g, third),
            ]
        )
        result += (
            lift2**total_parity
            * lift3**third
            * (-1) ** (total_parity * third)
            * inverse_ground_metric(third)
            * (left.T * ramond_level_one_inverse(h2, beta2, total_parity) * right)[0]
        )
    return sp.factor(sp.cancel(result))


def q3_coefficient(form_parity, first_structure, second_structure):
    result = 0
    for total_parity in (0, 1):
        second = (form_parity - total_parity) % 2
        third_l = total_parity
        third_g = 1 - total_parity
        left = sp.Matrix(
            [
                (h2 + h3 - h1)
                * ground_rho(form_parity, first_structure, second, third_l),
                third_g_rho(form_parity, first_structure, second, third_g),
            ]
        )
        right = sp.Matrix(
            [
                (h2 + h3 - h1)
                * ground_rho(form_parity, second_structure, second, third_l),
                third_g_rho(form_parity, second_structure, second, third_g),
            ]
        )
        result += (
            lift2**second
            * lift3**total_parity
            * (-1) ** (second * total_parity)
            * inverse_ground_metric(second)
            * (left.T * ramond_level_one_inverse(h3, beta3, total_parity) * right)[0]
        )
    return sp.factor(sp.cancel(result))


def ground_coefficient(form_parity, first_structure, second_structure):
    result = 0
    for second, third in product((0, 1), repeat=2):
        if (second + third) % 2 != form_parity:
            continue
        result += (
            lift2**second
            * lift3**third
            * (-1) ** (second * third)
            * inverse_ground_metric(second)
            * inverse_ground_metric(third)
            * ground_rho(form_parity, first_structure, second, third)
            * ground_rho(form_parity, second_structure, second, third)
        )
    return sp.expand(result)


def expected_ramond_finite(edge, first_structure, second_structure):
    difference = h1 - h2 - h3
    if edge == 2:
        weight = h2
        other = h3
    else:
        weight = h3
        other = h2
    if first_structure == second_structure == 1:
        numerator = 4 * difference**2 - 3 * h1 + 3 * other
        return 2 * numerator / (16 * weight + 3)
    if first_structure != second_structure:
        numerator = 4 * difference**2 - h1 + 2 * weight + other
        return 2 * numerator / (16 * weight + 3)
    numerator = 12 * difference**2 + 3 * h1 - 4 * weight - 3 * other
    return 2 * numerator / (3 * (16 * weight + 3))


def check_first_non_global_pole():
    ns_gram = sp.Matrix(
        [
            [c / 2 + 4 * h1, 6 * h1, 5 * h1],
            [6 * h1, 4 * h1 * (2 * h1 + 1), 4 * h1],
            [
                5 * h1,
                4 * h1,
                2 * h1 * (2 * h1 + 1 + 2 * c / 3),
            ],
        ]
    )
    ns_inverse = ns_gram.inv()
    assert sp.simplify(sp.limit(t**2 * ns_inverse[2, 2], t, sp.oo) - 1 / (8 * h1)) == 0
    assert sp.simplify(
        sp.limit(t**2 * ns_inverse[1, 2], t, sp.oo)
        + 1 / (8 * h1 * (2 * h1 + 1))
    ) == 0
    assert sp.limit(t**2 * ns_inverse[0, 2], t, sp.oo) == 0

    leading_components = {
        0: {(0, 0): -sp.Rational(1, 2), (1, 1): sp.Rational(1, 2)},
        1: {(0, 1): -sp.Rational(1, 2), (1, 0): I / 2},
    }
    for form_parity in (0, 1):
        sewn = 0
        for (second, third), value in leading_components[form_parity].items():
            sewn += (
                lift2**second
                * lift3**third
                * (-1) ** (second * third)
                * inverse_ground_metric(second)
                * inverse_ground_metric(third)
                * value**2
            )
        expected = (
            (1 + lift2 * lift3) / (32 * h1)
            if form_parity == 0
            else I * (lift2 - lift3) / (32 * h1)
        )
        assert sp.simplify(sewn / (8 * h1) - expected) == 0


def main():
    for form_parity in (0, 1):
        for first_structure, second_structure in product((1, -1), repeat=2):
            substitutions = {eta: first_structure, eta_prime: second_structure}
            q2 = q2_coefficient(form_parity, first_structure, second_structure)
            q3 = q3_coefficient(form_parity, first_structure, second_structure)
            ground = ground_coefficient(
                form_parity, first_structure, second_structure
            )
            q2_finite = sp.factor(sp.limit(q2, t, sp.oo))
            q3_finite = sp.factor(sp.limit(q3, t, sp.oo))
            assert sp.simplify(
                q2_finite
                - ground
                * expected_ramond_finite(2, first_structure, second_structure)
            ) == 0
            assert sp.simplify(
                q3_finite
                - ground
                * expected_ramond_finite(3, first_structure, second_structure)
            ) == 0
            assert sp.limit(q2 / t**2, t, sp.oo) == 0
            assert sp.limit(q3 / t**2, t, sp.oo) == 0
            assert sp.limit(t**2 * q2, t, 0) == 0
            assert sp.limit(t**2 * q3, t, 0) == 0
            print(
                "case",
                form_parity,
                first_structure,
                second_structure,
                "ground=",
                ground,
            )
            print("q2 finite at infinity =", q2_finite)
            print("q3 finite at infinity =", q3_finite)
            print("q2 infinity pole =", sp.factor(sp.limit(q2 / t**2, t, sp.oo)))
            print("q3 infinity pole =", sp.factor(sp.limit(q3 / t**2, t, sp.oo)))
            print("q2 zero pole =", sp.factor(sp.limit(t**2 * q2, t, 0)))
            print("q3 zero pole =", sp.factor(sp.limit(t**2 * q3, t, 0)))
    check_first_non_global_pole()
    print("all fixed-weight t-plane q-expansion checks passed")


if __name__ == "__main__":
    main()
