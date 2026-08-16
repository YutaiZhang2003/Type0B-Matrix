"""Exact coefficientwise analysis of the fixed-beta R sphere block at c=infinity.

At every fixed local level N the Ward--Gram coefficient F_N(c) is rational
in c.  This script extracts its polynomial part at infinity, performs the
exact multiplicative stripping

    C(c,z) = (1-z)**(c/12) B(c,z),

and verifies through level two that C_N(c) has a constant polynomial part
and a proper rational remainder O(1/c).

There is no branch point in the c-plane: each coefficient is meromorphic.
The resummed raw factor (1-z)**(-c/12) is entire in finite c and has an
essential singularity at c=infinity.  Its logarithmic branch belongs to the
z-plane.
"""

from __future__ import annotations

import sympy as sp

from symbolic_ramond_sphere_level4 import exact_coefficients


def polynomial_part_at_infinity(
    expression: sp.Expr, variable: sp.Symbol
) -> tuple[sp.Expr, sp.Expr]:
    """Return ``(polynomial_part, proper_remainder)`` in ``variable``."""

    numerator, denominator = sp.fraction(sp.cancel(expression))
    quotient, remainder = sp.div(
        sp.Poly(numerator, variable, domain=sp.EX),
        sp.Poly(denominator, variable, domain=sp.EX),
    )
    return sp.factor(quotient.as_expr()), sp.cancel(
        remainder.as_expr() / denominator
    )


def strip_universal_background(
    coefficients: tuple[sp.Expr, ...], c: sp.Symbol
) -> tuple[sp.Expr, ...]:
    """Multiply a local z-series by ``(1-z)**(c/12)`` exactly."""

    def generalized_binomial(value: sp.Expr, order: int) -> sp.Expr:
        return sp.prod(value - offset for offset in range(order)) / sp.factorial(
            order
        )

    stripped: list[sp.Expr] = []
    for level in range(len(coefficients)):
        stripped.append(
            sp.cancel(
                sum(
                    (-1) ** offset
                    * generalized_binomial(c / 12, offset)
                    * coefficients[level - offset]
                    for offset in range(level + 1)
                )
            )
        )
    return tuple(stripped)


def sphere_infinity_analysis() -> dict[str, sp.Expr]:
    """Construct and validate the exact level-one and level-two ledger."""

    symbols, coefficients = exact_coefficients(2)
    c, beta, beta2, beta3, h1, h4, eta2, eta3 = symbols
    central_scale = c / 12
    exponent = h1 + h4 + beta2**2 + beta3**2

    polynomial_1, remainder_1 = polynomial_part_at_infinity(
        coefficients[1], c
    )
    polynomial_2, remainder_2 = polynomial_part_at_infinity(
        coefficients[2], c
    )

    delta_2 = sp.factor(
        beta**4
        - beta**2 * (exponent + sp.Rational(1, 4))
        + beta * (beta2 * eta2 + beta3 * eta3) / 4
        + beta2**2 * (beta3**2 + h4)
        + beta3**2 * h1
        + h1 * h4
        - beta2 * beta3 * eta2 * eta3 / 4
    )

    expected_polynomial_1 = central_scale - exponent
    expected_polynomial_2 = (
        (central_scale - exponent)
        * (central_scale - exponent + 1)
        / 2
        + delta_2
    )
    assert sp.simplify(polynomial_1 - expected_polynomial_1) == 0
    assert sp.simplify(polynomial_2 - expected_polynomial_2) == 0

    # A proper O(1/c) remainder at level one is promoted to an O(1)
    # contribution at level two by the order-c background coefficient.
    promoted_remainder = sp.limit(
        central_scale * remainder_1, c, sp.oo
    )
    assert sp.simplify(promoted_remainder - delta_2) == 0

    stripped = strip_universal_background(coefficients, c)
    stripped_polynomial_1, stripped_remainder_1 = (
        polynomial_part_at_infinity(stripped[1], c)
    )
    stripped_polynomial_2, stripped_remainder_2 = (
        polynomial_part_at_infinity(stripped[2], c)
    )
    assert sp.simplify(stripped_polynomial_1 + exponent) == 0
    assert sp.simplify(
        stripped_polynomial_2 - exponent * (exponent - 1) / 2
    ) == 0
    assert sp.limit(stripped_remainder_1, c, sp.oo) == 0
    assert sp.limit(stripped_remainder_2, c, sp.oo) == 0

    return {
        "A": exponent,
        "Delta2": delta_2,
        "PolInf_F1": polynomial_1,
        "PolInf_F2": polynomial_2,
        "PolInf_C1": stripped_polynomial_1,
        "PolInf_C2": stripped_polynomial_2,
        "Remainder_F1": remainder_1,
        "Remainder_F2": remainder_2,
        "Remainder_C1": stripped_remainder_1,
        "Remainder_C2": stripped_remainder_2,
    }


def main() -> None:
    ledger = sphere_infinity_analysis()
    for name in (
        "A",
        "Delta2",
        "PolInf_F1",
        "PolInf_F2",
        "PolInf_C1",
        "PolInf_C2",
    ):
        print(f"{name} = {sp.sstr(ledger[name])}")
    print("proper remainders after stripping: O(1/c) at levels 1 and 2")
    print("c-plane classification: coefficientwise meromorphic; resummed essential at infinity")


if __name__ == "__main__":
    main()
