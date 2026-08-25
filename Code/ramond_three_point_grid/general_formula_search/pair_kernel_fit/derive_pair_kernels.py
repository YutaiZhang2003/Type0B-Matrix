#!/usr/bin/env python3
"""Exact low-level search for two-channel Ramond pair kernels.

The script reconstructs the dependence on ``Q=b+b**(-1)`` from several
exact rational values of ``b``.  This is only an interpolation device: every
identity is checked at an additional rational value which was not used in
the reconstruction.

No numerical fitting is used.  The output polynomials are the raw masters
after multiplication by the three standard one-leg denominators and removal
of the momentum-independent phase displayed in ``NORMALIZATIONS`` below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
GRID_DIR = HERE.parents[1]
if str(GRID_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_DIR))

import fit_signed_sectors as fit  # noqa: E402


I = sp.I
P1, P2, P3, Q = sp.symbols("P1 P2 P3 Q")
MOMENTA = (P1, P2, P3)

PAIR_12 = (sp.Rational(1, 2), sp.Rational(3, 4), sp.Rational(1, 4))
PAIR_13 = (sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(3, 4))
PAIR_23 = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))
TRIPLE = (sp.Rational(1, 2), sp.Rational(3, 4), sp.Rational(3, 4))

# (labels, epsilon_2, eta) -> phase removed from ``scaled_raw``.
NORMALIZATIONS = {
    (PAIR_12, 0, -1): sp.Pow(2, sp.Rational(3, 4)) * (1 - I),
    (PAIR_13, 1, 1): sp.Pow(2, sp.Rational(1, 4)) * (1 - I),
    (TRIPLE, 0, 1): sp.Pow(2, sp.Rational(3, 4)) * (1 - I),
    (TRIPLE, 1, -1): -2 * sp.Pow(2, sp.Rational(1, 4)) * (1 - I),
}

INTERPOLATION_B = tuple(
    map(sp.Rational, ("3/2", "5/3", "7/5", "4/3", "9/5"))
)
CHECK_B = sp.Rational(8, 5)


def normalized_master(labels, epsilon2, eta, b_value):
    """Return one exact cleared master with its constant phase removed."""

    phase = NORMALIZATIONS[(labels, epsilon2, eta)]
    value = fit.scaled_raw(
        labels,
        (epsilon2, 0, 0, eta),
        (b_value, P1, P2, P3),
    )
    return sp.cancel(value / phase)


def reconstruct_q_polynomial(values, q_degree):
    """Interpolate all momentum coefficients as polynomials in Q."""

    polynomial_values = [
        (b + 1 / b, sp.Poly(value, *MOMENTA)) for b, value in values
    ]
    monomials = sorted(
        set().union(*(set(poly.monoms()) for _, poly in polynomial_values)),
        reverse=True,
    )
    answer = 0
    used = polynomial_values[: q_degree + 1]
    for monomial in monomials:
        coefficient = sp.interpolate(
            [(q, poly.coeff_monomial(monomial)) for q, poly in used], Q
        )
        answer += coefficient * sp.prod(
            momentum**power for momentum, power in zip(MOMENTA, monomial)
        )
    return sp.expand(answer)


def exact_check(reconstructed, direct, b_value):
    q_value = b_value + 1 / b_value
    return sp.factor(sp.cancel(reconstructed.subs(Q, q_value) - direct))


def pair_cubic(labels, epsilon2, eta):
    values = [
        (b, normalized_master(labels, epsilon2, eta, b))
        for b in INTERPOLATION_B[:4]
    ]
    reconstructed = reconstruct_q_polynomial(values, 3)
    direct = normalized_master(labels, epsilon2, eta, CHECK_B)
    residual = exact_check(reconstructed, direct, CHECK_B)
    if residual != 0:
        raise AssertionError(f"pair interpolation residual: {residual}")
    return reconstructed


def universal_ns_r_kernel(u, v, e):
    """The two-channel polynomial extracted from either NS--R pair."""

    row = sp.Matrix([[u, Q]])
    kernel = sp.Matrix([[1 + Q * e, 1], [e**2, -1]])
    column = sp.Matrix([1, (v - Q) ** 2])
    return sp.expand((row * kernel * column)[0])


def derive_pairs():
    c12 = pair_cubic(PAIR_12, 0, -1)
    c13 = pair_cubic(PAIR_13, 1, 1)

    e2 = Q + 2 * P2
    e3 = Q + 2 * P3
    a = Q / 2 + P1 - P2 - P3
    z = Q / 2 + P1 - P2 + P3
    w = Q / 2 + P1 + P2 - P3

    formula12 = universal_ns_r_kernel(a, z, e2)
    formula13 = universal_ns_r_kernel(w, a, e3)
    residual12 = sp.factor(sp.expand(c12 - formula12))
    residual13 = sp.factor(sp.expand(c13 - formula13))
    if residual12 != 0 or residual13 != 0:
        raise AssertionError((residual12, residual13))

    print("C12 =", sp.factor(c12))
    print("C13 =", sp.factor(c13))
    print("pair-kernel residuals:", residual12, residual13)
    print(
        "universal K_NS,R(E) = Matrix([[1+Q*E,1],[E**2,-1]])"
    )
    print("C12=(a,Q) K_NS,R(E2) (1,(z-Q)^2)^T")
    print("C13=(w,Q) K_NS,R(E3) (1,(a-Q)^2)^T")


def triple_quartic():
    """Reconstruct the non-product quartic after its visible linear factor."""

    values = []
    for b in INTERPOLATION_B:
        q_value = b + 1 / b
        linear = q_value / 2 + P1 - P2 + P3
        quotient = sp.cancel(normalized_master(TRIPLE, 0, 1, b) / linear)
        if sp.denom(quotient) != 1:
            raise AssertionError("The triple master did not divide by its linear factor")
        values.append((b, quotient))
    reconstructed = reconstruct_q_polynomial(values, 4)

    q_check = CHECK_B + 1 / CHECK_B
    direct = sp.cancel(
        normalized_master(TRIPLE, 0, 1, CHECK_B)
        / (q_check / 2 + P1 - P2 + P3)
    )
    residual = exact_check(reconstructed, direct, CHECK_B)
    if residual != 0:
        raise AssertionError(f"triple interpolation residual: {residual}")
    print("F0 =", sp.factor(reconstructed))
    print("triple interpolation residual:", residual)
    return reconstructed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--triple", action="store_true", help="also reconstruct the slow quartic"
    )
    args = parser.parse_args()
    derive_pairs()
    if args.triple:
        triple_quartic()


if __name__ == "__main__":
    main()
