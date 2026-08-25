#!/usr/bin/env python3
"""Recover the Q-polynomials of the first mixed-channel master numerators.

``fit_signed_sectors.scaled_raw`` is fast with a rational value of ``b``
and symbolic momenta, but a direct calculation with symbolic ``b`` is much
slower.  The SCA Ward answer depends on ``b`` only through ``Q=b+b**(-1)``.
This helper evaluates at several rational b values and interpolates every
momentum coefficient in Q.  It is intended for discovering the elementary
two-channel kernels; it does not alter the state/Ward implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import sympy as sp


HERE = Path(__file__).resolve().parent
GRID_DIR = HERE.parent
sys.path.insert(0, str(GRID_DIR))

import fit_signed_sectors as fit  # noqa: E402


P1, P2, P3, Q = sp.symbols("P1 P2 P3 Q")

# Distinct Q values, all obtained from exact rational b values.
B_VALUES = tuple(
    map(
        sp.Rational,
        ("3/2", "5/3", "7/4", "4/3", "8/5", "9/7", "11/8", "13/9"),
    )
)

CASES = {
    "12": (
        (sp.Rational(1, 2), sp.Rational(3, 4), sp.Rational(1, 4)),
        (0, 0, 0, -1),
    ),
    "13": (
        (sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(3, 4)),
        (1, 0, 0, 1),
    ),
    "23": (
        (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4)),
        (0, 0, 0, -1),
    ),
}


def _phase_normalized(labels, discrete, b_value):
    """Return the monic momentum polynomial and its exact leading scale."""

    value = sp.cancel(
        fit.scaled_raw(labels, discrete, (b_value, P1, P2, P3))
    )
    polynomial = sp.Poly(value, P1, P2, P3, domain="EX")
    leading = polynomial.LC()
    return sp.Poly(sp.cancel(value / leading), P1, P2, P3), sp.factor(leading)


def interpolate_case(case_name: str):
    labels, discrete = CASES[case_name]
    samples = []
    leading_scales = []
    monomials = None
    for b_value in B_VALUES:
        polynomial, leading = _phase_normalized(labels, discrete, b_value)
        samples.append((sp.cancel(b_value + 1 / b_value), polynomial))
        leading_scales.append(leading)
        current = set(polynomial.monoms())
        monomials = current if monomials is None else monomials | current

    answer = 0
    for monomial in sorted(monomials, reverse=True):
        points = [
            (q_value, polynomial.coeff_monomial(monomial))
            for q_value, polynomial in samples
        ]
        coefficient = sp.factor(sp.interpolate(points, Q))
        answer += coefficient * P1 ** monomial[0] * P2 ** monomial[1] * P3 ** monomial[2]
    answer = sp.factor(answer)

    # Verify all interpolation samples exactly.
    for q_value, polynomial in samples:
        residual = sp.Poly(
            sp.expand(answer.subs(Q, q_value) - polynomial.as_expr()),
            P1,
            P2,
            P3,
        )
        if not residual.is_zero:
            raise AssertionError((case_name, q_value, residual.as_expr()))
    return labels, discrete, answer, tuple(leading_scales)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=tuple(CASES) + ("all",), default="all", nargs="?")
    args = parser.parse_args()
    names = CASES if args.case == "all" else (args.case,)
    for name in names:
        labels, discrete, polynomial, scales = interpolate_case(name)
        print(f"case={name}, labels={labels}, discrete={discrete}")
        print(f"leading scales={scales}")
        print(polynomial)


if __name__ == "__main__":
    main()
