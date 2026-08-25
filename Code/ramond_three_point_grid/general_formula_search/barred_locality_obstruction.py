#!/usr/bin/env python3
"""Exact audit of the barred Ramond locality contraction.

Suchanek's normalized Ramond forms use opposite phases in the two chiral
halves,

    rho_1^(eta)     = rho^(+-)     + i eta rho^(-+),
    bar rho_1^(eta) = bar rho^(+-) - i eta bar rho^(-+).

Consequently a local contraction is not the same-holomorphic Hadamard square
``A_f o A_f``.  This script applies the spin-frame involution i -> -i to the
second chiral table and checks two facts exactly:

1. after the common local-leg normalization is calibrated at the ground
   state, the diagonal contraction has the coefficient 2 appearing in the
   published correlator 2 C_R^(eta);
2. all four crossed spin pairings have rank one on
   (R_0 bar R_1, R_1 bar R_0).  Hence the crossed local fields do not provide
   two independent Bell-tomography equations.  The invertible system obtained
   from A_f o A_f is only an algebraic same-chiral diagnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
CONSTRUCTIVE = HERE.parent / "constructive_formula"
if str(CONSTRUCTIVE) not in sys.path:
    sys.path.insert(0, str(CONSTRUCTIVE))

import bell_tomography as bell  # noqa: E402


I = sp.I
SQRT2 = sp.sqrt(2)


EXPECTED_DIAGONAL = {
    (0, 0): (-2 + 2 * I, -4 - 4 * I),
    (0, 1): (4 + 4 * I, -8 + 8 * I),
    (1, 0): (4 + 4 * I, -2 + 2 * I),
    (1, 1): (8 - 8 * I, 4 + 4 * I),
}


def coefficient_row(expression, variables):
    """Return coefficients of R0*barR1 and R1*barR0."""

    r0, r1, bar_r0, bar_r1 = variables
    polynomial = sp.Poly(sp.expand(expression), *variables)
    return sp.Matrix(
        [[
            sp.factor(polynomial.coeff_monomial(r0 * bar_r1)),
            sp.factor(polynomial.coeff_monomial(r1 * bar_r0)),
        ]]
    )


def exact_rows():
    r0, r1, bar_r0, bar_r1 = sp.symbols("R0 R1 barR0 barR1")
    variables = (r0, r1, bar_r0, bar_r1)

    for p2, n2 in ((0, sp.Rational(1, 4)), (1, sp.Rational(3, 4))):
        for p3, n3 in ((0, sp.Rational(1, 4)), (1, sp.Rational(3, 4))):
            holomorphic = bell.amplitude_matrices(r0, r1, p2, p3, 1)
            barred = bell.amplitude_matrices(
                bar_r0, bar_r1, p2, p3, 1, barred=True
            )

            diagonal = bell.local_contraction(
                bell.diagonal_pairing(p2),
                bell.diagonal_pairing(p3),
                holomorphic,
                barred,
            )
            polynomial = sp.Poly(
                sp.expand(diagonal), r0, r1, bar_r0, bar_r1
            )
            diagonal_row = (
                sp.factor(polynomial.coeff_monomial(r0 * bar_r0)),
                sp.factor(polynomial.coeff_monomial(r1 * bar_r1)),
            )
            assert diagonal_row == EXPECTED_DIAGONAL[(p2, p3)]

            crossed_rows = []
            for sign2 in (1, -1):
                for sign3 in (1, -1):
                    crossed = bell.local_contraction(
                        bell.crossed_pairing(p2, sign2),
                        bell.crossed_pairing(p3, sign3),
                        holomorphic,
                        barred,
                    )
                    crossed_rows.append(
                        coefficient_row(crossed, variables)
                    )

            stacked = sp.Matrix.vstack(*crossed_rows)
            assert stacked.rank() == 1
            for first in range(len(crossed_rows)):
                for second in range(first + 1, len(crossed_rows)):
                    determinant = sp.Matrix.vstack(
                        crossed_rows[first], crossed_rows[second]
                    ).det()
                    assert sp.simplify(determinant) == 0

            print(
                f"(p2,p3)=({p2},{p3}): "
                f"D={diagonal_row}; crossed rank={stacked.rank()}"
            )


def published_ground_anchor():
    """Check the momentum-independent ground coefficient exactly."""

    level = sp.Rational(1, 4)
    b_value = sp.Rational(3, 2)
    momenta = sp.symbols("P1 P2 P3", real=True)
    for eta in (1, -1):
        masters = [
            bell.grid.enlarged_raw_three_point(
                0,
                level,
                level,
                epsilon2,
                0,
                0,
                eta,
                b_value,
                *momenta,
            )[1]
            for epsilon2 in (0, 1)
        ]
        barred = [bell.spin_frame_bar(master) for master in masters]
        diagonal = bell.forward_local_data(
            masters[0],
            masters[1],
            barred[0],
            barred[1],
            level,
            level,
            eta,
        )[0]
        assert sp.simplify(-diagonal / 4) == 2
        print(f"ground eta={eta:+d}: (-D_spin-frame/4)=2")


def main():
    exact_rows()
    published_ground_anchor()
    print("barred crossed-pairing rank audit: passed")


if __name__ == "__main__":
    main()
