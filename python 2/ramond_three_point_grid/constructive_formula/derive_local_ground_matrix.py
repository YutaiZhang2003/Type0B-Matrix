#!/usr/bin/env python3
"""Derive the local Ramond field in the two chiral branch copies.

The calculation translates the zero-mode conventions of
Schomerus--Suchanek (arXiv:1210.1856) into the spin frame used by the
Ramond three-point grid.  It is an exact finite-dimensional calculation;
no three-point data are fitted.

With t=exp(pi*i/4), the script proves

  Phi^(+1/2) = -t W_+^0 bar(W_+^0) + 2/t W_+^1 bar(W_+^1),
  Phi^(-1/2) =  t W_-^0 bar(W_-^0) - 2/t W_-^1 bar(W_-^1).

Only the finite-dimensional change of basis is asserted here.  A local
three-point contraction must use the anti-holomorphic form with the opposite
``i eta'' phase.  The correct barred contraction and its rank obstruction are
audited in ``bell_tomography.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
GRID_DIR = HERE.parent
if str(GRID_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_DIR))

import compute_grid as grid  # noqa: E402


I = sp.I
SQRT2 = sp.sqrt(2)
T = (1 + I) / SQRT2

# A chiral enlarged Ramond ground is indexed by (auxiliary, physical),
# with 0 denoting + and 1 denoting -.  This order separates the even and
# odd total-parity subspaces.
ENLARGED_BASIS = ((0, 0), (1, 1), (1, 0), (0, 1))
BRANCH_COLUMNS = ((1, 0), (-1, 0), (1, 1), (-1, 1))


def branch_vectors():
    """Return W_sign^epsilon in the enlarged ground basis."""

    return {
        (1, 0): {(0, 0): 1, (1, 1): -T},
        (-1, 0): {(0, 0): 1, (1, 1): T},
        (1, 1): {(1, 0): 1 / SQRT2, (0, 1): T / SQRT2},
        (-1, 1): {(1, 0): 1 / SQRT2, (0, 1): -T / SQRT2},
    }


def branch_change_of_basis():
    vectors = branch_vectors()
    return sp.Matrix(
        [
            [vectors[column].get(state, 0) for column in BRANCH_COLUMNS]
            for state in ENLARGED_BASIS
        ]
    )


def local_spin_embeddings():
    """Local spin fields satisfying the SS left/right zero-mode rules.

    A dictionary entry (a,b) is the coefficient of |a> tensor |bar b>.
    The choice is fixed so Phi^(+1/2) uses the positive branch and
    Phi^(-1/2) the reflected negative branch.
    """

    sigma_plus = {(0, 1): I, (1, 0): 1}
    sigma_minus = {(0, 0): T, (1, 1): I * T}
    capital_sigma_plus = {(0, 1): 1, (1, 0): I}
    capital_sigma_minus = {(0, 0): -1, (1, 1): 1}
    return sigma_plus, sigma_minus, capital_sigma_plus, capital_sigma_minus


def reordered_local_matrix(relative_sign):
    """Coefficient matrix after (u_L,u_R,w_L,w_R)->((u,w)_L,(u,w)_R)."""

    sigma_p, sigma_m, capital_p, capital_m = local_spin_embeddings()
    answer = sp.zeros(4, 4)
    for auxiliary, physical, coefficient in (
        (sigma_p, capital_p, 1),
        (sigma_m, capital_m, relative_sign),
    ):
        for (u_left, u_right), aux_coefficient in auxiliary.items():
            for (w_left, w_right), physical_coefficient in physical.items():
                # The right auxiliary state crosses the left physical state.
                koszul = (-1) ** (u_right * w_left)
                row = ENLARGED_BASIS.index((u_left, w_left))
                column = ENLARGED_BASIS.index((u_right, w_right))
                answer[row, column] += (
                    coefficient
                    * koszul
                    * aux_coefficient
                    * physical_coefficient
                )
    return answer


def reordered_crossed_matrix(relative_sign):
    """Matrix for sigma^+ Sigma^- plus/minus sigma^- Sigma^+."""

    sigma_p, sigma_m, capital_p, capital_m = local_spin_embeddings()
    answer = sp.zeros(4, 4)
    for auxiliary, physical, coefficient in (
        (sigma_p, capital_m, 1),
        (sigma_m, capital_p, relative_sign),
    ):
        for (u_left, u_right), aux_coefficient in auxiliary.items():
            for (w_left, w_right), physical_coefficient in physical.items():
                koszul = (-1) ** (u_right * w_left)
                row = ENLARGED_BASIS.index((u_left, w_left))
                column = ENLARGED_BASIS.index((u_right, w_right))
                answer[row, column] += (
                    coefficient
                    * koszul
                    * aux_coefficient
                    * physical_coefficient
                )
    return answer


def local_matrix_in_branch_basis(relative_sign):
    transition = branch_change_of_basis()
    return sp.simplify(
        transition.inv()
        * reordered_local_matrix(relative_sign)
        * transition.inv().T
    )


def crossed_matrix_in_branch_basis(relative_sign):
    transition = branch_change_of_basis()
    return sp.simplify(
        transition.inv()
        * reordered_crossed_matrix(relative_sign)
        * transition.inv().T
    )


def crossed_parity_block(relative_sign):
    """Rows (+,epsilon), columns (-,bar epsilon), epsilon=0,1."""

    full = crossed_matrix_in_branch_basis(relative_sign)
    return full.extract((0, 2), (1, 3))


def extend_crossed_block(block, mode_count):
    """Apply M paired nonzero chi modes, omitting their common leg factor."""

    swap = sp.Matrix([[0, 1], [1, 0]])
    parity_sign = sp.diag(1, (-1) ** mode_count)
    permutation = swap ** (mode_count % 2)
    return sp.simplify(permutation * parity_sign * block * permutation)


def hard_quadratic_check():
    """Legacy same-chiral square; this is not a local correlator."""
    p1, p2, p3 = sp.symbols("P1 P2 P3")
    b_value = sp.Rational(3, 2)
    level = sp.Rational(3, 4)

    # M=1: the ground even and odd copies are interchanged.  In the order
    # epsilon=(0,1), the local diagonal coefficient vector is (-2/t,-t).
    local_vector = sp.Matrix([-2 / T, -T])
    for eta in (1, -1):
        master = grid.enlarged_raw_three_point(
            0, level, level, 0, 0, 0, eta, b_value, p1, p2, p3
        )[1]
        coefficients = []
        for form_parity in (0, 1):
            amplitude = sp.zeros(2, 2)
            for epsilon2 in (0, 1):
                for epsilon3 in (0, 1):
                    amplitude[epsilon2, epsilon3] = (
                        grid.enlarged_raw_three_point(
                            0,
                            level,
                            level,
                            epsilon2,
                            epsilon3,
                            form_parity,
                            eta,
                            b_value,
                            p1,
                            p2,
                            p3,
                        )[1]
                    )
            quadratic = (
                local_vector.T
                * amplitude.multiply_elementwise(amplitude)
                * local_vector
            )[0]
            coefficient = sp.factor(sp.cancel(quadratic / master**2))
            coefficients.append(coefficient)
        assert coefficients == [-8 * I, -8]
        print(f"eta={eta:+d}: f=0 coefficient={coefficients[0]}")
        print(f"eta={eta:+d}: f=1 coefficient={coefficients[1]}")
        print(f"eta={eta:+d}: total coefficient={sum(coefficients)}")


def primary_quadratic_check():
    """Legacy same-chiral square; this is not a local correlator."""
    p1, p2, p3 = sp.symbols("P1 P2 P3")
    b_value = sp.Rational(3, 2)
    level = sp.Rational(1, 4)
    local_vector = sp.Matrix([-T, 2 / T])
    for eta in (1, -1):
        master = grid.enlarged_raw_three_point(
            0, level, level, 0, 0, 0, eta, b_value, p1, p2, p3
        )[1]
        coefficients = []
        for form_parity in (0, 1):
            amplitude = sp.zeros(2, 2)
            for epsilon2 in (0, 1):
                for epsilon3 in (0, 1):
                    amplitude[epsilon2, epsilon3] = (
                        grid.enlarged_raw_three_point(
                            0,
                            level,
                            level,
                            epsilon2,
                            epsilon3,
                            form_parity,
                            eta,
                            b_value,
                            p1,
                            p2,
                            p3,
                        )[1]
                    )
            quadratic = (
                local_vector.T
                * amplitude.multiply_elementwise(amplitude)
                * local_vector
            )[0]
            coefficients.append(
                sp.factor(sp.cancel(quadratic / master**2))
            )
        assert coefficients == [2 * I, 2]
        print(f"ground eta={eta:+d}: f=0 coefficient={coefficients[0]}")
        print(f"ground eta={eta:+d}: f=1 coefficient={coefficients[1]}")
        print(f"ground eta={eta:+d}: total coefficient={sum(coefficients)}")


def ns_ground_discrete_factor():
    """Same-chiral diagnostic when the NS branch is n_1=0.

    In addition to the phase relations proved by the full grid, this check
    uses the special n_1=0 relation between R_1 and R_0.  That relation is
    false for an excited NS branch and is not used by the general matrix
    formula above.  This does not use the barred local form and must not be
    interpreted as a published local-field correlator.
    """

    for m2 in (0, 1):
        for m3 in (0, 1):
            coefficients = [0, 0]
            local_vectors = []
            for mode_count in (m2, m3):
                parity = mode_count % 2
                vector = [0, 0]
                vector[parity] = -T
                vector[1 - parity] = 2 * (-1) ** mode_count / T
                local_vectors.append(vector)
            for eta in (1, -1):
                r2 = (
                    I
                    * eta
                    * (-1) ** (m2 + m3)
                    * 2 ** (sp.Rational((-1) ** (m2 + 1), 2))
                )
                r3 = 2 ** (sp.Rational((-1) ** (m3 + 1), 2))
                for form_parity in (0, 1):
                    value = 0
                    for epsilon2 in (0, 1):
                        for epsilon3 in (0, 1):
                            spin_phase = (
                                eta
                                * (-1) ** (m2 + 1 + epsilon2)
                                / T
                            )
                            amplitude_ratio = (
                                r2**epsilon2
                                * r3**epsilon3
                                * (-1) ** (epsilon3 * form_parity)
                                * spin_phase**form_parity
                            )
                            value += (
                                local_vectors[0][epsilon2]
                                * local_vectors[1][epsilon3]
                                * amplitude_ratio**2
                            )
                    expected = (
                        2 ** (m2 + m3 + 1)
                        * (I if form_parity == 0 else 1)
                        * (-I) ** (m2 + m3)
                    )
                    assert sp.simplify(value - expected) == 0
                    coefficients[form_parity] = expected
            print(
                f"(M2 mod 2,M3 mod 2)=({m2},{m3}): "
                f"f=0 {coefficients[0]}, f=1 {coefficients[1]}, "
                f"total {sp.simplify(sum(coefficients))}"
            )


def general_two_master_rows():
    """Legacy same-chiral row, not the barred local-field row."""

    r0, r1 = sp.symbols("R0 R1")
    expected_rows = {
        (0, 0): (2 * I, -4),
        (0, 1): (4, 8 * I),
        (1, 0): (4, 2 * I),
        (1, 1): (-8 * I, 4),
    }
    for m2 in (0, 1):
        for m3 in (0, 1):
            local_vectors = []
            for mode_count in (m2, m3):
                parity = mode_count % 2
                vector = [0, 0]
                vector[parity] = -T
                vector[1 - parity] = 2 * (-1) ** mode_count / T
                local_vectors.append(vector)
            r3 = 2 ** (sp.Rational((-1) ** (m3 + 1), 2))
            answer = 0
            for epsilon2, master in enumerate((r0, r1)):
                for epsilon3 in (0, 1):
                    # Summing the squares of f=0 and f=1 produces 1-i:
                    # (s_epsilon^eta)^2=t^{-2}=-i.
                    answer += (
                        (1 - I)
                        * local_vectors[0][epsilon2]
                        * local_vectors[1][epsilon3]
                        * master**2
                        * r3 ** (2 * epsilon3)
                    )
            row = expected_rows[(m2, m3)]
            expected = row[0] * r0**2 + row[1] * r1**2
            assert sp.simplify(answer - expected) == 0
            print(
                f"general row ({m2},{m3}): "
                f"Q=({row[0]}) R0^2 + ({row[1]}) R1^2"
            )


def crossed_outer_product_formula():
    """Legacy same-chiral crossed form, not a local correlator."""

    r0, r1 = sp.symbols("R0 R1")
    for m2 in (0, 1):
        for m3 in (0, 1):
            r3 = 2 ** (sp.Rational((-1) ** (m3 + 1), 2))
            for sign2 in (1, -1):
                for sign3 in (1, -1):
                    block2 = extend_crossed_block(
                        crossed_parity_block(sign2), m2
                    )
                    block3 = extend_crossed_block(
                        crossed_parity_block(sign3), m3
                    )
                    xi2 = sp.simplify(block2[0, 1] + block2[1, 0])
                    xi3 = sp.simplify(block3[0, 1] + block3[1, 0])
                    direct = sp.simplify(
                        (1 - I) * r3 * xi2 * xi3 * r0 * r1
                    )
                    sign2_index = 0 if sign2 == 1 else 1
                    sign3_index = 0 if sign3 == 1 else 1
                    expected = sp.simplify(
                        2 ** (sp.Rational(1, 2) + m3)
                        * (1 - I)
                        * I
                        ** (
                            m2
                            + m3
                            + sign2_index
                            + sign3_index
                        )
                        * r0
                        * r1
                    )
                    assert sp.simplify(direct - expected) == 0
            coefficient = sp.simplify(
                2 ** (sp.Rational(1, 2) + m3)
                * (1 - I)
                * I ** (m2 + m3)
            )
            print(
                f"crossed ++ row ({m2},{m3}): "
                f"Q_cross=({coefficient}) R0 R1"
            )


def outer_product_inversion():
    """Legacy same-chiral inversion; invalid for the barred local forms."""

    print("outer-product inversion:")
    print("  X=R0^2, Y=R0*R1, Z=R1^2, with X*Z=Y^2")
    print("  D=a*X+b*Z, C=c*Y")
    print("  Y=C/c, X=(D +/- sqrt(D^2-4*a*b*Y^2))/(2*a), Z=Y^2/X")


def grid_audit():
    """Check only the legacy same-chiral row, not a locality identity."""

    b_value = sp.Rational(3, 2)
    p1 = sp.Rational(1, 3)
    p2 = sp.Rational(2, 5)
    p3 = sp.Rational(3, 7)
    expected_rows = {
        (0, 0): (2 * I, -4),
        (0, 1): (4, 8 * I),
        (1, 0): (4, 2 * I),
        (1, 1): (-8 * I, 4),
    }
    checked = 0
    for n1 in grid.GRID_NS_LEVELS:
        for n2 in grid.GRID_R_LEVELS:
            for n3 in grid.GRID_R_LEVELS:
                m2 = int(2 * n2 - sp.Rational(1, 2)) % 2
                m3 = int(2 * n3 - sp.Rational(1, 2)) % 2
                local_vectors = []
                for mode_count in (m2, m3):
                    vector = [0, 0]
                    vector[mode_count] = -T
                    vector[1 - mode_count] = 2 * (-1) ** mode_count / T
                    local_vectors.append(sp.Matrix(vector))
                for eta in (1, -1):
                    amplitude_sum = sp.zeros(2, 2)
                    masters = []
                    for epsilon2 in (0, 1):
                        masters.append(
                            grid.enlarged_raw_three_point(
                                n1,
                                n2,
                                n3,
                                epsilon2,
                                0,
                                0,
                                eta,
                                b_value,
                                p1,
                                p2,
                                p3,
                            )[1]
                        )
                    for form_parity in (0, 1):
                        amplitude = sp.zeros(2, 2)
                        for epsilon2 in (0, 1):
                            for epsilon3 in (0, 1):
                                amplitude[epsilon2, epsilon3] = (
                                    grid.enlarged_raw_three_point(
                                        n1,
                                        n2,
                                        n3,
                                        epsilon2,
                                        epsilon3,
                                        form_parity,
                                        eta,
                                        b_value,
                                        p1,
                                        p2,
                                        p3,
                                    )[1]
                                )
                        amplitude_sum += amplitude.multiply_elementwise(amplitude)
                    direct = (
                        local_vectors[0].T
                        * amplitude_sum
                        * local_vectors[1]
                    )[0]
                    row = expected_rows[(m2, m3)]
                    expected = row[0] * masters[0] ** 2 + row[1] * masters[1] ** 2
                    residual = sp.cancel(direct - expected)
                    if residual != 0:
                        raise AssertionError(
                            (n1, n2, n3, eta, residual)
                        )
                    checked += 1
    assert checked == 108
    print(f"full master-sector quadratic audit: {checked}/108 passed")


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    plus = local_matrix_in_branch_basis(1)
    minus = local_matrix_in_branch_basis(-1)
    expected_plus = sp.diag(-T, 0, 2 / T, 0)
    expected_minus = sp.diag(0, T, 0, -2 / T)
    assert sp.simplify(plus - expected_plus) == sp.zeros(4)
    assert sp.simplify(minus - expected_minus) == sp.zeros(4)
    crossed_plus = crossed_matrix_in_branch_basis(1)
    crossed_minus = crossed_matrix_in_branch_basis(-1)
    expected_crossed_plus = sp.Matrix(
        [
            [0, 0, 0, -T],
            [0, 0, 1 / T, 0],
            [0, -1 / T, 0, 0],
            [-T, 0, 0, 0],
        ]
    )
    expected_crossed_minus = sp.Matrix(
        [
            [0, 0, 0, 1 / T],
            [0, 0, -T, 0],
            [0, -T, 0, 0],
            [-1 / T, 0, 0, 0],
        ]
    )
    assert sp.simplify(crossed_plus - expected_crossed_plus) == sp.zeros(4)
    assert sp.simplify(crossed_minus - expected_crossed_minus) == sp.zeros(4)
    print("column order: (+,0), (-,0), (+,1), (-,1)")
    print("Phi^(+1/2):")
    print(plus)
    print("Phi^(-1/2):")
    print(minus)
    print("sigma^+ Sigma^- + sigma^- Sigma^+:")
    print(crossed_plus)
    print("sigma^+ Sigma^- - sigma^- Sigma^+:")
    print(crossed_minus)
    print("barred local contractions are audited in bell_tomography.py")


if __name__ == "__main__":
    main()
