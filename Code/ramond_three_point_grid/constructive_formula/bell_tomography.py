#!/usr/bin/env python3
"""Exact locality-rank audit for the two Ramond parity-copy masters.

The anti-holomorphic Ramond form has the opposite ``i eta'' phase from the
holomorphic form.  Consequently the published local diagonal field and the
crossed spin products do *not* give an invertible four-by-four linear map on

    (R_0 bar(R_0), R_0 bar(R_1), R_1 bar(R_0), R_1 bar(R_1)).

This module derives the coefficient rows with the correct spin-frame
involution i -> -i.  Keeping the even and odd forms separately, and keeping
the two oriented products sigma^+ Sigma^- and sigma^- Sigma^+ separately,
still gives rank two.  The physical even-plus-odd sum additionally cancels
all mixed diagonal/crossed contractions.  Thus there is no linear ``Bell
tomography'' inverse from local fields alone.

There is nevertheless a useful convention-dependent nonlinear inverse.  The
direct chiral states obey

    bar(R_0)/R_0 = lambda,
    bar(R_1)/R_1 = -lambda,
    lambda = i eta (-1)^(2 n_1 + M_2 + M_3 + 1),

where M_j=2 n_j-1/2.  After imposing this known spin-frame phase and the
rank-one identity, one diagonal row and one crossed row determine the three
chiral products R_0^2, R_0 R_1, R_1^2 up to the two roots of a quadratic.
Analytic continuation from the ground state selects a root.

The diagonal D_+ and D_- matrices below are exactly the reflected local
fields derived from the zero-mode conventions of arXiv:1210.1856.  X_+,
X_-, and their two orientations are engineered momentum--winding
representatives; they are not formulas stated in that paper.
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

# Full branch order: (+,0), (-,0), (+,1), (-,1).
D_PLUS = sp.diag(-T, 0, 2 / T, 0)
D_MINUS = sp.diag(0, T, 0, -2 / T)
X_PLUS = sp.Matrix(
    [
        [0, 0, 0, -T],
        [0, 0, 1 / T, 0],
        [0, -1 / T, 0, 0],
        [-T, 0, 0, 0],
    ]
)
X_MINUS = sp.Matrix(
    [
        [0, 0, 0, 1 / T],
        [0, 0, -T, 0],
        [0, -T, 0, 0],
        [-1 / T, 0, 0, 0],
    ]
)
ORIENTED_SIGMA_PLUS = sp.simplify((X_PLUS + X_MINUS) / 2)
ORIENTED_SIGMA_MINUS = sp.simplify((X_PLUS - X_MINUS) / 2)

PLUS_INDICES = (0, 2)
MINUS_INDICES = (1, 3)
SWAP = sp.Matrix([[0, 1], [1, 0]])


def mode_parity(branch_label):
    """Return M mod 2 for n=M/2+1/4."""

    count = int(2 * sp.Rational(branch_label) - sp.Rational(1, 2))
    if count < 0:
        count = -count
    return count % 2


def ramond_mode_count(branch_label):
    count = int(2 * abs(sp.Rational(branch_label)) - sp.Rational(1, 2))
    if count < 0:
        raise ValueError("The Ramond magnitude must be at least 1/4")
    return count


def sign_block(matrix, left_sign, right_sign):
    rows = PLUS_INDICES if left_sign == 1 else MINUS_INDICES
    columns = PLUS_INDICES if right_sign == 1 else MINUS_INDICES
    return matrix.extract(rows, columns)


def extend_pairing(block, mode_count):
    """Transport a ground pairing through M paired nonzero chi modes.

    The common scalar from derivatives, the paper's n_k, and the conversion
    between chi conventions is a leg normalization and is deliberately not
    included.  It cancels in normalized ratios and is restored from the
    explicit N_alpha^(k) factors in the double-Liouville representative.
    """

    mode_count = int(mode_count)
    permutation = SWAP ** (mode_count % 2)
    left_sign = sp.diag(1, (-1) ** mode_count)
    return sp.simplify(permutation * left_sign * block * permutation)


def diagonal_pairing(mode_count):
    return extend_pairing(sign_block(D_PLUS, 1, 1), mode_count)


def crossed_pairing(mode_count, relative_sign=1):
    matrix = X_PLUS if relative_sign == 1 else X_MINUS
    return extend_pairing(sign_block(matrix, 1, -1), mode_count)


def oriented_pairing(mode_count, orientation):
    """Return one ordered crossed product before the local +/- sum.

    ``orientation=0`` is sigma^+ Sigma^- and ``orientation=1`` is
    sigma^- Sigma^+.
    """

    if orientation == 0:
        matrix = ORIENTED_SIGMA_PLUS
    elif orientation == 1:
        matrix = ORIENTED_SIGMA_MINUS
    else:
        raise ValueError("orientation must be 0 or 1")
    return extend_pairing(sign_block(matrix, 1, -1), mode_count)


def amplitude_ratio_matrices(m2, m3, eta):
    """Return A_f[e2,e3]/R_e2 for f=0,1."""

    eta = int(eta)
    r3 = 2 ** (sp.Rational((-1) ** (m3 + 1), 2))
    answer = []
    for form_parity in (0, 1):
        matrix = sp.zeros(2, 2)
        for epsilon2 in (0, 1):
            spin_phase = eta * (-1) ** (m2 + 1 + epsilon2) / T
            for epsilon3 in (0, 1):
                matrix[epsilon2, epsilon3] = (
                    r3**epsilon3
                    * (-1) ** (epsilon3 * form_parity)
                    * spin_phase**form_parity
                )
        answer.append(matrix)
    return tuple(answer)


def spin_frame_bar(expression):
    """Apply i->-i without conjugating momenta or the coupling."""

    expression = sp.sympify(expression)
    answer = sp.conjugate(expression)
    answer = answer.xreplace(
        {sp.conjugate(symbol): symbol for symbol in expression.free_symbols}
    )
    return sp.factor(sp.cancel(answer))


def amplitude_matrices(r0, r1, m2, m3, eta, barred=False):
    masters = (sp.sympify(r0), sp.sympify(r1))
    answer = []
    for ratios in amplitude_ratio_matrices(m2, m3, eta):
        if barred:
            ratios = ratios.applyfunc(spin_frame_bar)
        answer.append(
            sp.Matrix(
                2,
                2,
                lambda epsilon2, epsilon3: (
                    masters[epsilon2] * ratios[epsilon2, epsilon3]
                ),
            )
        )
    return tuple(answer)


def local_contraction(
    left_pairing,
    right_pairing,
    left_amplitudes,
    right_amplitudes,
    form_parity=None,
):
    """Contract local leg pairings against chiral amplitude tables.

    If ``form_parity`` is omitted the two form parities are summed.  Passing
    0 or 1 keeps Suchanek's even or odd chiral Ramond form separate.
    """

    total = 0
    if form_parity is None:
        pairs = zip(left_amplitudes, right_amplitudes)
    elif form_parity in (0, 1):
        pairs = ((left_amplitudes[form_parity], right_amplitudes[form_parity]),)
    else:
        raise ValueError("form_parity must be 0, 1, or None")
    for left, right in pairs:
        for epsilon2 in (0, 1):
            for epsilon2_bar in (0, 1):
                for epsilon3 in (0, 1):
                    for epsilon3_bar in (0, 1):
                        total += (
                            left_pairing[epsilon2, epsilon2_bar]
                            * right_pairing[epsilon3, epsilon3_bar]
                            * left[epsilon2, epsilon3]
                            * right[epsilon2_bar, epsilon3_bar]
                        )
    return sp.factor(sp.cancel(total))


def forward_local_data(r0, r1, bar_r0, bar_r1, n2, n3, eta):
    """Return D, X_{++}, and X_{+-} local contractions."""

    m2 = ramond_mode_count(n2)
    m3 = ramond_mode_count(n3)
    amplitudes = amplitude_matrices(r0, r1, m2, m3, eta)
    barred_amplitudes = amplitude_matrices(
        bar_r0, bar_r1, m2, m3, eta, barred=True
    )
    diagonal = local_contraction(
        diagonal_pairing(m2),
        diagonal_pairing(m3),
        amplitudes,
        barred_amplitudes,
    )
    crossed_plus_plus = local_contraction(
        crossed_pairing(m2, 1),
        crossed_pairing(m3, 1),
        amplitudes,
        barred_amplitudes,
    )
    crossed_plus_minus = local_contraction(
        crossed_pairing(m2, 1),
        crossed_pairing(m3, -1),
        amplitudes,
        barred_amplitudes,
    )
    return diagonal, crossed_plus_plus, crossed_plus_minus


def tomography_coefficients(n2, n3, eta):
    """Return coefficient rows on (R0 barR0,R0 barR1,R1 barR0,R1 barR1)."""

    r0, r1, bar_r0, bar_r1 = sp.symbols("R0 R1 barR0 barR1")
    contractions = forward_local_data(
        r0, r1, bar_r0, bar_r1, n2, n3, eta
    )
    monomials = (
        r0 * bar_r0,
        r0 * bar_r1,
        r1 * bar_r0,
        r1 * bar_r1,
    )
    rows = []
    for contraction in contractions:
        polynomial = sp.Poly(
            sp.expand(contraction), r0, r1, bar_r0, bar_r1
        )
        row = tuple(
            sp.simplify(polynomial.coeff_monomial(monomial))
            for monomial in monomials
        )
        residual = sp.expand(
            contraction - sum(coefficient * monomial for coefficient, monomial in zip(row, monomials))
        )
        if residual != 0:
            raise AssertionError(residual)
        rows.append(row)
    return tuple(rows)


def coefficient_row(expression, variables):
    """Return a bilinear coefficient row in the standard four-entry order."""

    r0, r1, bar_r0, bar_r1 = variables
    monomials = (
        r0 * bar_r0,
        r0 * bar_r1,
        r1 * bar_r0,
        r1 * bar_r1,
    )
    polynomial = sp.Poly(sp.expand(expression), *variables)
    row = tuple(
        sp.factor(polynomial.coeff_monomial(monomial))
        for monomial in monomials
    )
    residual = sp.expand(
        expression
        - sum(value * monomial for value, monomial in zip(row, monomials))
    )
    if residual != 0:
        raise AssertionError(residual)
    return row


def separated_measurement_matrix(n2, n3, eta, physical_sum=False):
    """Return every D/O1/O2 pairing row and its label.

    With ``physical_sum=False`` the f=0 and f=1 rows are kept separately.
    With ``physical_sum=True`` they are summed, as in the local physical
    vertex.  The former is the most generous linear data one could extract
    by splitting Suchanek's even and odd chiral forms.
    """

    m2 = ramond_mode_count(n2)
    m3 = ramond_mode_count(n3)
    r0, r1, bar_r0, bar_r1 = sp.symbols("R0 R1 barR0 barR1")
    variables = (r0, r1, bar_r0, bar_r1)
    amplitudes = amplitude_matrices(r0, r1, m2, m3, eta)
    barred = amplitude_matrices(bar_r0, bar_r1, m2, m3, eta, barred=True)
    left_pairings = (
        ("D", diagonal_pairing(m2)),
        ("O1", oriented_pairing(m2, 0)),
        ("O2", oriented_pairing(m2, 1)),
    )
    right_pairings = (
        ("D", diagonal_pairing(m3)),
        ("O1", oriented_pairing(m3, 0)),
        ("O2", oriented_pairing(m3, 1)),
    )
    labels = []
    rows = []
    form_parities = (None,) if physical_sum else (0, 1)
    for form_parity in form_parities:
        for left_name, left in left_pairings:
            for right_name, right in right_pairings:
                contraction = local_contraction(
                    left,
                    right,
                    amplitudes,
                    barred,
                    form_parity=form_parity,
                )
                labels.append((form_parity, left_name, right_name))
                rows.append(coefficient_row(contraction, variables))
    return tuple(labels), sp.Matrix(rows)


def spin_frame_phase(n1, n2, n3, eta):
    """Return lambda=bar(R0)/R0 in the direct-state phase convention."""

    twice_n1 = int(2 * sp.Rational(n1))
    exponent = (
        twice_n1
        + ramond_mode_count(n2)
        + ramond_mode_count(n3)
        + 1
    )
    return I * int(eta) * (-1) ** exponent


def phase_reduced_inverse(diagonal, crossed, n1, n2, n3, eta):
    """Recover the two chiral-product candidates from D and X++.

    Write X=R0^2, Y=R0 R1, Z=R1^2.  The known spin-frame phases give

        D/lambda = a X - b Z,
        X++/[lambda (d-c)] = Y,

    when the coefficient rows are D=(a,0,0,b) and X++=(0,c,d,0).
    The identity X Z=Y^2 then leaves the displayed quadratic ambiguity.
    """

    diagonal_row, crossed_row, _ = tomography_coefficients(n2, n3, eta)
    a, b = diagonal_row[0], diagonal_row[3]
    c, d = crossed_row[1], crossed_row[2]
    if diagonal_row[1:3] != (0, 0):
        raise AssertionError(diagonal_row)
    if crossed_row[0] != 0 or crossed_row[3] != 0 or d == c:
        raise AssertionError(crossed_row)
    lam = spin_frame_phase(n1, n2, n3, eta)
    delta = sp.factor(sp.cancel(sp.sympify(diagonal) / lam))
    y = sp.factor(sp.cancel(sp.sympify(crossed) / (lam * (d - c))))
    discriminant = sp.factor(sp.cancel(delta**2 + 4 * a * b * y**2))
    root = sp.sqrt(discriminant)
    candidates = []
    for sign in (1, -1):
        x = sp.factor(sp.cancel((delta + sign * root) / (2 * a)))
        z = sp.factor(sp.cancel(y**2 / x))
        candidates.append(sp.Matrix([[x, y], [y, z]]))
    return tuple(candidates)


def candidate_contains(candidates, expected):
    for candidate in candidates:
        if all(
            sp.simplify(sp.cancel(candidate[row, column] - expected[row, column]))
            == 0
            for row in (0, 1)
            for column in (0, 1)
        ):
            return True
    return False


def audit_one(labels, sample):
    n1, n2, n3 = map(sp.Rational, labels)
    b_value, p1, p2, p3 = sample
    checked = 0
    for eta in (1, -1):
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
        barred_masters = [spin_frame_bar(master) for master in masters]
        diagonal, crossed_plus_plus, crossed_plus_minus = forward_local_data(
            masters[0],
            masters[1],
            barred_masters[0],
            barred_masters[1],
            n2,
            n3,
            eta,
        )
        lam = spin_frame_phase(n1, n2, n3, eta)
        if sp.simplify(barred_masters[0] - lam * masters[0]) != 0:
            raise AssertionError((labels, eta, 0, masters[0]))
        if sp.simplify(barred_masters[1] + lam * masters[1]) != 0:
            raise AssertionError((labels, eta, 1, masters[1]))
        candidates = phase_reduced_inverse(
            diagonal,
            crossed_plus_plus,
            n1,
            n2,
            n3,
            eta,
        )
        expected = sp.Matrix(
            [
                [masters[0] ** 2, masters[0] * masters[1]],
                [masters[0] * masters[1], masters[1] ** 2],
            ]
        )
        if not candidate_contains(candidates, expected):
            raise AssertionError((labels, eta, masters, candidates, expected))
        checked += 1
    return checked


def grid_audit(include_high=False):
    sample = (
        sp.Rational(3, 2),
        sp.Rational(1, 3),
        sp.Rational(2, 5),
        sp.Rational(3, 7),
    )
    checked = 0
    for n1 in grid.GRID_NS_LEVELS:
        for n2 in grid.GRID_R_LEVELS:
            for n3 in grid.GRID_R_LEVELS:
                checked += audit_one((n1, n2, n3), sample)
    print(f"grid phase-reduced recovery: {checked}/54 eta sectors passed")
    if include_high:
        high_labels = (
            (sp.Rational(3, 2), sp.Rational(3, 4), sp.Rational(3, 4)),
            (sp.Rational(3, 2), sp.Rational(3, 4), sp.Rational(5, 4)),
            (sp.Rational(3, 2), sp.Rational(5, 4), sp.Rational(3, 4)),
        )
        high_checked = sum(audit_one(labels, sample) for labels in high_labels)
        print(f"n1=3/2 phase-reduced recovery: {high_checked}/6 eta sectors passed")


def local_rank_certificate():
    """Prove that all split local rows have rank two, not four."""

    for p2, n2 in ((0, sp.Rational(1, 4)), (1, sp.Rational(3, 4))):
        for p3, n3 in ((0, sp.Rational(1, 4)), (1, sp.Rational(3, 4))):
            labels, separated = separated_measurement_matrix(n2, n3, 1)
            physical_labels, physical = separated_measurement_matrix(
                n2, n3, 1, physical_sum=True
            )
            if separated.rank() != 2 or physical.rank() != 2:
                raise AssertionError((p2, p3, separated.rank(), physical.rank()))

            # A literal four-by-four map requested by the tomography idea:
            # D-D and O1-O1, each kept at f=0 and f=1.  Its determinant is
            # zero.  Adding every other orientation and every mixed row does
            # not raise the rank above two, as certified by ``separated``.
            canonical_labels = (
                (0, "D", "D"),
                (1, "D", "D"),
                (0, "O1", "O1"),
                (1, "O1", "O1"),
            )
            canonical = sp.Matrix(
                [separated.row(labels.index(label)) for label in canonical_labels]
            )
            if canonical.det() != 0:
                raise AssertionError(canonical)

            # In the physical f=0+f=1 sum all D/O mixed contractions cancel.
            for index, (_, left, right) in enumerate(physical_labels):
                if (left == "D") != (right == "D"):
                    if any(sp.simplify(value) != 0 for value in physical.row(index)):
                        raise AssertionError((p2, p3, physical_labels[index]))
            print(
                f"(M2 mod 2,M3 mod 2)=({p2},{p3}): "
                f"split rank={separated.rank()}, physical rank={physical.rank()}, "
                "canonical det=0"
            )


def print_symbolic_coefficients():
    for p2, n2 in ((0, sp.Rational(1, 4)), (1, sp.Rational(3, 4))):
        for p3, n3 in ((0, sp.Rational(1, 4)), (1, sp.Rational(3, 4))):
            diagonal, crossed_plus_plus, crossed_plus_minus = (
                tomography_coefficients(n2, n3, 1)
            )
            print(
                f"(M2 mod 2,M3 mod 2)=({p2},{p3}): "
                f"D={diagonal}, X++={crossed_plus_plus}, "
                f"X+-={crossed_plus_minus}"
            )


def published_ground_anchor():
    """Check the calibrated ground coefficient in SS Eq. (NSRR2).

    This fixes the common local-leg scale at the ground state.  It does not
    by itself derive the descendant normalization factors N_alpha^(k).
    """

    level = sp.Rational(1, 4)
    b_value = sp.Rational(3, 2)
    momenta = tuple(sp.symbols("P1 P2 P3", real=True))
    for eta in (1, -1):
        masters = [
            grid.enlarged_raw_three_point(
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
        barred = [spin_frame_bar(master) for master in masters]
        diagonal = forward_local_data(
            masters[0], masters[1], barred[0], barred[1], level, level, eta
        )[0]
        coefficient = sp.simplify(-diagonal / 4)
        if coefficient != 2:
            raise AssertionError((eta, masters, diagonal, coefficient))
        print(f"published ground anchor eta={eta:+d}: (-D/4)={coefficient}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-audit", action="store_true")
    parser.add_argument("--high-check", action="store_true")
    args = parser.parse_args()
    print_symbolic_coefficients()
    local_rank_certificate()
    published_ground_anchor()
    if args.grid_audit or args.high_check:
        grid_audit(include_high=args.high_check)


if __name__ == "__main__":
    main()
