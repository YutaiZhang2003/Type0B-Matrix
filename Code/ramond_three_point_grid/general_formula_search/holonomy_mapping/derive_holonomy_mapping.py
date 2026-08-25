#!/usr/bin/env python3
"""Exact low-level map between Ramond branches and local components.

This file keeps three changes of basis separate.

1.  At n=+/-1/4 the two branch sheets are Hadamard combinations of the
    two aligned auxiliary/physical spin components.  The matrices ``C0``
    and ``C1`` below are extracted directly from
    ``check_ramond_branching.branch_in_abstract_basis`` after converting
    the paper ground basis to the SCblock ``w^+,w^-`` basis.

2.  A local non-chiral field is diagonal in a fixed branch sheet but sums
    the two parity copies.  Its coefficients are recorded by the diagonal
    2 by 2 matrices ``local_sheet_matrix``.  This gives both Phi^(+k) and
    Phi^(-k), including all eighth-root phases.

3.  The four hard NS--R--R masters form a 2 by 2 matrix with rows the
    branch parity epsilon and columns the physical Ramond form eta.  A
    fixed 2 by 2 projection isolates the two eta/holonomy components.  The
    eta=+ component is the product K.  The eta=- component is a projection
    of a second 2 by 2 kernel; every non-universal entry of that kernel is
    a product of one-leg polynomials.

The final Hadamard rotation from eta eigencomponents to the two chamber
components of arXiv:1510.01773 is also displayed.  That source calls the
two supports A and B, not geometric holonomies; identifying them with
geometric holonomies is therefore an additional interpretation.  The
rotation itself is exact and produces K+/-i H (up to known phases), so it
does not turn the crossed answer into scalar ell products.

All identities are symbolic.  With ``--ward`` the four hard formulas are
also compared directly with the state/Ward evaluator at b=3/2 and
symbolic P1,P2,P3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
GRID_DIR = HERE.parents[1]
RAMOND_DIR = GRID_DIR.parent / "ramond_branching_coefficient_check"
for directory in (GRID_DIR, RAMOND_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import check_ramond_branching as branch  # noqa: E402


I = sp.I
SQRT2 = sp.sqrt(2)
T = (1 + I) / SQRT2
FOURTH_ROOT_TWO = sp.Pow(2, sp.Rational(1, 4))


def _flatten_abstract(branch_label, parity):
    """Return a branch in the common auxiliary x SCblock ground basis.

    ``check_ramond_branching`` uses the paper ground state

        |Delta,-> = -exp(-pi*i/4) w^- = -w^-/t.

    The same factor multiplies every PBW word whose ground label is one.
    """

    _, sectors = branch.branch_in_abstract_basis(branch_label, parity)
    answer = {}
    for (auxiliary_modes, auxiliary_ground), (_, basis, coefficients) in sectors.items():
        for state, coefficient in zip(basis, coefficients):
            if coefficient == 0:
                continue
            physical_ground = state[2]
            if physical_ground:
                coefficient *= -1 / T
            key = (auxiliary_modes, auxiliary_ground, state)
            answer[key] = sp.simplify(coefficient)
    return answer


def ground_sheet_matrices():
    """Extract the exact sheet-to-spin-component matrices at n=1/4.

    Columns are the positive and reflected sheets.  Rows are

      epsilon=0: (u^+ w^+, u^- w^-),
      epsilon=1: (u^- w^+, u^+ w^-).
    """

    plus_even = _flatten_abstract(sp.Rational(1, 4), 0)
    minus_even = _flatten_abstract(-sp.Rational(1, 4), 0)
    plus_odd = _flatten_abstract(sp.Rational(1, 4), 1)
    minus_odd = _flatten_abstract(-sp.Rational(1, 4), 1)

    even_keys = (
        ((), 0, ((), (), 0)),
        ((), 1, ((), (), 1)),
    )
    odd_keys = (
        ((), 1, ((), (), 0)),
        ((), 0, ((), (), 1)),
    )
    even = sp.Matrix(
        [[plus_even.get(key, 0), minus_even.get(key, 0)] for key in even_keys]
    )
    odd = sp.Matrix(
        [[plus_odd.get(key, 0), minus_odd.get(key, 0)] for key in odd_keys]
    )

    expected_even = sp.Matrix([[1, 1], [-T, T]])
    expected_odd = sp.Matrix([[1, 1], [T, -T]]) / SQRT2
    assert sp.simplify(even - expected_even) == sp.zeros(2)
    assert sp.simplify(odd - expected_odd) == sp.zeros(2)
    return even, odd


def _apply_main_chi(expression, mode, realization):
    """Apply chi_mode=auxiliary_mode-i*physical_mode to a Fock expression."""

    answer = {}
    for state, outer in expression.items():
        aux_modes, aux_ground, bosons, phys_modes, phys_ground = state

        aux_final, aux_coefficient = branch.apply_auxiliary(
            mode, (aux_modes, aux_ground)
        )
        if aux_coefficient:
            final = (
                aux_final[0],
                aux_final[1],
                bosons,
                phys_modes,
                phys_ground,
            )
            branch.add_term(answer, final, outer * aux_coefficient)

        phys_final, phys_coefficient = branch.apply_fermion(
            mode,
            (bosons, phys_modes, phys_ground),
            realization,
        )
        if phys_coefficient:
            auxiliary_parity = (len(aux_modes) + aux_ground) % 2
            final = (
                aux_modes,
                aux_ground,
                phys_final[0],
                phys_final[1],
                phys_final[2],
            )
            branch.add_term(
                answer,
                final,
                outer * (-I) * (-1) ** auxiliary_parity * phys_coefficient,
            )
    return answer


def verify_three_quarter_mode_swap():
    """Prove chi_-1 W_(s/4)^epsilon=-W_(3s/4)^(1-epsilon)."""

    for sheet in (1, -1):
        for parity in (0, 1):
            realization, _, ground = branch.expand_chi_string(
                sheet * sp.Rational(1, 4), parity
            )
            _, _, target = branch.expand_chi_string(
                sheet * sp.Rational(3, 4), 1 - parity
            )
            calculated = _apply_main_chi(ground, -1, realization)
            for state in set(calculated) | set(target):
                residual = sp.simplify(
                    calculated.get(state, 0) + target.get(state, 0)
                )
                if residual != 0:
                    raise AssertionError((sheet, parity, state, residual))


def local_coefficients(mode_count, sheet):
    """Coefficients of the two parity bilinears in Phi^(sheet*k).

    At mode_count=0, k=1/2 and n=1/4.  At mode_count=1, k=3/2 and
    n=3/4.  The recursion follows directly from the odd tensor action:

      c'_(1-epsilon)=(-1)^epsilon c_epsilon.

    The two minus signs in chi_-1 W=-W cancel between left and right; the
    displayed sign is the Koszul sign from moving the right odd operator
    past the left state.
    """

    if sheet not in (1, -1):
        raise ValueError("sheet must be +1 or -1")
    coefficients = sp.Matrix([-T, 2 / T])
    if sheet == -1:
        coefficients = -coefficients
    swap = sp.Matrix([[0, -1], [1, 0]])
    for _ in range(mode_count):
        coefficients = sp.simplify(swap * coefficients)
    return coefficients


def local_sheet_matrix(mode_count, parity):
    """2 by 2 map from sheet bilinears to local-field components.

    With B_s^epsilon=W_(s*n)^epsilon bar(W_(s*n)^epsilon),

      (Phi_(epsilon)^(+k), Phi_(epsilon)^(-k))^T
          = D_(mode_count,epsilon) (B_+^epsilon,B_-^epsilon)^T.

    Summing epsilon=0,1 gives the complete local fields.
    """

    positive = local_coefficients(mode_count, 1)[parity]
    negative = local_coefficients(mode_count, -1)[parity]
    return sp.diag(positive, negative)


def hard_polynomials():
    """Return K, H and the two-state crossed kernel in Q variables."""

    q, p1, p2, p3 = sp.symbols("Q P1 P2 P3")
    x_plus = q / 2 + p1 + p2 + p3
    x_reflected = q / 2 - p1 + p2 + p3
    x_minus = q / 2 + p1 - p2 - p3
    quadratic = lambda x: x**2 + q * x + 1

    product = sp.expand(quadratic(x_plus) * quadratic(x_reflected))
    crossed_line = sp.expand(x_plus * (x_minus - q))
    even2 = q + 2 * p2
    even3 = q + 2 * p3
    odd2 = sp.expand(quadratic(even2))
    odd3 = sp.expand(quadratic(even3))

    leg2 = sp.Matrix([[odd2, even2], [even2, 1]])
    leg3 = sp.Matrix([[odd3, even3], [even3, 1]])
    hadamard = leg2.multiply_elementwise(leg3)
    universal_flip = sp.Matrix([[0, 1], [1, 0]])
    kernel = hadamard + universal_flip
    boundary = sp.Matrix([1, crossed_line])
    hard = sp.expand((boundary.T * kernel * boundary)[0])
    return (q, p1, p2, p3), product, hard, kernel, leg2, leg3, boundary


def hard_master_matrix():
    """The four denominator-cleared hard masters as a 2 by 2 matrix."""

    variables, product, hard, _, _, _, _ = hard_polynomials()
    parity_from_holonomy = sp.Matrix(
        [[1, 1], [I * SQRT2, -I * SQRT2]]
    )
    local_phases = sp.diag(-(1 + I), -(1 - I))
    components = sp.diag(product, hard)
    masters = sp.simplify(
        FOURTH_ROOT_TWO * parity_from_holonomy * local_phases * components
    )
    return variables, masters, parity_from_holonomy, components


def hard_raw_and_chamber_matrices(parity):
    """Return the raw eta vector and its exact A/B Hadamard matrix.

    The hard raw masters (before clearing the two Ramond leg factors) are

      R_e^+ = -a_e K/(d_2 d_3),
      R_e^- = -b_e H/(d_2 d_3),

    where

      a_e=(1+i)(i sqrt(2))^e,
      b_e=(1-i)(-i sqrt(2))^e.

    If U=2^(-1/2)[[1,1],[1,-1]], the symmetric A/B component matrix is
    U diag(R_e^+,R_e^-) U^T.  This is the normalized form of the exact
    two-support relation S_e=(-1)^(2e) S_A+S_B in arXiv:1510.01773,
    eq. (2.9), with a harmless sign choice for the second component.
    """

    if parity not in (0, 1):
        raise ValueError("parity must be zero or one")
    _, product, hard, _, leg2, leg3, _ = hard_polynomials()
    denominator = sp.expand(leg2[0, 0] * leg3[0, 0])
    a = (1 + I) * (I * SQRT2) ** parity
    b = (1 - I) * (-I * SQRT2) ** parity
    eta_diagonal = -sp.diag(a * product, b * hard) / denominator
    hadamard = sp.Matrix([[1, 1], [1, -1]]) / SQRT2
    chamber = sp.simplify(hadamard * eta_diagonal * hadamard.T)
    expected = -sp.Matrix(
        [
            [a * product + b * hard, a * product - b * hard],
            [a * product - b * hard, a * product + b * hard],
        ]
    ) / (2 * denominator)
    assert sp.simplify(chamber - expected) == sp.zeros(2)
    return eta_diagonal, chamber, hadamard


def spin_basis_channel_matrix(
    j1,
    j13_second,
    j3,
    external_epsilon1,
    external_epsilon2,
    external_epsilon3,
):
    """Exact momentum-dependent channel matrix from arXiv:1510.01773 (2.11).

    The invariant label is epsilon=0,1/2.  In the ordered chamber basis
    (g^31,g^13), the two spin-basis matrix elements are

      (g^0,g^(1/2))^T = diag(a_0,a_(1/2)) [[1,1],[1,-1]]
                         (g^31,g^13)^T.

    This function is recorded separately from the hard Ward identity: the
    paper proves the two-channel invariant algebra but does not identify its
    A/B chambers with geometric Ramond holonomies.
    """

    common = (
        sp.sin(sp.pi * (j1 - external_epsilon1))
        * sp.sin(sp.pi * (j3 + external_epsilon3))
    )
    a_zero = common * sp.sin(
        sp.pi * (j13_second / 2 + external_epsilon2)
    )
    a_half = common * sp.sin(
        sp.pi * (j13_second / 2 - sp.Rational(1, 2) + external_epsilon2)
    )
    return sp.diag(a_zero, a_half) * sp.Matrix([[1, 1], [1, -1]])


def translated_noncommon_spin_matrix():
    """The non-common part of (2.11) in the present momentum labels."""

    q, p1, p2, p3, epsilon2 = sp.symbols("Q P1 P2 P3 epsilon2")
    theta = -sp.Rational(1, 4) + (p1 + p3 - p2) / (2 * q) + epsilon2 / 2
    matrix = sp.diag(
        sp.sin(sp.pi * theta),
        sp.sin(sp.pi * (theta - sp.Rational(1, 2))),
    ) * sp.Matrix([[1, 1], [1, -1]])
    return (q, p1, p2, p3, epsilon2), matrix


def verify_hard_identities(run_ward=False):
    """Check H/K, component projection, and optionally the direct Ward sum."""

    variables, product, hard, kernel, leg2, leg3, boundary = hard_polynomials()
    q, p1, p2, p3 = variables

    # Expanded form of the hard crossed polynomial.
    crossed_line = boundary[1]
    even2 = q + 2 * p2
    even3 = q + 2 * p3
    odd2 = leg2[0, 0]
    odd3 = leg3[0, 0]
    expected_hard = sp.expand(
        crossed_line**2
        + 2 * crossed_line * (1 + even2 * even3)
        + odd2 * odd3
    )
    assert sp.expand(hard - expected_hard) == 0
    assert kernel == leg2.multiply_elementwise(leg3) + sp.Matrix([[0, 1], [1, 0]])

    # Compare with the independent certificate polynomial.
    import certify_master_ell_ansatz as certificate  # noqa: WPS433,E402

    certificate_variables, certificate_product, certificate_hard = (
        certificate.hard_polynomials()
    )
    substitution = dict(zip(certificate_variables, variables))
    assert sp.expand(certificate_product.subs(substitution) - product) == 0
    assert sp.expand(certificate_hard.subs(substitution) - hard) == 0

    _, masters, parity_from_holonomy, components = hard_master_matrix()
    projected = sp.simplify(
        parity_from_holonomy.inv() * masters / FOURTH_ROOT_TWO
    )
    expected_projected = sp.diag(-(1 + I) * product, -(1 - I) * hard)
    assert sp.simplify(projected - expected_projected) == sp.zeros(2)
    assert components == sp.diag(product, hard)

    # The chamber rotation is exact, but its entries are the known
    # irreducible combinations H+/-iK rather than new ell products.
    for parity in (0, 1):
        eta_diagonal, chamber, hadamard = hard_raw_and_chamber_matrices(parity)
        assert sp.simplify(
            hadamard.T * chamber * hadamard - eta_diagonal
        ) == sp.zeros(2)

    if run_ward:
        import fit_signed_sectors as fit  # noqa: WPS433,E402

        b_value = sp.Rational(3, 2)
        substitutions = {q: b_value + 1 / b_value}
        labels = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))
        for row, epsilon in enumerate((0, 1)):
            for column, eta in enumerate((1, -1)):
                direct = fit.scaled_raw(
                    labels,
                    (epsilon, 0, 0, eta),
                    (b_value, p1, p2, p3),
                )
                residual = sp.factor(
                    sp.cancel(direct - masters[row, column].subs(substitutions))
                )
                if residual != 0:
                    raise AssertionError((epsilon, eta, residual))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ward",
        action="store_true",
        help="also run the state/Ward evaluator at symbolic momenta",
    )
    arguments = parser.parse_args()

    even, odd = ground_sheet_matrices()
    verify_three_quarter_mode_swap()
    verify_hard_identities(arguments.ward)

    print("sheet columns: (+n,-n)")
    print("C_0 rows (u+ w+,u- w-):")
    print(even)
    print("C_0^(-1):")
    print(sp.simplify(even.inv()))
    print("C_1 rows (u- w+,u+ w-):")
    print(odd)
    print("C_1^(-1):")
    print(sp.simplify(odd.inv()))
    print("chi_-1 W_(s/4)^epsilon = -W_(3s/4)^(1-epsilon): exact")

    for mode_count, label in ((0, "1/2"), (1, "3/2")):
        print(f"Phi^(+/-{label}) parity coefficient matrices:")
        for parity in (0, 1):
            print(f"  epsilon={parity}: {local_sheet_matrix(mode_count, parity)}")

    _, _, _, kernel, leg2, leg3, boundary = hard_polynomials()
    _, masters, projection, _ = hard_master_matrix()
    print("hard parity-from-holonomy matrix V:")
    print(projection)
    print("hard holonomy projectors V^(-1):")
    print(sp.simplify(projection.inv()))
    print("hard crossed kernel K_23=M_2 Hadamard M_3+sigma_x:")
    print(kernel)
    print("boundary vector (1,L):")
    print(boundary.T)
    print("H=(1,L) K_23 (1,L)^T: exact")
    _, spin_matrix = translated_noncommon_spin_matrix()
    print("non-common spin-basis A/B matrix from arXiv:1510.01773 (2.11):")
    print(spin_matrix)
    for parity in (0, 1):
        _, _, hadamard = hard_raw_and_chamber_matrices(parity)
        a = sp.simplify((1 + I) * (I * SQRT2) ** parity)
        b = sp.simplify((1 - I) * (-I * SQRT2) ** parity)
        print(f"hard normalized A/B rotation U (epsilon={parity}):")
        print(hadamard)
        print(
            "raw A/B matrix = -1/(2*d2*d3) "
            f"[[({a})*K+({b})*H, ({a})*K-({b})*H], symmetric]"
        )
    print("all four hard master entries: exact" + (" direct Ward check" if arguments.ward else ""))


if __name__ == "__main__":
    main()
