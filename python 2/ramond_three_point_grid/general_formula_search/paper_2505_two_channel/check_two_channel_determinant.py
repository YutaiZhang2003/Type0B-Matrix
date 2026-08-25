#!/usr/bin/env python3
"""Compare arXiv:2505.23122's RRN determinant with the hard Ramond master.

The paper's physical field is

    U_R = Theta^- R^+ + i Theta^+ R^- .

Because the NS-bottom RRN constants are diagonal in the Ramond signs, two
insertions give

    C_L^+ C_M^- - C_L^- C_M^+.

This file checks the resulting two-channel algebra and records the exact
determinant representation of the first crossed master.  It deliberately
does not identify the paper's N=1 matter theory with a free Majorana
fermion: those are different chiral theories.
"""

from __future__ import annotations

import sympy as sp


def paper_channel_identity():
    """Expand the paper's C^+ C^- - C^- C^+ combination exactly."""

    a_l, b_l, a_m, b_m = sp.symbols("A_L B_L A_M B_M")
    c_l_plus = (a_l + b_l) / 2
    c_l_minus = (a_l - b_l) / 2
    c_m_plus = (a_m + b_m) / 2
    c_m_minus = (a_m - b_m) / 2
    physical = sp.expand(c_l_plus * c_m_minus - c_l_minus * c_m_plus)
    expected = sp.expand((b_l * a_m - a_l * b_m) / 2)
    assert sp.expand(physical - expected) == 0
    return physical


def hard_data():
    q, p1, p2, p3 = sp.symbols("Q P_1 P_2 P_3")
    e2 = q + 2 * p2
    e3 = q + 2 * p3
    d2 = e2**2 + q * e2 + 1
    d3 = e3**2 + q * e3 + 1
    x_plus_plus = q / 2 + p1 + p2 + p3
    x_minus_minus = q / 2 + p1 - p2 - p3
    ell_line = sp.expand(x_plus_plus * (x_minus_minus - q))
    hard = sp.expand(
        ell_line**2 + 2 * ell_line * (1 + e2 * e3) + d2 * d3
    )
    return (q, p1, p2, p3), ell_line, e2, e3, d2, d3, hard


def hard_determinant_identity():
    """Check H=det([[L,d2],[-d3,L+2(1+E2 E3)]])."""

    variables, ell_line, e2, e3, d2, d3, hard = hard_data()
    channel_matrix = sp.Matrix(
        [[ell_line, d2], [-d3, ell_line + 2 * (1 + e2 * e3)]]
    )
    assert sp.expand(channel_matrix.det() - hard) == 0
    return variables, channel_matrix, hard


def hard_raw_phase_identity():
    """Restore the fixed conventions of R_0^(-) in the current notes."""

    _, _, _, _, d2, d3, hard = hard_data()
    raw = sp.cancel(-(1 - sp.I) * hard / (d2 * d3))
    matrix_raw = sp.cancel(
        -(1 - sp.I)
        * sp.Matrix([[hard, 0], [0, -sp.I * sp.sqrt(2) * hard]])
        / (d2 * d3)
    )
    assert sp.simplify(matrix_raw[0, 0] - raw) == 0
    assert sp.simplify(matrix_raw[1, 1] / raw + sp.I * sp.sqrt(2)) == 0
    return raw


def hard_chamber_projection():
    """Use the paper's antisymmetric pairing to recover the two chambers.

    Order each channel vector as (+,-), and put J=[[0,1],[-1,0]].
    The two normalized auxiliary vectors below yield the sum and difference
    of the two fixed-chiral masters.  This is exactly the Hadamard chamber
    matrix, with no complex conjugation.
    """

    _, _, _, _, d2, d3, hard = hard_data()
    q, p1, p2, p3 = hard_data()[0]
    x_plus = q / 2 + p1 + p2 + p3
    x_reflected = q / 2 - p1 + p2 + p3
    product = sp.expand(
        (x_plus**2 + q * x_plus + 1)
        * (x_reflected**2 + q * x_reflected + 1)
    )
    r_plus = -(1 + sp.I) * product / (d2 * d3)
    r_minus = -(1 - sp.I) * hard / (d2 * d3)
    ramond = sp.Matrix([r_plus, r_minus])
    symplectic = sp.Matrix([[0, 1], [-1, 0]])
    auxiliary_a = sp.Matrix([-1, 1]) / sp.sqrt(2)
    auxiliary_b = sp.Matrix([1, 1]) / sp.sqrt(2)
    projected = sp.Matrix(
        [
            (ramond.T * symplectic * auxiliary_a)[0],
            (ramond.T * symplectic * auxiliary_b)[0],
        ]
    )
    expected = sp.Matrix(
        [(r_plus + r_minus) / sp.sqrt(2),
         (r_plus - r_minus) / sp.sqrt(2)]
    )
    assert sp.simplify(projected - expected) == sp.zeros(2, 1)
    return projected


def main():
    paper = paper_channel_identity()
    _, channel_matrix, _ = hard_determinant_identity()
    hard_raw_phase_identity()
    hard_chamber_projection()
    print("2505.23122 two-channel contraction:", paper)
    print("hard channel matrix:")
    sp.pprint(channel_matrix)
    print("hard determinant residual=0")
    print("R_0^(-) and R_1^(-) phase residuals=0")
    print("antisymmetric-pairing chamber projection residual=0")
    print(
        "scope: the determinant algebra matches; the paper's N=1 matter "
        "factor is not the auxiliary Majorana or ordinary imaginary "
        "Liouville factor."
    )


if __name__ == "__main__":
    main()
