#!/usr/bin/env python3
"""Exact two-channel certificate for the first non-factorized Ramond master.

This module constructs the hard ``(0,3/4,3/4)`` numerator from finite
Coulomb data *before* importing the independently stored expanded answer.
Put

    E_j = Q + 2 P_j,
    d_j = 2^(-1/8) ell(E_j,3) = E_j^2 + Q E_j + 1.

The two boundary states on one Ramond leg have the fusion-polynomial matrix

    M_j = [[d_j, E_j], [E_j, 1]].

The two-leg kernel is the entrywise product ``M_2 o M_3`` together with the
universal zero-mode exchange ``sigma_x``.  Contracting it with the two
boundary amplitudes ``(1,L)`` gives

    H_C = (1,L) (M_2 o M_3 + sigma_x) (1,L)^T.

No occurrence of the known polynomial ``H`` enters this construction.  The
audit imports the old expanded polynomial only afterwards and proves that
``H_C-H`` vanishes identically.  The other charge channel is the ordinary
factorized product ``K``.  The exact two-by-two Fourier map then reconstructs
both fixed-eta hard masters, including every phase.

Scope matters.  This is a finite hard-channel certificate for the proposed
two-state Coulomb kernel.  The repository does not yet contain an
independent arbitrary-mode implementation of the complementary Ramond
Coulomb vertex.  In particular this file does not promote the level-zero
Majorana ground change to excited states and does not claim an all-level
branching formula.
"""

from __future__ import annotations

from pathlib import Path
import sys

import sympy as sp


I = sp.I
SQRT2 = sp.sqrt(2)
Q, P1, P2, P3 = sp.symbols("Q P1 P2 P3")
NORMALIZED_ZERO_MODE = sp.Matrix(((0, 1), (1, 0)))


def stripped_ell3(argument):
    """Return ``2^(-1/8) ell(argument,3)`` in the notes' convention."""

    argument = sp.sympify(argument)
    return sp.expand(argument**2 + Q * argument + 1)


def one_leg_coulomb_matrix(momentum):
    """The two boundary-channel fusion-polynomial matrix on one R leg.

    Its entries, from upper left to lower right, are the stripped
    ``ell(E,3)``, ``ell(E,2)``, and ``ell(E,1)`` factors.  Thus this function
    uses one-leg Coulomb data only; it does not know the hard polynomial.
    """

    even = sp.expand(Q + 2 * sp.sympify(momentum))
    odd = stripped_ell3(even)
    return sp.Matrix(((odd, even), (even, sp.Integer(1))))


def zero_mode_exchange():
    """The universal exchange of the two Ramond boundary channels.

    In the ground doublet ``psi_0=2^(-1/2) X``.  The conventional
    ``sqrt(2)`` attached to the bordered odd functional therefore leaves
    precisely the normalized Clifford generator ``X``.  This datum is
    momentum independent and is fixed before the two one-leg matrices are
    combined.
    """

    return NORMALIZED_ZERO_MODE.copy()


def crossed_coulomb_kernel():
    """Combine the two one-leg matrices and the zero-mode exchange."""

    second = one_leg_coulomb_matrix(P2)
    third = one_leg_coulomb_matrix(P3)
    return second.multiply_elementwise(third) + zero_mode_exchange()


def boundary_vector():
    """The two hard-channel boundary amplitudes ``(1,L)^T``."""

    x_plus_plus = Q / 2 + P1 + P2 + P3
    x_minus_minus = Q / 2 + P1 - P2 - P3
    line = sp.expand(x_plus_plus * (x_minus_minus - Q))
    return sp.Matrix((1, line))


def charge_channel_numerators():
    """Return the independently constructed ordinary and crossed channels.

    ``K`` is the ordinary scalar Coulomb product.  ``H_C`` is obtained from
    the two-state kernel.  The symbol or stored expression ``H`` is not an
    input to either calculation.
    """

    x_plus_plus = Q / 2 + P1 + P2 + P3
    x_minus_minus = Q / 2 + P1 - P2 - P3
    ordinary = sp.expand(
        stripped_ell3(x_plus_plus) * stripped_ell3(Q - x_minus_minus)
    )
    boundary = boundary_vector()
    crossed = sp.expand(
        (boundary.T * crossed_coulomb_kernel() * boundary)[0]
    )
    return ordinary, crossed


def canonical_chart_values():
    """Return the two phase-resolved charge-chart values for ``f=0``.

    The common Ramond leg denominator is ``d_2 d_3``.  These are the exact
    inverse Fourier transform of the ordinary/crossed eigenchannels:

        C   = -(K+H_C)/(d_2 d_3),
        C_Z =  i(H_C-K)/(d_2 d_3).

    The name ``C_Z`` denotes the second charge-chart component.  It must not
    be confused with multiplying an arbitrary excited Majorana Pfaffian by
    the endpoint matrix ``Z``; those operations agree only at level zero.
    """

    ordinary, crossed = charge_channel_numerators()
    second = one_leg_coulomb_matrix(P2)[0, 0]
    third = one_leg_coulomb_matrix(P3)[0, 0]
    denominator = sp.expand(second * third)
    canonical = sp.factor(sp.cancel(-(ordinary + crossed) / denominator))
    with_z = sp.factor(sp.cancel(I * (crossed - ordinary) / denominator))
    return canonical, with_z


def eta_resolution_matrix(form_parity):
    """Fixed hard-channel map from ``(C,C_Z)`` to ``(R_f^+,R_f^-)``."""

    form_parity = int(form_parity)
    if form_parity == 0:
        return sp.Matrix(
            (
                ((1 + I) / 2, (1 - I) / 2),
                ((1 - I) / 2, (1 + I) / 2),
            )
        )
    if form_parity == 1:
        fock_minus = -(1 - I) / SQRT2
        return sp.Matrix(
            (
                (fock_minus * (1 - I) / 2, -fock_minus * (1 + I) / 2),
                (fock_minus * (1 + I) / 2, -fock_minus * (1 - I) / 2),
            )
        )
    raise ValueError("form_parity must be 0 or 1")


def hard_masters(form_parity=0):
    """Reconstruct ``(R_f^+,R_f^-)`` from the two charge components."""

    form_parity = int(form_parity)
    values = sp.Matrix(canonical_chart_values())
    # The odd NS form uses J rather than K as its canonical ground matrix.
    # In the SCblock/Fock frame its two hard charge components are both
    # (1+i) times the even-form components.  This is the fixed zero-mode
    # conversion; all momentum dependence remains in the two chart values.
    if form_parity == 1:
        values *= 1 + I
    return sp.simplify(eta_resolution_matrix(form_parity) * values)


def _reference_polynomials():
    """Load the independently stored expanded Ward/state certificate."""

    grid_directory = Path(__file__).resolve().parents[2] / "ramond_three_point_grid"
    if str(grid_directory) not in sys.path:
        sys.path.insert(0, str(grid_directory))
    import certify_master_ell_ansatz as reference  # noqa: E402,WPS433

    variables, ordinary, crossed = reference.hard_polynomials()
    substitution = dict(zip(variables, (Q, P1, P2, P3)))
    return (
        sp.expand(ordinary.subs(substitution, simultaneous=True)),
        sp.expand(crossed.subs(substitution, simultaneous=True)),
    )


def audit():
    """Run exact symbolic checks and print a compact certificate report."""

    ordinary, crossed = charge_channel_numerators()

    # Check the one-leg entries directly against the defining screening
    # products.  This occurs before the independent hard polynomial is
    # loaded below.
    b, momentum = sp.symbols("b momentum", nonzero=True)
    q_substitution = b + 1 / b
    even = Q + 2 * momentum
    direct_ell3 = (even + b) * (even + 1 / b)
    if sp.factor(
        one_leg_coulomb_matrix(momentum)[0, 0].subs(Q, q_substitution)
        - direct_ell3.subs(Q, q_substitution)
    ) != 0:
        raise AssertionError("the stripped ell(E,3) leg entry failed")
    if zero_mode_exchange() ** 2 != sp.eye(2):
        raise AssertionError("the normalized Ramond zero mode failed X^2=1")

    reference_ordinary, reference_crossed = _reference_polynomials()
    if sp.expand(ordinary - reference_ordinary) != 0:
        raise AssertionError("ordinary K channel disagrees with the reference")
    if sp.expand(crossed - reference_crossed) != 0:
        raise AssertionError("crossed H channel disagrees with the reference")

    d2 = one_leg_coulomb_matrix(P2)[0, 0]
    d3 = one_leg_coulomb_matrix(P3)[0, 0]
    denominator = sp.expand(d2 * d3)
    expected_zero = sp.Matrix(
        (
            -(1 + I) * ordinary / denominator,
            -(1 - I) * crossed / denominator,
        )
    )
    zero = hard_masters(0)
    if any(sp.factor(sp.cancel(value)) != 0 for value in zero - expected_zero):
        raise AssertionError("f=0 phase reconstruction failed")

    # The f=1 hard masters obey R_1^eta=i sqrt(2) eta R_0^eta.
    expected_one = sp.diag(I * SQRT2, -I * SQRT2) * zero
    one = hard_masters(1)
    if any(sp.factor(sp.cancel(value)) != 0 for value in one - expected_one):
        raise AssertionError("f=1 phase reconstruction failed")

    # On the zero-screening ordinary plane x_{++}=0, the crossed channel
    # reduces to the two Ramond leg factors.  The factorized channel is a
    # different Coulomb representative and does not do so; conflating the
    # two is exactly the same-plane error guarded against by this check.
    plane = {P1: -Q / 2 - P2 - P3}
    if sp.expand(crossed.subs(plane) - denominator) != 0:
        raise AssertionError("crossed zero-screening normalization failed")
    if sp.expand(ordinary.subs(plane) - denominator) == 0:
        raise AssertionError("the two distinct charge channels collapsed")

    if not sp.Poly(16 * crossed, Q, P1, P2, P3).is_irreducible:
        raise AssertionError("the crossed hard numerator unexpectedly factorized")

    print("hard two-channel construction: symbolic K residual = 0")
    print("hard two-channel construction: symbolic H residual = 0")
    print("fixed-eta reconstruction: f=0 and f=1 phases exact")
    print("x_{++}=0 plane: H=d_2*d_3 and K differs generically")
    print("crossed H is irreducible over Q")
    print("scope: finite fusion-polynomial kernel; generic complementary vertex unproved")


if __name__ == "__main__":
    audit()
