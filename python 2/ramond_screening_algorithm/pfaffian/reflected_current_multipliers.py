"""Exact first reflected-current multiplier in the repository spin frame.

After the level-one reflection block has been applied, a physical
``psi_-1`` endpoint contains a ``c_-1`` endpoint.  The normalized third-leg
current form is the scalar below.  It is written before imposing a Coulomb
neutrality condition so that the two charge choices can be compared
without changing conventions.

This module contains no SCA state, PBW transition, or Ward evaluator.  The
remaining hard crossed obstruction is not in this scalar: on the
complementary charge plane the current terms cancel in the complete chi
sum, while the physical cross-leg fermion covariance still differs from
the ordinary two-spin kernel.
"""

from __future__ import annotations

import sympy as sp


I = sp.I


def third_current_multiplier(q, p1, p2, p3, eta):
    """Return the normalized form with ``c_-1`` on the third Ramond leg.

    The momenta are the actual momenta of the displayed NS--R--R form.  In
    particular, if the third branch is displayed on a reflected sheet,
    ``p3`` is already its reflected momentum.  ``eta`` is ``+1`` or ``-1``.
    """

    eta = int(eta)
    if eta not in (-1, 1):
        raise ValueError("eta must be +1 or -1")
    denominator = 4 * p3**2 + 6 * p3 * q + 2 * q**2 + 1
    numerator = (
        (p3 + q) * (4 * p1**2 - 4 * p2**2 - 4 * p3**2 + q**2)
        + q
        - 2 * eta * p2
    )
    return sp.factor(I * numerator / (2 * denominator))


def ordinary_charge_multiplier(q, p2):
    """Third-current multiplier for the ``P2`` Coulomb charge choice."""

    return I * (q / 2 + p2)


def complementary_charge_multiplier(q, p2):
    """Third-current multiplier for the complementary ``-P2`` charge."""

    return I * (q / 2 - p2)


def audit_charge_planes():
    """Prove the two exact neutrality-plane reductions symbolically."""

    q, p2, p3 = sp.symbols("Q P_2 P_3")
    ordinary_p1 = -q / 2 - p2 - p3
    complementary_p1 = -q / 2 + p2 - p3
    ordinary_residual = sp.factor(
        third_current_multiplier(q, ordinary_p1, p2, p3, -1)
        - ordinary_charge_multiplier(q, p2)
    )
    complementary_residual = sp.factor(
        third_current_multiplier(q, complementary_p1, p2, p3, 1)
        - complementary_charge_multiplier(q, p2)
    )
    if ordinary_residual != 0 or complementary_residual != 0:
        raise AssertionError((ordinary_residual, complementary_residual))
    print("level-one current: ordinary and complementary charge planes exact")


if __name__ == "__main__":
    audit_charge_planes()

