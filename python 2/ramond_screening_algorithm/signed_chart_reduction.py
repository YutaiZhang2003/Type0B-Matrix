#!/usr/bin/env python3
"""Raw endpoint-sign bookkeeping for consecutive Ramond chi strings.

For the formal binary path expansion, and only before the free-field state
is transported to the abstract SCA module, the coefficients obey

    coefficient[-n,P; path]
      = (-1)**physical_ground coefficient[+n,-P; path].

The matrix ``Z=diag(1,-1)`` records this ground-label sign.  It is not the
2013 reflection operator.  Starting at ``n=3/4`` the two appropriate
free-field-to-SCA transition matrices turn the raw identity into nonzero
``L_-1`` residuals.  The reflected fermion mixes bosonic and fermionic
oscillators, exactly as stated in arXiv:1312.4520.

Consequently the helpers here may label paths or ground sectors, but they
must not be used as a negative-branch SCA transform or a signed Coulomb-value
callback.  See ``reflection.audit_signed_state_obstruction``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import sympy as sp


PYTHON_DIRECTORY = Path(__file__).resolve().parents[1]
CHI_DIRECTORY = PYTHON_DIRECTORY / "nsrr_chi_branching"
if str(CHI_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CHI_DIRECTORY))

import nsrr_chi_formula as chi  # noqa: E402


Z = sp.diag(1, -1)


@dataclass(frozen=True)
class SignedRamondChart:
    """Raw-path chart data for one nonzero Ramond branch label.

    ``magnitude`` is the positive branch label used for the consecutive
    ``chi^-`` string.  ``momentum_sign`` multiplies the original momentum.
    ``ground_twist`` is zero or one and says whether ``Z`` acts on that
    path boundary index.  These fields do not define an SCA reflection.
    """

    magnitude: sp.Rational
    momentum_sign: int
    ground_twist: int


def signed_ramond_chart(branch_label) -> SignedRamondChart:
    """Return the all-positive raw-path chart for ``branch_label``."""

    branch_label = sp.Rational(branch_label)
    if branch_label == 0:
        raise ValueError("There is no Ramond branch at zero.")
    sign = 1 if branch_label > 0 else -1
    return SignedRamondChart(
        magnitude=abs(branch_label),
        momentum_sign=sign,
        ground_twist=int(sign < 0),
    )


def signed_ns_chart(branch_label):
    """Return ``(|n|,sgn(n))`` using ``w_-n(P)=w_n(-P)``.

    At zero the momentum sign is chosen to be one.
    """

    branch_label = sp.Rational(branch_label)
    sign = -1 if branch_label < 0 else 1
    return abs(branch_label), sign


def twist_ground_matrix(matrix, second_negative=False, third_negative=False):
    """Apply raw endpoint signs to a 2 by 2 Ramond ground matrix.

    Rows are the second-leg ground index and columns are the third-leg
    ground index.  This is an exact path/ground operation, not the action of
    the reflected SCA vertex on excited states.
    """

    matrix = sp.Matrix(matrix)
    if matrix.shape != (2, 2):
        raise ValueError("the Ramond ground matrix must be 2 by 2")
    if second_negative:
        matrix = Z * matrix
    if third_negative:
        matrix = matrix * Z
    return matrix


def endpoint_twist(state):
    """The coefficient of ``Z_0`` on a free-Fock endpoint."""

    physical_ground = int(state[3])
    return -1 if physical_ground else 1


def audit(maximum_quarters=7):
    """Check the raw sign rule on several finite consecutive strings.

    This deliberately compares the raw free-Fock paths, before any SCA PBW
    conversion or Ward contraction.  It therefore tests only path
    bookkeeping and nothing about the reflected abstract state.
    """

    checked = 0
    for numerator in range(1, int(maximum_quarters) + 1, 2):
        level = sp.Rational(numerator, 4)
        for epsilon in (0, 1):
            negative = dict(chi.ramond_fock_paths(-level, epsilon))
            positive = dict(chi.ramond_fock_paths(level, epsilon))
            if set(negative) != set(positive):
                raise AssertionError((level, epsilon, "endpoint support"))
            for state, coefficient in negative.items():
                residual = sp.simplify(
                    coefficient - endpoint_twist(state) * positive[state]
                )
                if residual != 0:
                    raise AssertionError((level, epsilon, state, residual))
                checked += 1

    gamma = sp.Matrix([[sp.Symbol("a"), sp.Symbol("b")],
                       [sp.Symbol("c"), sp.Symbol("d")]])
    expected = sp.Matrix([[sp.Symbol("a"), -sp.Symbol("b")],
                          [-sp.Symbol("c"), sp.Symbol("d")]])
    if twist_ground_matrix(gamma, True, True) != expected:
        raise AssertionError("two-boundary Z action failed")

    print(
        "raw signed Ramond paths: "
        f"{checked} exact free-Fock endpoint coefficients passed"
    )
    print(
        "scope: endpoint coefficients only; abstract SCA reflection and "
        "signed Coulomb values still require the reflection operator"
    )


if __name__ == "__main__":
    audit()
