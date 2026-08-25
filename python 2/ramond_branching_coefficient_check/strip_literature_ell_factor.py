#!/usr/bin/env python3
"""Strip the Hadasz--Jaskolski scalar screening/ell channel in the first hard R case.

In the NS screening-charge derivation of arXiv:1312.4520, the zero loci and
degree determine a four-ell polynomial, while a generalized Selberg contour
integral fixes the remaining momentum-independent constant.  This script
tests the literal scalar four-ell analogue against the chiral Ramond PBW
result at

    (n1,n2,n3) = (0,3/4,3/4).

The exact Ward calculation is re-run at two rational samples.  The symbolic
output then displays the residual after division by the ell channel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
GRID_DIR = HERE.parent / "ramond_three_point_grid"
for directory in (HERE, GRID_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import certify_master_ell_ansatz as certificate  # noqa: E402
import compute_ramond_kappa as ell_data  # noqa: E402


LABELS = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))


def symbolic_data():
    """Return the scalar channel K, crossed channel H, and leg factors."""

    variables, scalar, crossed = certificate.hard_polynomials()
    q_value, _, p2, p3 = variables
    e2 = q_value + 2 * p2
    e3 = q_value + 2 * p3
    a2 = (2 * p2) ** 2 + q_value * (2 * p2) + 1
    a3 = (2 * p3) ** 2 + q_value * (2 * p3) + 1
    d2 = e2**2 + q_value * e2 + 1
    d3 = e3**2 + q_value * e3 + 1
    denominator = sp.expand(a2 * a3 * d2 * d3)
    return variables, scalar, crossed, e2, e3, d2, d3, denominator


def check_scalar_ell_channel():
    """Check that the literal four-ell ratio reduces to K^2/leg factors."""

    variables, scalar, _, _, _, _, _, denominator = symbolic_data()
    q_value, p1, p2, p3 = variables
    for sample in certificate.WARD_SAMPLES:
        b_value, p1_value, p2_value, p3_value = sample
        substitutions = {
            q_value: b_value + 1 / b_value,
            p1: p1_value,
            p2: p2_value,
            p3: p3_value,
        }
        numerator = ell_data.numerator_product(
            *LABELS, p1_value, p2_value, p3_value, b_value
        )
        legs = sp.prod(
            ell_data.leg_product(momentum, label, b_value)
            for label, momentum in zip(
                LABELS, (p1_value, p2_value, p3_value)
            )
        )
        literal = sp.factor(sp.cancel(numerator**2 / legs))
        reduced = sp.factor(
            sp.cancel(
                scalar.subs(substitutions) ** 2
                / denominator.subs(substitutions)
            )
        )
        if sp.factor(sp.cancel(literal - reduced)) != 0:
            raise AssertionError((sample, literal, reduced))


def main():
    certificate.exact_gatekeeper_certificate(symbolic_ward=False)
    check_scalar_ell_channel()

    variables, scalar, crossed, e2, e3, d2, d3, _ = symbolic_data()
    q_value, p1, p2, p3 = variables
    x_pp = q_value / 2 + p1 + p2 + p3
    x_mm = q_value / 2 + p1 - p2 - p3
    boundary = x_pp * (x_mm - q_value)
    kernel = sp.Matrix(
        [[d2 * d3, e2 * e3 + 1], [e2 * e3 + 1, 1]]
    )
    kernel_form = sp.expand(
        (sp.Matrix([[1, boundary]]) * kernel * sp.Matrix([1, boundary]))[0]
    )
    if sp.expand(crossed - kernel_form) != 0:
        raise AssertionError("The two-state kernel representation failed.")

    print("Hadasz--Jaskolski four-ell square: K^2/(a2*a3*d2*d3), residual=0")
    print("stripped unsquared factors: U_plus=1, U_minus=H/K")
    print("stripped squared factors: U_plus^2=1, U_minus^2=(H/K)^2")
    print(
        "H=(1,L)*[[d2*d3,E2*E3+1],[E2*E3+1,1]]*(1,L)^T, "
        "L=x_pp*(x_mm-Q)"
    )


if __name__ == "__main__":
    main()
