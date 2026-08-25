#!/usr/bin/env python3
"""Independent exact audit of the genuine positive hard Coulomb evaluator.

The production module ``native_hard_screening`` contains no Ward evaluator
and imports no hard ``K`` or ``H`` polynomial.  This file alone compares its
literal chi/Pfaffian/Selberg values with the abstract SCA three-form after
both have been evaluated on the same charge-neutral plane.

The comparison is made at screening counts zero through three.  Even count
selects ``eta=-`` and odd count selects ``eta=+`` in the repository spin
frame.  The last value, ``N=3``, is the natural maximal-screening hard node.
"""

from __future__ import annotations

import sympy as sp

from python.ramond_three_point_grid import compute_grid as grid

from .native_hard_screening import (
    HARD,
    hard_neutrality_momentum,
    hard_screening_value,
)


SAMPLES = (
    (sp.Rational(3, 2), sp.Rational(2, 5), sp.Rational(3, 7)),
    (sp.Rational(4, 3), sp.Rational(1, 6), sp.Rational(2, 7)),
)


def audit():
    checked = 0
    for b, p2, p3 in SAMPLES:
        for screenings in range(4):
            eta = -1 if screenings % 2 == 0 else 1
            p1 = hard_neutrality_momentum(b, p2, p3, screenings)
            for form_parity in (0, 1):
                calculated = hard_screening_value(
                    screenings,
                    form_parity,
                    eta,
                    b,
                    p2,
                    p3,
                )
                expected = grid.enlarged_raw_three_point(
                    0,
                    HARD,
                    HARD,
                    0,
                    0,
                    form_parity,
                    eta,
                    b,
                    p1,
                    p2,
                    p3,
                )[1]
                residual = sp.factor(sp.cancel(calculated - expected))
                if residual != 0:
                    raise AssertionError(
                        (b, screenings, form_parity, eta, residual)
                    )
                checked += 1
    print(
        f"native positive hard screening: {checked} exact residuals zero; "
        "N=0,1,2,3 and both f"
    )
    print("natural hard node N=3: genuine chi/Pfaffian/Selberg callback passed")
    return checked


if __name__ == "__main__":
    audit()

