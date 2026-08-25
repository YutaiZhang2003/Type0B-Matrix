#!/usr/bin/env python3
"""Exact audit of the two Coulomb charts in the first hard channel.

The production calculation in :mod:`reflected_hard_oracle` contains no
super-Virasoro state and no PBW transition.  This audit alone imports the
stored Ward evaluator and compares

    (n_1,n_2,n_3)=(0,3/4,-3/4)

on the ordinary and complementary zero-screening charge planes.  Two
independent rational points, both Ramond copies on both legs, and both
form parities give 32 exact comparisons.

This is a glue/phase audit, not an independent prediction of the crossed
polynomial.  ``hard_complementary_pair_multiplier`` was obtained by
solving the known hard ``H`` polynomial for the missing mode-one
two-spin entry.  The audit checks that this one entry, together with the
direct reflection recurrence and Coulomb current, reproduces every copy
and parity component without any further fit.
"""

from __future__ import annotations

import sympy as sp

from .audit_ground_covariance import grid
from .reflected_hard_oracle import (
    charge_plane_p1,
    hard_mixed_sheet_value,
)


HARD = sp.Rational(3, 4)
SAMPLES = (
    (sp.Rational(3, 2), sp.Rational(2, 5), -sp.Rational(3, 7)),
    (sp.Rational(4, 3), sp.Rational(1, 6), -sp.Rational(2, 7)),
)


def audit() -> int:
    checked = 0
    for b_value, p2, p3 in SAMPLES:
        q_value = sp.cancel(b_value + 1 / b_value)
        for eta in (-1, 1):
            p1 = charge_plane_p1(q_value, p2, p3, eta)
            for epsilon2 in (0, 1):
                for epsilon3 in (0, 1):
                    for form_parity in (0, 1):
                        calculated = hard_mixed_sheet_value(
                            epsilon2,
                            epsilon3,
                            form_parity,
                            eta,
                            q_value,
                            p2,
                            p3,
                        )
                        expected = grid.enlarged_raw_three_point(
                            0,
                            HARD,
                            -HARD,
                            epsilon2,
                            epsilon3,
                            form_parity,
                            eta,
                            b_value,
                            p1,
                            p2,
                            p3,
                        )[1]
                        residual = sp.factor(sp.cancel(calculated - expected))
                        if residual != 0:
                            raise AssertionError(
                                (
                                    b_value,
                                    eta,
                                    epsilon2,
                                    epsilon3,
                                    form_parity,
                                    residual,
                                )
                            )
                        checked += 1
    print(
        f"hard Coulomb charts: {checked}/32 exact Ward values; "
        "complementary mode-one hook is H-calibrated"
    )
    return checked


if __name__ == "__main__":
    audit()
