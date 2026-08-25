#!/usr/bin/env python3
"""Exact audit of the two canonical NS--R--R Majorana functionals.

This file does not construct a super-Virasoro state.  It checks the cubic
Pfaffian implementation against the literal Wick expansion and verifies the
two-by-two ground change used to recover fixed SCblock ``eta`` structures.
"""

from __future__ import annotations

import itertools
import time

import sympy as sp

from ..pfaffian.native_spin_kernel import (
    canonical_ising_value,
    check_ground_change,
    resolved_ground_value,
    scblock_fock_ground_matrix,
    z_boundary_value,
)
from ..pfaffian.audit_ground_covariance import grid


def audit():
    check_ground_change()
    ground_checks = 0
    for form_parity, eta, ground2, ground3 in itertools.product(
        (0, 1), (1, -1), (0, 1), (0, 1)
    ):
        calculated = resolved_ground_value(
            form_parity, eta, ground2, ground3
        )
        expected = scblock_fock_ground_matrix(
            form_parity, eta
        )[ground2, ground3]
        if sp.simplify(calculated - expected) != 0:
            raise AssertionError(
                (form_parity, eta, ground2, ground3, calculated, expected)
            )
        ground_checks += 1

    first_strings = (
        (),
        (sp.Rational(1, 2),),
        (sp.Rational(3, 2),),
        (sp.Rational(3, 2), sp.Rational(1, 2)),
    )
    # Every auxiliary path in the complete W_(7/4) chi string is one subset
    # of (3,2,1), not just its strict endpoint.  Include all eight subsets
    # on both Ramond legs.
    ramond_strings = tuple(
        subset
        for count in range(4)
        for subset in itertools.combinations(
            (sp.Integer(3), sp.Integer(2), sp.Integer(1)), count
        )
    )
    pfaffian_checks = 0
    signed_checks = 0
    for form_parity, first, second, ground2, third, ground3 in itertools.product(
        (0, 1), first_strings, ramond_strings, (0, 1), ramond_strings, (0, 1)
    ):
        calculated = canonical_ising_value(
            form_parity, first, second, ground2, third, ground3
        )
        expected = grid.fermion_value(
            form_parity, first, second, ground2, third, ground3
        )
        if sp.simplify(calculated - expected) != 0:
            raise AssertionError(
                (form_parity, first, second, ground2, third, ground3)
            )
        pfaffian_checks += 1

        signed = z_boundary_value(
            form_parity, first, second, ground2, third, ground3
        )
        if sp.simplify(signed - (-1) ** ground3 * expected) != 0:
            raise AssertionError(
                ("Z", form_parity, first, second, ground2, third, ground3)
            )
        signed_checks += 1

    # The complete W_(7/4) endpoint contains the strict Ramond string
    # (3,2,1).  Time a representative eight-field evaluation to make the
    # absence of a binary path expansion explicit.
    start = time.perf_counter()
    benchmark = canonical_ising_value(
        0,
        (sp.Rational(3, 2), sp.Rational(1, 2)),
        (3, 2, 1),
        0,
        (3, 2, 1),
        0,
    )
    elapsed = time.perf_counter() - start
    print(f"two-chart ground change: {ground_checks} exact entries")
    print(
        f"canonical/boundary-Z kernels: {pfaffian_checks}+{signed_checks} "
        "exact mode-string checks"
    )
    print(
        "eight-field W_7/4-scale Pfaffian: "
        f"{elapsed:.3f}s, value={benchmark}"
    )


if __name__ == "__main__":
    audit()
