#!/usr/bin/env python3
"""Exact check of the human-note auxiliary Majorana block.

The defining side is the Fock/Pfaffian sum in Eq. (6.16) of
``Human Notes/SCblock.tex``.  The resummed side is the four-term Fredholm
determinant combination obtained by expanding principal minors of the
three-slot sphere contraction kernel.  No Schottky multiplier, period
matrix, or literature BPZ convention enters this check.
"""

from __future__ import annotations

import argparse
from math import comb
from typing import Sequence

import sympy as sp

from python.free_majorana_pair_of_pants import (
    majorana_three_point,
    ns_fermion_states_at_twice_level,
)


Level = tuple[int, int, int]


def level_tuples(cutoff: int) -> tuple[Level, ...]:
    return tuple(
        (first, second, third)
        for first in range(cutoff + 1)
        for second in range(cutoff + 1 - first)
        for third in range(cutoff + 1 - first - second)
    )


def human_orientation_sign(levels: Sequence[int]) -> int:
    """The literal quadratic sign in Eq. (6.16), with no linear bit."""

    first, second, third = (int(level) % 2 for level in levels)
    exponent = first * second + first * third + second * third
    return -1 if exponent else 1


def direct_human_coefficients(cutoff: int) -> dict[Level, sp.Integer]:
    """Evaluate Eq. (6.16) directly in the NS Fock basis."""

    states = tuple(
        ns_fermion_states_at_twice_level(level) for level in range(cutoff + 1)
    )
    result: dict[Level, sp.Integer] = {}
    for levels in level_tuples(cutoff):
        coefficient = 0
        for first in states[levels[0]]:
            for second in states[levels[1]]:
                for third in states[levels[2]]:
                    rho = majorana_three_point(first, second, third)
                    coefficient += rho * rho
        result[levels] = sp.Integer(
            human_orientation_sign(levels) * coefficient
        )
    return result


def sphere_kernel(max_mode: int) -> tuple[sp.Matrix, tuple[tuple[int, int], ...]]:
    """Return the antisymmetric contraction kernel in slots (infinity,1,0)."""

    indices = tuple(
        (slot, mode) for slot in range(3) for mode in range(1, max_mode + 1)
    )

    def upper(left: tuple[int, int], right: tuple[int, int]) -> int:
        left_slot, left_mode = left
        right_slot, right_mode = right
        if left_slot == right_slot:
            return 0
        if left_slot > right_slot:
            return -upper(right, left)
        if (left_slot, right_slot) == (0, 1):
            return (
                comb(left_mode - 1, right_mode - 1)
                if left_mode >= right_mode
                else 0
            )
        if (left_slot, right_slot) == (0, 2):
            return int(left_mode == right_mode)
        if (left_slot, right_slot) == (1, 2):
            return (-1) ** (left_mode - 1) * comb(
                left_mode + right_mode - 2, left_mode - 1
            )
        raise AssertionError("unreachable slot pair")

    kernel = sp.Matrix(
        [[upper(left, right) for right in indices] for left in indices]
    )
    if kernel.T != -kernel:
        raise AssertionError("sphere contraction kernel is not antisymmetric")
    return kernel, indices


def resummed_coefficients(cutoff: int) -> dict[Level, sp.Integer]:
    """Expand the four Fredholm determinants through the requested degree."""

    variables = sp.symbols("x_1 x_2 x_3")
    max_mode = (cutoff + 1) // 2
    kernel, indices = sphere_kernel(max_mode)

    def delta(slot_signs: tuple[int, int, int]) -> sp.Expr:
        weights = sp.diag(
            *[
                slot_signs[slot] * variables[slot] ** (2 * mode - 1)
                for slot, mode in indices
            ]
        )
        return sp.expand(
            (sp.eye(len(indices)) + kernel * weights).det(method="domain-ge")
        )

    resummed = sp.expand(
        (
            -delta((1, 1, 1))
            + delta((-1, 1, 1))
            + delta((1, -1, 1))
            + delta((1, 1, -1))
        )
        / 2
    )
    polynomial = sp.Poly(resummed, *variables)
    result = {levels: sp.S.Zero for levels in level_tuples(cutoff)}
    for monomial, coefficient in polynomial.terms():
        if sum(monomial) <= cutoff:
            result[monomial] = sp.Integer(coefficient)
    return result


def run(cutoff: int) -> None:
    direct = direct_human_coefficients(cutoff)
    resummed = resummed_coefficients(cutoff)
    mismatches = {
        levels: (direct.get(levels, 0), resummed.get(levels, 0))
        for levels in sorted(
            set(direct) | set(resummed), key=lambda item: (sum(item), item)
        )
        if direct.get(levels, 0) != resummed.get(levels, 0)
    }
    if mismatches:
        for levels, values in mismatches.items():
            print(f"{levels}: direct={values[0]}, determinant={values[1]}")
        raise AssertionError("human Fock sum and determinant resummation disagree")
    nonzero = sum(value != 0 for value in direct.values())
    print(
        "PASS: exact human-definition/Fredholm comparison through "
        f"total twice-level {cutoff} ({nonzero} nonzero coefficients)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.cutoff < 0:
        parser.error("--cutoff must be non-negative")
    run(arguments.cutoff)
