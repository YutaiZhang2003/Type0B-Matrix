#!/usr/bin/env python3
"""Test the naive Wick-Pfaffian idea on the first crossed Ramond state.

The positive n=3/4, epsilon=0 state is the two-letter string
``chi_0^- chi_-1^-``.  If the enlarged chiral vertex were Gaussian in the
chi fields, its four-letter matrix element on the two Ramond legs would be
the Pfaffian of the six two-letter matrix elements.  This script constructs
those one- and two-letter Fock endpoints, transports only the physical
endpoint to the abstract Ramond module, and evaluates the exact chiral form.

The result is a useful gatekeeper for proposed determinant formulas: the
ordinary Pfaffian misses a nonzero connected four-chi kernel.  Consequently
the hard quartic is not obtainable from the free-fermion two-point kernel
alone; any correct finite determinant/path formula must include this extra
physical Ramond vertex kernel.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import sympy as sp


HERE = Path(__file__).resolve().parent
GRID_DIR = HERE.parent
if str(GRID_DIR) not in sys.path:
    sys.path.insert(0, str(GRID_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import compute_grid as grid  # noqa: E402
import path_sum_formula as paths  # noqa: E402


I = sp.I
SQRT2 = sp.sqrt(2)
EIGHTH_MINUS = (1 - I) / SQRT2


def _add(expression, state, coefficient):
    coefficient = sp.cancel(coefficient)
    if coefficient == 0:
        return
    expression[state] = sp.cancel(expression.get(state, 0) + coefficient)
    if expression[state] == 0:
        del expression[state]


def chi_fock_endpoints(tokens):
    """Expand an ordered list ``(mode,opposite_zero)`` of chi operators."""

    expression = {((), 0, (), 0): sp.Integer(1)}
    for mode, opposite_zero in reversed(tokens):
        next_expression = {}
        for (aux_modes, aux_ground, phys_modes, phys_ground), outer in expression.items():
            aux_final, aux_coefficient = paths._fermion_action(
                mode, aux_modes, aux_ground
            )
            if aux_coefficient:
                _add(
                    next_expression,
                    (aux_final[0], aux_final[1], phys_modes, phys_ground),
                    outer * aux_coefficient,
                )

            phys_final, phys_coefficient = paths._fermion_action(
                mode,
                phys_modes,
                phys_ground,
                zero_sign=-1 if opposite_zero else 1,
            )
            if phys_coefficient:
                aux_parity = (len(aux_modes) + aux_ground) % 2
                _add(
                    next_expression,
                    (aux_modes, aux_ground, phys_final[0], phys_final[1]),
                    outer * (-I) * (-1) ** aux_parity * phys_coefficient,
                )
        expression = next_expression
    return expression


def abstract_components(tokens, q_value, momentum):
    """Convert physical Fock endpoints to the abstract Ramond PBW basis."""

    by_auxiliary = defaultdict(lambda: defaultdict(lambda: sp.Integer(0)))
    for (aux_modes, aux_ground, phys_modes, phys_ground), coefficient in (
        chi_fock_endpoints(tokens).items()
    ):
        by_auxiliary[(aux_modes, aux_ground)][
            ((), phys_modes, phys_ground)
        ] += coefficient

    substitutions = {
        grid.ramond_branch.Q: q_value,
        grid.ramond_branch.P: momentum,
    }
    answer = []
    for auxiliary_state, physical_expression in by_auxiliary.items():
        one_state = next(iter(physical_expression))
        level = sum(one_state[0]) + sum(one_state[1])
        basis, transition = grid.ramond_branch.transition(level, -1)
        transition = transition.subs(substitutions, simultaneous=True)
        vector = sp.zeros(len(basis), 1)
        row = {state: index for index, state in enumerate(basis)}
        for state, coefficient in physical_expression.items():
            vector[row[state]] += coefficient
        coefficients = transition.inv() * vector
        for index, (virasoro_modes, supercurrent_modes, ground) in enumerate(basis):
            coefficient = sp.cancel(coefficients[index])
            if ground == 1:
                coefficient *= -EIGHTH_MINUS
            coefficient = sp.cancel(coefficient)
            if coefficient == 0:
                continue
            word = tuple(("L", -mode) for mode in virasoro_modes) + tuple(
                ("G", -mode) for mode in supercurrent_modes
            )
            answer.append(
                (
                    auxiliary_state[0],
                    auxiliary_state[1],
                    word,
                    ground,
                    coefficient,
                )
            )
    return tuple(answer)


def chi_amplitude(second_tokens, third_tokens, eta, sample):
    """Exact f=0 matrix element of two arbitrary short chi strings."""

    b_value, p1, p2, p3 = sample
    q_value = sp.cancel(b_value + 1 / b_value)
    evaluator = grid.PhysicalNRREvaluator(
        0,
        eta,
        (q_value**2 / 4 - p1**2) / 2,
        sp.Rational(1, 16) + q_value**2 / 8 - p2**2 / 2,
        sp.Rational(1, 16) + q_value**2 / 8 - p3**2 / 2,
        p2 / SQRT2,
        p3 / SQRT2,
        sp.Rational(3, 2) + 3 * q_value**2,
    )
    second = abstract_components(second_tokens, q_value, p2)
    third = abstract_components(third_tokens, q_value, p3)
    auxiliary_form_parity = (len(second_tokens) + len(third_tokens)) % 2

    answer = 0
    for aux2, aux_ground2, word2, phys_ground2, coefficient2 in second:
        physical_parity2 = grid.state_parity(word2, phys_ground2)
        for aux3, aux_ground3, word3, phys_ground3, coefficient3 in third:
            auxiliary_parity3 = (len(aux3) + aux_ground3) % 2
            answer += (
                (-1) ** (physical_parity2 * auxiliary_parity3)
                * coefficient2
                * coefficient3
                * grid.fermion_value_virasoro(
                    auxiliary_form_parity,
                    (),
                    aux2,
                    aux_ground2,
                    aux3,
                    aux_ground3,
                )
                * evaluator.value((), word2, phys_ground2, word3, phys_ground3)
            )
    return sp.factor(sp.cancel(answer))


def pfaffian_residual(eta, sample):
    """Return (direct four-point value, Pfaffian, connected kernel)."""

    fields = (
        (((0, False),), ()),
        (((-1, False),), ()),
        ((), ((0, False),)),
        ((), ((-1, False),)),
    )
    vacuum = chi_amplitude((), (), eta, sample)
    pair = {}
    for first in range(4):
        for second in range(first + 1, 4):
            tokens2 = fields[first][0] + fields[second][0]
            tokens3 = fields[first][1] + fields[second][1]
            pair[first, second] = sp.cancel(
                chi_amplitude(tokens2, tokens3, eta, sample) / vacuum
            )
    pfaffian = sp.cancel(
        pair[0, 1] * pair[2, 3]
        - pair[0, 2] * pair[1, 3]
        + pair[0, 3] * pair[1, 2]
    )
    direct = sp.cancel(
        chi_amplitude(
            ((0, False), (-1, False)),
            ((0, False), (-1, False)),
            eta,
            sample,
        )
        / vacuum
    )
    return direct, pfaffian, sp.factor(sp.cancel(direct - pfaffian))


def audit():
    labels = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))
    for sample_number, sample in enumerate(grid.SAMPLES, start=1):
        for eta in (1, -1):
            direct, pfaffian, connected = pfaffian_residual(eta, sample)
            branch = grid.enlarged_raw_three_point(
                *labels, 0, 0, 0, eta, *sample
            )[1]
            branch_residual = sp.factor(sp.cancel(direct - branch))
            if branch_residual != 0:
                raise AssertionError((sample_number, eta, branch_residual))
            if connected == 0:
                raise AssertionError("The naive Pfaffian unexpectedly worked")
            print(
                f"sample={sample_number} eta={eta:+d} "
                f"branch-residual=0 connected-four-chi={connected}"
            )


if __name__ == "__main__":
    audit()
