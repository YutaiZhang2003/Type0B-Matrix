#!/usr/bin/env python3
"""Derive exact boundary L_-1 actions in the common free-field basis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / "ramond_branching_coefficient_check" / "check_ramond_branching.py"


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


rhelper = load_module("ramond_boundary_symbolic_helper", HELPER)

b, P, Q = sp.symbols("b P Q", nonzero=True)
I = sp.I


def add(out, state, coefficient):
    coefficient = sp.factor(sp.cancel(coefficient))
    if coefficient == 0:
        return
    out[state] = sp.factor(sp.cancel(out.get(state, 0) + coefficient))
    if out[state] == 0:
        del out[state]


def apply_expression(expression, action):
    answer = {}
    for state, outer in expression.items():
        for final, inner in action(state).items():
            add(answer, final, outer * inner)
    return answer


def combine(*terms):
    answer = {}
    for coefficient, expression in terms:
        for state, value in expression.items():
            add(answer, state, coefficient * value)
    return answer


def solve(target, columns):
    keys = sorted(set(target).union(*(set(column) for column in columns)), key=repr)
    matrix = sp.Matrix(
        [[column.get(key, 0) for column in columns] for key in keys]
    )
    vector = sp.Matrix([target.get(key, 0) for key in keys])
    matrix = matrix.applyfunc(
        lambda value: sp.factor(sp.cancel(value.subs(Q, b + 1 / b)))
    )
    vector = vector.applyfunc(
        lambda value: sp.factor(sp.cancel(value.subs(Q, b + 1 / b)))
    )
    sample_matrix = matrix.subs(
        {b: sp.Rational(3, 2), P: sp.Rational(2, 5)}
    )
    independent_rows = sample_matrix.T.rref()[1]
    square = matrix[list(independent_rows), :]
    values = sp.Matrix([vector[index] for index in independent_rows])
    unknowns = sp.symbols(f"coefficient_0:{len(columns)}")
    solution = sp.solve_linear_system(square.row_join(values), *unknowns)
    if solution is None:
        raise AssertionError("The selected symbolic subsystem has no solution.")
    coefficients = [sp.factor(sp.cancel(solution[unknown])) for unknown in unknowns]
    residual = matrix * sp.Matrix(coefficients) - vector
    if any(sp.factor(sp.cancel(value)) != 0 for value in residual):
        raise AssertionError("The proposed symbolic descendant space is incomplete.")
    return coefficients


# NS states are (auxiliary half-integer modes in twice-mode units,
# physical bosons, physical half-integer modes in twice-mode units).
def ns_fermion(mode2, state):
    bosons, fermions = state
    if mode2 < 0:
        created = -mode2
        if created in fermions:
            return None, 0
        crossings = sum(existing > created for existing in fermions)
        return (bosons, tuple(sorted(fermions + (created,), reverse=True))), (-1) ** crossings
    if mode2 not in fermions:
        return None, 0
    position = fermions.index(mode2)
    return (bosons, fermions[:position] + fermions[position + 1 :]), (-1) ** position


def ns_auxiliary(mode2, modes):
    if mode2 < 0:
        created = -mode2
        if created in modes:
            return None, 0
        crossings = sum(existing > created for existing in modes)
        return tuple(sorted(modes + (created,), reverse=True)), (-1) ** crossings
    if mode2 not in modes:
        return None, 0
    position = modes.index(mode2)
    return modes[:position] + modes[position + 1 :], (-1) ** position


def ns_c(mode, state):
    bosons, fermions = state
    if mode < 0:
        return (tuple(sorted(bosons + (-mode,), reverse=True)), fermions), 1
    count = bosons.count(mode)
    if not count:
        return None, 0
    remaining = list(bosons)
    remaining.remove(mode)
    return (tuple(remaining), fermions), mode * count


def ns_two(first, second, state):
    middle, right = second(state)
    if not right:
        return None, 0
    final, left = first(middle)
    return final, right * left


def ns_physical_l_minus_one(state):
    bosons, fermions = state
    answer = {}
    indices = {-1}
    indices.update(bosons)
    indices.update(-1 - occupied for occupied in bosons)
    for mode in indices:
        if mode in (0, -1):
            continue
        final, coefficient = ns_two(
            lambda current, k=-1 - mode: ns_c(k, current),
            lambda current, k=mode: ns_c(k, current),
            state,
        )
        if coefficient:
            add(answer, final, sp.Rational(1, 2) * coefficient)
    indices2 = set(fermions)
    indices2.update(-2 - occupied for occupied in fermions)
    indices2.add(-1)
    for mode2 in indices2:
        final, coefficient = ns_two(
            lambda current, r2=-2 - mode2: ns_fermion(r2, current),
            lambda current, r2=mode2: ns_fermion(r2, current),
            state,
        )
        if coefficient:
            add(answer, final, sp.Rational(mode2, 4) * coefficient)
    final, coefficient = ns_c(-1, state)
    if coefficient:
        add(answer, final, I * (-Q / 2 - P) * coefficient)
    return answer


def ns_physical_g(mode2, state):
    bosons, fermions = state
    answer = {}
    indices = set(bosons)
    indices.update((mode2 - occupied) // 2 for occupied in fermions)
    if mode2 < 0:
        indices.update(range(mode2 // 2 + 1, 0))
    for mode in indices:
        if mode == 0:
            continue
        final, coefficient = ns_two(
            lambda current, k=mode: ns_c(k, current),
            lambda current, r2=mode2 - 2 * mode: ns_fermion(r2, current),
            state,
        )
        if coefficient:
            add(answer, final, coefficient)
    final, coefficient = ns_fermion(mode2, state)
    if coefficient:
        add(answer, final, I * (sp.Rational(mode2, 2) * Q - P) * coefficient)
    return answer


def ns_l(expression):
    return apply_expression(
        expression,
        lambda state: {
            (state[0], final[0], final[1]): coefficient
            for final, coefficient in ns_physical_l_minus_one((state[1], state[2])).items()
        },
    )


def ns_lf(expression):
    def action(state):
        auxiliary, bosons, fermions = state
        answer = {}
        for mode2 in set(auxiliary) | {-1} | {-2 - occupied for occupied in auxiliary}:
            middle, right = ns_auxiliary(mode2, auxiliary)
            if not right:
                continue
            final, left = ns_auxiliary(-2 - mode2, middle)
            if left:
                add(answer, (final, bosons, fermions), sp.Rational(mode2, 4) * right * left)
        return answer

    return apply_expression(expression, action)


def ns_u(expression):
    def action(state):
        auxiliary, bosons, fermions = state
        physical = (bosons, fermions)
        lower = -2 - sum(auxiliary)
        upper = 2 * sum(bosons) + sum(fermions)
        if lower % 2 == 0:
            lower += 1
        if upper % 2 == 0:
            upper -= 1
        answer = {}
        for mode2 in range(lower, upper + 1, 2):
            aux_final, aux_coefficient = ns_auxiliary(-2 - mode2, auxiliary)
            if not aux_coefficient:
                continue
            for physical_final, physical_coefficient in ns_physical_g(mode2, physical).items():
                add(
                    answer,
                    (aux_final, physical_final[0], physical_final[1]),
                    (-1) ** len(auxiliary) * aux_coefficient * physical_coefficient,
                )
        return answer

    return apply_expression(expression, action)


def ns_embedded(copy, expression):
    denominator = 1 / b - b
    if copy == 1:
        return combine(
            ((1 / b) / denominator, ns_l(expression)),
            (-(1 / b + 2 * b) / denominator, ns_lf(expression)),
            (1 / denominator, ns_u(expression)),
        )
    return combine(
        (-b / denominator, ns_l(expression)),
        ((b + 2 / b) / denominator, ns_lf(expression)),
        (-1 / denominator, ns_u(expression)),
    )


def derive_ns():
    vacuum = {((), (), ()): 1}
    plus_coefficient = P + Q / 2
    minus_coefficient = Q / 2 - P
    plus = {((1,), (), ()): plus_coefficient, ((), (), (1,)): -I * plus_coefficient}
    minus = {((1,), (), ()): minus_coefficient, ((), (), (1,)): -I * plus_coefficient}

    print("NS n=0", solve(ns_l(vacuum), [ns_embedded(1, vacuum), ns_embedded(2, vacuum)]))
    columns = [
        ns_embedded(1, plus),
        ns_embedded(2, plus),
        ns_embedded(1, minus),
        ns_embedded(2, minus),
    ]
    print("NS n=1/2", solve(ns_l(plus), columns))
    print("NS n=-1/2", solve(ns_l(minus), columns))


# Ramond states use the five-component state convention of the numerical
# backend.  Every symbolic branch is transported to realization -1.
def ramond_branch(label, parity):
    _, sectors = rhelper.branch_in_abstract_basis(label, parity)
    answer = {}
    for auxiliary_state, (level, ordered_basis, abstract_coefficients) in sectors.items():
        fixed_basis, fixed_transition = rhelper.transition(level, -1)
        if fixed_basis != ordered_basis:
            raise AssertionError("The Ramond bases disagree.")
        fock = (fixed_transition * abstract_coefficients).subs(
            rhelper.Q, Q, simultaneous=True
        ).subs(rhelper.P, P, simultaneous=True)
        for state, coefficient in zip(ordered_basis, fock):
            if coefficient != 0:
                add(
                    answer,
                    (auxiliary_state[0], auxiliary_state[1], state[0], state[1], state[2]),
                    coefficient,
                )
    return answer


def ramond_l(expression):
    def action(state):
        auxiliary, auxiliary_ground, bosons, fermions, ground = state
        answer = {}
        physical = (bosons, fermions, ground)
        for final, coefficient in rhelper.apply_L_to_state(
            -1, physical, -1, momentum=P
        ).items():
            coefficient = coefficient.subs(rhelper.Q, Q, simultaneous=True).subs(
                rhelper.P, P, simultaneous=True
            )
            add(
                answer,
                (auxiliary, auxiliary_ground, final[0], final[1], final[2]),
                coefficient,
            )
        return answer

    return apply_expression(expression, action)


def ramond_lf(expression):
    def action(state):
        modes, auxiliary_ground, bosons, fermions, ground = state
        answer = {}
        for mode in set(modes) | {-1, 0} | {-1 - occupied for occupied in modes}:
            middle, right = rhelper.apply_auxiliary(mode, (modes, auxiliary_ground))
            if not right:
                continue
            final, left = rhelper.apply_auxiliary(-1 - mode, middle)
            if left:
                add(
                    answer,
                    (final[0], final[1], bosons, fermions, ground),
                    sp.Rational(mode, 2) * right * left,
                )
        return answer

    return apply_expression(expression, action)


def ramond_g_zero(state):
    bosons, fermions, ground = state
    if bosons or fermions:
        raise AssertionError("Only the ground-state G_0 action is needed here.")
    return {(bosons, fermions, 1 - ground): -I * P / sp.sqrt(2)}


def ramond_u(expression):
    def action(state):
        modes, auxiliary_ground, bosons, fermions, ground = state
        physical = (bosons, fermions, ground)
        answer = {}
        for mode in (-1, 0):
            aux_final, aux_coefficient = rhelper.apply_auxiliary(
                -1 - mode, (modes, auxiliary_ground)
            )
            if not aux_coefficient:
                continue
            physical_answer = (
                rhelper.apply_G_to_state(-1, physical, -1, momentum=P)
                if mode == -1
                else ramond_g_zero(physical)
            )
            for physical_final, physical_coefficient in physical_answer.items():
                physical_coefficient = physical_coefficient.subs(
                    rhelper.Q, Q, simultaneous=True
                ).subs(rhelper.P, P, simultaneous=True)
                add(
                    answer,
                    (
                        aux_final[0],
                        aux_final[1],
                        physical_final[0],
                        physical_final[1],
                        physical_final[2],
                    ),
                    (-1) ** (len(modes) + auxiliary_ground)
                    * aux_coefficient
                    * physical_coefficient,
                )
        return answer

    return apply_expression(expression, action)


def ramond_embedded(copy, expression):
    denominator = 1 / b - b
    if copy == 1:
        return combine(
            ((1 / b) / denominator, ramond_l(expression)),
            (-(1 / b + 2 * b) / denominator, ramond_lf(expression)),
            (1 / denominator, ramond_u(expression)),
        )
    return combine(
        (-b / denominator, ramond_l(expression)),
        ((b + 2 / b) / denominator, ramond_lf(expression)),
        (-1 / denominator, ramond_u(expression)),
    )


def derive_ramond():
    for parity in (0, 1):
        plus = ramond_branch(sp.Rational(1, 4), parity)
        minus = ramond_branch(-sp.Rational(1, 4), parity)
        plus_high = ramond_branch(sp.Rational(3, 4), parity)
        minus_high = ramond_branch(-sp.Rational(3, 4), parity)
        print(
            f"R n=1/4 alpha={parity}",
            solve(
                ramond_l(plus),
                [ramond_embedded(1, plus), ramond_embedded(2, plus), minus_high],
            ),
        )
        print(
            f"R n=-1/4 alpha={parity}",
            solve(
                ramond_l(minus),
                [ramond_embedded(1, minus), ramond_embedded(2, minus), plus_high],
            ),
        )


if __name__ == "__main__":
    derive_ns()
    derive_ramond()
