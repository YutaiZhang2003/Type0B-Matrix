#!/usr/bin/env python3
"""Compute Ramond branching coefficients recursively for a requested target.

The program is self-contained.  It constructs the branch states from the
chi strings, constructs the two embedded Virasoro algebras from L, L^F, and
U, solves the required L_{+/-1} branch decompositions, and recursively reduces
the target to the boundary branching coefficients.  It does not import any
old code, stored coefficient, or boundary relation not stated in SCblock.tex.

The implemented recursion component has a nonnegative integral NS label and
Ramond labels on the quarter lattice Z/2 + 1/4.  The connected Ramond
reflection component containing each target label is generated automatically.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import time
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import mpmath as mp
from scipy import linalg as scipy_linalg


HERE = Path(__file__).resolve().parent
TOLERANCE = 1.0e-13
RANK_TOLERANCE = 1.0e-11
MP_DPS = 0


def set_multiprecision(dps: int):
    global MP_DPS
    MP_DPS = int(dps)
    if MP_DPS:
        mp.mp.dps = MP_DPS


def arithmetic_tolerance():
    if not MP_DPS:
        return TOLERANCE
    return mp.power(10, -max(20, MP_DPS - 20))


def real_number(value):
    if not MP_DPS:
        return float(value)
    if isinstance(value, Fraction):
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def complex_number(value=0):
    if not MP_DPS:
        return complex(value)
    if isinstance(value, mp.mpc):
        return value
    if isinstance(value, mp.mpf):
        return mp.mpc(value)
    converted = complex(value)
    return mp.mpc(str(converted.real), str(converted.imag))


def scalar_sqrt(value):
    return mp.sqrt(value) if MP_DPS else cmath.sqrt(value)


def scalar_power_of_two(exponent):
    return mp.power(2, real_number(exponent)) if MP_DPS else 2 ** float(exponent)


def parse_number(text: str) -> float:
    return real_number(Fraction(text))


def parse_label(text: str) -> Fraction:
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError(f"invalid rational label: {text}") from error


def validate_target(target):
    n1, n2, n3 = (Fraction(value) for value in target)
    if n1 < 0 or n1.denominator != 1:
        raise ValueError("n1 must be a nonnegative integer for this recursion component.")
    for name, value in (("n2", n2), ("n3", n3)):
        four_times = 4 * value
        if four_times.denominator != 1 or int(four_times) % 2 == 0:
            raise ValueError(f"{name} must lie in (1/2) Z + 1/4.")
    return n1, n2, n3


def ns_label_closure(target: Fraction):
    target = Fraction(target)
    if target < 0 or target.denominator != 1:
        raise ValueError("NS labels must be nonnegative integers.")
    return tuple(Fraction(label) for label in range(int(target) + 1))


@lru_cache(None)
def partitions(total: int, largest: int | None = None):
    if total == 0:
        return ((),)
    if largest is None or largest > total:
        largest = total
    answer = []
    for first in range(largest, 0, -1):
        for rest in partitions(total - first, first):
            answer.append((first,) + rest)
    return tuple(answer)


@lru_cache(None)
def strict_partitions(total: int, largest: int | None = None):
    if total == 0:
        return ((),)
    if largest is None or largest > total:
        largest = total
    answer = []
    for first in range(largest, 0, -1):
        for rest in strict_partitions(total - first, first - 1):
            answer.append((first,) + rest)
    return tuple(answer)


@lru_cache(None)
def strict_odd_partitions(total: int, largest: int | None = None):
    """Partitions into distinct positive odd integers.

    NS supercurrent modes are stored in doubled units, so a part ``k``
    denotes ``G_{-k/2}``.  This is the PBW ordering used by the Human Note.
    """

    total = int(total)
    if total < 0:
        return ()
    if total == 0:
        return ((),)
    if largest is None or largest > total:
        largest = total
    largest = int(largest)
    if largest % 2 == 0:
        largest -= 1
    answer = []
    for first in range(largest, 0, -2):
        for rest in strict_odd_partitions(total - first, first - 2):
            answer.append((first,) + rest)
    return tuple(answer)


def partition_pairs(level: int):
    return tuple(
        (first, second)
        for first_level in range(level + 1)
        for first in partitions(first_level)
        for second in partitions(level - first_level)
    )


def add_term(expression, state, coefficient):
    value = expression.get(state, complex_number()) + coefficient
    if abs(value) <= arithmetic_tolerance():
        expression.pop(state, None)
    else:
        expression[state] = value


def combine(*terms):
    answer = {}
    for coefficient, expression in terms:
        for state, value in expression.items():
            add_term(answer, state, coefficient * value)
    return answer


def apply_expression(expression, action):
    answer = {}
    for state, outer in expression.items():
        for final, inner in action(state).items():
            add_term(answer, final, outer * inner)
    return answer


def max_abs(expression):
    return max((abs(value) for value in expression.values()), default=0.0)


def ell(x: complex, index: int, b: float) -> complex:
    index = int(index)
    q = b + 1 / b
    if index == 0:
        return complex_number(1)
    if index < 0:
        reflected = ell(q - x, -index, b)
        return ((-1) ** ((-index) // 2) * reflected) if index % 2 == 0 else reflected
    answer = scalar_power_of_two(Fraction(1, 8)) if index % 2 else real_number(1)
    wanted_parity = index % 2
    for r in range(index):
        for s in range(index - r):
            if (r + s) % 2 == wanted_parity:
                answer *= x + r * b + s / b
    return answer


class FreeFieldModule:
    """Auxiliary fermion tensor a free-field SCA module."""

    def __init__(self, sector: str, b: float, momentum: complex, realization=-1):
        if sector not in ("NS", "R"):
            raise ValueError("sector must be NS or R")
        self.sector = sector
        self.b = real_number(b)
        self.q = self.b + 1 / self.b
        # Physical continuum states enter this free-field realization through
        # the analytic continuation P_note=i*P_physical.
        self.momentum = complex_number(momentum)
        self.realization = int(realization)

    @staticmethod
    def _fermion(mode, modes, ground=None, zero_sign=1):
        if mode < 0:
            created = -mode
            if created in modes:
                return None, 0.0
            crossings = sum(existing > created for existing in modes)
            final = tuple(sorted(modes + (created,), reverse=True))
            if ground is None:
                return final, float((-1) ** crossings)
            return (final, ground), float((-1) ** crossings)
        if mode > 0:
            if mode not in modes:
                return None, 0.0
            position = modes.index(mode)
            final = modes[:position] + modes[position + 1 :]
            if ground is None:
                return final, float((-1) ** position)
            return (final, ground), float((-1) ** position)
        if ground is None:
            raise AssertionError("The NS fermion has no zero mode.")
        coefficient = (-1) ** len(modes) * zero_sign / scalar_sqrt(real_number(2))
        return (modes, 1 - ground), coefficient

    @staticmethod
    def _boson(mode, bosons):
        if mode < 0:
            return tuple(sorted(bosons + (-mode,), reverse=True)), 1.0
        if mode == 0:
            raise AssertionError("The bosonic zero mode is already evaluated.")
        count = bosons.count(mode)
        if count == 0:
            return None, 0.0
        remaining = list(bosons)
        remaining.remove(mode)
        return tuple(remaining), float(mode * count)

    def physical_level_units(self, physical):
        bosons, fermions, *_ = physical
        if self.sector == "NS":
            return 2 * sum(bosons) + sum(fermions)
        return sum(bosons) + sum(fermions)

    def auxiliary_level_units(self, auxiliary):
        return sum(auxiliary[0] if self.sector == "R" else auxiliary)

    def auxiliary_parity(self, auxiliary):
        if self.sector == "NS":
            return len(auxiliary) % 2
        return (len(auxiliary[0]) + auxiliary[1]) % 2

    def apply_auxiliary(self, mode, auxiliary):
        if self.sector == "NS":
            return self._fermion(mode, auxiliary)
        return self._fermion(mode, auxiliary[0], auxiliary[1])

    def apply_physical_fermion(self, mode, physical):
        if self.sector == "NS":
            final, coefficient = self._fermion(mode, physical[1])
            if not coefficient:
                return None, 0.0
            return (physical[0], final), coefficient
        zero_sign = 1 if self.realization == -1 else -1
        final, coefficient = self._fermion(
            mode, physical[1], physical[2], zero_sign=zero_sign
        )
        if not coefficient:
            return None, 0.0
        return (physical[0], final[0], final[1]), coefficient

    def apply_c(self, mode, physical):
        final, coefficient = self._boson(mode, physical[0])
        if not coefficient:
            return None, 0.0
        return (final,) + physical[1:], coefficient

    @staticmethod
    def apply_two(first, second, state):
        middle, right = second(state)
        if not right:
            return None, 0.0
        final, left = first(middle)
        if not left:
            return None, 0.0
        return final, right * left

    @lru_cache(None)
    def physical_l_on_state(self, mode: int, physical):
        bosons, fermions, *_ = physical
        answer = {}

        bosonic_indices = set(bosons)
        bosonic_indices.update(mode - occupied for occupied in bosons)
        if mode < 0:
            bosonic_indices.update(range(mode + 1, 0))
        for summation_mode in bosonic_indices:
            if summation_mode in (0, mode):
                continue
            final, coefficient = self.apply_two(
                lambda current, k=mode - summation_mode: self.apply_c(k, current),
                lambda current, k=summation_mode: self.apply_c(k, current),
                physical,
            )
            if coefficient:
                add_term(answer, final, 0.5 * coefficient)

        if self.sector == "NS":
            doubled_mode = 2 * mode
            indices = set(fermions)
            indices.update(doubled_mode - occupied for occupied in fermions)
            if mode < 0:
                indices.update(range(doubled_mode + 1, 0, 2))
            for r2 in indices:
                final, coefficient = self.apply_two(
                    lambda current, s2=doubled_mode - r2: self.apply_physical_fermion(s2, current),
                    lambda current, s2=r2: self.apply_physical_fermion(s2, current),
                    physical,
                )
                if coefficient:
                    add_term(answer, final, r2 * coefficient / 4)
        else:
            indices = set(fermions)
            indices.update(mode - occupied for occupied in fermions)
            if mode < 0:
                indices.update(range(mode, 1))
            for r in indices:
                final, coefficient = self.apply_two(
                    lambda current, s=mode - r: self.apply_physical_fermion(s, current),
                    lambda current, s=r: self.apply_physical_fermion(s, current),
                    physical,
                )
                if coefficient:
                    add_term(answer, final, 0.5 * r * coefficient)

        final, coefficient = self.apply_c(mode, physical)
        if coefficient:
            momentum_term = self.q * mode + 2 * self.realization * self.momentum
            add_term(answer, final, 0.5j * momentum_term * coefficient)
        return tuple(answer.items())

    @lru_cache(None)
    def physical_g_on_state(self, mode, physical):
        bosons, fermions, *_ = physical
        answer = {}
        indices = set(bosons)
        if self.sector == "NS":
            mode2 = int(mode)
            indices.update((mode2 - occupied) // 2 for occupied in fermions)
            if mode2 < 0:
                indices.update(range(mode2 // 2 + 1, 0))
            for summation_mode in indices:
                if summation_mode == 0:
                    continue
                final, coefficient = self.apply_two(
                    lambda current, k=summation_mode: self.apply_c(k, current),
                    lambda current, r2=mode2 - 2 * summation_mode: self.apply_physical_fermion(r2, current),
                    physical,
                )
                if coefficient:
                    add_term(answer, final, coefficient)
            final, coefficient = self.apply_physical_fermion(mode2, physical)
            if coefficient:
                r = mode2 / 2
                add_term(
                    answer,
                    final,
                    1j * (self.q * r + self.realization * self.momentum) * coefficient,
                )
        else:
            mode = int(mode)
            indices.update(mode - occupied for occupied in fermions)
            if mode != 0:
                indices.add(mode)
            if mode < 0:
                indices.update(range(mode, 0))
            for summation_mode in indices:
                if summation_mode == 0:
                    continue
                final, coefficient = self.apply_two(
                    lambda current, k=summation_mode: self.apply_c(k, current),
                    lambda current, r=mode - summation_mode: self.apply_physical_fermion(r, current),
                    physical,
                )
                if coefficient:
                    add_term(answer, final, coefficient)
            final, coefficient = self.apply_physical_fermion(mode, physical)
            if coefficient:
                add_term(
                    answer,
                    final,
                    1j * (self.q * mode + self.realization * self.momentum) * coefficient,
                )
        return tuple(answer.items())

    def split_state(self, state):
        if self.sector == "NS":
            return state[0], (state[1], state[2])
        return (state[0], state[1]), (state[2], state[3], state[4])

    def join_state(self, auxiliary, physical):
        if self.sector == "NS":
            return (auxiliary, physical[0], physical[1])
        return (auxiliary[0], auxiliary[1], physical[0], physical[1], physical[2])

    def apply_l(self, mode: int, expression):
        def action(state):
            auxiliary, physical = self.split_state(state)
            return {
                self.join_state(auxiliary, final): coefficient
                for final, coefficient in self.physical_l_on_state(mode, physical)
            }

        return apply_expression(expression, action)

    def apply_lf(self, mode: int, expression):
        def action(state):
            auxiliary, physical = self.split_state(state)
            modes = auxiliary if self.sector == "NS" else auxiliary[0]
            answer = {}
            if self.sector == "NS":
                doubled_mode = 2 * mode
                indices = set(modes)
                indices.update(doubled_mode - occupied for occupied in modes)
                if mode < 0:
                    indices.update(range(doubled_mode + 1, 0, 2))
                factor = lambda r: r / 4
                complement = lambda r: doubled_mode - r
            else:
                indices = set(modes)
                indices.update(mode - occupied for occupied in modes)
                if mode < 0:
                    indices.update(range(mode, 1))
                factor = lambda r: r / 2
                complement = lambda r: mode - r
            for r in indices:
                middle, right = self.apply_auxiliary(r, auxiliary)
                if not right:
                    continue
                final, left = self.apply_auxiliary(complement(r), middle)
                if left:
                    add_term(
                        answer,
                        self.join_state(final, physical),
                        factor(r) * right * left,
                    )
            return answer

        return apply_expression(expression, action)

    def apply_u(self, mode: int, expression):
        def action(state):
            auxiliary, physical = self.split_state(state)
            answer = {}
            if self.sector == "NS":
                lower = 2 * mode - self.auxiliary_level_units(auxiliary)
                upper = self.physical_level_units(physical)
                if lower % 2 == 0:
                    lower += 1
                if upper % 2 == 0:
                    upper -= 1
                modes = range(lower, upper + 1, 2)
                aux_mode = lambda r: 2 * mode - r
            else:
                lower = mode - self.auxiliary_level_units(auxiliary)
                upper = self.physical_level_units(physical)
                modes = range(lower, upper + 1)
                aux_mode = lambda r: mode - r
            sign = (-1) ** self.auxiliary_parity(auxiliary)
            for r in modes:
                auxiliary_final, auxiliary_coefficient = self.apply_auxiliary(
                    aux_mode(r), auxiliary
                )
                if not auxiliary_coefficient:
                    continue
                for physical_final, physical_coefficient in self.physical_g_on_state(
                    r, physical
                ):
                    add_term(
                        answer,
                        self.join_state(auxiliary_final, physical_final),
                        sign * auxiliary_coefficient * physical_coefficient,
                    )
            return answer

        return apply_expression(expression, action)

    def apply_embedded(self, copy: int, mode: int, expression):
        denominator = 1 / self.b - self.b
        physical = self.apply_l(mode, expression)
        auxiliary = self.apply_lf(mode, expression)
        mixed = self.apply_u(mode, expression)
        if copy == 1:
            return combine(
                ((1 / self.b) / denominator, physical),
                (-(1 / self.b + 2 * self.b) / denominator, auxiliary),
                (1 / denominator, mixed),
            )
        if copy == 2:
            return combine(
                (-self.b / denominator, physical),
                ((self.b + 2 / self.b) / denominator, auxiliary),
                (-1 / denominator, mixed),
            )
        raise ValueError("copy must be 1 or 2")

    def descendant(self, primary, first_partition, second_partition):
        answer = primary
        for mode in reversed(second_partition):
            answer = self.apply_embedded(2, -mode, answer)
        for mode in reversed(first_partition):
            answer = self.apply_embedded(1, -mode, answer)
        return answer

    def ns_branch(self, label: Fraction):
        """Construct the positive-chart Human-Note NS chi-string state."""

        label = Fraction(label)
        if (
            self.sector != "NS"
            or label < 0
            or (2 * label).denominator != 1
        ):
            raise ValueError(
                "The positive NS chart requires a nonnegative half-integral label."
            )
        expression = {((), (), ()): 1.0 + 0.0j}
        if label == 0:
            return expression
        four_label = int(4 * label)
        operators = tuple(-mode2 for mode2 in range(1, four_label, 2))
        for mode2 in reversed(operators):
            next_expression = {}
            for state, outer in expression.items():
                auxiliary, physical = self.split_state(state)
                auxiliary_final, auxiliary_coefficient = self.apply_auxiliary(
                    mode2, auxiliary
                )
                if auxiliary_coefficient:
                    add_term(
                        next_expression,
                        self.join_state(auxiliary_final, physical),
                        outer * auxiliary_coefficient,
                    )
                physical_final, physical_coefficient = self.apply_physical_fermion(
                    mode2, physical
                )
                if physical_coefficient:
                    add_term(
                        next_expression,
                        self.join_state(auxiliary, physical_final),
                        outer
                        * (-1j)
                        * (-1) ** self.auxiliary_parity(auxiliary)
                        * physical_coefficient,
                    )
            expression = next_expression
        scale = scalar_power_of_two(-2 * label) * ell(
            self.q + 2 * self.momentum, four_label, self.b
        )
        return {state: scale * value for state, value in expression.items()}

    @lru_cache(None)
    def _ns_level_transition(self, realization: int, level_units: int):
        """Map NS SCA PBW columns to free-field oscillator rows.

        ``level_units`` is twice the physical NS level.  This is the NS
        analogue of :meth:`_level_transition` below and is built directly
        from the same Human-Note free-field generators used everywhere else
        in this module.
        """

        temporary = FreeFieldModule(
            "NS", self.b, self.momentum, int(realization)
        )
        rows = tuple(
            (bosons, fermions)
            for fermion_level in range(level_units + 1)
            if (level_units - fermion_level) % 2 == 0
            for bosons in partitions((level_units - fermion_level) // 2)
            for fermions in strict_odd_partitions(fermion_level)
        )
        columns = []
        for virasoro_level in range(level_units // 2 + 1):
            for virasoro_modes in partitions(virasoro_level):
                for supercurrent_modes in strict_odd_partitions(
                    level_units - 2 * virasoro_level
                ):
                    expression = {((), ()): complex_number(1)}
                    for mode2 in reversed(supercurrent_modes):
                        expression = apply_expression(
                            expression,
                            lambda state, mode2=mode2: dict(
                                temporary.physical_g_on_state(-mode2, state)
                            ),
                        )
                    for mode in reversed(virasoro_modes):
                        expression = apply_expression(
                            expression,
                            lambda state, mode=mode: dict(
                                temporary.physical_l_on_state(-mode, state)
                            ),
                        )
                    columns.append(
                        [expression.get(row, complex_number()) for row in rows]
                    )
        row_count = len(rows)
        column_count = len(columns)
        if row_count != column_count:
            raise AssertionError("The NS free-field/SCA transition is not square.")
        if MP_DPS:
            matrix = mp.matrix(
                [
                    [columns[column][row] for column in range(column_count)]
                    for row in range(row_count)
                ]
            )
        else:
            matrix = np.asarray(columns, dtype=np.complex128).T
            if np.linalg.matrix_rank(matrix, tol=RANK_TOLERANCE) != row_count:
                raise AssertionError(
                    "The NS free-field/SCA transition is singular."
                )
        return rows, matrix

    def _raw_r_branch(self, label: Fraction, parity: int):
        if self.sector != "R":
            raise ValueError("Ramond branch requested from an NS module.")
        sign = 1 if label > 0 else -1
        native_realization = -sign
        largest = int(2 * abs(label) - Fraction(1, 2))
        operators = [(0, native_realization)] + [
            (-mode, native_realization) for mode in range(1, largest + 1)
        ]
        if len(operators) % 2 != parity:
            operators.append((0, -native_realization))
        expression = {((), 0, (), (), 0): 1.0 + 0.0j}
        for mode, realization in reversed(operators):
            next_expression = {}
            for state, outer in expression.items():
                auxiliary, physical = self.split_state(state)
                auxiliary_final, auxiliary_coefficient = self.apply_auxiliary(
                    mode, auxiliary
                )
                if auxiliary_coefficient:
                    add_term(
                        next_expression,
                        self.join_state(auxiliary_final, physical),
                        outer * auxiliary_coefficient,
                    )
                old_realization = self.realization
                self.realization = realization
                physical_final, physical_coefficient = self.apply_physical_fermion(
                    mode, physical
                )
                self.realization = old_realization
                if physical_coefficient:
                    add_term(
                        next_expression,
                        self.join_state(auxiliary, physical_final),
                        outer
                        * (-1j)
                        * (-1) ** self.auxiliary_parity(auxiliary)
                        * physical_coefficient,
                    )
            expression = next_expression
        return native_realization, expression

    @lru_cache(None)
    def _level_transition(self, realization: int, level: int):
        temporary = FreeFieldModule("R", self.b, self.momentum, realization)
        rows = tuple(
            (bosons, fermions, ground)
            for boson_level in range(level + 1)
            for bosons in partitions(boson_level)
            for fermions in strict_partitions(level - boson_level)
            for ground in (0, 1)
        )
        columns = []
        for virasoro_level in range(level + 1):
            for virasoro_modes in partitions(virasoro_level):
                for supercurrent_modes in strict_partitions(level - virasoro_level):
                    for ground in (0, 1):
                        expression = {((), (), ground): 1.0 + 0.0j}
                        for mode in reversed(supercurrent_modes):
                            expression = apply_expression(
                                expression,
                                lambda state, mode=mode: dict(
                                    temporary.physical_g_on_state(-mode, state)
                                ),
                            )
                        for mode in reversed(virasoro_modes):
                            expression = apply_expression(
                                expression,
                                lambda state, mode=mode: dict(
                                    temporary.physical_l_on_state(-mode, state)
                                ),
                            )
                        columns.append(
                            [expression.get(row, 0.0j) for row in rows]
                        )
        row_count = len(rows)
        column_count = len(columns)
        if row_count != column_count:
            raise AssertionError("The Ramond free-field/SCA transition is not square.")
        if MP_DPS:
            matrix = mp.matrix(
                [[columns[column][row] for column in range(column_count)] for row in range(row_count)]
            )
        else:
            matrix = np.asarray(columns, dtype=np.complex128).T
            if np.linalg.matrix_rank(matrix, tol=RANK_TOLERANCE) != matrix.shape[0]:
                raise AssertionError("The Ramond free-field/SCA transition is singular.")
        return rows, matrix

    def r_branch(self, label, parity: int):
        label = Fraction(label)
        native, raw = self._raw_r_branch(label, parity)
        if native == self.realization:
            return raw
        grouped = {}
        for state, coefficient in raw.items():
            auxiliary, physical = self.split_state(state)
            grouped.setdefault(auxiliary, {})[physical] = coefficient
        answer = {}
        for auxiliary, physical_expression in grouped.items():
            level = self.physical_level_units(next(iter(physical_expression)))
            if level == 0:
                converted = physical_expression
            else:
                rows, native_matrix = self._level_transition(native, level)
                _, target_matrix = self._level_transition(self.realization, level)
                values = [physical_expression.get(row, complex_number()) for row in rows]
                if MP_DPS:
                    vector = mp.matrix(values)
                    abstract = mp.lu_solve(native_matrix, vector)
                    target = target_matrix * abstract
                    if not all(mp.isfinite(value) for value in target):
                        raise FloatingPointError(
                            f"Non-finite Ramond reflection conversion at n={label}."
                        )
                else:
                    vector = np.asarray(values, dtype=np.complex128)
                    abstract = np.linalg.solve(native_matrix, vector)
                    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                        target = target_matrix @ abstract
                    if not np.all(np.isfinite(target)):
                        raise FloatingPointError(
                            f"Non-finite Ramond reflection conversion at n={label}."
                        )
                converted = {
                    row: value
                    for row, value in zip(rows, target)
                    if abs(value) > arithmetic_tolerance()
                }
            for physical, coefficient in converted.items():
                add_term(answer, self.join_state(auxiliary, physical), coefficient)
        return answer


def sparse_inner(left, right):
    if len(left) > len(right):
        return mp.conj(sparse_inner(right, left))
    return mp.fsum(
        mp.conj(value) * right.get(state, 0)
        for state, value in left.items()
        if state in right
    )


def multiprecision_span_fit(target, columns):
    """Fit an exact descendant span by certified mixed-precision refinement.

    A pivoted QR of the double-precision shadow chooses a well-conditioned
    square set of oscillator rows.  The coefficients and residuals are then
    accumulated at ``MP_DPS`` digits while a cached double LU supplies only
    the iterative corrections.  The returned residual is evaluated in
    multiprecision on *all* oscillator rows, not merely on the selected set.
    """

    column_count = len(columns)
    keys = sorted(set(target).union(*(set(column) for column in columns)), key=repr)
    norms = [mp.sqrt(mp.re(sparse_inner(column, column))) for column in columns]
    if any(norm == 0 for norm in norms):
        raise AssertionError("A descendant column vanished at the sample point.")

    shadow = np.empty((len(keys), column_count), dtype=np.complex128)
    for column_index, (column, norm) in enumerate(zip(columns, norms)):
        inverse_norm = 1 / norm
        shadow[:, column_index] = [
            complex(column.get(key, 0) * inverse_norm) for key in keys
        ]

    # Pivoting the columns of A^T selects independent rows of A.  Since the
    # descendant identity is exact, a full-rank square row restriction fixes
    # the unique coefficients; the all-row residual below certifies it.
    _, _, pivots = scipy_linalg.qr(
        shadow.T, mode="economic", pivoting=True, check_finite=False
    )
    selected_indices = np.asarray(pivots[:column_count], dtype=int)
    selected = shadow[selected_indices, :]
    singular_values = scipy_linalg.svdvals(selected, check_finite=False)
    rank = int(np.count_nonzero(singular_values > RANK_TOLERANCE))
    if rank != column_count:
        raise np.linalg.LinAlgError(
            f"Pivoted oscillator restriction has rank {rank}/{column_count}."
        )
    selected_keys = [keys[index] for index in selected_indices]
    selected_vector = np.asarray(
        [complex(target.get(key, 0)) for key in selected_keys],
        dtype=np.complex128,
    )
    lu, lu_pivots = scipy_linalg.lu_factor(selected, check_finite=False)
    initial = scipy_linalg.lu_solve(
        (lu, lu_pivots), selected_vector, check_finite=False
    )
    scaled_coefficients = [complex_number(value) for value in initial]

    selected_target_norm = mp.sqrt(
        mp.fsum(abs(target.get(key, 0)) ** 2 for key in selected_keys)
    )
    selected_scale = max(selected_target_norm, mp.mpf(1))
    refinement_tolerance = mp.power(10, -max(20, MP_DPS - 15))
    selected_relative_residual = mp.inf
    refinement_iterations = 0
    for iteration in range(1, 31):
        residual = [
            target.get(key, 0)
            - mp.fsum(
                column.get(key, 0) * coefficient / norm
                for column, coefficient, norm in zip(
                    columns, scaled_coefficients, norms
                )
            )
            for key in selected_keys
        ]
        selected_relative_residual = (
            mp.sqrt(mp.fsum(abs(value) ** 2 for value in residual))
            / selected_scale
        )
        if selected_relative_residual <= refinement_tolerance:
            break
        correction = scipy_linalg.lu_solve(
            (lu, lu_pivots),
            np.asarray([complex(value) for value in residual], dtype=np.complex128),
            check_finite=False,
        )
        scaled_coefficients = [
            coefficient + complex_number(delta)
            for coefficient, delta in zip(scaled_coefficients, correction)
        ]
        refinement_iterations = iteration
    else:
        raise FloatingPointError(
            "Mixed-precision descendant refinement did not reach the requested tolerance."
        )

    coefficients = [
        scaled_coefficients[index] / norms[index]
        for index in range(column_count)
    ]
    all_row_residual = {key: -target.get(key, 0) for key in keys}
    for coefficient, column in zip(coefficients, columns):
        for state, value in column.items():
            all_row_residual[state] += coefficient * value
    absolute = mp.sqrt(mp.fsum(abs(value) ** 2 for value in all_row_residual.values()))
    target_norm = mp.sqrt(mp.re(sparse_inner(target, target)))
    condition_number = float(singular_values[0] / singular_values[-1])
    return {
        "coefficients": coefficients,
        "rows": len(keys),
        "columns": column_count,
        "rank": rank,
        "absolute_residual": float(absolute),
        "relative_residual": float(absolute / target_norm) if target_norm else float(absolute),
        "smallest_singular_value": float(singular_values[-1]),
        "scaled_condition_number": condition_number,
        "selected_rows": column_count,
        "refinement_iterations": refinement_iterations,
        "selected_relative_residual": float(selected_relative_residual),
        "solver": f"mixed-precision-pivoted-refinement-{MP_DPS}dps",
    }


def span_fit(target, columns):
    if MP_DPS:
        return multiprecision_span_fit(target, columns)
    keys = sorted(set(target).union(*(set(column) for column in columns)), key=repr)
    matrix = np.zeros((len(keys), len(columns)), dtype=np.complex128)
    vector = np.asarray([target.get(key, 0.0j) for key in keys], dtype=np.complex128)
    for column_index, column in enumerate(columns):
        matrix[:, column_index] = [column.get(key, 0.0j) for key in keys]
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms == 0):
        raise AssertionError("A descendant column vanished at the sample point.")
    normalized = matrix / norms
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        normalized, vector, rcond=RANK_TOLERANCE
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        residual = normalized @ coefficients - vector
    if not np.all(np.isfinite(residual)):
        raise FloatingPointError("The scaled descendant fit produced a non-finite residual.")
    absolute = float(np.linalg.norm(residual))
    target_norm = float(np.linalg.norm(vector))
    relative = absolute / target_norm if target_norm else absolute
    return {
        "coefficients": coefficients / norms,
        "rows": len(keys),
        "columns": len(columns),
        "rank": int(rank),
        "absolute_residual": absolute,
        "relative_residual": relative,
        "smallest_singular_value": float(singular_values[rank - 1]),
        "solver": "numpy-lstsq-complex128",
    }


@dataclass(frozen=True)
class ActionTerm:
    label: Fraction
    first: tuple[int, ...]
    second: tuple[int, ...]
    coefficient: complex


def solve_ns_l1(module: FreeFieldModule, label: Fraction):
    label = Fraction(label)
    if label <= -1 and (2 * label).denominator == 1:
        # Reflect v_n(P)=v_{-n}(-P), solve on the positive chart, and map the
        # lower branch label back.  Virasoro descendants and coefficients are
        # unchanged by this identification.
        reflected = FreeFieldModule("NS", module.b, -module.momentum)
        positive_terms, fit = solve_ns_l1(reflected, -label)
        return [
            ActionTerm(-term.label, term.first, term.second, term.coefficient)
            for term in positive_terms
        ], fit
    if label < 1 or (2 * label).denominator != 1:
        raise ValueError("The NS L1 reduction requires n >= 1 in Z/2.")
    high = module.ns_branch(label)
    low = module.ns_branch(label - 1)
    level = int(4 * label - 3)
    pairs = partition_pairs(level)
    columns = [module.descendant(low, first, second) for first, second in pairs]
    fit = span_fit(module.apply_l(1, high), columns)
    terms = [
        ActionTerm(Fraction(label - 1), first, second, coefficient)
        for (first, second), coefficient in zip(pairs, fit["coefficients"])
    ]
    return terms, fit


def ramond_lminus_structure(label: Fraction):
    if label >= Fraction(3, 4):
        return label - 1, int(4 * label - 1)
    if label == Fraction(1, 4):
        return Fraction(-3, 4), 0
    if label == Fraction(-1, 4):
        return Fraction(3, 4), 0
    if label <= Fraction(-3, 4):
        return label + 1, int(-4 * label - 1)
    raise ValueError(f"No Ramond L_-1 structure is stated for n={label}.")


def ramond_label_closure(target: Fraction):
    """Return the closed reflection component containing ``target``."""

    target = Fraction(target)
    four_times = 4 * target
    if four_times.denominator != 1 or int(four_times) % 2 == 0:
        raise ValueError("Ramond labels must lie in (1/2) Z + 1/4.")
    labels = set()
    current = target
    while True:
        if current in labels:
            break
        labels.add(current)
        current, _ = ramond_lminus_structure(current)
    return tuple(sorted(labels))


def solve_ramond_lminus(module: FreeFieldModule, label: Fraction, parity: int):
    high = module.r_branch(label, parity)
    neighbor_label, level = ramond_lminus_structure(label)
    neighbor = module.r_branch(neighbor_label, parity)
    same = [
        module.descendant(high, (1,), ()),
        module.descendant(high, (), (1,)),
    ]
    pairs = partition_pairs(level)
    neighbor_columns = [
        module.descendant(neighbor, first, second) for first, second in pairs
    ]
    fit = span_fit(module.apply_l(-1, high), same + neighbor_columns)
    coefficients = fit["coefficients"]
    terms = [
        ActionTerm(label, (1,), (), coefficients[0]),
        ActionTerm(label, (), (1,), coefficients[1]),
    ]
    terms.extend(
        ActionTerm(neighbor_label, first, second, coefficient)
        for (first, second), coefficient in zip(pairs, coefficients[2:])
    )
    identity = combine(
        (1, module.apply_l(-1, high)),
        (-1, same[0]),
        (-1, same[1]),
        (1, module.apply_lf(-1, high)),
    )
    fit["inverse_identity_max_residual"] = max_abs(identity)
    return terms, fit


@lru_cache(None)
def canonicalize_word(word):
    word = tuple(word)
    for position in range(len(word) - 1):
        first, second = word[position : position + 2]
        if first >= second:
            continue
        answer = {}
        exchanged = word[:position] + (second, first) + word[position + 2 :]
        for final, coefficient in canonicalize_word(exchanged).items():
            answer[final] = answer.get(final, 0.0) + coefficient
        bracket = word[:position] + (first + second,) + word[position + 2 :]
        for final, coefficient in canonicalize_word(bracket).items():
            answer[final] = answer.get(final, 0.0) + (second - first) * coefficient
        return answer
    return {word: 1.0}


class VirasoroThreePoint:
    """Stripped ordinary Virasoro form in the slot order of the notes."""

    def __init__(self, weights, central_charge):
        self.weights = tuple(complex_number(value) for value in weights)
        self.central_charge = complex_number(central_charge)
        self._value_cache = {}
        self._act_cache = {}

    def act(self, slot, mode, word):
        key = (slot, mode, tuple(word))
        if key in self._act_cache:
            return self._act_cache[key]
        word = tuple(word)
        if mode < 0:
            answer = canonicalize_word((-mode,) + word)
        elif not word:
            answer = {(): self.weights[slot]} if mode == 0 else {}
        else:
            first, rest = word[0], word[1:]
            answer = {}
            for reduced, coefficient in self.act(slot, mode, rest).items():
                for final, ordering in canonicalize_word((first,) + reduced).items():
                    answer[final] = answer.get(final, 0.0j) + coefficient * ordering
            bracket_coefficient = mode + first
            replacement = mode - first
            if replacement < 0:
                for final, coefficient in canonicalize_word((-replacement,) + rest).items():
                    answer[final] = answer.get(final, 0.0j) + bracket_coefficient * coefficient
            elif replacement == 0:
                answer[rest] = answer.get(rest, 0.0j) + bracket_coefficient * (
                    self.weights[slot] + sum(rest)
                )
            else:
                for final, coefficient in self.act(slot, replacement, rest).items():
                    answer[final] = answer.get(final, 0.0j) + bracket_coefficient * coefficient
            if mode == first:
                answer[rest] = answer.get(rest, 0.0j) + self.central_charge * (
                    mode**3 - mode
                ) / 12
        self._act_cache[key] = answer
        return answer

    def value(self, word1=(), word2=(), word3=()):
        key = (tuple(word1), tuple(word2), tuple(word3))
        if key in self._value_cache:
            return self._value_cache[key]
        word1, word2, word3 = key
        if word2:
            n, rest2 = word2[0], word2[1:]
            if n == 1:
                exponent = (
                    self.weights[0]
                    + sum(word1)
                    - self.weights[1]
                    - sum(rest2)
                    - self.weights[2]
                    - sum(word3)
                )
                answer = exponent * self.value(word1, rest2, word3)
            else:
                answer = 0.0j
                maximum = max(sum(word1) - n, sum(word3) + 1, 0)
                for p in range(maximum + 1):
                    ward = math.comb(n - 2 + p, n - 2)
                    answer += ward * sum(
                        coefficient * self.value(reduced, rest2, word3)
                        for reduced, coefficient in self.act(0, n + p, word1).items()
                    )
                    answer += ward * (-1) ** n * sum(
                        coefficient * self.value(word1, rest2, reduced)
                        for reduced, coefficient in self.act(2, p - 1, word3).items()
                    )
        elif word1:
            n, rest1 = word1[0], word1[1:]
            answer = sum(
                coefficient * self.value(rest1, (), reduced)
                for reduced, coefficient in self.act(2, n, word3).items()
            )
            for mode in range(-1, n + 1):
                ward = math.comb(n + 1, mode + 1)
                answer += ward * sum(
                    coefficient * self.value(rest1, reduced, word3)
                    for reduced, coefficient in self.act(1, mode, ()).items()
                )
        elif word3:
            n, rest3 = word3[0], word3[1:]
            coefficient = self.weights[2] + sum(rest3) + n * self.weights[1] - self.weights[0]
            answer = coefficient * self.value((), (), rest3)
        else:
            answer = 1.0 + 0.0j
        self._value_cache[key] = answer
        return answer


class BranchWeights:
    def __init__(self, b: float, momenta):
        self.b = real_number(b)
        self.momenta = tuple(complex_number(value) for value in momenta)
        b = self.b
        self.b1 = scalar_sqrt(2 * b**2 / (1 - b**2))
        self.b2_inverse = scalar_sqrt(2 / (b**2 - 1))
        denominator1 = scalar_sqrt(2 - 2 * b**2)
        denominator2 = scalar_sqrt(2 - 2 / b**2)
        if abs(self.b1 / denominator1 + self.b2_inverse / denominator2) > abs(
            self.b1 / denominator1 - self.b2_inverse / denominator2
        ):
            self.b2_inverse = -self.b2_inverse
        # The chi-string states used below realize the correlated opposite
        # square-root branch.  This is fixed (rather than guessed) by
        # L_1^(i)L_-1^(i)v_n = 2 h_n^(i)v_n at a nonzero n.
        self.b1 = -self.b1
        self.b2_inverse = -self.b2_inverse
        self.denominators = (denominator1, denominator2)
        q1 = self.b1 + 1 / self.b1
        b2 = 1 / self.b2_inverse
        q2 = b2 + self.b2_inverse
        self.q_copies = (q1, q2)
        self.central_charges = (1 + 6 * q1**2, 1 + 6 * q2**2)

    def weight(self, leg: int, label: Fraction, copy: int):
        momentum = self.momenta[leg]
        label_value = real_number(label)
        if copy == 0:
            branched = momentum / self.denominators[0] + label_value * self.b1
        else:
            branched = momentum / self.denominators[1] + label_value * self.b2_inverse
        return self.q_copies[copy] ** 2 / 4 - branched**2

    def triple(self, labels, copy):
        return tuple(self.weight(leg, labels[leg], copy) for leg in range(3))


def ordinary_factor(weights: BranchWeights, labels, slot, term: ActionTerm):
    changed = list(labels)
    changed[slot] = term.label
    answer = term.coefficient
    for copy in (0, 1):
        evaluator = VirasoroThreePoint(
            weights.triple(tuple(changed), copy), weights.central_charges[copy]
        )
        words = [(), (), ()]
        words[slot] = term.first if copy == 0 else term.second
        answer *= evaluator.value(*words)
    return tuple(changed), answer


def check_branch_weights(module, weights, leg, labels, sector):
    checks = []
    for label in labels:
        primary = (
            module.ns_branch(int(label))
            if sector == "NS"
            else module.r_branch(label, 0)
        )
        for copy in (0, 1):
            acted = module.apply_embedded(
                copy + 1,
                1,
                module.apply_embedded(copy + 1, -1, primary),
            )
            fit = span_fit(acted, [primary])
            inferred = fit["coefficients"][0] / 2
            expected = weights.weight(leg, label, copy)
            checks.append(
                {
                    "label": format_fraction(label),
                    "copy": copy + 1,
                    "inferred": encode_complex(inferred),
                    "expected": encode_complex(expected),
                    "absolute_difference": float(abs(inferred - expected)),
                    "relative_state_residual": fit["relative_residual"],
                }
            )
    return checks


def direct_ground_value(
    second_module,
    third_module,
    second_label,
    third_label,
    alpha2,
    alpha3,
    eta,
):
    """Evaluate the tensor-ground form in the literal Human-Note basis.

    The free-field endpoint labelled ``physical=1`` is not itself ``w^-``.
    Equation (5.1), applied to the realization used here, gives

        |raw,1> = -exp(-i*pi/4) |w^->.

    The tensor sign is the ground specialization of the defining hatted form
    in Section 8, not the standard tensor-product Koszul convention.
    """
    second = second_module.r_branch(second_label, alpha2)
    third = third_module.r_branch(third_label, alpha3)
    form_parity = (alpha2 + alpha3) % 2
    raw_to_human_minus = -cmath.exp(-0.25j * math.pi)
    answer = 0.0j
    for state2, coefficient2 in second.items():
        for state3, coefficient3 in third.items():
            if state2[0] or state2[2] or state2[3] or state3[0] or state3[2] or state3[3]:
                raise AssertionError("A ground anchor contains an oscillator excitation.")
            auxiliary2, physical2 = state2[1], state2[4]
            auxiliary3, physical3 = state3[1], state3[4]
            if auxiliary2 != auxiliary3:
                continue
            auxiliary_form = 1 if auxiliary2 == 0 else -1
            tensor_sign = (-1) ** (
                (physical2 + form_parity) * auxiliary3
            )
            if form_parity == 0:
                if (physical2, physical3) == (0, 0):
                    physical_form = 1
                elif (physical2, physical3) == (1, 1):
                    physical_form = eta
                else:
                    physical_form = 0
            else:
                if (physical2, physical3) == (0, 1):
                    physical_form = 1
                elif (physical2, physical3) == (1, 0):
                    physical_form = 1j * eta
                else:
                    physical_form = 0
            answer += (
                coefficient2
                * coefficient3
                * (raw_to_human_minus if physical2 else 1)
                * (raw_to_human_minus if physical3 else 1)
                * tensor_sign
                * auxiliary_form
                * physical_form
            )
    return answer


def finite_ward_solution(
    weights,
    ns_l1,
    second_lminus,
    third_lminus,
    labels1,
    labels2,
    labels3,
    second_module,
    third_module,
    alpha2,
    alpha3,
    eta,
    ground_value_fn=None,
):
    """Close the finite system using only the first Ward identity in the notes."""
    labels1 = tuple(Fraction(value) for value in labels1)
    labels2 = tuple(Fraction(value) for value in labels2)
    labels3 = tuple(Fraction(value) for value in labels3)
    unknowns = tuple(
        (first, second, third)
        for first in labels1
        for second in labels2
        for third in labels3
    )
    index = {labels: position for position, labels in enumerate(unknowns)}
    rows = []
    right_hand_sides = []

    def append_equation(equation):
        if MP_DPS:
            norm = mp.sqrt(mp.fsum(abs(value) ** 2 for value in equation))
            normalized = [value / norm for value in equation] if norm else None
        else:
            equation = np.asarray(equation, dtype=np.complex128)
            norm = np.linalg.norm(equation)
            normalized = equation / norm if norm else None
        if norm > arithmetic_tolerance():
            rows.append(normalized)
            right_hand_sides.append(complex_number())

    for labels in unknowns:
        first_slot = [complex_number() for _ in unknowns]
        for term in ns_l1[labels[0]]:
            changed, coefficient = ordinary_factor(weights, labels, 0, term)
            if changed not in index:
                raise AssertionError(f"The NS label box is not closed at {changed}.")
            first_slot[index[changed]] += coefficient
        for term in second_lminus[labels[1]]:
            changed, coefficient = ordinary_factor(weights, labels, 1, term)
            if changed not in index:
                raise AssertionError(f"The second Ramond label box is not closed at {changed}.")
            first_slot[index[changed]] -= coefficient
        for term in third_lminus[labels[2]]:
            changed, coefficient = ordinary_factor(weights, labels, 2, term)
            if changed not in index:
                raise AssertionError(f"The third Ramond label box is not closed at {changed}.")
            first_slot[index[changed]] -= coefficient
        append_equation(first_slot)

    ground_labels2 = tuple(label for label in labels2 if abs(label) == Fraction(1, 4))
    ground_labels3 = tuple(label for label in labels3 if abs(label) == Fraction(1, 4))
    if len(ground_labels2) != 1 or len(ground_labels3) != 1:
        raise AssertionError("Each closed Ramond component must contain one ground label.")
    anchors = {}
    for second_label in ground_labels2:
        for third_label in ground_labels3:
            labels = (Fraction(0), second_label, third_label)
            value = (ground_value_fn or direct_ground_value)(
                second_module,
                third_module,
                second_label,
                third_label,
                alpha2,
                alpha3,
                eta,
            )
            anchors[labels] = value
            row = [complex_number() for _ in unknowns]
            row[index[labels]] = 1
            rows.append(row)
            right_hand_sides.append(value)

    if MP_DPS:
        matrix = mp.matrix(rows)
        vector = mp.matrix(right_hand_sides)
        row_count, column_count = matrix.rows, matrix.cols
        column_norms = [
            mp.sqrt(mp.fsum(abs(matrix[row, column]) ** 2 for row in range(row_count)))
            for column in range(column_count)
        ]
        if any(norm == 0 for norm in column_norms):
            raise AssertionError("A branching coefficient is absent from the Ward system.")
        scaled_matrix = mp.matrix(
            [
                [matrix[row, column] / column_norms[column] for column in range(column_count)]
                for row in range(row_count)
            ]
        )
        ward_gram = mp.matrix(column_count)
        ward_rhs = mp.matrix(column_count, 1)
        for first in range(column_count):
            ward_rhs[first] = mp.fsum(
                mp.conj(scaled_matrix[row, first]) * vector[row]
                for row in range(row_count)
            )
            for second in range(first + 1):
                value = mp.fsum(
                    mp.conj(scaled_matrix[row, first])
                    * scaled_matrix[row, second]
                    for row in range(row_count)
                )
                ward_gram[first, second] = value
                ward_gram[second, first] = mp.conj(value)
        eigenvalues, eigenvectors = mp.eighe(ward_gram)
        maximum_eigenvalue = max(abs(value) for value in eigenvalues)
        eigenvalue_minimum = float(min(mp.re(value) for value in eigenvalues))
        eigenvalue_maximum = float(max(mp.re(value) for value in eigenvalues))
        rank_threshold = maximum_eigenvalue * mp.power(10, -(MP_DPS - 15))
        active = [
            position
            for position, value in enumerate(eigenvalues)
            if value > rank_threshold
        ]
        rank = len(active)
        scaled_solution = mp.matrix(column_count, 1)
        for position in active:
            projection = mp.fsum(
                mp.conj(eigenvectors[row, position]) * ward_rhs[row]
                for row in range(column_count)
            ) / eigenvalues[position]
            for row in range(column_count):
                scaled_solution[row] += eigenvectors[row, position] * projection
        solution = [
            scaled_solution[column] / column_norms[column]
            for column in range(column_count)
        ]
        residual = matrix * mp.matrix(solution) - vector
        residual_norm = mp.sqrt(mp.fsum(abs(value) ** 2 for value in residual))
        vector_norm = mp.sqrt(mp.fsum(abs(value) ** 2 for value in vector))
        singular_values = np.asarray(
            [float(mp.sqrt(max(eigenvalues[position], 0))) for position in reversed(active)],
            dtype=float,
        )
        if not len(singular_values):
            singular_values = np.asarray([0.0])
        smallest_singular_value = (
            float(mp.sqrt(eigenvalues[active[0]])) if active else 0.0
        )
        scaled_condition_number = (
            float(mp.sqrt(eigenvalues[active[-1]] / eigenvalues[active[0]]))
            if active
            else math.inf
        )
        solve_rcond = None
        solver = f"mpmath-rank-revealing-normal-equations-{MP_DPS}dps"
    else:
        matrix = np.asarray(rows, dtype=np.complex128)
        vector = np.asarray(right_hand_sides, dtype=np.complex128)
        row_count, column_count = matrix.shape
        column_norms = np.linalg.norm(matrix, axis=0)
        if np.any(column_norms == 0):
            raise AssertionError("A branching coefficient is absent from the Ward system.")
        scaled_matrix = matrix / column_norms
        candidates = []
        for rcond in (1.0e-13, 1.0e-14, 1.0e-15, 1.0e-16):
            scaled_solution, _, rank, singular_values = np.linalg.lstsq(
                scaled_matrix, vector, rcond=rcond
            )
            solution = scaled_solution / column_norms
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                residual = matrix @ solution - vector
            if not np.all(np.isfinite(residual)):
                continue
            residual_norm = float(np.linalg.norm(residual))
            candidates.append(
                {
                    "solution": solution,
                    "rank": int(rank),
                    "singular_values": singular_values,
                    "residual": residual,
                    "residual_norm": residual_norm,
                    "rcond": rcond,
                }
            )
        if not candidates:
            raise FloatingPointError("The finite Ward solve produced no finite solution.")
        selected = max(
            candidates,
            key=lambda candidate: (
                candidate["rank"] == matrix.shape[1],
                candidate["rank"],
                -candidate["residual_norm"],
            ),
        )
        solution = selected["solution"]
        rank = selected["rank"]
        singular_values = selected["singular_values"]
        residual = selected["residual"]
        residual_norm = float(np.linalg.norm(residual))
        vector_norm = float(np.linalg.norm(vector))
        solve_rcond = selected["rcond"]
        solver = "numpy-adaptive-lstsq-complex128"
        smallest_singular_value = float(singular_values[rank - 1])
        scaled_condition_number = float(singular_values[0] / singular_values[-1])
        eigenvalue_minimum = None
        eigenvalue_maximum = None
    return {
        "unknowns": unknowns,
        "values": solution,
        "anchors": anchors,
        "rows": int(row_count),
        "columns": int(column_count),
        "rank": int(rank),
        "absolute_residual": float(residual_norm),
        "relative_residual": float(residual_norm / max(vector_norm, 1)),
        "smallest_singular_value": smallest_singular_value,
        "solve_rcond": solve_rcond,
        "scaled_condition_number": scaled_condition_number,
        "solver": solver,
        "hermitian_eigenvalue_minimum": eigenvalue_minimum,
        "hermitian_eigenvalue_maximum": eigenvalue_maximum,
    }


def encode_complex(value):
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def fit_summary(fit):
    return {
        key: value
        for key, value in fit.items()
        if key != "coefficients"
    }


def format_fraction(value: Fraction):
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def ns_norm_ratio(label: Fraction, b: float, momentum: float):
    """The ratio ||v_{n-1}||/||v_n|| displayed in SCblock.tex."""
    label = Fraction(label)
    q = b + 1 / b
    upper = int(4 * label)
    return 2 * scalar_sqrt(
        ell(2 * momentum, upper - 4, b)
        * ell(q + 2 * momentum, upper - 4, b)
        / (
            ell(2 * momentum, upper, b)
            * ell(q + 2 * momentum, upper, b)
        )
    )


def ramond_norm_ratio(
    label: Fraction,
    b: float,
    momentum: float,
    parity: int | None = None,
):
    """The ratio ||v_{n-1}^alpha||/||v_n^alpha|| in SCblock.tex.

    Supplying ``parity`` evaluates the ratio from the Human-Note norms.  This
    matters at the chart crossing 3/4 -> -1/4, where the reflected alpha=0
    norm contributes an additional factor of two.  The closed product below
    is retained as the parity-independent positive-chart formula.
    """
    label = Fraction(label)
    if parity is not None:
        return scalar_sqrt(
            ramond_norm_squared(label - 1, parity, b, momentum)
            / ramond_norm_squared(label, parity, b, momentum)
        )
    q = b + 1 / b
    upper = int(4 * label)
    return real_number(Fraction(1, 2)) * scalar_sqrt(
        ell(2 * momentum, upper - 4, b)
        * ell(q + 2 * momentum, upper, b)
        / (
            ell(q + 2 * momentum, upper - 4, b)
            * ell(2 * momentum, upper, b)
        )
    )


def ns_norm_squared(label: Fraction, b: float, momentum: float):
    """Human-Note NS branch norm, including its reflection prescription."""

    label = Fraction(label)
    if label < 0:
        # Human Notes/SCblock.tex: v_n(P)=v_{-n}(-P).
        return ns_norm_squared(-label, b, -momentum)
    q = b + 1 / b
    return (-1) ** int(2 * label) * scalar_power_of_two(-2 * label) * ell(
        2 * momentum, int(4 * label), b
    ) * ell(q + 2 * momentum, int(4 * label), b)


def ramond_norm_squared(label: Fraction, parity: int, b: float, momentum: float):
    """Human-Note Ramond branch norm, including the reflected chart."""

    label = Fraction(label)
    if label < 0:
        # Human Notes/SCblock.tex (5.1): the two components reflect with
        # opposite signs.  That relative sign drops out of the BPZ norm, so
        # both parities use the positive-label norm at reflected momentum.
        return ramond_norm_squared(-label, parity, b, -momentum)
    mode_count = int(2 * label - Fraction(1, 2))
    if parity == 0:
        power = 2 * (mode_count // 2) + 1
        discrete = scalar_power_of_two(power)
    else:
        power = 2 * ((mode_count + 1) // 2)
        discrete = -scalar_power_of_two(power)
    q = b + 1 / b
    return discrete * ell(2 * momentum, int(4 * label), b) / ell(
        q + 2 * momentum, int(4 * label), b
    )


def norm_product(labels, alpha2, alpha3, b, momenta):
    return (
        scalar_sqrt(ns_norm_squared(labels[0], b, momenta[0]))
        * scalar_sqrt(ramond_norm_squared(labels[1], alpha2, b, momenta[1]))
        * scalar_sqrt(ramond_norm_squared(labels[2], alpha3, b, momenta[2]))
    )


def same_branch_coefficient(terms, label, copy):
    first = (1,) if copy == 0 else ()
    second = () if copy == 0 else (1,)
    matches = [
        term.coefficient
        for term in terms
        if term.label == label and term.first == first and term.second == second
    ]
    if len(matches) != 1:
        raise AssertionError("The Ramond L_-1 decomposition has the wrong same-branch terms.")
    return matches[0]


def add_scaled(target, source, scale):
    for key, value in source.items():
        combined = target.get(key, 0.0j) + scale * value
        if abs(combined) <= arithmetic_tolerance():
            target.pop(key, None)
        else:
            target[key] = combined


class BranchingRecursion:
    """The normalized three-term recursion written in the main notes."""

    def __init__(
        self,
        b,
        momenta,
        weights,
        ns_actions,
        second_actions,
        third_actions,
        second_parity=None,
        third_parity=None,
    ):
        self.b = b
        self.momenta = momenta
        self.weights = weights
        self.ns_actions = ns_actions
        self.second_actions = second_actions
        self.third_actions = third_actions
        self.second_parity = second_parity
        self.third_parity = third_parity
        self.expansion_cache = {}
        self.node_data = {}

    @staticmethod
    def is_interior(labels):
        return (
            labels[0] >= 1
            and labels[1] >= Fraction(3, 4)
            and labels[2] >= Fraction(3, 4)
        )

    def node_coefficients(self, labels):
        labels = tuple(Fraction(value) for value in labels)
        if labels in self.node_data:
            data = self.node_data[labels]
            return tuple(data["coefficients"])
        if not self.is_interior(labels):
            raise ValueError("A boundary point has no recursion coefficients.")

        first_actions = self.ns_actions[labels[0]]
        second_actions = self.second_actions[labels[1]]
        third_actions = self.third_actions[labels[2]]
        lower_labels = (
            (labels[0] - 1, labels[1], labels[2]),
            (labels[0], labels[1] - 1, labels[2]),
            (labels[0], labels[1], labels[2] - 1),
        )

        first_sum = 0.0j
        for term in first_actions:
            changed, coefficient = ordinary_factor(self.weights, labels, 0, term)
            if changed != lower_labels[0]:
                raise AssertionError("An NS L_1 term did not land on n_1-1.")
            first_sum += coefficient

        second_sum = 0.0j
        for term in second_actions:
            if term.label == labels[1]:
                continue
            changed, coefficient = ordinary_factor(self.weights, labels, 1, term)
            if changed != lower_labels[1]:
                raise AssertionError("A Ramond L_-1 term did not land on n_2-1.")
            second_sum += coefficient

        third_sum = 0.0j
        for term in third_actions:
            if term.label == labels[2]:
                continue
            changed, coefficient = ordinary_factor(self.weights, labels, 2, term)
            if changed != lower_labels[2]:
                raise AssertionError("A Ramond L_-1 term did not land on n_3-1.")
            third_sum += coefficient

        denominator = 0.0j
        for copy in (0, 1):
            second_same = same_branch_coefficient(
                second_actions, labels[1], copy
            )
            third_same = same_branch_coefficient(third_actions, labels[2], copy)
            h1, h2, h3 = self.weights.triple(labels, copy)
            denominator += (second_same - third_same) * (h2 + h3 - h1)
        if abs(denominator) <= 1.0e-11:
            raise ZeroDivisionError(f"The recursion denominator vanished at {labels}.")

        coefficients = (
            -first_sum
            * ns_norm_ratio(labels[0], self.b, self.momenta[0])
            / denominator,
            second_sum
            * ramond_norm_ratio(
                labels[1], self.b, self.momenta[1], self.second_parity
            )
            / denominator,
            third_sum
            * ramond_norm_ratio(
                labels[2], self.b, self.momenta[2], self.third_parity
            )
            / denominator,
        )
        self.node_data[labels] = {
            "denominator": denominator,
            "first_descendant_sum": first_sum,
            "second_descendant_sum": second_sum,
            "third_descendant_sum": third_sum,
            "coefficients": coefficients,
        }
        return coefficients

    def expansion(self, labels):
        labels = tuple(Fraction(value) for value in labels)
        if labels in self.expansion_cache:
            return self.expansion_cache[labels]
        if not self.is_interior(labels):
            answer = {labels: 1.0 + 0.0j}
        else:
            children = (
                (labels[0] - 1, labels[1], labels[2]),
                (labels[0], labels[1] - 1, labels[2]),
                (labels[0], labels[1], labels[2] - 1),
            )
            answer = {}
            for coefficient, child in zip(self.node_coefficients(labels), children):
                add_scaled(answer, self.expansion(child), coefficient)
        self.expansion_cache[labels] = answer
        return answer


def boundary_entry(alpha2, alpha3, eta, labels, value=None):
    entry = {
        "alpha2": alpha2,
        "alpha3": alpha3,
        "eta": eta,
        "n1": format_fraction(labels[0]),
        "n2": format_fraction(labels[1]),
        "n3": format_fraction(labels[2]),
    }
    entry["B"] = (
        {"real": None, "imag": None} if value is None else encode_complex(value)
    )
    return entry


def main():
    parser = argparse.ArgumentParser(
        description="Compute the Ramond branching recursion at arbitrary supported labels."
    )
    parser.add_argument("--b", default="7/5")
    parser.add_argument("--p1", default="11/23")
    parser.add_argument("--p2", default="13/29")
    parser.add_argument("--p3", default="17/31")
    parser.add_argument("--n1", type=parse_label, default=Fraction(2))
    parser.add_argument("--n2", type=parse_label, default=Fraction(7, 4))
    parser.add_argument("--n3", type=parse_label, default=Fraction(5, 4))
    parser.add_argument(
        "--mp-dps",
        type=int,
        default=0,
        help="Use mpmath arithmetic at this many decimal digits (0 keeps complex128).",
    )
    parser.add_argument("--json", type=Path, default=HERE / "results.json")
    arguments = parser.parse_args()

    if arguments.mp_dps and arguments.mp_dps < 30:
        parser.error("--mp-dps must be 0 or at least 30.")
    set_multiprecision(arguments.mp_dps)

    try:
        target = validate_target((arguments.n1, arguments.n2, arguments.n3))
    except ValueError as error:
        parser.error(str(error))
    labels1 = ns_label_closure(target[0])
    labels2 = ramond_label_closure(target[1])
    labels3 = ramond_label_closure(target[2])

    started = time.perf_counter()
    b = parse_number(arguments.b)
    momenta = tuple(parse_number(value) for value in (arguments.p1, arguments.p2, arguments.p3))
    ns_module = FreeFieldModule("NS", b, momenta[0])
    second_module = FreeFieldModule("R", b, momenta[1])
    third_module = FreeFieldModule("R", b, momenta[2])

    decomposition_started = time.perf_counter()
    ns_l1 = {Fraction(0): []}
    decomposition_checks = {
        "NS_L1": {},
        "R2_Lminus1": {},
        "R3_Lminus1": {},
    }
    for label in labels1[1:]:
        terms, fit = solve_ns_l1(ns_module, int(label))
        ns_l1[label] = terms
        decomposition_checks["NS_L1"][format_fraction(label)] = fit_summary(fit)

    second_lminus_by_alpha = {}
    for alpha2 in (0, 1):
        lminus_actions = {}
        for label in labels2:
            terms, fit = solve_ramond_lminus(second_module, label, alpha2)
            lminus_actions[label] = terms
            decomposition_checks["R2_Lminus1"][f"{label};alpha={alpha2}"] = fit_summary(fit)
        second_lminus_by_alpha[alpha2] = lminus_actions

    third_lminus_by_alpha = {}
    for alpha3 in (0, 1):
        lminus_actions = {}
        for label in labels3:
            terms, fit = solve_ramond_lminus(third_module, label, alpha3)
            lminus_actions[label] = terms
            decomposition_checks["R3_Lminus1"][f"{label};alpha={alpha3}"] = fit_summary(fit)
        third_lminus_by_alpha[alpha3] = lminus_actions
    decomposition_time = time.perf_counter() - decomposition_started

    weights = BranchWeights(b, momenta)
    branch_weight_checks = {
        "NS": check_branch_weights(
            ns_module, weights, 0, labels1, "NS"
        ),
        "R2": check_branch_weights(
            second_module, weights, 1, labels2, "R"
        ),
        "R3": check_branch_weights(
            third_module, weights, 2, labels3, "R"
        ),
    }
    maximum_weight_difference = max(
        item["absolute_difference"]
        for checks in branch_weight_checks.values()
        for item in checks
    )
    maximum_weight_state_residual = max(
        item["relative_state_residual"]
        for checks in branch_weight_checks.values()
        for item in checks
    )
    all_results = []
    minimum_denominator = None
    maximum_ward_residual = 0.0
    maximum_recursive_disagreement = 0.0
    recursion_started = time.perf_counter()
    for alpha2 in (0, 1):
        for alpha3 in (0, 1):
            recursion = BranchingRecursion(
                b,
                momenta,
                weights,
                ns_l1,
                second_lminus_by_alpha[alpha2],
                third_lminus_by_alpha[alpha3],
                second_parity=alpha2,
                third_parity=alpha3,
            )
            expansion = recursion.expansion(target)
            if recursion.node_data:
                candidate = min(
                    abs(data["denominator"])
                    for data in recursion.node_data.values()
                )
                minimum_denominator = (
                    candidate
                    if minimum_denominator is None
                    else min(minimum_denominator, candidate)
                )
            expansion_entries = [
                {
                    "n1": format_fraction(labels[0]),
                    "n2": format_fraction(labels[1]),
                    "n3": format_fraction(labels[2]),
                    "coefficient": encode_complex(coefficient),
                }
                for labels, coefficient in sorted(expansion.items())
            ]
            evaluations = []
            for eta in (1, -1):
                ward = finite_ward_solution(
                    weights,
                    ns_l1,
                    second_lminus_by_alpha[alpha2],
                    third_lminus_by_alpha[alpha3],
                    labels1,
                    labels2,
                    labels3,
                    second_module,
                    third_module,
                    alpha2,
                    alpha3,
                    eta,
                )
                maximum_ward_residual = max(
                    maximum_ward_residual, ward["relative_residual"]
                )
                ward_index = {
                    labels: position for position, labels in enumerate(ward["unknowns"])
                }
                boundary_branching = {
                    labels: ward["values"][ward_index[labels]]
                    / norm_product(labels, alpha2, alpha3, b, momenta)
                    for labels in expansion
                }
                recursive_value = sum(
                    coefficient * boundary_branching[labels]
                    for labels, coefficient in expansion.items()
                )
                direct_value = ward["values"][ward_index[target]] / norm_product(
                    target, alpha2, alpha3, b, momenta
                )
                disagreement = float(
                    abs(recursive_value - direct_value)
                    / max(abs(direct_value), real_number(1))
                )
                maximum_recursive_disagreement = max(
                    maximum_recursive_disagreement, disagreement
                )
                evaluations.append(
                    {
                        "eta": eta,
                        "B_from_recursion": encode_complex(recursive_value),
                        "B_from_finite_Ward_solution": encode_complex(direct_value),
                        "relative_disagreement": disagreement,
                        "ward_system": {
                            "rows": ward["rows"],
                            "columns": ward["columns"],
                            "rank": ward["rank"],
                            "absolute_residual": ward["absolute_residual"],
                            "relative_residual": ward["relative_residual"],
                            "smallest_singular_value": ward["smallest_singular_value"],
                            "solve_rcond": ward["solve_rcond"],
                            "scaled_condition_number": ward["scaled_condition_number"],
                            "solver": ward["solver"],
                            "hermitian_eigenvalue_minimum": ward[
                                "hermitian_eigenvalue_minimum"
                            ],
                            "hermitian_eigenvalue_maximum": ward[
                                "hermitian_eigenvalue_maximum"
                            ],
                            "ground_anchors": [
                                {
                                    "n2": format_fraction(labels[1]),
                                    "n3": format_fraction(labels[2]),
                                    "value": encode_complex(value),
                                }
                                for labels, value in sorted(ward["anchors"].items())
                            ],
                        },
                        "boundary_values_used": [
                            boundary_entry(
                                alpha2,
                                alpha3,
                                eta,
                                labels,
                                boundary_branching[labels],
                            )
                            for labels in sorted(expansion)
                        ],
                    }
                )
            all_results.append(
                {
                    "alpha2": alpha2,
                    "alpha3": alpha3,
                    "f": (alpha2 + alpha3) % 2,
                    "boundary_terms": len(expansion),
                    "interior_nodes": len(recursion.node_data),
                    "expansion": expansion_entries,
                    "evaluations": evaluations,
                }
            )
            print(
                f"alpha2={alpha2}, alpha3={alpha3}: "
                f"{len(recursion.node_data)} interior nodes, "
                f"{len(expansion)} boundary coefficients, "
                f"max recursion disagreement={max(item['relative_disagreement'] for item in evaluations):.3e}",
                flush=True,
            )
    recursion_time = time.perf_counter() - recursion_started
    total_time = time.perf_counter() - started
    maximum_fit_residual = max(
        item["relative_residual"]
        for sector in decomposition_checks.values()
        for item in sector.values()
    )
    minimum_fit_rank_margin = min(
        item["rank"] - item["columns"]
        for sector in decomposition_checks.values()
        for item in sector.values()
    )
    all_ward_systems_full_rank = all(
        evaluation["ward_system"]["rank"]
        == evaluation["ward_system"]["columns"]
        for result in all_results
        for evaluation in result["evaluations"]
    )
    passed = bool(
        maximum_fit_residual < 1.0e-8
        and minimum_fit_rank_margin == 0
        and maximum_weight_difference < 1.0e-10
        and maximum_weight_state_residual < 1.0e-10
        and all_ward_systems_full_rank
        and maximum_ward_residual < 1.0e-8
        and maximum_recursive_disagreement < 1.0e-7
    )
    payload = {
        "convention": "Human Notes/SCblock.tex only",
        "external_ramond_convention_conversion_used": False,
        "pbw_double_virasoro_match_certified": False,
        "target": {
            "n1": format_fraction(target[0]),
            "n2": format_fraction(target[1]),
            "n3": format_fraction(target[2]),
        },
        "arithmetic": {
            "backend": "mpmath" if MP_DPS else "complex128",
            "decimal_digits": MP_DPS if MP_DPS else None,
        },
        "ward_label_sets": {
            "n1": [format_fraction(value) for value in labels1],
            "n2": [format_fraction(value) for value in labels2],
            "n3": [format_fraction(value) for value in labels3],
        },
        "point": {
            "b": arguments.b,
            "P1": arguments.p1,
            "P2": arguments.p2,
            "P3": arguments.p3,
            "Q": float(b + 1 / b),
            "c": float(real_number(Fraction(3, 2)) + 3 * (b + 1 / b) ** 2),
        },
        "decomposition_checks": decomposition_checks,
        "maximum_decomposition_relative_residual": maximum_fit_residual,
        "branch_weight_checks": branch_weight_checks,
        "maximum_branch_weight_absolute_difference": maximum_weight_difference,
        "maximum_branch_weight_state_residual": maximum_weight_state_residual,
        "minimum_recursion_denominator_absolute_value": (
            None if minimum_denominator is None else float(minimum_denominator)
        ),
        "maximum_finite_Ward_relative_residual": maximum_ward_residual,
        "maximum_recursion_vs_Ward_relative_disagreement": maximum_recursive_disagreement,
        "results": all_results,
        "decomposition_seconds": decomposition_time,
        "recursion_seconds": recursion_time,
        "total_seconds": total_time,
        "passed": passed,
    }
    arguments.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"maximum decomposition residual={maximum_fit_residual:.3e}")
    print(f"maximum branch-weight difference={maximum_weight_difference:.3e}")
    if minimum_denominator is None:
        print("minimum |recursion denominator|=not applicable (boundary target)")
    else:
        print(f"minimum |recursion denominator|={float(minimum_denominator):.3e}")
    print(f"maximum finite-Ward residual={maximum_ward_residual:.3e}")
    print(f"maximum recursion/Ward disagreement={maximum_recursive_disagreement:.3e}")
    print(f"total runtime={total_time:.3f} s")
    print("PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
