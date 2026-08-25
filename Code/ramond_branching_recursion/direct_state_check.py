#!/usr/bin/env python3
"""Direct low-state check of the Ramond branching recursion.

This file deliberately does not solve a lattice of branching-coefficient Ward
identities.  It constructs every branch primary as a state in

    auxiliary free fermion tensor physical SCA Verma module,

converts the physical free-field oscillator coordinates to the SCA PBW basis,
and evaluates the auxiliary and physical three-point functions separately.
The tensor-product sign is then inserted exactly as in SCblock.tex.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np

from compute_target import (
    HERE,
    TOLERANCE,
    ActionTerm,
    BranchWeights,
    BranchingRecursion,
    FreeFieldModule,
    VirasoroThreePoint,
    add_term,
    encode_complex,
    format_fraction,
    norm_product,
    max_abs,
    partitions,
    ramond_norm_squared,
    solve_ns_l1,
    solve_ramond_lminus,
    span_fit,
    strict_odd_partitions,
    strict_partitions,
    ns_norm_squared,
)


def generalized_binomial(upper, lower: int):
    if lower < 0:
        return 0.0
    answer = 1.0
    upper = complex(upper)
    for index in range(lower):
        answer *= (upper - index) / (index + 1)
    return answer


def expression_level(module: FreeFieldModule, expression):
    levels = {module.physical_level_units(state) for state in expression}
    if len(levels) != 1:
        raise AssertionError("A physical expression mixes levels.")
    return levels.pop()


class PBWModule:
    """Numerical change of basis between free oscillators and SCA PBW states."""

    def __init__(self, module: FreeFieldModule):
        self.module = module
        self.sector = module.sector
        self.q = module.q
        self.central_charge = 1.5 + 3 * self.q**2
        if self.sector == "NS":
            self.weight = 0.5 * (self.q**2 / 4 - module.momentum**2)
        else:
            self.weight = 1 / 16 + 0.5 * (
                self.q**2 / 4 - module.momentum**2
            )
        self._inner_cache = {}

    @lru_cache(None)
    def basis(self, level_units: int):
        if self.sector == "NS":
            rows, matrix = self.module._ns_level_transition(
                self.module.realization, level_units
            )
            metadata = tuple(
                (virasoro_modes, supercurrent_modes)
                for virasoro_level in range(level_units // 2 + 1)
                for virasoro_modes in partitions(virasoro_level)
                for supercurrent_modes in strict_odd_partitions(
                    level_units - 2 * virasoro_level
                )
            )
        else:
            rows, matrix = self.module._level_transition(
                self.module.realization, level_units
            )
            metadata = tuple(
                (virasoro_modes, supercurrent_modes, ground)
                for virasoro_level in range(level_units + 1)
                for virasoro_modes in partitions(virasoro_level)
                for supercurrent_modes in strict_partitions(
                    level_units - virasoro_level
                )
                for ground in (0, 1)
            )
        if len(metadata) != matrix.shape[1]:
            raise AssertionError("PBW metadata and transition matrix disagree.")
        return rows, matrix, metadata

    def level_units(self, state):
        if self.sector == "NS":
            virasoro_modes, supercurrent_modes = state
            return 2 * sum(virasoro_modes) + sum(supercurrent_modes)
        virasoro_modes, supercurrent_modes, _ = state
        return sum(virasoro_modes) + sum(supercurrent_modes)

    def parity(self, state):
        if self.sector == "NS":
            return len(state[1]) % 2
        return (len(state[1]) + state[2]) % 2

    def l0_weight(self, state):
        divisor = 2 if self.sector == "NS" else 1
        return self.weight + self.level_units(state) / divisor

    def has_oscillators(self, state):
        return bool(state[0] or state[1])

    def ground(self, state):
        return 0 if self.sector == "NS" else state[2]

    def from_fock(self, expression):
        if not expression:
            return {}
        level_units = expression_level(self.module, expression)
        rows, matrix, metadata = self.basis(level_units)
        vector = np.asarray(
            [expression.get(row, 0.0j) for row in rows], dtype=np.complex128
        )
        coefficients = np.linalg.solve(matrix, vector)
        residual = np.linalg.norm(matrix @ coefficients - vector)
        if residual > 2.0e-9 * max(np.linalg.norm(vector), 1.0):
            raise AssertionError("Free-field to PBW conversion lost accuracy.")
        return {
            state: coefficient
            for state, coefficient in zip(metadata, coefficients)
            if abs(coefficient) > TOLERANCE
        }

    def to_fock(self, state):
        level_units = self.level_units(state)
        rows, matrix, metadata = self.basis(level_units)
        column = metadata.index(state)
        return {
            row: coefficient
            for row, coefficient in zip(rows, matrix[:, column])
            if abs(coefficient) > TOLERANCE
        }

    @lru_cache(None)
    def act(self, kind: str, mode, state):
        if kind == "L" and mode == 0:
            return ((state, complex(self.l0_weight(state))),)
        input_expression = self.to_fock(state)
        if kind == "L":
            output = {}
            for initial, outer in input_expression.items():
                for final, inner in self.module.physical_l_on_state(
                    int(mode), initial
                ):
                    add_term(output, final, outer * inner)
        elif kind == "G":
            output = {}
            for initial, outer in input_expression.items():
                for final, inner in self.module.physical_g_on_state(mode, initial):
                    add_term(output, final, outer * inner)
        else:
            raise ValueError("kind must be L or G")
        return tuple(self.from_fock(output).items())

    def inner(self, left, right):
        key = (left, right)
        if key in self._inner_cache:
            return self._inner_cache[key]
        expression = {right: 1.0 + 0.0j}
        for mode in left[0]:
            next_expression = {}
            for state, outer in expression.items():
                for final, inner in self.act("L", mode, state):
                    add_term(next_expression, final, outer * inner)
            expression = next_expression
        for mode in left[1]:
            next_expression = {}
            for state, outer in expression.items():
                for final, inner in self.act("G", mode, state):
                    add_term(next_expression, final, outer * inner)
            expression = next_expression
        if self.sector == "NS":
            answer = expression.get(((), ()), 0.0j)
        else:
            answer = expression.get(((), (), left[2]), 0.0j)
        self._inner_cache[key] = answer
        return answer


def branch_in_pbw(module: FreeFieldModule, pbw: PBWModule, expression):
    grouped = {}
    for state, coefficient in expression.items():
        auxiliary, physical = module.split_state(state)
        add_term(grouped.setdefault(auxiliary, {}), physical, coefficient)
    answer = {}
    for auxiliary, physical_expression in grouped.items():
        for physical, coefficient in pbw.from_fock(physical_expression).items():
            add_term(answer, (auxiliary, physical), coefficient)
    return answer


def add_acted_triples(target, coefficient, slot, acted, states):
    for final, inner in acted:
        changed = list(states)
        changed[slot] = final
        add_term(target, tuple(changed), coefficient * inner)


class PhysicalThreePoint:
    """The canonical eta=(-1)^f NS-R-R SCA trilinear form.

    ``form_parity`` is the relative Human-Note label ``f``: it does not
    include the intrinsic parity of the NS primary.  ``primary_parity`` is
    included whenever an actual NS state crosses a supercurrent contour.
    Thus changing the primary parity changes the odd-NS Ward boundary but
    leaves the Virasoro Ward recursion and the ground component table fixed.
    """

    def __init__(
        self,
        modules,
        form_parity: int,
        eta: int,
        primary_parity: int = 0,
    ):
        self.modules = tuple(modules)
        self.form_parity = int(form_parity)
        self.primary_parity = int(primary_parity)
        self.eta = int(eta)
        if self.primary_parity not in (0, 1):
            raise ValueError("primary_parity must be 0 or 1")
        if self.eta != (-1) ** self.form_parity:
            raise ValueError("This evaluator implements the canonical eta=(-1)^f pair.")
        self.infinity_phase = -1j
        self.zero_phase = 1j
        self._cache = {}
        self._active = set()
        # The free-field Ramond ground basis used by FreeFieldModule is
        # f^0=w^+ and f^1=e^{3 pi i/4} w^- in the convention of SCblock.tex.
        self.ramond_odd_phase = cmath.exp(3j * math.pi / 4)

    def base_value(self, states):
        ground2 = self.modules[1].ground(states[1])
        ground3 = self.modules[2].ground(states[2])
        phase = self.ramond_odd_phase ** (ground2 + ground3)
        if self.form_parity == 0:
            if (ground2, ground3) == (0, 0):
                return 1.0 + 0.0j
            if (ground2, ground3) == (1, 1):
                return phase * self.eta
            return 0.0j
        if (ground2, ground3) == (0, 1):
            return phase
        if (ground2, ground3) == (1, 0):
            return phase * 1j * self.eta
        return 0.0j

    def _sum_action(self, states, slot, kind, mode):
        answer = 0.0j
        for final, coefficient in self.modules[slot].act(
            kind, mode, states[slot]
        ):
            changed = list(states)
            changed[slot] = final
            answer += coefficient * self.value(tuple(changed))
        return answer

    def _physical_g_ward(self, states, target_slot, rest_states):
        source_parities = tuple(
            module.parity(state)
            for module, state in zip(self.modules, rest_states)
        )
        koszul = (1, (-1) ** source_parities[0],
                  (-1) ** (source_parities[0] + source_parities[1]))
        primary_sign = (-1) ** self.primary_parity
        if target_slot in (0, 1):
            # In the first two generalized NS--R--R Ward identities the
            # intrinsic NS-primary parity occurs only in epsilon, multiplying
            # the contour contribution from the third puncture.
            koszul = (koszul[0], koszul[1], primary_sign * koszul[2])
        else:
            # Solving the third identity for its target moves that epsilon to
            # the other two punctures.
            koszul = (
                primary_sign * koszul[0],
                primary_sign * koszul[1],
                koszul[2],
            )
        equation = {}
        cutoff = 16

        if target_slot == 0:
            mode2 = states[0][1][0]
            m = (mode2 - 1) // 2
            for j in range(cutoff + 1):
                a = (-1) ** j * generalized_binomial(Fraction(1, 2), j)
                add_acted_triples(
                    equation, koszul[0] * self.infinity_phase * a, 0,
                    self.modules[0].act("G", 2 * j - mode2, rest_states[0]),
                    rest_states,
                )
                add_acted_triples(
                    equation,
                    koszul[1] * generalized_binomial(m + Fraction(1, 2), j),
                    1,
                    self.modules[1].act("G", j, rest_states[1]),
                    rest_states,
                )
                add_acted_triples(
                    equation,
                    koszul[2] * self.zero_phase * a,
                    2,
                    self.modules[2].act("G", m + j, rest_states[2]),
                    rest_states,
                )
        elif target_slot == 1:
            n = states[1][1][0]
            for j in range(cutoff + 1):
                c = (-1) ** j * generalized_binomial(Fraction(1, 2) - n, j)
                add_acted_triples(
                    equation, koszul[0] * self.infinity_phase * c, 0,
                    self.modules[0].act("G", 2 * (n + j) - 1, rest_states[0]),
                    rest_states,
                )
                add_acted_triples(
                    equation,
                    koszul[1] * generalized_binomial(Fraction(1, 2), j),
                    1,
                    self.modules[1].act("G", -n + j, rest_states[1]),
                    rest_states,
                )
                add_acted_triples(
                    equation,
                    koszul[2] * self.zero_phase * ((-1) ** n) * c,
                    2,
                    self.modules[2].act("G", j, rest_states[2]),
                    rest_states,
                )
        else:
            n = states[2][1][0]
            for j in range(cutoff + 1):
                a = (-1) ** j * generalized_binomial(Fraction(1, 2), j)
                add_acted_triples(
                    equation, koszul[0] * self.infinity_phase * a, 0,
                    self.modules[0].act("G", 2 * (n + j) - 1, rest_states[0]),
                    rest_states,
                )
                add_acted_triples(
                    equation,
                    koszul[1] * generalized_binomial(Fraction(1, 2) - n, j),
                    1,
                    self.modules[1].act("G", j, rest_states[1]),
                    rest_states,
                )
                add_acted_triples(
                    equation,
                    koszul[2] * self.zero_phase * a,
                    2,
                    self.modules[2].act("G", -n + j, rest_states[2]),
                    rest_states,
                )

        target_coefficient = equation.pop(states, 0.0j)
        if abs(target_coefficient) < 1.0e-10:
            raise AssertionError("The supercurrent Ward identity missed its target.")
        return -sum(
            coefficient * self.value(changed)
            for changed, coefficient in equation.items()
        ) / target_coefficient

    def value(self, states):
        states = tuple(states)
        if states in self._cache:
            return self._cache[states]
        absolute_parity = (
            self.primary_parity
            + sum(
                module.parity(state)
                for module, state in zip(self.modules, states)
            )
        ) % 2
        absolute_form_parity = (
            self.primary_parity + self.form_parity
        ) % 2
        if absolute_parity != absolute_form_parity:
            return 0.0j
        if states in self._active:
            raise RuntimeError(f"Cyclic SCA Ward reduction at {states!r}")
        self._active.add(states)
        try:
            l_words = [state[0] for state in states]
            if l_words[1]:
                n = l_words[1][0]
                rest = list(states)
                rest[1] = (l_words[1][1:],) + states[1][1:]
                rest = tuple(rest)
                if n == 1:
                    answer = (
                        self.modules[0].l0_weight(states[0])
                        - self.modules[1].l0_weight(rest[1])
                        - self.modules[2].l0_weight(states[2])
                    ) * self.value(rest)
                else:
                    answer = 0.0j
                    maximum = max(
                        self.modules[0].level_units(states[0]) + n + 2,
                        self.modules[2].level_units(states[2]) + 3,
                    )
                    for p in range(maximum + 1):
                        coefficient = math.comb(n - 2 + p, n - 2)
                        answer += coefficient * self._sum_action(
                            rest, 0, "L", n + p
                        )
                        answer += coefficient * (-1) ** n * self._sum_action(
                            rest, 2, "L", p - 1
                        )
            elif l_words[0]:
                n = l_words[0][0]
                rest = list(states)
                rest[0] = (l_words[0][1:],) + states[0][1:]
                rest = tuple(rest)
                answer = self._sum_action(rest, 2, "L", n)
                for p in range(-1, n + 1):
                    answer += math.comb(n + 1, p + 1) * self._sum_action(
                        rest, 1, "L", p
                    )
            elif l_words[2]:
                n = l_words[2][0]
                rest = list(states)
                rest[2] = (l_words[2][1:],) + states[2][1:]
                rest = tuple(rest)
                answer = self._sum_action(rest, 0, "L", n)
                maximum = self.modules[1].level_units(states[1]) + 3
                for p in range(maximum + 1):
                    answer -= generalized_binomial(1 - n, p) * self._sum_action(
                        rest, 1, "L", p - 1
                    )
            elif states[0][1]:
                rest = list(states)
                rest[0] = (states[0][0], states[0][1][1:])
                answer = self._physical_g_ward(states, 0, tuple(rest))
            elif states[1][1]:
                rest = list(states)
                rest[1] = (states[1][0], states[1][1][1:], states[1][2])
                answer = self._physical_g_ward(states, 1, tuple(rest))
            elif states[2][1]:
                rest = list(states)
                rest[2] = (states[2][0], states[2][1][1:], states[2][2])
                answer = self._physical_g_ward(states, 2, tuple(rest))
            else:
                answer = self.base_value(states)
        finally:
            self._active.remove(states)
        self._cache[states] = answer
        return answer


class AuxiliaryThreePoint:
    """NS-R-R free-fermion three-point function from its mode Ward identity."""

    def __init__(self, modules):
        self.modules = tuple(modules)
        self._cache = {}
        self._active = set()
        self._inner_cache = {}

    def parity(self, slot, state):
        return self.modules[slot].auxiliary_parity(state)

    def act(self, slot, mode, state):
        final, coefficient = self.modules[slot].apply_auxiliary(mode, state)
        return () if not coefficient else ((final, coefficient),)

    def base_value(self, states):
        ground2 = states[1][1]
        ground3 = states[2][1]
        if ground2 != ground3:
            return 0.0j
        return 1.0 + 0.0j if ground2 == 0 else -1.0 + 0.0j

    def inner(self, slot, left, right):
        key = (slot, left, right)
        if key in self._inner_cache:
            return self._inner_cache[key]
        module = self.modules[slot]
        modes = left if slot == 0 else left[0]
        expression = {right: 1.0 + 0.0j}
        for mode in modes:
            next_expression = {}
            for state, outer in expression.items():
                final, inner = module.apply_auxiliary(mode, state)
                if inner:
                    add_term(next_expression, final, -outer * inner)
            expression = next_expression
        if slot == 0:
            answer = expression.get((), 0.0j)
        else:
            ground = left[1]
            answer = expression.get(((), ground), 0.0j) * (-1) ** ground
        self._inner_cache[key] = answer
        return answer

    def _ward(self, states, target_slot, rest_states):
        parities = tuple(self.parity(slot, state) for slot, state in enumerate(rest_states))
        koszul = (1, (-1) ** parities[0], (-1) ** (parities[0] + parities[1]))
        equation = {}
        cutoff = 16
        if target_slot == 0:
            mode2 = states[0][0]
            k = (mode2 + 1) // 2
            for j in range(cutoff + 1):
                a = (-1) ** j * generalized_binomial(Fraction(-1, 2), j)
                add_acted_triples(
                    equation, koszul[0] * 1j * a, 0,
                    self.act(0, 2 * j - mode2, rest_states[0]), rest_states,
                )
                add_acted_triples(
                    equation,
                    koszul[1] * generalized_binomial(k - Fraction(1, 2), j),
                    1, self.act(1, j, rest_states[1]), rest_states,
                )
                add_acted_triples(
                    equation, koszul[2] * (-1j) * a, 2,
                    self.act(2, k + j, rest_states[2]), rest_states,
                )
        elif target_slot == 1:
            n = states[1][0][0]
            for j in range(cutoff + 1):
                d = generalized_binomial(Fraction(-1, 2), j)
                e = (-1) ** j * generalized_binomial(-n - Fraction(1, 2), j)
                add_acted_triples(
                    equation, koszul[0] * 1j * e, 0,
                    self.act(0, 2 * (n + j) + 1, rest_states[0]), rest_states,
                )
                add_acted_triples(
                    equation, koszul[1] * d, 1,
                    self.act(1, -n + j, rest_states[1]), rest_states,
                )
                add_acted_triples(
                    equation, koszul[2] * (-1j) * ((-1) ** n) * e, 2,
                    self.act(2, j, rest_states[2]), rest_states,
                )
        else:
            n = states[2][0][0]
            for j in range(cutoff + 1):
                a = (-1) ** j * generalized_binomial(Fraction(-1, 2), j)
                add_acted_triples(
                    equation, koszul[0] * 1j * a, 0,
                    self.act(0, 2 * (n + j) + 1, rest_states[0]), rest_states,
                )
                add_acted_triples(
                    equation,
                    koszul[1] * generalized_binomial(-n - Fraction(1, 2), j),
                    1, self.act(1, j, rest_states[1]), rest_states,
                )
                add_acted_triples(
                    equation, koszul[2] * (-1j) * a, 2,
                    self.act(2, -n + j, rest_states[2]), rest_states,
                )
        target_coefficient = equation.pop(states, 0.0j)
        if abs(target_coefficient) < 1.0e-10:
            raise AssertionError("The fermion Ward identity missed its target.")
        return -sum(
            coefficient * self.value(changed)
            for changed, coefficient in equation.items()
        ) / target_coefficient

    def value(self, states):
        states = tuple(states)
        if states in self._cache:
            return self._cache[states]
        if sum(self.parity(slot, state) for slot, state in enumerate(states)) % 2:
            return 0.0j
        if states in self._active:
            raise RuntimeError(f"Cyclic fermion Ward reduction at {states!r}")
        self._active.add(states)
        try:
            if states[0]:
                rest = (states[0][1:], states[1], states[2])
                answer = self._ward(states, 0, rest)
            elif states[1][0]:
                rest = (states[0], (states[1][0][1:], states[1][1]), states[2])
                answer = self._ward(states, 1, rest)
            elif states[2][0]:
                rest = (states[0], states[1], (states[2][0][1:], states[2][1]))
                answer = self._ward(states, 2, rest)
            else:
                answer = self.base_value(states)
        finally:
            self._active.remove(states)
        self._cache[states] = answer
        return answer


class AuxiliaryVirasoroThreePoint(AuxiliaryThreePoint):
    """Free-fermion form obtained from its exact Virasoro decomposition.

    This avoids using the branching recurrence under test.  Each auxiliary
    Fock state is expanded into descendants of the appropriate Ising primary:
    the NS vacuum (h=0), the NS fermion (h=1/2), or a Ramond spin ground
    (h=1/16).  Ordinary Virasoro Ward identities then evaluate the form.
    """

    def __init__(self, modules):
        super().__init__(modules)
        self._decomposition_cache = {}

    def _lf_action(self, slot, mode, expression):
        module = self.modules[slot]
        answer = {}
        for auxiliary, outer in expression.items():
            if slot == 0:
                full_state = (auxiliary, (), ())
            else:
                full_state = (auxiliary[0], auxiliary[1], (), (), 0)
            acted = module.apply_lf(-mode, {full_state: 1.0 + 0.0j})
            for final, inner in acted.items():
                auxiliary_final, _ = module.split_state(final)
                add_term(answer, auxiliary_final, outer * inner)
        return answer

    def _primary_data(self, slot, state):
        total_parity = self.parity(slot, state)
        if slot == 0:
            primary = () if total_parity == 0 else (1,)
            primary_units = total_parity
            total_units = sum(state)
            if (total_units - primary_units) % 2:
                raise AssertionError("An NS auxiliary state has the wrong level parity.")
            descendant_level = (total_units - primary_units) // 2
            weight = 0.0 if total_parity == 0 else 0.5
        else:
            primary = ((), total_parity)
            descendant_level = sum(state[0])
            weight = 1 / 16
        return total_parity, primary, descendant_level, weight

    @lru_cache(None)
    def _basis(self, slot, total_parity, descendant_level):
        if slot == 0:
            primary_units = total_parity
            total_units = primary_units + 2 * descendant_level
            rows = tuple(
                modes
                for modes in strict_odd_partitions(total_units)
                if len(modes) % 2 == total_parity
            )
            primary = () if total_parity == 0 else (1,)
        else:
            rows = tuple(
                (modes, (total_parity - len(modes)) % 2)
                for modes in strict_partitions(descendant_level)
            )
            primary = ((), total_parity)
        words = partitions(descendant_level)
        columns = []
        for word in words:
            expression = {primary: 1.0 + 0.0j}
            for mode in reversed(word):
                expression = self._lf_action(slot, mode, expression)
            columns.append([expression.get(row, 0.0j) for row in rows])
        matrix = np.asarray(columns, dtype=np.complex128).T
        return rows, words, matrix

    def decompose(self, slot, state):
        key = (slot, state)
        if key in self._decomposition_cache:
            return self._decomposition_cache[key]
        total_parity, _, descendant_level, weight = self._primary_data(slot, state)
        rows, words, matrix = self._basis(slot, total_parity, descendant_level)
        vector = np.asarray(
            [1.0 if row == state else 0.0 for row in rows],
            dtype=np.complex128,
        )
        if not rows:
            raise AssertionError("The auxiliary state lies in an empty Ising level.")
        coefficients, _, _, _ = np.linalg.lstsq(matrix, vector, rcond=1.0e-13)
        residual = np.linalg.norm(matrix @ coefficients - vector)
        if residual > 2.0e-11:
            raise AssertionError("The auxiliary Virasoro decomposition is incomplete.")
        answer = (
            weight,
            {
                word: coefficient
                for word, coefficient in zip(words, coefficients)
                if abs(coefficient) > TOLERANCE
            },
        )
        self._decomposition_cache[key] = answer
        return answer

    @staticmethod
    def primary_value(parities):
        first, second, third = parities
        if first == 0:
            if second != third:
                return 0.0j
            return 1.0 + 0.0j if second == 0 else -1.0 + 0.0j
        if (second, third) == (0, 1):
            return -1 / math.sqrt(2)
        if (second, third) == (1, 0):
            return 1 / math.sqrt(2)
        return 0.0j

    def value(self, states):
        states = tuple(states)
        if states in self._cache:
            return self._cache[states]
        parities = tuple(self.parity(slot, state) for slot, state in enumerate(states))
        primary = self.primary_value(parities)
        if not primary:
            self._cache[states] = 0.0j
            return 0.0j
        decompositions = tuple(
            self.decompose(slot, state) for slot, state in enumerate(states)
        )
        weights = tuple(item[0] for item in decompositions)
        virasoro = VirasoroThreePoint(weights, 0.5)
        answer = 0.0j
        for word1, coefficient1 in decompositions[0][1].items():
            for word2, coefficient2 in decompositions[1][1].items():
                for word3, coefficient3 in decompositions[2][1].items():
                    answer += (
                        primary
                        * coefficient1
                        * coefficient2
                        * coefficient3
                        * virasoro.value(word1, word2, word3)
                    )
        self._cache[states] = answer
        return answer


class DirectBranchingCoefficient:
    def __init__(self, b, momenta, primary_parity: int = 0):
        self.b = float(b)
        self.momenta = tuple(float(value) for value in momenta)
        self.primary_parity = int(primary_parity)
        if self.primary_parity not in (0, 1):
            raise ValueError("primary_parity must be 0 or 1")
        self.free_modules = (
            FreeFieldModule("NS", b, momenta[0]),
            FreeFieldModule("R", b, momenta[1]),
            FreeFieldModule("R", b, momenta[2]),
        )
        self.pbw_modules = tuple(PBWModule(module) for module in self.free_modules)
        self.auxiliary_form = AuxiliaryThreePoint(self.free_modules)
        self._reflected_ns_module = None
        self._reflected_ns_pbw = None
        self._branch_cache = {}
        self._physical_forms = {}

    def branch(self, slot, label, parity=0):
        label = Fraction(label)
        key = (slot, label, int(parity))
        if key in self._branch_cache:
            return self._branch_cache[key]
        module = self.free_modules[slot]
        pbw_module = self.pbw_modules[slot]
        if slot == 0 and label < 0:
            # Human Notes/SCblock.tex: v_n(P)=v_{-n}(-P).  The reflected
            # free-field realization has the same abstract NS PBW basis.
            if self._reflected_ns_module is None:
                self._reflected_ns_module = FreeFieldModule(
                    "NS", self.b, -self.momenta[0]
                )
                self._reflected_ns_pbw = PBWModule(self._reflected_ns_module)
            module = self._reflected_ns_module
            pbw_module = self._reflected_ns_pbw
            expression = module.ns_branch(-label)
        else:
            expression = (
                module.ns_branch(label)
                if slot == 0
                else module.r_branch(label, parity)
            )
        answer = branch_in_pbw(module, pbw_module, expression)
        self._branch_cache[key] = answer
        return answer

    def raw(self, labels, alpha2, alpha3, eta):
        labels = tuple(Fraction(label) for label in labels)
        form_parity = (int(2 * labels[0]) + alpha2 + alpha3) % 2
        key = (form_parity, int(eta))
        if key not in self._physical_forms:
            self._physical_forms[key] = PhysicalThreePoint(
                self.pbw_modules,
                form_parity,
                eta,
                primary_parity=self.primary_parity,
            )
        physical_form = self._physical_forms[key]
        branches = (
            self.branch(0, labels[0]),
            self.branch(1, labels[1], alpha2),
            self.branch(2, labels[2], alpha3),
        )
        answer = 0.0j
        for (auxiliary1, physical1), coefficient1 in branches[0].items():
            parity_physical1 = self.pbw_modules[0].parity(physical1)
            parity_auxiliary1 = self.free_modules[0].auxiliary_parity(
                auxiliary1
            )
            for (auxiliary2, physical2), coefficient2 in branches[1].items():
                parity_physical2 = self.pbw_modules[1].parity(physical2)
                for (auxiliary3, physical3), coefficient3 in branches[2].items():
                    auxiliary = self.auxiliary_form.value(
                        (auxiliary1, auxiliary2, auxiliary3)
                    )
                    if abs(auxiliary) <= TOLERANCE:
                        continue
                    parity_auxiliary3 = self.free_modules[2].auxiliary_parity(
                        auxiliary3
                    )
                    # Complete Human-Note Section 8 sign:
                    #
                    #   (-1)^[ A A_aux
                    #            + (B+|alpha|+p_1)(C_aux+c) ].
                    #
                    # There is no independent f(C_aux+c) factor.
                    hatted_form_sign = (-1) ** (
                        parity_physical1 * parity_auxiliary1
                        + (parity_physical2 + self.primary_parity)
                        * parity_auxiliary3
                    )
                    physical = physical_form.value(
                        (physical1, physical2, physical3)
                    )
                    answer += (
                        coefficient1
                        * coefficient2
                        * coefficient3
                        * hatted_form_sign
                        * auxiliary
                        * physical
                    )
        return answer

    def normalized(self, labels, alpha2, alpha3, eta):
        return self.raw(labels, alpha2, alpha3, eta) / norm_product(
            tuple(Fraction(label) for label in labels),
            alpha2,
            alpha3,
            self.b,
            self.momenta,
        )

    def direct_norm_squared(self, slot, label, parity=0):
        branch = self.branch(slot, label, parity)
        answer = 0.0j
        for (auxiliary_left, physical_left), coefficient_left in branch.items():
            for (auxiliary_right, physical_right), coefficient_right in branch.items():
                auxiliary_inner = self.auxiliary_form.inner(
                    slot, auxiliary_left, auxiliary_right
                )
                if abs(auxiliary_inner) <= TOLERANCE:
                    continue
                physical_inner = self.pbw_modules[slot].inner(
                    physical_left, physical_right
                )
                answer += (
                    coefficient_left
                    * coefficient_right
                    * auxiliary_inner
                    * physical_inner
                )
        return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", default="7/5")
    parser.add_argument("--p1", default="11/23")
    parser.add_argument("--p2", default="13/29")
    parser.add_argument("--p3", default="17/31")
    parser.add_argument("--primary-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--json", type=Path, default=HERE / "direct_state_results.json")
    arguments = parser.parse_args()
    b = float(Fraction(arguments.b))
    momenta = tuple(
        float(Fraction(value))
        for value in (arguments.p1, arguments.p2, arguments.p3)
    )
    evaluator = DirectBranchingCoefficient(
        b, momenta, primary_parity=arguments.primary_parity
    )
    started = time.perf_counter()

    ns_labels = tuple(
        Fraction(value, 2) for value in (-3, -2, -1, 0, 1, 2, 3)
    )
    ramond_labels = tuple(
        Fraction(value, 4) for value in (-5, -3, -1, 1, 3, 5)
    )
    weights = BranchWeights(b, momenta)

    state_checks = []
    maximum_norm_error = 0.0
    maximum_primary_residual = 0.0
    maximum_weight_error = 0.0

    def check_state(slot, label, parity):
        nonlocal maximum_norm_error, maximum_primary_residual, maximum_weight_error
        module = evaluator.free_modules[slot]
        branch = (
            module.ns_branch(label)
            if slot == 0
            else module.r_branch(label, parity)
        )
        converted = evaluator.branch(slot, label, parity)
        direct_norm = evaluator.direct_norm_squared(slot, label, parity)
        expected_norm = (
            ns_norm_squared(label, b, momenta[slot])
            if slot == 0
            else ramond_norm_squared(label, parity, b, momenta[slot])
        )
        norm_error = abs(direct_norm - expected_norm) / max(
            abs(expected_norm), 1.0
        )
        maximum_norm_error = max(maximum_norm_error, norm_error)

        if slot == 0:
            total_level = 2 * float(label) ** 2
        else:
            total_level = 2 * float(label) ** 2 - 1 / 8
        primary_residual = 0.0
        denominator = max(max_abs(branch), 1.0)
        for copy in (1, 2):
            for mode in range(1, int(math.floor(total_level)) + 2):
                primary_residual = max(
                    primary_residual,
                    max_abs(module.apply_embedded(copy, mode, branch)) / denominator,
                )
            inferred_state = module.apply_embedded(
                copy,
                1,
                module.apply_embedded(copy, -1, branch),
            )
            fit = span_fit(inferred_state, [branch])
            inferred_weight = fit["coefficients"][0] / 2
            expected_weight = weights.weight(slot, label, copy - 1)
            maximum_weight_error = max(
                maximum_weight_error, abs(inferred_weight - expected_weight)
            )
        maximum_primary_residual = max(
            maximum_primary_residual, primary_residual
        )
        state_checks.append(
            {
                "sector": "NS" if slot == 0 else "R",
                "slot": slot + 1,
                "label": format_fraction(label),
                "alpha": None if slot == 0 else parity,
                "free_field_terms": len(branch),
                "tensor_PBW_terms": len(converted),
                "direct_norm_squared": encode_complex(direct_norm),
                "formula_norm_squared": encode_complex(expected_norm),
                "relative_norm_error": float(norm_error),
                "maximum_positive_embedded_mode_residual": float(
                    primary_residual
                ),
            }
        )

    for label in ns_labels:
        check_state(0, label, 0)
    for slot in (1, 2):
        for parity in (0, 1):
            for label in ramond_labels:
                check_state(slot, label, parity)

    cases = (
        (Fraction(1), Fraction(3, 4), Fraction(3, 4)),
        (Fraction(3, 2), Fraction(5, 4), Fraction(3, 4)),
        (Fraction(3, 2), Fraction(3, 4), Fraction(5, 4)),
        (Fraction(-1), Fraction(-3, 4), Fraction(-3, 4)),
        (Fraction(-3, 2), Fraction(-5, 4), Fraction(-3, 4)),
        (Fraction(-3, 2), Fraction(-3, 4), Fraction(-5, 4)),
        (Fraction(-1), Fraction(3, 4), Fraction(3, 4)),
    )

    action_started = time.perf_counter()
    ns_actions = {}
    action_checks = []
    for label in sorted({labels[0] for labels in cases}):
        terms, fit = solve_ns_l1(evaluator.free_modules[0], label)
        ns_actions[label] = terms
        action_checks.append(
            {
                "sector": "NS",
                "slot": 1,
                "label": format_fraction(label),
                "alpha": None,
                "relative_residual": fit["relative_residual"],
                "rank": fit["rank"],
                "columns": fit["columns"],
            }
        )
    ramond_actions = ({}, {})
    for offset, slot in enumerate((1, 2)):
        for parity in (0, 1):
            ramond_actions[offset][parity] = {}
            for label in sorted({labels[slot] for labels in cases}):
                terms, fit = solve_ramond_lminus(
                    evaluator.free_modules[slot], label, parity
                )
                ramond_actions[offset][parity][label] = terms
                action_checks.append(
                    {
                        "sector": "R",
                        "slot": slot + 1,
                        "label": format_fraction(label),
                        "alpha": parity,
                        "relative_residual": fit["relative_residual"],
                        "rank": fit["rank"],
                        "columns": fit["columns"],
                    }
                )
    action_seconds = time.perf_counter() - action_started
    maximum_action_residual = max(
        item["relative_residual"] for item in action_checks
    )

    negative_target = (
        Fraction(-3, 2),
        Fraction(-5, 4),
        Fraction(-5, 4),
    )
    negative_recursion_smoke = []
    for alpha2 in (0, 1):
        for alpha3 in (0, 1):
            recursion = BranchingRecursion(
                b,
                momenta,
                weights,
                ns_actions,
                ramond_actions[0][alpha2],
                ramond_actions[1][alpha3],
                second_parity=alpha2,
                third_parity=alpha3,
            )
            expansion = recursion.expansion(negative_target)
            negative_recursion_smoke.append(
                {
                    "alpha2": alpha2,
                    "alpha3": alpha3,
                    "target": [
                        format_fraction(label) for label in negative_target
                    ],
                    "boundary_terms": [
                        {
                            "labels": [
                                format_fraction(label) for label in boundary
                            ],
                            "coefficient": encode_complex(coefficient),
                        }
                        for boundary, coefficient in sorted(expansion.items())
                    ],
                }
            )

    recursion_checks = []
    maximum_recursion_error = 0.0
    direct_seconds = 0.0
    for labels in cases:
        case_started = time.perf_counter()
        case_entries = []
        for alpha2 in (0, 1):
            for alpha3 in (0, 1):
                form_parity = (
                    int(2 * labels[0]) + alpha2 + alpha3
                ) % 2
                for eta in [(-1) ** form_parity]:
                    recursion = BranchingRecursion(
                        b,
                        momenta,
                        weights,
                        ns_actions,
                        ramond_actions[0][alpha2],
                        ramond_actions[1][alpha3],
                        second_parity=alpha2,
                        third_parity=alpha3,
                    )
                    coefficients = recursion.node_coefficients(labels)
                    children = (
                        (
                            recursion.neighbor(labels[0]),
                            labels[1],
                            labels[2],
                        ),
                        (
                            labels[0],
                            recursion.neighbor(labels[1]),
                            labels[2],
                        ),
                        (
                            labels[0],
                            labels[1],
                            recursion.neighbor(labels[2]),
                        ),
                    )
                    direct_target = evaluator.normalized(
                        labels, alpha2, alpha3, eta
                    )
                    direct_children = tuple(
                        evaluator.normalized(child, alpha2, alpha3, eta)
                        for child in children
                    )
                    recursive_target = sum(
                        coefficient * value
                        for coefficient, value in zip(
                            coefficients, direct_children
                        )
                    )
                    error = abs(direct_target - recursive_target) / max(
                        abs(direct_target), 1.0
                    )
                    maximum_recursion_error = max(maximum_recursion_error, error)
                    case_entries.append(
                        {
                            "alpha2": alpha2,
                            "alpha3": alpha3,
                            "f": form_parity,
                            "eta": eta,
                            "B_from_direct_states": encode_complex(direct_target),
                            "B_from_one_recursion_step": encode_complex(
                                recursive_target
                            ),
                            "relative_difference": float(error),
                            "children": [
                                {
                                    "labels": [
                                        format_fraction(value) for value in child
                                    ],
                                    "coefficient": encode_complex(coefficient),
                                    "B_from_direct_states": encode_complex(value),
                                }
                                for child, coefficient, value in zip(
                                    children, coefficients, direct_children
                                )
                            ],
                        }
                    )
        elapsed = time.perf_counter() - case_started
        direct_seconds += elapsed
        recursion_checks.append(
            {
                "labels": [format_fraction(label) for label in labels],
                "elapsed_seconds": elapsed,
                "checks": case_entries,
            }
        )
        print(
            f"{tuple(format_fraction(label) for label in labels)}: "
            f"{elapsed:.4f} s"
        )

    result = {
        "parameters": {"b": b, "momenta": list(momenta)},
        "three_point_scope": {
            "eta": "eta=(-1)^f",
            "chambers": [
                "positive labels",
                "negative labels",
                "the mixed NS test (-1,3/4,3/4)",
            ],
            "note": (
                "All requested individual negative states, norms, and action "
                "decompositions are checked.  The second coupled eta pair is "
                "not evaluated by reusing the canonical scalar reducer."
            ),
        },
        "cutoffs": {
            "NS": [format_fraction(label) for label in ns_labels],
            "R": [format_fraction(label) for label in ramond_labels],
        },
        "state_checks": state_checks,
        "action_checks": action_checks,
        "negative_recursion_smoke": negative_recursion_smoke,
        "recursion_checks": recursion_checks,
        "summary": {
            "state_count": len(state_checks),
            "recursion_comparison_count": sum(
                len(case["checks"]) for case in recursion_checks
            ),
            "maximum_relative_norm_error": float(maximum_norm_error),
            "maximum_primary_residual": float(maximum_primary_residual),
            "maximum_branch_weight_error": float(maximum_weight_error),
            "maximum_action_decomposition_residual": float(
                maximum_action_residual
            ),
            "maximum_direct_recursion_relative_difference": float(
                maximum_recursion_error
            ),
            "action_decomposition_seconds": action_seconds,
            "direct_comparison_seconds": direct_seconds,
            "total_elapsed_seconds": time.perf_counter() - started,
        },
    }
    arguments.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
