#!/usr/bin/env python3
"""Exact genus-two NS c-recursion check through total physical order three.

The calculation compares, coefficient by coefficient,

1. direct theta-graph sewing from exact PBW Gram matrices and exact NS Ward
   identities; and
2. the triangular c-recursion with the exact global osp(1|2) regular seed.

Twice-levels ``(n0,n1,ninf)`` obey ``n0+n1+ninf <= 6``.  This gives 84
identities and includes the first three NS Kac channels, ``(3,1)``, ``(2,2)``,
and ``(5,1)``.  It also includes the first three nontrivial theta-vacuum
coefficients and their polarized vacuum/global signs.
All arithmetic is in ``Q(c,h0,h1,hinf)``; no numerical evaluation, matrix
conditioning, tolerance, or coefficient pruning enters the comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from typing import Iterable, Mapping, Sequence

import sympy as sp


Mode = tuple[str, int]
State = tuple[Mode, ...]

C, H0, H1, HINF = sp.symbols("c h_0 h_1 h_infinity")


# These are the complete NS PBW bases needed at this cutoff, in the same
# HJS ordering used by the finite-level oracle.
PBW_BASES: Mapping[int, tuple[State, ...]] = {
    0: ((),),
    1: ((('G', -1),),),
    2: ((('L', -2),),),
    3: ((('G', -3),), (('G', -1), ('L', -2))),
    4: (
        (('L', -4),),
        (('L', -2), ('L', -2)),
        (('G', -1), ('G', -3)),
    ),
    5: (
        (('G', -5),),
        (('G', -3), ('L', -2)),
        (('G', -1), ('L', -4)),
        (('G', -1), ('L', -2), ('L', -2)),
    ),
    6: (
        (('L', -6),),
        (('L', -2), ('L', -4)),
        (('L', -2), ('L', -2), ('L', -2)),
        (('G', -1), ('G', -5)),
        (('G', -1), ('G', -3), ('L', -2)),
    ),
}


def state_twice_level(state: State) -> int:
    return sum(-index for _, index in state if index < 0)


def state_parity(state: State) -> int:
    return sum(kind == "G" for kind, _ in state) % 2


def mode_parity(mode: Mode) -> int:
    return int(mode[0] == "G")


def zone(mode: Mode) -> int:
    if mode[1] < 0:
        return 0
    if mode[1] == 0:
        return 1
    return 2


def rising(value: sp.Expr, order: int) -> sp.Expr:
    result = sp.S.One
    for offset in range(order):
        result *= value + offset
    return sp.expand(result)


def falling(value: sp.Expr, order: int) -> sp.Expr:
    result = sp.S.One
    for offset in range(order):
        result *= value - offset
    return sp.expand(result)


def generalized_binomial(value: sp.Expr, order: int) -> sp.Expr:
    if order < 0:
        return sp.S.Zero
    result = sp.S.One
    for offset in range(order):
        result *= (value - offset) / (offset + 1)
    return sp.cancel(result)


class ExactNSVermaModule:
    """Small exact NS Verma module generated from the algebra itself."""

    def __init__(self, *, c: sp.Expr, weight: sp.Expr) -> None:
        self.c = c
        self.weight = weight

    def basis(self, twice_level: int) -> tuple[State, ...]:
        return PBW_BASES.get(twice_level, ())

    @staticmethod
    def bpz(state: State) -> State:
        return tuple((kind, -index) for kind, index in reversed(state))

    def super_bracket(
        self, left: Mode, right: Mode
    ) -> tuple[tuple[sp.Expr, Mode | None], ...]:
        left_kind, left_twice = left
        right_kind, right_twice = right
        left_index = sp.Rational(left_twice, 2)
        right_index = sp.Rational(right_twice, 2)
        terms: list[tuple[sp.Expr, Mode | None]] = []
        if left_kind == "L" and right_kind == "L":
            terms.append((left_index - right_index, ("L", left_twice + right_twice)))
            if left_twice + right_twice == 0:
                terms.append(
                    (self.c * (left_index**3 - left_index) / 12, None)
                )
        elif left_kind == "L" and right_kind == "G":
            terms.append(
                (left_index / 2 - right_index, ("G", left_twice + right_twice))
            )
        elif left_kind == "G" and right_kind == "L":
            terms.append(
                (left_index - right_index / 2, ("G", left_twice + right_twice))
            )
        else:
            terms.append((sp.Integer(2), ("L", left_twice + right_twice)))
            if left_twice + right_twice == 0:
                terms.append(
                    (self.c * (left_index**2 - sp.Rational(1, 4)) / 3, None)
                )
        return tuple((coefficient, mode) for coefficient, mode in terms if coefficient != 0)

    @lru_cache(maxsize=None)
    def expectation(self, word: State) -> sp.Expr:
        for index in range(len(word) - 1):
            left = word[index]
            right = word[index + 1]
            if zone(left) <= zone(right):
                continue
            swapped_sign = -1 if mode_parity(left) and mode_parity(right) else 1
            swapped = word[:index] + (right, left) + word[index + 2 :]
            result = swapped_sign * self.expectation(swapped)
            for coefficient, replacement in self.super_bracket(left, right):
                reduced = (
                    word[:index]
                    + (() if replacement is None else (replacement,))
                    + word[index + 2 :]
                )
                result += coefficient * self.expectation(reduced)
            return sp.expand(result)
        if any(index != 0 for _, index in word):
            return sp.S.Zero
        if any(kind == "G" for kind, _ in word):
            raise ValueError("the NS module has no G_0 mode")
        return self.weight ** len(word)

    def inner_product(self, left: State, right: State) -> sp.Expr:
        return self.expectation(self.bpz(left) + right)

    @lru_cache(maxsize=None)
    def gram_matrix(self, twice_level: int) -> sp.Matrix:
        basis = self.basis(twice_level)
        return sp.Matrix(
            [[self.inner_product(left, right) for right in basis] for left in basis]
        )

    @lru_cache(maxsize=None)
    def mode_action(self, mode: Mode, state: State) -> tuple[tuple[State, sp.Expr], ...]:
        target_level = state_twice_level(state) - mode[1]
        if target_level < 0:
            return ()
        target_basis = self.basis(target_level)
        if not target_basis:
            return ()
        overlaps = sp.Matrix(
            [
                self.expectation(self.bpz(test_state) + (mode,) + state)
                for test_state in target_basis
            ]
        )
        coordinates = self.gram_matrix(target_level).inv() * overlaps
        return tuple(
            (basis_state, sp.cancel(coefficient))
            for basis_state, coefficient in zip(target_basis, coordinates)
            if coefficient != 0
        )


def exact_osp_norm(weight: sp.Expr, n: int, epsilon: int) -> sp.Expr:
    return sp.factorial(n) * rising(2 * weight, n + epsilon)


def exact_osp_two_chain_kernel(
    *,
    k: int,
    m: int,
    epsilon1: int,
    epsilon2: int,
    epsilon3: int,
    d1: sp.Expr,
    d2: sp.Expr,
    d3: sp.Expr,
) -> sp.Expr:
    a_value = d2 + d3 - d1
    b_value = d1 + d2 - d3
    c_value = d1 - d2 + d3
    s_value = d1 + d2 + d3
    bits = (epsilon1, epsilon2, epsilon3)
    if bits in ((1, 0, 0), (1, 1, 0)):
        reflection_sign = -1 if epsilon2 else 1
        return reflection_sign * exact_osp_two_chain_kernel(
            k=m,
            m=k,
            epsilon1=0,
            epsilon2=epsilon2,
            epsilon3=1,
            d1=d3,
            d2=d2,
            d3=d1,
        )
    result = sp.S.Zero
    for p in range(min(k, m) + 1):
        common0 = sp.binomial(k, p) * falling(m, p) * falling(2 * d3 + m - 1, p)
        common1 = sp.binomial(k, p) * falling(m, p) * falling(2 * d3 + m, p)
        if bits == (0, 0, 0):
            term = common0 * rising(a_value, m - p) * rising(
                b_value + p - m, k - p
            )
        elif bits == (0, 1, 0):
            term = common0 * rising(a_value + sp.Rational(1, 2), m - p) * rising(
                b_value + sp.Rational(1, 2) + p - m, k - p
            )
        elif bits == (0, 0, 1):
            term = common1 * rising(a_value + sp.Rational(1, 2), m - p) * rising(
                b_value - sp.Rational(1, 2) + p - m, k - p
            )
        elif bits == (1, 0, 1):
            term = c_value * common1 * rising(a_value, m - p) * rising(
                b_value + p - m, k - p
            )
        elif bits == (0, 1, 1):
            term = -common1 * rising(a_value, m - p + 1) * rising(
                b_value + p - m, k - p
            )
        elif bits == (1, 1, 1):
            term = (s_value - sp.Rational(1, 2)) * common1 * rising(
                a_value + sp.Rational(1, 2), m - p
            ) * rising(b_value + sp.Rational(1, 2) + p - m, k - p)
        else:
            raise AssertionError(f"unhandled osp fermion labels {bits}")
        result += term
    return sp.expand(result)


def exact_osp_three_point(
    *,
    n1: int,
    n2: int,
    n3: int,
    epsilon1: int,
    epsilon2: int,
    epsilon3: int,
    d1: sp.Expr,
    d2: sp.Expr,
    d3: sp.Expr,
) -> sp.Expr:
    exponent = (
        d1
        - d2
        - d3
        + sp.Rational(epsilon1 - epsilon2 - epsilon3, 2)
        + n1
        - n3
    )
    return sp.expand(
        falling(exponent, n2)
        * exact_osp_two_chain_kernel(
            k=n1,
            m=n3,
            epsilon1=epsilon1,
            epsilon2=epsilon2,
            epsilon3=epsilon3,
            d1=d1,
            d2=d2,
            d3=d3,
        )
    )


def is_primary_component_state(state: State) -> bool:
    return state in ((), (("G", -1),))


def global_labels(state: State) -> tuple[int, int]:
    return (
        sum(kind == "L" for kind, _ in state),
        sum(kind == "G" for kind, _ in state),
    )


class ExactNSDescendantThreeForm:
    """Exact Ward-identity three-form at (infinity,1,0)."""

    def __init__(self, *, c: sp.Expr, weights: Sequence[sp.Expr]) -> None:
        self.c = c
        self.weights = tuple(weights)
        self.modules = tuple(
            ExactNSVermaModule(c=c, weight=weight) for weight in self.weights
        )
        self._reflection_partner: ExactNSDescendantThreeForm | None = None

    @staticmethod
    def state_weight(primary_weight: sp.Expr, state: State) -> sp.Expr:
        return primary_weight + sp.Rational(state_twice_level(state), 2)

    def reflected(self) -> "ExactNSDescendantThreeForm":
        if self._reflection_partner is None:
            reflected = ExactNSDescendantThreeForm(
                c=self.c, weights=(self.weights[2], self.weights[1], self.weights[0])
            )
            reflected._reflection_partner = self
            self._reflection_partner = reflected
        return self._reflection_partner

    def linear_action(
        self, *, slot: int, mode: Mode, states: tuple[State, State, State]
    ) -> sp.Expr:
        result = sp.S.Zero
        for acted_state, coefficient in self.modules[slot].mode_action(mode, states[slot]):
            changed = list(states)
            changed[slot] = acted_state
            result += coefficient * self.value(*changed)
        return sp.cancel(result)

    def reorder_leading_global_fermion(
        self, bra: State, middle: State, ket: State
    ) -> sp.Expr:
        next_mode = bra[1]
        remainder = bra[2:]
        if next_mode[0] == "G":
            reordered = (next_mode, ("G", -1)) + remainder
            l_mode = ("L", next_mode[1] - 1)
            return -self.value(reordered, middle, ket) + 2 * self.value(
                (l_mode,) + remainder, middle, ket
            )
        coefficient = sp.Rational(-next_mode[1] - 2, 4)
        reordered = (next_mode, ("G", -1)) + remainder
        g_mode = ("G", next_mode[1] - 1)
        return self.value(reordered, middle, ket) + coefficient * self.value(
            (g_mode,) + remainder, middle, ket
        )

    @lru_cache(maxsize=None)
    def value(self, bra: State, middle: State, ket: State) -> sp.Expr:
        if all(is_primary_component_state(state) for state in (bra, middle, ket)):
            (n3, e3), (n2, e2), (n1, e1) = (
                global_labels(state) for state in (bra, middle, ket)
            )
            return exact_osp_three_point(
                n1=n3,
                n2=n2,
                n3=n1,
                epsilon1=e3,
                epsilon2=e2,
                epsilon3=e1,
                d1=self.weights[0],
                d2=self.weights[1],
                d3=self.weights[2],
            )
        states = (bra, middle, ket)
        if middle:
            mode = middle[0]
            tail = middle[1:]
            if mode[0] == "G":
                k = sp.Rational(-mode[1], 2)
                parity_13 = (state_parity(bra) + state_parity(ket)) % 2
                result = sp.S.Zero
                max_first = max(
                    0, math.floor(state_twice_level(bra) / 2 - float(k))
                )
                for m in range(max_first + 1):
                    result += generalized_binomial(k - sp.Rational(3, 2) + m, m) * self.linear_action(
                        slot=0,
                        mode=("G", int(2 * (k + m))),
                        states=(bra, tail, ket),
                    )
                max_second = max(0, math.floor(state_twice_level(ket) / 2 + 0.5))
                ward_sign = -((-1) ** (parity_13 + int(k + sp.Rational(1, 2))))
                for m in range(max_second + 1):
                    result += ward_sign * generalized_binomial(
                        k - sp.Rational(3, 2) + m, m
                    ) * self.linear_action(
                        slot=2,
                        mode=("G", 2 * m - 1),
                        states=(bra, tail, ket),
                    )
                return sp.cancel(result)
            n = -mode[1] // 2
            if n == 1:
                exponent = (
                    self.state_weight(self.weights[0], bra)
                    - self.state_weight(self.weights[1], tail)
                    - self.state_weight(self.weights[2], ket)
                )
                return sp.expand(exponent * self.value(bra, tail, ket))
            result = sp.S.Zero
            max_first = max(0, state_twice_level(bra) // 2 - n)
            for m in range(max_first + 1):
                result += generalized_binomial(n - 2 + m, n - 2) * self.linear_action(
                    slot=0,
                    mode=("L", 2 * (n + m)),
                    states=(bra, tail, ket),
                )
            max_second = max(0, state_twice_level(ket) // 2 + 1)
            for m in range(max_second + 1):
                result += ((-1) ** n) * generalized_binomial(
                    n - 2 + m, n - 2
                ) * self.linear_action(
                    slot=2,
                    mode=("L", 2 * (m - 1)),
                    states=(bra, tail, ket),
                )
            return sp.cancel(result)
        if bra:
            mode = bra[0]
            tail = bra[1:]
            if mode == ("G", -1) and tail:
                return sp.cancel(self.reorder_leading_global_fermion(bra, middle, ket))
            if mode[0] == "G":
                k = sp.Rational(-mode[1], 2)
                if k <= sp.Rational(1, 2):
                    return self.reflected().value(ket, middle, bra)
                parity_13 = (state_parity(tail) + state_parity(ket)) % 2
                result = ((-1) ** (parity_13 + 1)) * self.linear_action(
                    slot=2,
                    mode=("G", int(2 * k)),
                    states=(tail, middle, ket),
                )
                upper = int(k - sp.Rational(1, 2))
                for m in range(-1, upper + 1):
                    result += generalized_binomial(k + sp.Rational(1, 2), m + 1) * self.linear_action(
                        slot=1,
                        mode=("G", 2 * m + 1),
                        states=(tail, middle, ket),
                    )
                return sp.cancel(result)
            n = -mode[1] // 2
            result = self.linear_action(
                slot=2, mode=("L", 2 * n), states=(tail, middle, ket)
            )
            for m in range(-1, n + 1):
                result += sp.binomial(n + 1, m + 1) * self.linear_action(
                    slot=1,
                    mode=("L", 2 * m),
                    states=(tail, middle, ket),
                )
            return sp.cancel(result)
        if ket:
            return self.reflected().value(ket, middle, bra)
        return sp.S.One


def theta_orientation_sign(levels: Sequence[int]) -> int:
    """Return the literal sign in the human-note all-NS definition.

    At fixed twice-levels, every PBW state on an edge has parity equal to
    that level modulo two.  Thus this is precisely

        (-1)^(A C + A E + C E)

    from Eq. (NSblock), with no additional linear infinity-edge factor.
    """

    e0, e1, einf = (int(level) % 2 for level in levels)
    exponent = e0 * e1 + e0 * einf + e1 * einf
    return -1 if exponent % 2 else 1


class ExactDirectThetaOracle:
    def __init__(self, *, c: sp.Expr, weights: Sequence[sp.Expr]) -> None:
        self.c = c
        self.weights = tuple(weights)
        self.modules = tuple(
            ExactNSVermaModule(c=c, weight=weight) for weight in self.weights
        )
        self.form = ExactNSDescendantThreeForm(c=c, weights=self.weights)

    @lru_cache(maxsize=None)
    def inverse_gram(self, edge: int, twice_level: int) -> sp.Matrix:
        return self.modules[edge].gram_matrix(twice_level).inv()

    @lru_cache(maxsize=None)
    def coefficient(self, levels: tuple[int, int, int]) -> sp.Expr:
        bases = tuple(self.modules[edge].basis(level) for edge, level in enumerate(levels))
        inverses = tuple(self.inverse_gram(edge, level) for edge, level in enumerate(levels))
        tensor: dict[tuple[int, int, int], sp.Expr] = {}
        for i0, state0 in enumerate(bases[0]):
            for i1, state1 in enumerate(bases[1]):
                for i2, state2 in enumerate(bases[2]):
                    tensor[i0, i1, i2] = self.form.value(state0, state1, state2)
        contracted = sp.S.Zero
        for a in range(len(bases[0])):
            for b in range(len(bases[1])):
                for d in range(len(bases[2])):
                    for ap in range(len(bases[0])):
                        for bp in range(len(bases[1])):
                            for dp in range(len(bases[2])):
                                contracted += (
                                    tensor[a, b, d]
                                    * inverses[0][a, ap]
                                    * inverses[1][b, bp]
                                    * inverses[2][d, dp]
                                    * tensor[ap, bp, dp]
                                )
        return sp.cancel(theta_orientation_sign(levels) * contracted)


def exact_global_theta_coefficient(
    *, weights: Sequence[sp.Expr], levels: Sequence[int], sectors: tuple[int, int]
) -> sp.Expr:
    parity = sum(levels) % 2
    if sectors != (parity, parity):
        return sp.S.Zero
    occupations = tuple(level // 2 for level in levels)
    fermions = tuple(level % 2 for level in levels)
    rho = exact_osp_three_point(
        n1=occupations[0],
        n2=occupations[1],
        n3=occupations[2],
        epsilon1=fermions[0],
        epsilon2=fermions[1],
        epsilon3=fermions[2],
        d1=weights[0],
        d2=weights[1],
        d3=weights[2],
    )
    denominator = sp.prod(
        exact_osp_norm(weight, occupation, fermion)
        for weight, occupation, fermion in zip(weights, occupations, fermions)
    )
    return sp.cancel(theta_orientation_sign(levels) * rho**2 / denominator)


THETA_VACUUM_ORDER3: Mapping[tuple[int, int, int], int] = {
    (0, 0, 0): 1,
    (0, 3, 3): 1,
    (3, 0, 3): 1,
    (3, 3, 0): -1,
}


def theta_cross_sign(vacuum_levels: Sequence[int], global_levels: Sequence[int]) -> int:
    v0, v1, vinf = (int(level) % 2 for level in vacuum_levels)
    r0, r1, rinf = (int(level) % 2 for level in global_levels)
    exponent = (
        v0 * r1
        + r0 * v1
        + v0 * rinf
        + r0 * vinf
        + v1 * rinf
        + r1 * vinf
    )
    return -1 if exponent % 2 else 1


def exact_regular_theta_coefficient(
    *, weights: Sequence[sp.Expr], levels: Sequence[int], sectors: tuple[int, int]
) -> sp.Expr:
    """Exact vacuum/global convolution through physical total order three."""

    total = sp.S.Zero
    for vacuum_levels, vacuum_coefficient in THETA_VACUUM_ORDER3.items():
        global_levels = tuple(
            int(levels[edge]) - vacuum_levels[edge] for edge in range(3)
        )
        if min(global_levels) < 0:
            continue
        total += (
            theta_cross_sign(vacuum_levels, global_levels)
            * vacuum_coefficient
            * exact_global_theta_coefficient(
                weights=weights, levels=global_levels, sectors=sectors
            )
        )
    return sp.cancel(total)


def fusion_pair(weights: Sequence[sp.Expr], edge: int) -> tuple[sp.Expr, sp.Expr]:
    if edge == 0:
        return weights[1], weights[2]
    if edge == 1:
        return weights[2], weights[0]
    if edge == 2:
        return weights[0], weights[1]
    raise ValueError("theta edge must be 0, 1, or 2")


def paired_shift_fusion(lambda_i_sq: sp.Expr, lambda_j_sq: sp.Expr, shift_sq: sp.Expr) -> sp.Expr:
    return sp.expand(
        ((lambda_i_sq + shift_sq - lambda_j_sq) ** 2 - 4 * shift_sq * lambda_i_sq)
        / 64
    )


def exact_theta_residue(
    *,
    r: int,
    s: int,
    edge: int,
    weights: Sequence[sp.Expr],
    sector: int,
) -> tuple[sp.Expr, sp.Expr]:
    """Return exact ``(c_pole, J*A*P^2)`` for the low-order poles."""

    h = weights[edge]
    first_weight, second_weight = fusion_pair(weights, edge)
    if (r, s) == (3, 1):
        x = -h - sp.Rational(1, 2)  # b^2 on the selected branch
        q_squared = x + 2 + 1 / x
        lambda_i_sq = q_squared - 8 * first_weight
        lambda_j_sq = q_squared - 8 * second_weight
        if sector == 0:
            polynomial = paired_shift_fusion(lambda_i_sq, lambda_j_sq, 4 * x)
        else:
            polynomial = (lambda_i_sq - lambda_j_sq) / 8
        pole_c = 6 - 3 * h - 6 / (2 * h + 1)
        jacobian_times_slope = 6 / (2 * h + 1) ** 2
    elif (r, s) == (2, 2):
        q_squared = -sp.Rational(8, 3) * h
        lambda_i_sq = q_squared - 8 * first_weight
        lambda_j_sq = q_squared - 8 * second_weight
        shift_sq = q_squared if sector == 0 else q_squared - 4
        polynomial = paired_shift_fusion(lambda_i_sq, lambda_j_sq, shift_sq)
        pole_c = sp.Rational(3, 2) - 8 * h
        jacobian_times_slope = sp.Rational(9, 4) / (h * (2 * h + 3))
    elif (r, s) == (5, 1):
        x = -(h + 1) / 3  # b^2 on the selected branch
        q_squared = x + 2 + 1 / x
        lambda_i_sq = q_squared - 8 * first_weight
        lambda_j_sq = q_squared - 8 * second_weight
        if sector == 0:
            polynomial = (
                (lambda_i_sq - lambda_j_sq)
                / 8
                * paired_shift_fusion(lambda_i_sq, lambda_j_sq, 16 * x)
            )
        else:
            polynomial = paired_shift_fusion(lambda_i_sq, lambda_j_sq, 4 * x)
        pole_c = sp.Rational(13, 2) - h - 9 / (h + 1)
        jacobian_times_slope = sp.Rational(9, 8) / (
            h * (h + 1) ** 2 * (h + 2)
        )
    else:
        raise ValueError("unsupported symbolic low-order Kac label")
    return sp.cancel(pole_c), sp.factor(jacobian_times_slope * polynomial**2)


class ExactThetaRecursion:
    def __init__(self) -> None:
        self._cache: dict[
            tuple[sp.Expr, tuple[sp.Expr, ...], tuple[int, ...], tuple[int, int]],
            sp.Expr,
        ] = {}

    def coefficient(
        self,
        *,
        c: sp.Expr,
        weights: tuple[sp.Expr, sp.Expr, sp.Expr],
        levels: tuple[int, int, int],
        sectors: tuple[int, int],
    ) -> sp.Expr:
        key = (c, weights, levels, sectors)
        if key in self._cache:
            return self._cache[key]
        parity = sum(levels) % 2
        if sectors != (parity, parity):
            return sp.S.Zero
        total = exact_regular_theta_coefficient(
            weights=weights, levels=levels, sectors=sectors
        )
        for edge, edge_level in enumerate(levels):
            for r, s, rs in ((3, 1, 3), (2, 2, 4), (5, 1, 5)):
                if rs > edge_level:
                    continue
                pole_c, residue = exact_theta_residue(
                    r=r,
                    s=s,
                    edge=edge,
                    weights=weights,
                    sector=sectors[0],
                )
                shifted_levels = list(levels)
                shifted_levels[edge] -= rs
                shifted_weights = list(weights)
                shifted_weights[edge] += sp.Rational(rs, 2)
                shifted_sectors = (
                    (sectors[0] ^ 1, sectors[1] ^ 1) if rs % 2 else sectors
                )
                orientation_transport = sp.Rational(
                    theta_orientation_sign(levels),
                    theta_orientation_sign(shifted_levels),
                )
                total += (
                    orientation_transport
                    * residue
                    / (c - pole_c)
                    * self.coefficient(
                        c=pole_c,
                        weights=tuple(shifted_weights),
                        levels=tuple(shifted_levels),
                        sectors=shifted_sectors,
                    )
                )
        result = sp.cancel(total)
        self._cache[key] = result
        return result


def level_tuples(max_total_twice_level: int) -> Iterable[tuple[int, int, int]]:
    for total in range(max_total_twice_level + 1):
        for level0 in range(total + 1):
            for level1 in range(total - level0 + 1):
                yield level0, level1, total - level0 - level1


@dataclass(frozen=True)
class SymbolicCheckSummary:
    coefficient_count: int
    zero_difference_count: int
    max_total_twice_level: int
    checked_poles: tuple[str, ...]
    exceptional_double_pole_denominator: str
    vacuum_seed_identities: Mapping[str, str]
    representative_identities: Mapping[str, str]


def run_checks() -> SymbolicCheckSummary:
    weights = (H0, H1, HINF)
    direct = ExactDirectThetaOracle(c=C, weights=weights)
    recursive = ExactThetaRecursion()
    differences: dict[tuple[int, int, int], sp.Expr] = {}
    coefficients: dict[tuple[int, int, int], sp.Expr] = {}
    for levels in level_tuples(6):
        direct_value = direct.coefficient(levels)
        parity = sum(levels) % 2
        recursive_value = recursive.coefficient(
            c=C,
            weights=weights,
            levels=levels,
            sectors=(parity, parity),
        )
        difference = sp.cancel(sp.together(direct_value - recursive_value))
        differences[levels] = difference
        coefficients[levels] = sp.factor(direct_value)
        if difference != 0:
            raise AssertionError(
                f"symbolic mismatch at twice-level {levels}: {sp.factor(difference)}"
            )
    level_300 = (3, 0, 0)
    level_310 = (3, 1, 0)
    regular_300 = exact_global_theta_coefficient(
        weights=weights, levels=level_300, sectors=(1, 1)
    )
    regular_310 = exact_global_theta_coefficient(
        weights=weights, levels=level_310, sectors=(0, 0)
    )
    vacuum_seed_identities: dict[str, str] = {}
    for levels, expected_correction in {
        (0, 3, 3): 1,
        (3, 0, 3): 1,
        (3, 3, 0): -1,
    }.items():
        direct_large_c = sp.cancel(sp.limit(coefficients[levels], C, sp.oo))
        regular_value = exact_regular_theta_coefficient(
            weights=weights, levels=levels, sectors=(0, 0)
        )
        global_value = exact_global_theta_coefficient(
            weights=weights, levels=levels, sectors=(0, 0)
        )
        if sp.cancel(direct_large_c - regular_value) != 0:
            raise AssertionError(f"symbolic large-c mismatch at twice-level {levels}")
        correction = sp.cancel(regular_value - global_value)
        if correction != expected_correction:
            raise AssertionError(
                f"theta vacuum seed mismatch at {levels}: {correction}"
            )
        vacuum_seed_identities[str(levels)] = str(correction)

    # A collision of the two (3,1) edge singularities is not described by
    # merely adding two simple residues.  At h_1=h_0 the already-cancelled
    # direct PBW answer has a genuine squared Kac denominator.
    coincident_330 = sp.cancel(coefficients[(3, 3, 0)].subs(H1, H0))
    double_pole_denominator = sp.factor(sp.denom(coincident_330))
    expected_double_pole_denominator = 4 * H0**2 * (
        (2 * H0 + 1) * C + 6 * H0**2 - 9 * H0
    ) ** 2
    if sp.expand(
        double_pole_denominator - expected_double_pole_denominator
    ) != 0:
        raise AssertionError(
            "the exact D_(3,3,0) coincident-pole denominator changed"
        )

    return SymbolicCheckSummary(
        coefficient_count=len(differences),
        zero_difference_count=sum(value == 0 for value in differences.values()),
        max_total_twice_level=6,
        checked_poles=("(3,1)", "(2,2)", "(5,1)"),
        exceptional_double_pole_denominator=str(double_pole_denominator),
        vacuum_seed_identities=vacuum_seed_identities,
        representative_identities={
            "D_(3,0,0)-U_(3,0,0)": str(
                sp.factor(coefficients[level_300] - regular_300)
            ),
            "D_(3,1,0)-U_(3,1,0)": str(
                sp.factor(coefficients[level_310] - regular_310)
            ),
            "D_(3,0,0)-R_(3,0,0)": str(differences[level_300]),
            "D_(3,1,0)-R_(3,1,0)": str(differences[level_310]),
            "D_(4,0,0)-R_(4,0,0)": str(differences[(4, 0, 0)]),
            "D_(5,0,0)-R_(5,0,0)": str(differences[(5, 0, 0)]),
            "D_(3,3,0)-R_(3,3,0)": str(differences[(3, 3, 0)]),
            "D_(6,0,0)-R_(6,0,0)": str(differences[(6, 0, 0)]),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary = run_checks()
    if args.json:
        print(json.dumps(summary.__dict__, indent=2))
        return
    print("symbolic genus-two NS c-recursion check: PASS")
    print(f"  exact zero identities: {summary.zero_difference_count}/{summary.coefficient_count}")
    print(f"  maximum total twice-level: {summary.max_total_twice_level}")
    print(f"  Kac channels: {', '.join(summary.checked_poles)}")
    print(
        "  D_(3,3,0) at h_1=h_0 denominator: "
        f"{summary.exceptional_double_pole_denominator}"
    )
    print(f"  exact theta-vacuum seed: {dict(summary.vacuum_seed_identities)}")
    for identity, value in summary.representative_identities.items():
        print(f"  {identity} = {value}")


if __name__ == "__main__":
    main()
