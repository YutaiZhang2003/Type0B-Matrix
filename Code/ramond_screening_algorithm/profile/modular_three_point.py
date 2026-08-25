#!/usr/bin/env python3
"""Exact finite-field NS--R--R three-point evaluator.

This is the scalable audit backend for the Ramond branching problem.  It
never constructs a symbolic formula for ``v_n`` or ``W_n``.  Instead it

1. expands only the short ordered chi strings (there are ``2**(2n)`` paths);
2. converts all physical Fock endpoints to PBW coordinates in batches over
   a prime field;
3. evaluates the NS--R--R Ward recursion in that same prime field; and
4. contracts the auxiliary Majorana factor directly by its Pfaffian.

Repeating at several primes and applying CRT/rational reconstruction to the
*final scalar* is an exact characteristic-zero algorithm.  In particular,
there is no floating-point loss of accuracy and no symbolic expression swell
in the branch states.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from math import comb
from pathlib import Path
import argparse
import sys
import time

import numpy as np
import sympy as sp


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GRID = ROOT / "python" / "ramond_three_point_grid"
CHI = ROOT / "python" / "nsrr_chi_branching"
for directory in (HERE, GRID, CHI):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import compute_grid as grid  # noqa: E402
import nsrr_chi_formula as chi  # noqa: E402
from modular_transition import (  # noqa: E402
    _sympy_mod,
    ns_target_rhs,
    ns_reference,
    ns_transition,
    ramond_reference,
    ramond_target_rhs,
    ramond_transition_block,
    ramond_transition_sector,
    rational_mod,
    roots,
    solve_mod,
)


def _add(out, key, coefficient, prime):
    coefficient %= prime
    if not coefficient:
        return
    value = (out.get(key, 0) + coefficient) % prime
    if value:
        out[key] = value
    elif key in out:
        del out[key]


def _mul_rational(value, numerator, denominator, prime):
    return value * (numerator % prime) * pow(denominator % prime, -1, prime) % prime


@lru_cache(None)
def _binomial_mod(numerator, denominator, lower, prime):
    return rational_mod(
        sp.binomial(sp.Rational(numerator, denominator), int(lower)), prime
    )


def _level2(word):
    return sum(-mode2 for _, mode2 in word)


def _parity(word, ground=0):
    return (sum(kind == "G" for kind, _ in word) + int(ground)) % 2


class ModularNRR:
    """The triangular NS--R--R Ward evaluator over ``GF(prime)``.

    Modes are stored doubled, so every mode label is an integer.  This avoids
    allocating SymPy rationals in the hot recursion.
    """

    def __init__(
        self,
        form_parity,
        eta,
        weights,
        momenta,
        central_charge,
        root_i,
        prime,
    ):
        self.form_parity = int(form_parity)
        self.eta = int(eta)
        self.weights = tuple(int(value) % prime for value in weights)
        self.momenta = tuple(int(value) % prime for value in momenta)
        self.central_charge = int(central_charge) % prime
        self.root_i = int(root_i) % prime
        self.prime = int(prime)
        self.inv2 = pow(2, -1, prime)
        self.inv3 = pow(3, -1, prime)
        self.inv4 = pow(4, -1, prime)
        self.inv12 = pow(12, -1, prime)
        self.inv96 = pow(96, -1, prime)

    def bracket(self, first, second):
        first_kind, first_mode2 = first
        second_kind, second_mode2 = second
        p = self.prime
        answer = []
        if first_kind == "L" and second_kind == "L":
            coefficient = (first_mode2 - second_mode2) * self.inv2 % p
            answer.append((coefficient, (("L", first_mode2 + second_mode2),)))
            if first_mode2 + second_mode2 == 0:
                central = (
                    self.central_charge
                    * first_mode2
                    * (first_mode2 * first_mode2 - 4)
                    * self.inv96
                ) % p
                answer.append((central, ()))
        elif first_kind == "L" and second_kind == "G":
            coefficient = (first_mode2 - 2 * second_mode2) * self.inv4 % p
            answer.append((coefficient, (("G", first_mode2 + second_mode2),)))
        elif first_kind == "G" and second_kind == "L":
            coefficient = (2 * first_mode2 - second_mode2) * self.inv4 % p
            answer.append((coefficient, (("G", first_mode2 + second_mode2),)))
        else:
            answer.append((2, (("L", first_mode2 + second_mode2),)))
            if first_mode2 + second_mode2 == 0:
                central = (
                    self.central_charge
                    * (first_mode2 * first_mode2 - 1)
                    * self.inv12
                ) % p
                answer.append((central, ()))
        return tuple(answer)

    @staticmethod
    def canonical_key(operator):
        kind, mode2 = operator
        return (0 if kind == "L" else 1, mode2)

    @lru_cache(None)
    def canonicalize(self, word):
        if any(mode2 >= 0 for _, mode2 in word):
            raise AssertionError(word)
        p = self.prime
        for position in range(len(word) - 1):
            first, second = word[position], word[position + 1]
            if first == second and first[0] == "G":
                reduced = word[:position] + (("L", 2 * first[1]),) + word[position + 2 :]
                return self.canonicalize(reduced)
            if self.canonical_key(first) <= self.canonical_key(second):
                continue
            out = {}
            sign = -1 if first[0] == second[0] == "G" else 1
            exchanged = word[:position] + (second, first) + word[position + 2 :]
            for canonical, coefficient in self.canonicalize(exchanged).items():
                _add(out, canonical, sign * coefficient, p)
            for bracket_coefficient, replacement in self.bracket(first, second):
                reduced = word[:position] + replacement + word[position + 2 :]
                for canonical, coefficient in self.canonicalize(reduced).items():
                    _add(out, canonical, bracket_coefficient * coefficient, p)
            return out
        return {word: 1}

    def _g0(self, slot, ground):
        # beta=P/sqrt(2), and G_0 w^0=i beta (1-i)/sqrt(2) w^1,
        # G_0 w^1=i beta (1+i)/sqrt(2) w^0.
        phase = 1 + self.root_i if ground == 0 else -1 + self.root_i
        return self.momenta[slot] * phase * self.inv2 % self.prime, 1 - ground

    @lru_cache(None)
    def act(self, slot, kind, mode2, word, ground):
        p = self.prime
        if mode2 < 0:
            out = {}
            for canonical, coefficient in self.canonicalize(((kind, mode2),) + word).items():
                _add(out, (canonical, ground), coefficient, p)
            return out

        if not word:
            if kind == "L":
                if mode2 == 0:
                    return {((), ground): self.weights[slot]}
                return {}
            if slot == 0:
                if mode2 == 0:
                    raise AssertionError("NS has no G_0")
                return {}
            if mode2 == 0:
                coefficient, flipped = self._g0(slot, ground)
                return {((), flipped): coefficient}
            return {}

        first, rest = word[0], word[1:]
        out = {}
        sign = -1 if kind == first[0] == "G" else 1
        for (reduced_word, reduced_ground), coefficient in self.act(
            slot, kind, mode2, rest, ground
        ).items():
            for canonical, canonical_coefficient in self.canonicalize(
                (first,) + reduced_word
            ).items():
                _add(
                    out,
                    (canonical, reduced_ground),
                    sign * coefficient * canonical_coefficient,
                    p,
                )
        for bracket_coefficient, replacement in self.bracket((kind, mode2), first):
            if not replacement:
                _add(out, (rest, ground), bracket_coefficient, p)
                continue
            replacement_kind, replacement_mode2 = replacement[0]
            for key, coefficient in self.act(
                slot, replacement_kind, replacement_mode2, rest, ground
            ).items():
                _add(out, key, bracket_coefficient * coefficient, p)
        return out

    def state_sum(self, word1, states2, states3):
        answer = 0
        p = self.prime
        for (word2, ground2), coefficient2 in states2.items():
            for (word3, ground3), coefficient3 in states3.items():
                answer += coefficient2 * coefficient3 * self.raw_value(
                    word1, word2, ground2, word3, ground3
                )
        return answer % p

    def first_state_sum(self, states1, word2, ground2, word3, ground3):
        answer = 0
        p = self.prime
        for (word1, _), coefficient in states1.items():
            answer += coefficient * self.raw_value(
                word1, word2, ground2, word3, ground3
            )
        return answer % p

    def binomial_half(self, numerator, denominator, lower):
        return _binomial_mod(numerator, denominator, lower, self.prime)

    @lru_cache(None)
    def raw_value(self, word1, word2, ground2, word3, ground3):
        pmod = self.prime

        if word1 and word1[0][0] == "G":
            _, mode2 = word1[0]
            rest1 = word1[1:]
            r2 = -mode2
            answer = 0
            parity_sign = -1 if _parity(word3, ground3) else 1
            maximum2 = max(r2 + _level2(rest1), _level2(word2), _level2(word3), 0)
            maximum = maximum2 // 2 + 3
            for lower in range(maximum + 1):
                acted2 = self.act(1, "G", 2 * lower, word2, ground2)
                answer += (
                    parity_sign
                    * self.binomial_half(r2, 2, lower)
                    * self.state_sum(rest1, acted2, {(word3, ground3): 1})
                )
                if lower:
                    acted1 = self.act(0, "G", 2 * lower - r2, rest1, 0)
                    answer -= (
                        self.binomial_half(1, 2, lower)
                        * (-1) ** lower
                        * self.first_state_sum(
                            acted1, word2, ground2, word3, ground3
                        )
                    )
                acted3 = self.act(
                    2, "G", r2 - 1 + 2 * lower, word3, ground3
                )
                answer += (
                    -self.root_i
                    * self.binomial_half(1, 2, lower)
                    * (-1) ** lower
                    * self.state_sum(rest1, {(word2, ground2): 1}, acted3)
                )
            return answer % pmod

        if word2:
            kind, mode2 = word2[0]
            rest2 = word2[1:]
            if kind == "L":
                n = -mode2 // 2
                if n == 1:
                    exponent = (
                        self.weights[0]
                        - self.weights[1]
                        - self.weights[2]
                        + (_level2(word1) - _level2(rest2) - _level2(word3))
                        * self.inv2
                    ) % pmod
                    return exponent * self.raw_value(
                        word1, rest2, ground2, word3, ground3
                    ) % pmod
                answer = 0
                maximum2 = max(_level2(word1) - 2 * n, _level2(word3) + 2, 0)
                maximum = maximum2 // 2
                for lower in range(maximum + 1):
                    ward = self.binomial_half(n - 2 + lower, 1, n - 2)
                    acted1 = self.act(0, "L", 2 * (n + lower), word1, 0)
                    answer += ward * self.first_state_sum(
                        acted1, rest2, ground2, word3, ground3
                    )
                    acted3 = self.act(2, "L", 2 * (lower - 1), word3, ground3)
                    answer += ward * (-1) ** n * self.state_sum(
                        word1, {(rest2, ground2): 1}, acted3
                    )
                return answer % pmod

            n = -mode2 // 2
            answer = 0
            parity_sign = -1 if _parity(word3, ground3) else 1
            maximum2 = max(
                2 * n + _level2(rest2),
                _level2(word1) - 2 * n + 1,
                _level2(word3),
                0,
            )
            maximum = maximum2 // 2 + 2
            for lower in range(maximum + 1):
                acted1 = self.act(
                    0, "G", 2 * lower + 2 * n - 1, word1, 0
                )
                answer += (
                    parity_sign
                    * self.binomial_half(1 - 2 * n, 2, lower)
                    * (-1) ** lower
                    * self.first_state_sum(
                        acted1, rest2, ground2, word3, ground3
                    )
                )
                acted3 = self.act(2, "G", 2 * lower, word3, ground3)
                answer += (
                    self.root_i
                    * parity_sign
                    * self.binomial_half(1 - 2 * n, 2, lower)
                    * (-1) ** (n + lower)
                    * self.state_sum(word1, {(rest2, ground2): 1}, acted3)
                )
                if lower:
                    acted2 = self.act(
                        1, "G", 2 * (lower - n), rest2, ground2
                    )
                    answer -= self.binomial_half(1, 2, lower) * self.state_sum(
                        word1, acted2, {(word3, ground3): 1}
                    )
            return answer % pmod

        if word1:
            kind, mode2 = word1[0]
            rest1 = word1[1:]
            if kind == "L":
                n = -mode2 // 2
                answer = self.state_sum(
                    rest1,
                    {((), ground2): 1},
                    self.act(2, "L", 2 * n, word3, ground3),
                )
                for middle_mode in range(-1, n + 1):
                    ward = self.binomial_half(n + 1, 1, middle_mode + 1)
                    answer += ward * self.state_sum(
                        rest1,
                        self.act(1, "L", 2 * middle_mode, (), ground2),
                        {(word3, ground3): 1},
                    )
                return answer % pmod

            r2 = -mode2
            answer = 0
            parity_sign = -1 if _parity(word3, ground3) else 1
            maximum2 = max(r2 + _level2(rest1), _level2(word3), 0)
            maximum = maximum2 // 2 + 3
            for lower in range(maximum + 1):
                acted2 = self.act(1, "G", 2 * lower, (), ground2)
                answer += (
                    parity_sign
                    * self.binomial_half(r2, 2, lower)
                    * self.state_sum(rest1, acted2, {(word3, ground3): 1})
                )
                if lower:
                    acted1 = self.act(0, "G", 2 * lower - r2, rest1, 0)
                    answer -= (
                        self.binomial_half(1, 2, lower)
                        * (-1) ** lower
                        * self.first_state_sum(
                            acted1, (), ground2, word3, ground3
                        )
                    )
                acted3 = self.act(
                    2, "G", r2 - 1 + 2 * lower, word3, ground3
                )
                answer += (
                    -self.root_i
                    * self.binomial_half(1, 2, lower)
                    * (-1) ** lower
                    * self.state_sum(rest1, {((), ground2): 1}, acted3)
                )
            return answer % pmod

        if word3:
            kind, mode2 = word3[0]
            rest3 = word3[1:]
            if kind == "L":
                n = -mode2 // 2
                coefficient = (
                    self.weights[2]
                    - self.weights[0]
                    + _level2(rest3) * self.inv2
                    + n * self.weights[1]
                ) % pmod
                return coefficient * self.raw_value(
                    (), (), ground2, rest3, ground3
                ) % pmod

            m = -mode2 // 2
            parity_sign = -1 if _parity(rest3, ground3) else 1
            answer = 0
            maximum = (2 * m + _level2(rest3)) // 2 + 2
            for lower in range(maximum + 1):
                acted2 = self.act(1, "G", 2 * lower, (), ground2)
                answer += (
                    -self.root_i
                    * parity_sign
                    * self.binomial_half(1 - 2 * m, 2, lower)
                    * self.state_sum((), acted2, {(rest3, ground3): 1})
                )
                acted1 = self.act(
                    0, "G", 2 * m - 1 + 2 * lower, (), 0
                )
                answer += (
                    self.root_i
                    * (-1) ** lower
                    * self.binomial_half(1, 2, lower)
                    * self.first_state_sum(
                        acted1, (), ground2, rest3, ground3
                    )
                )
                if lower:
                    acted3 = self.act(
                        2, "G", 2 * (-m + lower), rest3, ground3
                    )
                    answer -= (
                        self.binomial_half(1, 2, lower)
                        * (-1) ** lower
                        * self.state_sum((), {((), ground2): 1}, acted3)
                    )
            return answer % pmod

        if self.form_parity == 0:
            if (ground2, ground3) == (0, 0):
                return 1
            if (ground2, ground3) == (1, 1):
                return self.eta % pmod
            return 0
        if (ground2, ground3) == (0, 1):
            return 1
        if (ground2, ground3) == (1, 0):
            return self.root_i * self.eta % pmod
        return 0


def _pack_vector(expression, prime):
    """Canonical immutable representation of a sparse PBW vector."""

    cleaned = []
    for (color, word, ground), coefficient in expression.items():
        coefficient %= prime
        if coefficient:
            cleaned.append((color, word, int(ground), coefficient))
    return tuple(sorted(cleaned, key=lambda item: (repr(item[0]), item[1], item[2])))


class VectorWardContraction:
    """Apply Ward identities to whole homogeneous vectors.

    The scalar recursion evaluates one PBW triple at a time.  Here linearity
    is used before recursion: a common leading mode is removed from an entire
    homogeneous vector, and positive modes act on the other entire vector.
    The top-level chi expansion therefore has one call per auxiliary path
    triple, not one call per PBW component triple.
    """

    def __init__(self, evaluator, ground_pairing=None):
        self.evaluator = evaluator
        self.prime = evaluator.prime
        self.ground_pairing = ground_pairing

    @staticmethod
    def level2(vector):
        levels = {_level2(word) for _, word, _, _ in vector}
        if not levels:
            return 0
        if len(levels) != 1:
            raise AssertionError(("inhomogeneous level", levels))
        return next(iter(levels))

    @staticmethod
    def parity(vector):
        parities = {_parity(word, ground) for _, word, ground, _ in vector}
        if not parities:
            return 0
        if len(parities) != 1:
            raise AssertionError(("inhomogeneous parity", parities))
        return next(iter(parities))

    def add_scaled(self, target, vector, scalar=1):
        for color, word, ground, coefficient in vector:
            _add(
                target,
                (color, word, ground),
                scalar * coefficient,
                self.prime,
            )

    @lru_cache(None)
    def apply(self, slot, kind, mode2, vector):
        out = {}
        for color, word, ground, outer in vector:
            for key, inner in self.evaluator.act(
                slot, kind, mode2, word, ground
            ).items():
                final_word, final_ground = key
                _add(
                    out,
                    (color, final_word, final_ground),
                    outer * inner,
                    self.prime,
                )
        return _pack_vector(out, self.prime)

    def split_leading(self, vector, require_kind=None):
        """Strip one common leading operator and return the residual vector."""

        selected = next(
            (
                word[0]
                for _, word, _, _ in vector
                if word and (require_kind is None or word[0][0] == require_kind)
            ),
            None,
        )
        if selected is None:
            return None, (), vector
        stripped = {}
        residual = {}
        for color, word, ground, coefficient in vector:
            if word and word[0] == selected:
                _add(
                    stripped,
                    (color, word[1:], ground),
                    coefficient,
                    self.prime,
                )
            else:
                _add(
                    residual,
                    (color, word, ground),
                    coefficient,
                    self.prime,
                )
        return (
            selected,
            _pack_vector(stripped, self.prime),
            _pack_vector(residual, self.prime),
        )

    def ground_value(self, first, second, third):
        answer = 0
        for color1, word1, ground1, coefficient1 in first:
            if word1 or ground1:
                raise AssertionError((word1, ground1))
            for color2, word2, ground2, coefficient2 in second:
                if word2:
                    raise AssertionError(word2)
                for color3, word3, ground3, coefficient3 in third:
                    if word3:
                        raise AssertionError(word3)
                    if self.ground_pairing is None:
                        pairing = self.evaluator.raw_value(
                            (), (), ground2, (), ground3
                        )
                    else:
                        pairing = self.ground_pairing(
                            color1,
                            color2,
                            color3,
                            ground1,
                            ground2,
                            ground3,
                        )
                    answer += (
                        coefficient1
                        * coefficient2
                        * coefficient3
                        * pairing
                    )
        return answer % self.prime

    @lru_cache(None)
    def value(self, first, second, third):
        if not first or not second or not third:
            return 0
        ev = self.evaluator
        pmod = self.prime

        operator, rest_first, residual = self.split_leading(first, "G")
        if operator is not None:
            answer = self.value(residual, second, third) if residual else 0
            _, mode2 = operator
            r2 = -mode2
            parity_sign = -1 if self.parity(third) else 1
            maximum2 = max(
                r2 + self.level2(rest_first),
                self.level2(second),
                self.level2(third),
                0,
            )
            maximum = maximum2 // 2 + 3
            for lower in range(maximum + 1):
                answer += (
                    parity_sign
                    * ev.binomial_half(r2, 2, lower)
                    * self.value(
                        rest_first,
                        self.apply(1, "G", 2 * lower, second),
                        third,
                    )
                )
                if lower:
                    answer -= (
                        ev.binomial_half(1, 2, lower)
                        * (-1) ** lower
                        * self.value(
                            self.apply(0, "G", 2 * lower - r2, rest_first),
                            second,
                            third,
                        )
                    )
                answer += (
                    -ev.root_i
                    * ev.binomial_half(1, 2, lower)
                    * (-1) ** lower
                    * self.value(
                        rest_first,
                        second,
                        self.apply(2, "G", r2 - 1 + 2 * lower, third),
                    )
                )
            return answer % pmod

        operator, rest_second, residual = self.split_leading(second)
        if operator is not None:
            answer = self.value(first, residual, third) if residual else 0
            kind, mode2 = operator
            if kind == "L":
                n = -mode2 // 2
                if n == 1:
                    exponent = (
                        ev.weights[0]
                        - ev.weights[1]
                        - ev.weights[2]
                        + (self.level2(first) - self.level2(rest_second) - self.level2(third))
                        * ev.inv2
                    ) % pmod
                    answer += exponent * self.value(first, rest_second, third)
                    return answer % pmod
                maximum2 = max(
                    self.level2(first) - 2 * n,
                    self.level2(third) + 2,
                    0,
                )
                for lower in range(maximum2 // 2 + 1):
                    ward = ev.binomial_half(n - 2 + lower, 1, n - 2)
                    answer += ward * self.value(
                        self.apply(0, "L", 2 * (n + lower), first),
                        rest_second,
                        third,
                    )
                    answer += ward * (-1) ** n * self.value(
                        first,
                        rest_second,
                        self.apply(2, "L", 2 * (lower - 1), third),
                    )
                return answer % pmod

            n = -mode2 // 2
            parity_sign = -1 if self.parity(third) else 1
            maximum2 = max(
                2 * n + self.level2(rest_second),
                self.level2(first) - 2 * n + 1,
                self.level2(third),
                0,
            )
            for lower in range(maximum2 // 2 + 3):
                answer += (
                    parity_sign
                    * ev.binomial_half(1 - 2 * n, 2, lower)
                    * (-1) ** lower
                    * self.value(
                        self.apply(0, "G", 2 * lower + 2 * n - 1, first),
                        rest_second,
                        third,
                    )
                )
                answer += (
                    ev.root_i
                    * parity_sign
                    * ev.binomial_half(1 - 2 * n, 2, lower)
                    * (-1) ** (n + lower)
                    * self.value(
                        first,
                        rest_second,
                        self.apply(2, "G", 2 * lower, third),
                    )
                )
                if lower:
                    answer -= ev.binomial_half(1, 2, lower) * self.value(
                        first,
                        self.apply(1, "G", 2 * (lower - n), rest_second),
                        third,
                    )
            return answer % pmod

        operator, rest_first, residual = self.split_leading(first)
        if operator is not None:
            answer = self.value(residual, second, third) if residual else 0
            kind, mode2 = operator
            if kind != "L":
                raise AssertionError(operator)
            n = -mode2 // 2
            answer += self.value(
                rest_first,
                second,
                self.apply(2, "L", 2 * n, third),
            )
            for middle_mode in range(-1, n + 1):
                ward = ev.binomial_half(n + 1, 1, middle_mode + 1)
                answer += ward * self.value(
                    rest_first,
                    self.apply(1, "L", 2 * middle_mode, second),
                    third,
                )
            return answer % pmod

        operator, rest_third, residual = self.split_leading(third)
        if operator is not None:
            answer = self.value(first, second, residual) if residual else 0
            kind, mode2 = operator
            if kind == "L":
                n = -mode2 // 2
                coefficient = (
                    ev.weights[2]
                    - ev.weights[0]
                    + self.level2(rest_third) * ev.inv2
                    + n * ev.weights[1]
                ) % pmod
                answer += coefficient * self.value(first, second, rest_third)
                return answer % pmod

            m = -mode2 // 2
            parity_sign = -1 if self.parity(rest_third) else 1
            maximum = (2 * m + self.level2(rest_third)) // 2 + 2
            for lower in range(maximum + 1):
                answer += (
                    -ev.root_i
                    * parity_sign
                    * ev.binomial_half(1 - 2 * m, 2, lower)
                    * self.value(
                        first,
                        self.apply(1, "G", 2 * lower, second),
                        rest_third,
                    )
                )
                answer += (
                    ev.root_i
                    * (-1) ** lower
                    * ev.binomial_half(1, 2, lower)
                    * self.value(
                        self.apply(0, "G", 2 * m - 1 + 2 * lower, first),
                        second,
                        rest_third,
                    )
                )
                if lower:
                    answer -= (
                        ev.binomial_half(1, 2, lower)
                        * (-1) ** lower
                        * self.value(
                            first,
                            second,
                            self.apply(2, "G", 2 * (-m + lower), rest_third),
                        )
                    )
            return answer % pmod

        return self.ground_value(first, second, third)


class DenseMiddleWardContraction:
    """Eliminate the middle Ramond state by finite-field matrix blocks.

    For a fixed middle PBW word the NS--R--R form is a matrix between a
    homogeneous NS basis and a homogeneous Ramond basis.  Ward identities
    then act by left and right multiplication with sparse representation
    matrices.  This variable-elimination order computes a block once and
    reuses it for every auxiliary Fock path; it never forms the tensor of
    individual component triples.

    The scalar recursion gives priority to a leading NS ``G`` mode.  Such
    rows are therefore overwritten by the identical leading-``G`` identity
    after the middle-word matrix recurrence is evaluated.  This small detail
    is required for the branch-cut convention of :class:`ModularNRR`.
    """

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.prime = evaluator.prime
        self.block_calls = 0
        self.block_entries = 0
        self.base_entries = 0

    @lru_cache(None)
    def basis(self, slot, level2, parity):
        slot = int(slot)
        level2 = int(level2)
        parity = int(parity)
        if level2 < 0:
            return ()
        answer = []
        if slot == 0:
            for virasoro_modes, supercurrent_modes2 in ns_reference.basis(level2):
                word = tuple(
                    ("L", -2 * int(mode)) for mode in virasoro_modes
                ) + tuple(
                    ("G", -int(mode2)) for mode2 in supercurrent_modes2
                )
                if _parity(word, 0) == parity:
                    answer.append((word, 0))
        else:
            if level2 % 2:
                return ()
            for virasoro_modes, supercurrent_modes, ground in (
                ramond_reference.basis(level2 // 2)
            ):
                word = tuple(
                    ("L", -2 * int(mode)) for mode in virasoro_modes
                ) + tuple(
                    ("G", -2 * int(mode)) for mode in supercurrent_modes
                )
                if _parity(word, ground) == parity:
                    answer.append((word, int(ground)))
        return tuple(answer)

    @lru_cache(None)
    def basis_index(self, slot, level2, parity):
        return {
            state: index
            for index, state in enumerate(self.basis(slot, level2, parity))
        }

    @lru_cache(None)
    def action_matrix(self, slot, kind, mode2, level2, parity):
        """Matrix of one SCA mode, with columns indexed by source states."""

        slot = int(slot)
        level2 = int(level2)
        parity = int(parity)
        mode2 = int(mode2)
        target_level2 = level2 - mode2
        target_parity = parity ^ (kind == "G")
        source = self.basis(slot, level2, parity)
        target = self.basis(slot, target_level2, target_parity)
        matrix = np.zeros((len(target), len(source)), dtype=np.int64)
        if not source or not target:
            return matrix
        row = self.basis_index(slot, target_level2, target_parity)
        for column, (word, ground) in enumerate(source):
            for state, coefficient in self.evaluator.act(
                slot, kind, mode2, word, ground
            ).items():
                if state not in row:
                    raise AssertionError(
                        ("mode image outside PBW basis", slot, kind, mode2, state)
                    )
                matrix[row[state], column] = (
                    matrix[row[state], column] + coefficient
                ) % self.prime
        return matrix

    def _left_action(self, matrix, block):
        if not matrix.size or not block.size:
            return np.zeros(
                (matrix.shape[1], block.shape[1]), dtype=np.int64
            )
        return matrix.T @ block % self.prime

    def _right_action(self, block, matrix):
        if not block.size or not matrix.size:
            return np.zeros(
                (block.shape[0], matrix.shape[1]), dtype=np.int64
            )
        return block @ matrix % self.prime

    @lru_cache(None)
    def base_block(self, ground2, level1, parity1, level3, parity3):
        """Middle-primary block; scalar Ward values are only a 2-leg table."""

        first = self.basis(0, level1, parity1)
        third = self.basis(2, level3, parity3)
        answer = np.zeros((len(first), len(third)), dtype=np.int64)
        for row, (word1, _ground1) in enumerate(first):
            for column, (word3, ground3) in enumerate(third):
                answer[row, column] = self.evaluator.raw_value(
                    word1, (), int(ground2), word3, ground3
                )
        self.base_entries += answer.size
        return answer

    def _state_row(self, block, slot, level2, parity, state):
        index = self.basis_index(slot, level2, parity).get(state)
        if index is None:
            raise AssertionError(("missing PBW row", slot, level2, parity, state))
        return block[index]

    def _leading_g_row(
        self,
        word1,
        word2,
        ground2,
        level3,
        parity3,
    ):
        """Exact exceptional row in the branch-cut Ward convention.

        Only NS PBW words with no leading Virasoro mode enter here.  Their
        number is the number of strict odd partitions, far smaller than the
        full NS basis.  Evaluating these rows with the already memoized scalar
        oracle avoids imposing an invalid reordered square-root Ward identity
        while retaining dense elimination for every ordinary row.
        """

        return np.fromiter(
            (
                self.evaluator.raw_value(
                    word1, word2, int(ground2), word3, ground3
                )
                for word3, ground3 in self.basis(2, level3, parity3)
            ),
            dtype=np.int64,
            count=len(self.basis(2, level3, parity3)),
        )

    @lru_cache(None)
    def block(self, word2, ground2, level1, parity1, level3, parity3):
        """Return the physical form matrix for one middle PBW state."""

        word2 = tuple(word2)
        ground2 = int(ground2)
        level1 = int(level1)
        parity1 = int(parity1)
        level3 = int(level3)
        parity3 = int(parity3)
        if level1 < 0 or level3 < 0:
            return np.zeros(
                (
                    len(self.basis(0, level1, parity1)),
                    len(self.basis(2, level3, parity3)),
                ),
                dtype=np.int64,
            )
        if not word2:
            return self.base_block(
                ground2, level1, parity1, level3, parity3
            )

        self.block_calls += 1
        ev = self.evaluator
        pmod = self.prime
        first_basis = self.basis(0, level1, parity1)
        third_basis = self.basis(2, level3, parity3)
        answer = np.zeros(
            (len(first_basis), len(third_basis)), dtype=np.int64
        )
        kind, mode2 = word2[0]
        rest2 = word2[1:]
        if kind == "L":
            n = -mode2 // 2
            if n == 1:
                exponent = (
                    ev.weights[0]
                    - ev.weights[1]
                    - ev.weights[2]
                    + (
                        level1
                        - _level2(rest2)
                        - level3
                    )
                    * ev.inv2
                ) % pmod
                answer = exponent * self.block(
                    rest2,
                    ground2,
                    level1,
                    parity1,
                    level3,
                    parity3,
                ) % pmod
            else:
                maximum2 = max(level1 - 2 * n, level3 + 2, 0)
                for lower in range(maximum2 // 2 + 1):
                    ward = ev.binomial_half(n - 2 + lower, 1, n - 2)
                    first_action = self.action_matrix(
                        0,
                        "L",
                        2 * (n + lower),
                        level1,
                        parity1,
                    )
                    target_level1 = level1 - 2 * (n + lower)
                    reduced = self.block(
                        rest2,
                        ground2,
                        target_level1,
                        parity1,
                        level3,
                        parity3,
                    )
                    answer += ward * self._left_action(first_action, reduced)

                    third_mode2 = 2 * (lower - 1)
                    third_action = self.action_matrix(
                        2,
                        "L",
                        third_mode2,
                        level3,
                        parity3,
                    )
                    target_level3 = level3 - third_mode2
                    reduced = self.block(
                        rest2,
                        ground2,
                        level1,
                        parity1,
                        target_level3,
                        parity3,
                    )
                    answer += (
                        ward
                        * (-1) ** n
                        * self._right_action(reduced, third_action)
                    )
        else:
            n = -mode2 // 2
            parity_sign = -1 if parity3 else 1
            maximum2 = max(
                2 * n + _level2(rest2),
                level1 - 2 * n + 1,
                level3,
                0,
            )
            for lower in range(maximum2 // 2 + 3):
                common = (
                    parity_sign
                    * ev.binomial_half(1 - 2 * n, 2, lower)
                    * (-1) ** lower
                )
                first_mode2 = 2 * lower + 2 * n - 1
                first_action = self.action_matrix(
                    0,
                    "G",
                    first_mode2,
                    level1,
                    parity1,
                )
                reduced = self.block(
                    rest2,
                    ground2,
                    level1 - first_mode2,
                    parity1 ^ 1,
                    level3,
                    parity3,
                )
                answer += common * self._left_action(first_action, reduced)

                third_mode2 = 2 * lower
                third_action = self.action_matrix(
                    2,
                    "G",
                    third_mode2,
                    level3,
                    parity3,
                )
                reduced = self.block(
                    rest2,
                    ground2,
                    level1,
                    parity1,
                    level3 - third_mode2,
                    parity3 ^ 1,
                )
                answer += (
                    ev.root_i
                    * parity_sign
                    * ev.binomial_half(1 - 2 * n, 2, lower)
                    * (-1) ** (n + lower)
                    * self._right_action(reduced, third_action)
                )
                if lower:
                    for (reduced2, reduced_ground2), coefficient in ev.act(
                        1,
                        "G",
                        2 * (lower - n),
                        rest2,
                        ground2,
                    ).items():
                        answer -= (
                            ev.binomial_half(1, 2, lower)
                            * coefficient
                            * self.block(
                                reduced2,
                                reduced_ground2,
                                level1,
                                parity1,
                                level3,
                                parity3,
                            )
                        )

        # The branch-cut recursion moves a leading NS G before the middle
        # state.  Replace precisely those rows by that exact recurrence.
        for row, (word1, _ground1) in enumerate(first_basis):
            if word1 and word1[0][0] == "G":
                answer[row] = self._leading_g_row(
                    word1,
                    word2,
                    ground2,
                    level3,
                    parity3,
                )
        answer %= pmod
        self.block_entries += answer.size
        return answer

    def coefficient_matrix(self, slot, sector, vector, colors):
        """Dense physical-basis by auxiliary-color coefficient matrix."""

        level2, parity = sector
        states = self.basis(slot, level2, parity)
        row = self.basis_index(slot, level2, parity)
        color_index = {color: index for index, color in enumerate(colors)}
        answer = np.zeros((len(states), len(colors)), dtype=np.int64)
        for color, word, ground, coefficient in vector:
            answer[row[(word, ground)], color_index[color]] = (
                answer[row[(word, ground)], color_index[color]] + coefficient
            ) % self.prime
        return answer

    def pack_middle(self, expression):
        cleaned = []
        for (word, ground), coefficient in expression.items():
            coefficient %= self.prime
            if coefficient:
                cleaned.append((word, int(ground), coefficient))
        return tuple(sorted(cleaned, key=lambda item: (item[0], item[1])))

    def split_middle(self, vector):
        selected = next(
            (word[0] for word, _ground, _coefficient in vector if word),
            None,
        )
        if selected is None:
            return None, (), vector
        stripped = {}
        residual = {}
        for word, ground, coefficient in vector:
            target = stripped if word and word[0] == selected else residual
            final_word = word[1:] if word and word[0] == selected else word
            _add(target, (final_word, ground), coefficient, self.prime)
        return selected, self.pack_middle(stripped), self.pack_middle(residual)

    @lru_cache(None)
    def apply_middle(self, kind, mode2, vector):
        out = {}
        for word, ground, outer in vector:
            for state, inner in self.evaluator.act(
                1, kind, mode2, word, ground
            ).items():
                _add(out, state, outer * inner, self.prime)
        return self.pack_middle(out)

    def _leading_g_vector_row(
        self, word1, middle, level3, parity3
    ):
        return np.fromiter(
            (
                sum(
                    coefficient
                    * self.evaluator.raw_value(
                        word1, word2, ground2, word3, ground3
                    )
                    for word2, ground2, coefficient in middle
                )
                % self.prime
                for word3, ground3 in self.basis(2, level3, parity3)
            ),
            dtype=np.int64,
            count=len(self.basis(2, level3, parity3)),
        )

    @lru_cache(None)
    def vector_block(self, middle, level1, parity1, level3, parity3):
        """Block for a whole sparse middle vector, before auxiliary summation."""

        level1 = int(level1)
        parity1 = int(parity1)
        level3 = int(level3)
        parity3 = int(parity3)
        first_basis = self.basis(0, level1, parity1)
        third_basis = self.basis(2, level3, parity3)
        shape = (len(first_basis), len(third_basis))
        if not middle or level1 < 0 or level3 < 0:
            return np.zeros(shape, dtype=np.int64)
        operator, rest_middle, residual = self.split_middle(middle)
        if operator is None:
            answer = np.zeros(shape, dtype=np.int64)
            for word, ground, coefficient in middle:
                if word:
                    raise AssertionError(word)
                answer += coefficient * self.base_block(
                    ground, level1, parity1, level3, parity3
                )
            return answer % self.prime

        answer = (
            self.vector_block(
                residual, level1, parity1, level3, parity3
            ).copy()
            if residual
            else np.zeros(shape, dtype=np.int64)
        )
        ev = self.evaluator
        pmod = self.prime
        kind, mode2 = operator
        rest_level2 = _level2(rest_middle[0][0]) if rest_middle else 0
        if any(_level2(word) != rest_level2 for word, _, _ in rest_middle):
            raise AssertionError("inhomogeneous stripped middle vector")
        if kind == "L":
            n = -mode2 // 2
            if n == 1:
                exponent = (
                    ev.weights[0]
                    - ev.weights[1]
                    - ev.weights[2]
                    + (level1 - rest_level2 - level3) * ev.inv2
                ) % pmod
                answer += exponent * self.vector_block(
                    rest_middle, level1, parity1, level3, parity3
                )
            else:
                maximum2 = max(level1 - 2 * n, level3 + 2, 0)
                for lower in range(maximum2 // 2 + 1):
                    ward = ev.binomial_half(n - 2 + lower, 1, n - 2)
                    first_mode2 = 2 * (n + lower)
                    first_action = self.action_matrix(
                        0, "L", first_mode2, level1, parity1
                    )
                    reduced = self.vector_block(
                        rest_middle,
                        level1 - first_mode2,
                        parity1,
                        level3,
                        parity3,
                    )
                    answer += ward * self._left_action(first_action, reduced)

                    third_mode2 = 2 * (lower - 1)
                    third_action = self.action_matrix(
                        2, "L", third_mode2, level3, parity3
                    )
                    reduced = self.vector_block(
                        rest_middle,
                        level1,
                        parity1,
                        level3 - third_mode2,
                        parity3,
                    )
                    answer += (
                        ward
                        * (-1) ** n
                        * self._right_action(reduced, third_action)
                    )
        else:
            n = -mode2 // 2
            parity_sign = -1 if parity3 else 1
            maximum2 = max(
                2 * n + rest_level2,
                level1 - 2 * n + 1,
                level3,
                0,
            )
            for lower in range(maximum2 // 2 + 3):
                first_mode2 = 2 * lower + 2 * n - 1
                first_action = self.action_matrix(
                    0, "G", first_mode2, level1, parity1
                )
                reduced = self.vector_block(
                    rest_middle,
                    level1 - first_mode2,
                    parity1 ^ 1,
                    level3,
                    parity3,
                )
                answer += (
                    parity_sign
                    * ev.binomial_half(1 - 2 * n, 2, lower)
                    * (-1) ** lower
                    * self._left_action(first_action, reduced)
                )

                third_mode2 = 2 * lower
                third_action = self.action_matrix(
                    2, "G", third_mode2, level3, parity3
                )
                reduced = self.vector_block(
                    rest_middle,
                    level1,
                    parity1,
                    level3 - third_mode2,
                    parity3 ^ 1,
                )
                answer += (
                    ev.root_i
                    * parity_sign
                    * ev.binomial_half(1 - 2 * n, 2, lower)
                    * (-1) ** (n + lower)
                    * self._right_action(reduced, third_action)
                )
                if lower:
                    acted_middle = self.apply_middle(
                        "G", 2 * (lower - n), rest_middle
                    )
                    answer -= ev.binomial_half(
                        1, 2, lower
                    ) * self.vector_block(
                        acted_middle,
                        level1,
                        parity1,
                        level3,
                        parity3,
                    )

        for row, (word1, _ground1) in enumerate(first_basis):
            if word1 and word1[0][0] == "G":
                answer[row] = self._leading_g_vector_row(
                    word1, middle, level3, parity3
                )
        return answer % pmod

    def contract_sector(
        self,
        sector1,
        vector1,
        sector2,
        vector2,
        sector3,
        vector3,
        colors,
        auxiliary_tensor,
    ):
        """Contract one homogeneous sector triple by middle-leg elimination."""

        pmod = self.prime
        coefficients1 = self.coefficient_matrix(
            0, sector1, vector1, colors[0]
        )
        coefficients2 = self.coefficient_matrix(
            1, sector2, vector2, colors[1]
        )
        coefficients3 = self.coefficient_matrix(
            2, sector3, vector3, colors[2]
        )
        active_middle = np.flatnonzero(np.any(coefficients2, axis=1))
        if not len(active_middle):
            return 0
        middle_basis = self.basis(1, *sector2)
        blocks = np.stack(
            [
                self.block(
                    middle_basis[index][0],
                    middle_basis[index][1],
                    sector1[0],
                    sector1[1],
                    sector3[0],
                    sector3[1],
                )
                for index in active_middle
            ],
            axis=0,
        )
        # The largest sum in this benchmark has 40 terms.  With p<2^20,
        # signed int64 multiplication is exact before the modular reduction.
        colored_blocks = np.tensordot(
            coefficients2[active_middle].T,
            blocks,
            axes=(1, 0),
        ) % pmod
        answer = 0
        for middle_color, matrix in enumerate(colored_blocks):
            if not np.any(matrix):
                continue
            reduced = coefficients1.T @ matrix % pmod
            reduced = reduced @ coefficients3 % pmod
            answer += int(
                np.sum(
                    reduced * auxiliary_tensor[:, middle_color, :]
                    % pmod,
                    dtype=np.int64,
                )
            )
        return answer % pmod


def _auxiliary_color_tensor(
    colors,
    auxiliary_form_parity,
    prime,
    root_i,
    root_two,
):
    """Corrected auxiliary sewing tensor, including all Koszul signs."""

    answer = np.zeros(tuple(map(len, colors)), dtype=np.int64)
    for first_index, color1 in enumerate(colors[0]):
        auxiliary1, physical_parity1 = color1
        for second_index, color2 in enumerate(colors[1]):
            auxiliary2, ground2, physical_parity2 = color2
            auxiliary_parity2 = (len(auxiliary2) + ground2) % 2
            for third_index, color3 in enumerate(colors[2]):
                auxiliary3, ground3, _physical_parity3 = color3
                auxiliary_parity3 = (len(auxiliary3) + ground3) % 2
                koszul = -1 if (
                    physical_parity1
                    * (auxiliary_parity2 + auxiliary_parity3)
                    + physical_parity2 * auxiliary_parity3
                ) % 2 else 1
                answer[first_index, second_index, third_index] = (
                    koszul
                    * _auxiliary_mod(
                        auxiliary_form_parity,
                        auxiliary1,
                        auxiliary2,
                        ground2,
                        auxiliary3,
                        ground3,
                        prime,
                        root_i,
                        root_two,
                    )
                ) % prime
    return answer


def _group_components(first, second, third, prime):
    grouped_first = defaultdict(dict)
    for auxiliary, word, ground, coefficient in first:
        _add(grouped_first[auxiliary], (None, word, ground), coefficient, prime)
    grouped_second = defaultdict(dict)
    for modes, auxiliary_ground, word, physical_ground, coefficient in second:
        _add(
            grouped_second[(modes, auxiliary_ground)],
            (None, word, physical_ground),
            coefficient,
            prime,
        )
    grouped_third = defaultdict(dict)
    for modes, auxiliary_ground, word, physical_ground, coefficient in third:
        _add(
            grouped_third[(modes, auxiliary_ground)],
            (None, word, physical_ground),
            coefficient,
            prime,
        )
    return (
        {key: _pack_vector(value, prime) for key, value in grouped_first.items()},
        {key: _pack_vector(value, prime) for key, value in grouped_second.items()},
        {key: _pack_vector(value, prime) for key, value in grouped_third.items()},
    )


def _entangled_vectors(first, second, third, prime):
    """Keep the auxiliary endpoint as a spectator color on each PBW term."""

    vectors = (defaultdict(dict), defaultdict(dict), defaultdict(dict))
    for auxiliary, word, ground, coefficient in first:
        physical_parity = _parity(word, ground)
        sector = (_level2(word), physical_parity)
        color = (auxiliary, physical_parity)
        _add(
            vectors[0][sector],
            (color, word, ground),
            coefficient,
            prime,
        )
    for modes, auxiliary_ground, word, physical_ground, coefficient in second:
        physical_parity = _parity(word, physical_ground)
        sector = (_level2(word), physical_parity)
        color = (modes, auxiliary_ground, physical_parity)
        _add(
            vectors[1][sector],
            (color, word, physical_ground),
            coefficient,
            prime,
        )
    for modes, auxiliary_ground, word, physical_ground, coefficient in third:
        physical_parity = _parity(word, physical_ground)
        sector = (_level2(word), physical_parity)
        color = (modes, auxiliary_ground, physical_parity)
        _add(
            vectors[2][sector],
            (color, word, physical_ground),
            coefficient,
            prime,
        )
    return tuple(
        {
            sector: _pack_vector(expression, prime)
            for sector, expression in vector.items()
            if expression
        }
        for vector in vectors
    )


def _all_subsets(items):
    items = tuple(items)
    for size in range(len(items) + 1):
        yield from combinations(items, size)


def _ns_components(label, q_value, momentum, root_i, prime):
    label = sp.Rational(label)
    if label == 0:
        return (((), (), 0, 1),)
    effective_momentum = momentum if label > 0 else -momentum
    modes2 = tuple(range(int(4 * abs(label) - 1), 0, -2))
    grouped = defaultdict(list)
    for physical in _all_subsets(modes2):
        physical = tuple(sorted(physical, reverse=True))
        auxiliary = tuple(mode for mode in modes2 if mode not in physical)
        crossings = sum(
            physical_mode > auxiliary_mode
            for physical_mode in physical
            for auxiliary_mode in auxiliary
        )
        coefficient = pow(-root_i, len(physical), prime)
        if crossings % 2:
            coefficient = -coefficient
        grouped[sum(physical)].append((physical, auxiliary, coefficient % prime))

    answer = []
    for level2, targets in grouped.items():
        basis, matrix = ns_transition(
            level2, q_value, effective_momentum % prime, root_i, prime
        )
        unique = tuple(dict.fromkeys(target for target, _, _ in targets))
        rhs = np.concatenate(
            [ns_target_rhs(basis, target) for target in unique], axis=1
        )
        solutions = solve_mod(matrix, rhs, prime)
        columns = {target: index for index, target in enumerate(unique)}
        for target, auxiliary, path_coefficient in targets:
            column = columns[target]
            for row, (virasoro_modes, supercurrent_modes2) in enumerate(basis):
                coefficient = int(solutions[row, column]) * path_coefficient % prime
                if not coefficient:
                    continue
                word = tuple(("L", -2 * int(mode)) for mode in virasoro_modes)
                word += tuple(("G", -int(mode2)) for mode2 in supercurrent_modes2)
                answer.append((auxiliary, word, 0, coefficient))
    return tuple(answer)


def _ramond_components(label, parity, q_value, momentum, root_i, root_two, prime):
    label = sp.Rational(label)
    realization = -1 if label > 0 else 1
    paths = []
    targets_by_block = defaultdict(set)
    for state, coefficient in chi.ramond_fock_paths(label, int(parity)):
        auxiliary_modes, auxiliary_ground, physical_modes, physical_ground = state
        target = tuple(map(int, physical_modes))
        key = (
            sum(target),
            (len(target) + int(physical_ground)) % 2,
        )
        coefficient_mod = _sympy_mod(coefficient, prime, root_i, root_two)
        record = (
            tuple(map(int, auxiliary_modes)),
            int(auxiliary_ground),
            target,
            int(physical_ground),
            coefficient_mod,
        )
        paths.append((key, record))
        targets_by_block[key].add(target)

    solved = {}
    for (level, block_parity), targets in targets_by_block.items():
        basis, matrix = ramond_transition_sector(
            level,
            block_parity,
            realization,
            q_value,
            momentum,
            root_i,
            root_two,
            prime,
        )
        targets = tuple(sorted(targets))
        target_records = tuple(
            (target, ground)
            for target in targets
            for ground in (0, 1)
            if (len(target) + ground) % 2 == block_parity
            and ((), tuple(sorted(target, reverse=True)), ground) in basis
        )
        rhs = np.concatenate(
            [
                ramond_target_rhs(basis, target, ground)
                for target, ground in target_records
            ],
            axis=1,
        )
        solutions = solve_mod(matrix, rhs, prime)
        for column, (target, ground) in enumerate(target_records):
            solved[(level, block_parity, target, ground)] = (
                basis,
                solutions[:, column],
            )

    eighth_conversion = _sympy_mod(-chi.EIGHTH_MINUS, prime, root_i, root_two)
    answer = []
    for key, record in paths:
        auxiliary_modes, auxiliary_ground, target, physical_ground, path_coefficient = record
        basis, solution = solved[key + (target, physical_ground)]
        for row, (virasoro_modes, supercurrent_modes, ground) in enumerate(basis):
            coefficient = int(solution[row]) * path_coefficient % prime
            if ground == 1:
                coefficient = coefficient * eighth_conversion % prime
            if not coefficient:
                continue
            word = tuple(("L", -2 * int(mode)) for mode in virasoro_modes)
            word += tuple(("G", -2 * int(mode)) for mode in supercurrent_modes)
            answer.append(
                (auxiliary_modes, auxiliary_ground, word, int(ground), coefficient)
            )
    return tuple(answer)


@lru_cache(None)
def _canonicalize_virasoro_mod(word, prime):
    """Canonicalize a negative Virasoro word without dropping brackets."""

    word = tuple(int(mode) for mode in word)
    for position in range(len(word) - 1):
        first, second = word[position : position + 2]
        if first >= second:
            continue
        out = {}
        exchanged = word[:position] + (second, first) + word[position + 2 :]
        for canonical, coefficient in _canonicalize_virasoro_mod(
            exchanged, prime
        ).items():
            _add(out, canonical, coefficient, prime)
        bracket = word[:position] + (first + second,) + word[position + 2 :]
        for canonical, coefficient in _canonicalize_virasoro_mod(
            bracket, prime
        ).items():
            _add(out, canonical, (second - first) * coefficient, prime)
        return out
    return {word: 1}


class ModularAuxiliaryVirasoro:
    """The corrected Ising Virasoro sewing functional over ``GF(prime)``.

    A single instance is shared by every Fock endpoint with the same four
    primary labels.  This is important: constructing one symbolic Ward
    evaluator per endpoint triple repeats the same descendant calculation
    thousands of times at the high benchmark.
    """

    def __init__(
        self,
        first_primary,
        form_parity,
        second_primary,
        third_primary,
        prime,
        root_i,
        root_two,
    ):
        self.first_primary = int(first_primary)
        self.form_parity = int(form_parity)
        self.second_primary = int(second_primary)
        self.third_primary = int(third_primary)
        self.prime = int(prime)
        self.root_i = int(root_i) % prime
        self.root_two = int(root_two) % prime
        self.weights = (
            self.first_primary * pow(2, -1, prime) % prime,
            pow(16, -1, prime),
            pow(16, -1, prime),
        )
        self.central_charge = pow(2, -1, prime)
        self.inv12 = pow(12, -1, prime)

    def base(self):
        # Gamma_f has nonzero entries only when row xor column = f.
        if self.first_primary == 0:
            parity = self.form_parity
            phase = 1
        else:
            parity = 1 - self.form_parity
            phase = -self.root_i * pow(self.root_two, -1, self.prime)
        if self.second_primary ^ self.third_primary != parity:
            return 0
        if parity == 0:
            matrix_entry = 1 if self.second_primary == 0 else -1
        else:
            matrix_entry = 1 if self.second_primary == 0 else -1
        return phase * matrix_entry % self.prime

    @lru_cache(None)
    def act(self, slot, mode, word):
        mode = int(mode)
        word = tuple(int(value) for value in word)
        pmod = self.prime
        if mode < 0:
            return _canonicalize_virasoro_mod((-mode,) + word, pmod)
        if not word:
            if mode == 0:
                return {(): self.weights[slot]}
            return {}
        first, rest = word[0], word[1:]
        out = {}
        for reduced, coefficient in self.act(slot, mode, rest).items():
            for canonical, ordering in _canonicalize_virasoro_mod(
                (first,) + reduced, pmod
            ).items():
                _add(out, canonical, coefficient * ordering, pmod)
        bracket_coefficient = mode + first
        replacement = mode - first
        if replacement < 0:
            for reduced, coefficient in _canonicalize_virasoro_mod(
                (-replacement,) + rest, pmod
            ).items():
                _add(out, reduced, bracket_coefficient * coefficient, pmod)
        elif replacement == 0:
            _add(
                out,
                rest,
                bracket_coefficient
                * (self.weights[slot] + sum(rest)),
                pmod,
            )
        else:
            for reduced, coefficient in self.act(
                slot, replacement, rest
            ).items():
                _add(out, reduced, bracket_coefficient * coefficient, pmod)
        if mode == first:
            _add(
                out,
                rest,
                self.central_charge * (mode**3 - mode) * self.inv12,
                pmod,
            )
        return out

    @lru_cache(None)
    def value(self, word1, word2, word3):
        pmod = self.prime
        if word2:
            n, rest2 = int(word2[0]), word2[1:]
            if n == 1:
                exponent = (
                    self.weights[0]
                    + sum(word1)
                    - self.weights[1]
                    - sum(rest2)
                    - self.weights[2]
                    - sum(word3)
                ) % pmod
                return exponent * self.value(word1, rest2, word3) % pmod
            answer = 0
            maximum = max(sum(word1) - n, sum(word3) + 1, 0)
            for lower in range(maximum + 1):
                ward = comb(n - 2 + lower, n - 2) % pmod
                for reduced1, coefficient in self.act(
                    0, n + lower, word1
                ).items():
                    answer += (
                        ward
                        * coefficient
                        * self.value(reduced1, rest2, word3)
                    )
                for reduced3, coefficient in self.act(
                    2, lower - 1, word3
                ).items():
                    answer += (
                        ward
                        * (-1) ** n
                        * coefficient
                        * self.value(word1, rest2, reduced3)
                    )
            return answer % pmod
        if word1:
            n, rest1 = int(word1[0]), word1[1:]
            answer = sum(
                coefficient * self.value(rest1, (), reduced3)
                for reduced3, coefficient in self.act(2, n, word3).items()
            )
            for middle_mode in range(-1, n + 1):
                ward = comb(n + 1, middle_mode + 1) % pmod
                for reduced2, coefficient in self.act(
                    1, middle_mode, ()
                ).items():
                    answer += (
                        ward
                        * coefficient
                        * self.value(rest1, reduced2, word3)
                    )
            return answer % pmod
        if word3:
            n, rest3 = int(word3[0]), word3[1:]
            coefficient = (
                self.weights[2]
                + sum(rest3)
                + n * self.weights[1]
                - self.weights[0]
            ) % pmod
            return coefficient * self.value((), (), rest3) % pmod
        return self.base()


@lru_cache(None)
def _auxiliary_vector_mod(
    sector, modes, ground, prime, root_i, root_two
):
    primary, basis, coefficients = grid.auxiliary_to_virasoro(
        sector,
        tuple(sp.Rational(mode) for mode in modes),
        int(ground),
    )
    vector = tuple(
        (tuple(word), _sympy_mod(coefficient, prime, root_i, root_two))
        for word, coefficient in zip(basis, coefficients)
        if coefficient != 0
    )
    return int(primary), vector


@lru_cache(None)
def _modular_auxiliary_evaluator(
    first_primary,
    form_parity,
    second_primary,
    third_primary,
    prime,
    root_i,
    root_two,
):
    return ModularAuxiliaryVirasoro(
        first_primary,
        form_parity,
        second_primary,
        third_primary,
        prime,
        root_i,
        root_two,
    )


@lru_cache(None)
def _auxiliary_mod(
    form_parity,
    first_modes2,
    second_modes,
    second_ground,
    third_modes,
    third_ground,
    prime,
    root_i,
    root_two,
):
    # This is exactly the corrected Virasoro-primary sewing frame, evaluated
    # natively in the prime field.  It is intentionally not the distinct
    # global-coordinate Majorana Pfaffian.
    first_primary, first = _auxiliary_vector_mod(
        "NS",
        tuple(sp.Rational(mode2, 2) for mode2 in first_modes2),
        0,
        prime,
        root_i,
        root_two,
    )
    second_primary, second = _auxiliary_vector_mod(
        "R",
        second_modes,
        second_ground,
        prime,
        root_i,
        root_two,
    )
    third_primary, third = _auxiliary_vector_mod(
        "R",
        third_modes,
        third_ground,
        prime,
        root_i,
        root_two,
    )
    evaluator = _modular_auxiliary_evaluator(
        first_primary,
        int(form_parity),
        second_primary,
        third_primary,
        prime,
        root_i,
        root_two,
    )
    answer = 0
    for word1, coefficient1 in first:
        for word2, coefficient2 in second:
            for word3, coefficient3 in third:
                answer += (
                    coefficient1
                    * coefficient2
                    * coefficient3
                    * evaluator.value(word1, word2, word3)
                )
    phase = grid.auxiliary_virasoro_transport_phase(
        tuple(sp.Rational(mode2, 2) for mode2 in first_modes2),
        second_modes,
    )
    return phase * answer % prime


def raw_three_point_mod(
    n1,
    n2,
    n3,
    epsilon2,
    epsilon3,
    form_parity,
    eta,
    sample,
    prime=1_000_033,
    return_profile=False,
    backend="dense-middle",
):
    """Return the exact residue of one raw branching three-point function."""

    began = time.perf_counter()
    root_i, root_two = roots(prime)
    b, p1, p2, p3 = map(sp.Rational, sample)
    q = b + 1 / b
    q_value = rational_mod(q, prime)
    momenta = tuple(rational_mod(value, prime) for value in (0, p2, p3))
    weights = (
        (q**2 / 4 - p1**2) / 2,
        sp.Rational(1, 16) + q**2 / 8 - p2**2 / 2,
        sp.Rational(1, 16) + q**2 / 8 - p3**2 / 2,
    )
    central_charge = sp.Rational(3, 2) + 3 * q**2
    evaluator = ModularNRR(
        form_parity,
        eta,
        tuple(rational_mod(value, prime) for value in weights),
        momenta,
        rational_mod(central_charge, prime),
        root_i,
        prime,
    )

    first = _ns_components(
        n1, q_value, rational_mod(p1, prime), root_i, prime
    )
    second = _ramond_components(
        n2, epsilon2, q_value, rational_mod(p2, prime), root_i, root_two, prime
    )
    third = _ramond_components(
        n3, epsilon3, q_value, rational_mod(p3, prime), root_i, root_two, prime
    )
    built = time.perf_counter() - began
    auxiliary_form_parity = (
        int(2 * sp.Rational(n1)) + int(epsilon2) + int(epsilon3) - int(form_parity)
    ) % 2

    vectors1, vectors2, vectors3 = _entangled_vectors(
        first, second, third, prime
    )
    colors = tuple(
        tuple(
            sorted(
                {item[0] for vector in vectors.values() for item in vector},
                key=repr,
            )
        )
        for vectors in (vectors1, vectors2, vectors3)
    )
    pairing_counter = [0]
    if backend == "dense-middle":
        auxiliary_tensor = _auxiliary_color_tensor(
            colors,
            auxiliary_form_parity,
            prime,
            root_i,
            root_two,
        )
        pairing_counter[0] = int(np.count_nonzero(auxiliary_tensor))
        dense_ward = DenseMiddleWardContraction(evaluator)
        answer = 0
        sector_triples = 0
        for sector1, vector1 in vectors1.items():
            for sector2, vector2 in vectors2.items():
                for sector3, vector3 in vectors3.items():
                    # Fermion parity is preserved by the trilinear form.
                    if (
                        sector1[1] + sector2[1] + sector3[1]
                    ) % 2 != int(form_parity):
                        continue
                    sector_triples += 1
                    answer += dense_ward.contract_sector(
                        sector1,
                        vector1,
                        sector2,
                        vector2,
                        sector3,
                        vector3,
                        colors,
                        auxiliary_tensor,
                    )
        answer %= prime
        vector_cache_misses = 0
        dense_block_misses = dense_ward.block.cache_info().misses
        dense_block_entries = dense_ward.block_entries
        dense_base_entries = dense_ward.base_entries
    elif backend == "vector":
        @lru_cache(None)
        def ground_pairing(
            color1, color2, color3, ground1, ground2, ground3
        ):
            auxiliary1, physical_parity1 = color1
            auxiliary2, ground_a2, physical_parity2 = color2
            auxiliary3, ground_a3, _ = color3
            auxiliary_parity2 = (len(auxiliary2) + ground_a2) % 2
            auxiliary_parity3 = (len(auxiliary3) + ground_a3) % 2
            auxiliary_value = _auxiliary_mod(
                auxiliary_form_parity,
                auxiliary1,
                auxiliary2,
                ground_a2,
                auxiliary3,
                ground_a3,
                prime,
                root_i,
                root_two,
            )
            if not auxiliary_value:
                return 0
            physical_ground_value = evaluator.raw_value(
                (), (), ground2, (), ground3
            )
            if not physical_ground_value:
                return 0
            pairing_counter[0] += 1
            koszul = -1 if (
                physical_parity1 * (auxiliary_parity2 + auxiliary_parity3)
                + physical_parity2 * auxiliary_parity3
            ) % 2 else 1
            return koszul * auxiliary_value * physical_ground_value % prime

        vector_ward = VectorWardContraction(evaluator, ground_pairing)
        answer = 0
        for vector1 in vectors1.values():
            for vector2 in vectors2.values():
                for vector3 in vectors3.values():
                    answer += vector_ward.value(vector1, vector2, vector3)
        answer %= prime
        vector_cache_misses = vector_ward.value.cache_info().misses
        dense_block_misses = 0
        dense_block_entries = 0
        dense_base_entries = 0
        sector_triples = len(vectors1) * len(vectors2) * len(vectors3)
    else:
        raise ValueError(f"unknown contraction backend: {backend}")
    contracted = time.perf_counter() - began - built
    answer %= prime
    if not return_profile:
        return auxiliary_form_parity, answer
    profile = {
        "components": (len(first), len(second), len(third)),
        "candidate_triples": len(first) * len(second) * len(third),
        "auxiliary_path_triples": len(
            {item[0] for vector in vectors1.values() for item in vector}
        ) * len(
            {item[0] for vector in vectors2.values() for item in vector}
        ) * len(
            {item[0] for vector in vectors3.values() for item in vector}
        ),
        "nonzero_ground_pairings": pairing_counter[0],
        "contraction_backend": backend,
        "contracted_sector_triples": sector_triples,
        "vector_ward_cache_misses": vector_cache_misses,
        "dense_block_cache_misses": dense_block_misses,
        "dense_block_entries": dense_block_entries,
        "dense_base_entries": dense_base_entries,
        "build_seconds": built,
        "contract_seconds": contracted,
        "total_seconds": time.perf_counter() - began,
    }
    return auxiliary_form_parity, answer, profile


def _expected_mod(labels, discrete, sample, prime):
    root_i, root_two = roots(prime)
    expected = grid.enlarged_raw_three_point(
        *labels, *discrete, *sample
    )[1]
    return _sympy_mod(expected, prime, root_i, root_two)


def low_audit(prime=1_000_033):
    cases = (
        ((0, sp.Rational(1, 4), sp.Rational(1, 4)), (0, 0, 0, 1)),
        ((sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(3, 4)), (1, 0, 1, -1)),
        ((1, sp.Rational(3, 4), sp.Rational(3, 4)), (0, 1, 0, -1)),
    )
    for labels, discrete in cases:
        calculated = raw_three_point_mod(
            *labels, *discrete, grid.SAMPLES[0], prime=prime
        )[1]
        expected = _expected_mod(labels, discrete, grid.SAMPLES[0], prime)
        if calculated != expected:
            raise AssertionError((labels, discrete, calculated, expected))
    print(f"modular three-point audit: {len(cases)} exact residues agree")


def benchmark(prime=1_000_033):
    labels = (sp.Integer(2), sp.Rational(7, 4), sp.Rational(7, 4))
    discrete = (0, 0, 0, 1)
    _, residue, profile = raw_three_point_mod(
        *labels,
        *discrete,
        grid.SAMPLES[0],
        prime=prime,
        return_profile=True,
    )
    print("benchmark labels", labels, "discrete", discrete, "residue", residue)
    for key, value in profile.items():
        print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--prime", type=int, default=1_000_033)
    arguments = parser.parse_args()
    low_audit(arguments.prime)
    if arguments.benchmark:
        benchmark(arguments.prime)


if __name__ == "__main__":
    main()
