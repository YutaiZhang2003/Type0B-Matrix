#!/usr/bin/env python3
r"""KLT period data for the first heterotic five-point wall resonance.

This module converts every holomorphic/antiholomorphic Wick term from
``spin23_linear_dilaton_fivepoint`` into powers of boundary divisors on an
ordered real three-simplex.  It also supplies the exact six-point sine
momentum kernel.  Numerical meromorphic continuation of the resulting real
periods is kept in the generic twisted-period backend.

Six KLT labels are used:

``1=out1, 2=out2, 3=out3, 4=soft singlet/screen,``
``5=out4, 6=incoming``.

The original gauge is ``z1=0, z5=1, z6=infinity``.  Positive cycles put the
three moving labels between zero and one.  Negative cycles are compactified
by ``Q=-Y/(1-Y)`` before being written as simplex divisors.  No picture or
ghost factors are silently changed by this compactification.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from itertools import permutations
import math
from typing import Iterable, Literal, Mapping, Sequence

from spin23_linear_dilaton_fivepoint import (
    FermionMatchingTerm,
    SCREEN_LABEL,
    liouville_charge,
    regulated_fermion_matching_terms,
)
from spin23_sphere_fivepoint_amplitude import SphereFivePointState


Chirality = Literal["holomorphic", "antiholomorphic"]
GapDivisor = tuple[int, ...]
KLTWord = tuple[int, int, int]
KLT_WORDS: tuple[KLTWord, ...] = tuple(permutations((2, 3, 4)))  # type: ignore[assignment]

# The nine finite-puncture exponents are sufficient to move every endpoint
# hyperplane encountered by the ordered periods.  These fixed, generic
# coefficients define only a direction of analytic continuation; the
# regulator is removed after the complete KLT contraction.
DEFAULT_TWIST_REGULATOR_DIRECTION: dict[tuple[int, int], complex] = {
    pair: complex(0.137 * (index + 1), 0.071 * (index + 2))
    for index, pair in enumerate(
        (
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
            (4, 5),
        )
    )
}

_WORLD_TO_KLT = {
    1: 1,
    2: 2,
    3: 3,
    4: 5,
    5: 6,
    SCREEN_LABEL: 4,
}


def _add_power(
    powers: dict[GapDivisor, complex],
    divisor: Iterable[int],
    power: complex,
) -> None:
    key = tuple(sorted(int(index) for index in divisor))
    if not key:
        raise ValueError("a collision divisor must contain at least one gap")
    powers[key] = powers.get(key, 0.0j) + complex(power)
    if abs(powers[key]) < 1.0e-15:
        del powers[key]


def _between(first_position: int, second_position: int) -> GapDivisor:
    low, high = sorted((int(first_position), int(second_position)))
    if low == high:
        raise ValueError("two distinct punctures cannot occupy one position")
    return tuple(range(low, high))


@dataclass(frozen=True)
class OrderedSimplexTerm:
    """One coefficient times powers of sums of the four simplex gaps."""

    coefficient: complex
    powers: tuple[tuple[GapDivisor, complex], ...]

    def evaluate_gaps(self, gaps: Sequence[float]) -> complex:
        normalized = tuple(float(value) for value in gaps)
        if len(normalized) != 4 or any(value <= 0 for value in normalized):
            raise ValueError("four positive ordered-simplex gaps are required")
        if abs(sum(normalized) - 1.0) > 1.0e-10:
            raise ValueError("simplex gaps must sum to one")
        result = complex(self.coefficient)
        for divisor, power in self.powers:
            distance = sum(normalized[index] for index in divisor)
            result *= cmath.exp(complex(power) * math.log(distance))
        return result

    @classmethod
    def from_mapping(
        cls,
        coefficient: complex,
        powers: Mapping[GapDivisor, complex],
    ) -> "OrderedSimplexTerm":
        return cls(
            coefficient=complex(coefficient),
            powers=tuple(
                sorted(
                    (
                        (tuple(divisor), complex(power))
                        for divisor, power in powers.items()
                    ),
                    key=lambda entry: entry[0],
                )
            ),
        )


def sixpoint_twist_exponents(
    states: Sequence[SphereFivePointState],
    soft_momentum: complex,
) -> dict[tuple[int, int], complex]:
    r"""Return ``kappa_ij=-(alpha_i alpha_j+k_i k_j)`` for six labels."""

    normalized = tuple(states)
    if len(normalized) != 5:
        raise ValueError("five physical states are required")
    soft = complex(soft_momentum)
    labels = {
        1: normalized[0],
        2: normalized[1],
        3: normalized[2],
        5: normalized[3],
        6: normalized[4],
    }
    charges = {
        label: liouville_charge(state.liouville_momentum)
        for label, state in labels.items()
    }
    times = {
        label: complex(state.time_momentum) for label, state in labels.items()
    }
    charges[4] = liouville_charge(soft)
    times[4] = soft
    result: dict[tuple[int, int], complex] = {}
    for first in range(1, 7):
        for second in range(first + 1, 7):
            result[(first, second)] = -(
                charges[first] * charges[second]
                + times[first] * times[second]
            )
    return result


def regulated_twist_exponents(
    states: Sequence[SphereFivePointState],
    soft_momentum: complex,
    regulator: complex,
    *,
    direction: Mapping[tuple[int, int], complex] = DEFAULT_TWIST_REGULATOR_DIRECTION,
) -> dict[tuple[int, int], complex]:
    r"""Return one coherent off-shell deformation of the KLT twist.

    Some ordered periods lie on endpoint poles even when the soft momentum is
    nonzero.  They are not to be assigned separate finite parts: their poles
    cancel against the sine kernel and other chambers.  ``regulator`` shifts
    the same base exponents in both chiral period systems.  The caller must
    remove it only after forming the complete KLT bilinear.
    """

    result = sixpoint_twist_exponents(states, soft_momentum)
    epsilon = complex(regulator)
    for raw_pair, coefficient in direction.items():
        pair = tuple(sorted((int(raw_pair[0]), int(raw_pair[1]))))
        if pair not in result:
            raise ValueError(f"twist-regulator pair {pair!r} is not a six-point pair")
        result[pair] += epsilon * complex(coefficient)
    return result


def _kappa(
    exponents: Mapping[tuple[int, int], complex],
    first: int,
    second: int,
) -> complex:
    return complex(exponents[tuple(sorted((int(first), int(second))))])


def klt_sine_kernel(
    exponents: Mapping[tuple[int, int], complex],
) -> tuple[tuple[complex, ...], ...]:
    r"""Return ``S[gamma|sigma]_1`` in the six-word basis.

    Rows are ``gamma`` and columns are ``sigma`` in :data:`KLT_WORDS` order.
    The raw-area KLT contraction is ``-L^T S^T R`` (equivalently the explicit
    double sum ``-sum_sigma,gamma L_sigma S[gamma|sigma] R_gamma``).
    """

    rows: list[tuple[complex, ...]] = []
    for gamma in KLT_WORDS:
        row: list[complex] = []
        for sigma in KLT_WORDS:
            positions = {label: index for index, label in enumerate(sigma)}
            value = 1.0 + 0.0j
            for index, label in enumerate(gamma):
                argument = _kappa(exponents, 1, label)
                for later in gamma[index + 1 :]:
                    if positions[later] < positions[label]:
                        argument += _kappa(exponents, label, later)
                value *= cmath.sin(math.pi * argument)
            row.append(value)
        rows.append(tuple(row))
    return tuple(rows)


def _mapped_matching_terms(
    states: Sequence[SphereFivePointState],
    soft_momentum: complex,
    chirality: Chirality,
) -> tuple[FermionMatchingTerm, ...]:
    holomorphic, antiholomorphic = regulated_fermion_matching_terms(
        states, soft_momentum
    )
    source = holomorphic if chirality == "holomorphic" else antiholomorphic
    return tuple(
        FermionMatchingTerm(
            coefficient=term.coefficient,
            edges=tuple(
                (_WORLD_TO_KLT[first], _WORLD_TO_KLT[second])
                for first, second in term.edges
            ),
        )
        for term in source
    )


def _positive_base_powers(
    exponents: Mapping[tuple[int, int], complex],
    word: KLTWord,
) -> tuple[dict[GapDivisor, complex], dict[int, int]]:
    finite_order = (1,) + tuple(word) + (5,)
    positions = {label: index for index, label in enumerate(finite_order)}
    powers: dict[GapDivisor, complex] = {}
    for offset, first in enumerate(finite_order):
        for second in finite_order[offset + 1 :]:
            _add_power(
                powers,
                _between(positions[first], positions[second]),
                _kappa(exponents, first, second),
            )
    return powers, positions


def _positive_edge(
    powers: dict[GapDivisor, complex],
    coefficient: complex,
    positions: Mapping[int, int],
    edge: tuple[int, int],
) -> complex:
    first, second = edge
    if first == 6 or second == 6:
        # lim z_inf/(z_first-z_second) is +1 when infinity is first, -1
        # when it is second.
        return coefficient if first == 6 else -coefficient
    if positions[first] < positions[second]:
        coefficient = -coefficient
    _add_power(
        powers,
        _between(positions[first], positions[second]),
        -1,
    )
    return coefficient


def positive_period_terms(
    states: Sequence[SphereFivePointState],
    soft_momentum: complex,
    word: KLTWord,
    chirality: Chirality,
    *,
    exponents: Mapping[tuple[int, int], complex] | None = None,
) -> tuple[OrderedSimplexTerm, ...]:
    """Return divisor terms for ``0<X1<X2<X3<1`` on ``B_word``."""

    if tuple(word) not in KLT_WORDS:
        raise ValueError("word must be a permutation of KLT labels (2,3,4)")
    twist = (
        sixpoint_twist_exponents(states, soft_momentum)
        if exponents is None
        else exponents
    )
    common, positions = _positive_base_powers(twist, word)
    result: list[OrderedSimplexTerm] = []
    for matching in _mapped_matching_terms(states, soft_momentum, chirality):
        powers = dict(common)
        coefficient = complex(matching.coefficient)
        for edge in matching.edges:
            coefficient = _positive_edge(powers, coefficient, positions, edge)
        result.append(OrderedSimplexTerm.from_mapping(coefficient, powers))
    return tuple(result)


def positive_scalar_period_term(
    exponents: Mapping[tuple[int, int], complex],
    word: KLTWord,
) -> OrderedSimplexTerm:
    """Return the positive-cycle term for a scalar Koba--Nielsen factor.

    This small public helper is useful for end-to-end KLT normalization tests
    which should not depend on heterotic fermion Wick contractions.
    """

    if tuple(word) not in KLT_WORDS:
        raise ValueError("word must be a permutation of KLT labels (2,3,4)")
    powers, _ = _positive_base_powers(exponents, word)
    return OrderedSimplexTerm.from_mapping(1.0 + 0.0j, powers)


def _negative_geometry(
    word: KLTWord,
) -> tuple[dict[int, int], dict[int, int]]:
    # Original negative positions obey gamma1<gamma2<gamma3<0.  After
    # Q=-Y/(1-Y), their positive-simplex order is reversed.
    q_order = (1, word[2], word[1], word[0], 0)
    q_positions = {label: index for index, label in enumerate(q_order[:-1])}
    y_ranks = {word[0]: -3, word[1]: -2, word[2]: -1, 1: 0, 5: 1}
    return q_positions, y_ranks


def _q_zero_divisor(q_positions: Mapping[int, int], label: int) -> GapDivisor:
    return tuple(range(0, q_positions[label]))


def _q_one_divisor(q_positions: Mapping[int, int], label: int) -> GapDivisor:
    return tuple(range(q_positions[label], 4))


def _negative_base_powers(
    exponents: Mapping[tuple[int, int], complex],
    word: KLTWord,
) -> tuple[dict[GapDivisor, complex], dict[int, int], dict[int, int]]:
    powers: dict[GapDivisor, complex] = {}
    q_positions, y_ranks = _negative_geometry(word)
    for label in word:
        zero = _q_zero_divisor(q_positions, label)
        one = _q_one_divisor(q_positions, label)
        _add_power(powers, zero, _kappa(exponents, 1, label))
        _add_power(
            powers,
            one,
            -_kappa(exponents, 1, label)
            - _kappa(exponents, label, 5)
            - 2,
        )
    for first_index, first in enumerate(word):
        for second in word[first_index + 1 :]:
            _add_power(
                powers,
                _between(q_positions[first], q_positions[second]),
                _kappa(exponents, first, second),
            )
            _add_power(
                powers,
                _q_one_divisor(q_positions, first),
                -_kappa(exponents, first, second),
            )
            _add_power(
                powers,
                _q_one_divisor(q_positions, second),
                -_kappa(exponents, first, second),
            )
    return powers, q_positions, y_ranks


def _negative_edge(
    powers: dict[GapDivisor, complex],
    coefficient: complex,
    q_positions: Mapping[int, int],
    y_ranks: Mapping[int, int],
    edge: tuple[int, int],
) -> complex:
    first, second = edge
    if first == 6 or second == 6:
        return coefficient if first == 6 else -coefficient
    if y_ranks[first] < y_ranks[second]:
        coefficient = -coefficient
    moving = set(q_positions) - {1}
    if first in moving and second in moving:
        _add_power(
            powers,
            _between(q_positions[first], q_positions[second]),
            -1,
        )
        _add_power(powers, _q_one_divisor(q_positions, first), 1)
        _add_power(powers, _q_one_divisor(q_positions, second), 1)
    elif first in moving or second in moving:
        label = first if first in moving else second
        fixed = second if first in moving else first
        if fixed == 1:
            _add_power(powers, _q_zero_divisor(q_positions, label), -1)
            _add_power(powers, _q_one_divisor(q_positions, label), 1)
        elif fixed == 5:
            _add_power(powers, _q_one_divisor(q_positions, label), 1)
        else:
            raise AssertionError("unexpected fixed label on a negative cycle")
    # The fixed 0--1 distance is one and needs no divisor.
    return coefficient


def negative_period_terms(
    states: Sequence[SphereFivePointState],
    soft_momentum: complex,
    word: KLTWord,
    chirality: Chirality,
    *,
    exponents: Mapping[tuple[int, int], complex] | None = None,
) -> tuple[OrderedSimplexTerm, ...]:
    """Return compactified divisor terms for the negative cycle ``C_word``."""

    if tuple(word) not in KLT_WORDS:
        raise ValueError("word must be a permutation of KLT labels (2,3,4)")
    twist = (
        sixpoint_twist_exponents(states, soft_momentum)
        if exponents is None
        else exponents
    )
    common, q_positions, y_ranks = _negative_base_powers(twist, word)
    result: list[OrderedSimplexTerm] = []
    for matching in _mapped_matching_terms(states, soft_momentum, chirality):
        powers = dict(common)
        coefficient = complex(matching.coefficient)
        for edge in matching.edges:
            coefficient = _negative_edge(
                powers, coefficient, q_positions, y_ranks, edge
            )
        result.append(OrderedSimplexTerm.from_mapping(coefficient, powers))
    return tuple(result)


def negative_scalar_period_term(
    exponents: Mapping[tuple[int, int], complex],
    word: KLTWord,
) -> OrderedSimplexTerm:
    """Return the compactified negative-cycle scalar Koba--Nielsen term."""

    if tuple(word) not in KLT_WORDS:
        raise ValueError("word must be a permutation of KLT labels (2,3,4)")
    powers, _, _ = _negative_base_powers(exponents, word)
    return OrderedSimplexTerm.from_mapping(1.0 + 0.0j, powers)


def contract_klt_periods(
    left_periods: Sequence[complex],
    right_periods: Sequence[complex],
    sine_kernel: Sequence[Sequence[complex]],
) -> complex:
    """Apply the raw-area six-point KLT contraction, including its minus sign."""

    left = tuple(map(complex, left_periods))
    right = tuple(map(complex, right_periods))
    kernel = tuple(tuple(map(complex, row)) for row in sine_kernel)
    if len(left) != 6 or len(right) != 6 or len(kernel) != 6 or any(
        len(row) != 6 for row in kernel
    ):
        raise ValueError("six left periods, six right periods, and a 6x6 kernel are required")
    return -sum(
        left[sigma] * kernel[gamma][sigma] * right[gamma]
        for sigma in range(6)
        for gamma in range(6)
    )


__all__ = [
    "Chirality",
    "DEFAULT_TWIST_REGULATOR_DIRECTION",
    "GapDivisor",
    "KLTWord",
    "KLT_WORDS",
    "OrderedSimplexTerm",
    "contract_klt_periods",
    "klt_sine_kernel",
    "negative_period_terms",
    "negative_scalar_period_term",
    "positive_period_terms",
    "positive_scalar_period_term",
    "regulated_twist_exponents",
    "sixpoint_twist_exponents",
]
