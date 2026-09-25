#!/usr/bin/env python3
r"""KLT pilot for the triple-external-contact part of the ``s=3`` wall.

When all three finite picture-zero legs are singlets, one charge-three
component replaces each of them by the auxiliary-field term in ``W_a``.
There are then no integrated wall screens.  The remaining integral has only
the two ordinary five-point moduli and is the cheapest nontrivial ``s=3``
component.  This module evaluates it by a two-moving-point KLT pairing and a
six-sector meromorphic Hepp continuation.

The result is a component, not the full residue.  It must eventually be
combined with the other eleven all-singlet components using one fused
regulator before the regulator is removed.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from itertools import permutations
import math
from typing import Mapping, Sequence

import sympy as sp

from spin23_s3_soft_resonance import (
    S3Component,
    liouville_charge,
    s3_component_action_coefficient,
    soft_s3_family,
)
from spin23_fivepoint_resonance_channels import FivePointChannelProjection
from spin23_twisted_periods import meromorphic_cube_integral, sine_momentum_kernel


Word = tuple[int, int]
WORDS: tuple[Word, Word] = ((2, 3), (3, 2))
GapDivisor = tuple[int, ...]


@dataclass(frozen=True)
class TriangleTerm:
    """One coefficient times powers of sums of three ordered gaps."""

    coefficient: complex
    powers: tuple[tuple[GapDivisor, complex], ...]


@dataclass(frozen=True)
class TrianglePeriod:
    value: complex
    absolute_error: float
    sector_values: tuple[complex, ...]
    sector_errors: tuple[float, ...]
    converged: bool


@dataclass(frozen=True)
class TripleContactKLTResult:
    raw_area_value: complex
    component_value: complex
    action_coefficient: complex
    picture_factor: float
    left_periods: tuple[complex, complex]
    right_periods: tuple[complex, complex]
    absolute_error_bound: float
    periods_converged: bool


def _add_power(
    powers: dict[GapDivisor, complex], divisor: Sequence[int], value: complex
) -> None:
    key = tuple(sorted(int(index) for index in divisor))
    powers[key] = powers.get(key, 0.0j) + complex(value)
    if abs(powers[key]) < 1.0e-14:
        del powers[key]


def _between(first: int, second: int) -> GapDivisor:
    low, high = sorted((int(first), int(second)))
    return tuple(range(low, high))


def _kappa(
    exponents: Mapping[tuple[int, int], complex], first: int, second: int
) -> complex:
    return complex(exponents[tuple(sorted((first, second)))])


def positive_triangle_term(
    exponents: Mapping[tuple[int, int], complex],
    word: Word,
    *,
    coefficient: complex = 1.0,
) -> TriangleTerm:
    """Return the scalar period on ``0<X1<X2<1``."""

    if tuple(word) not in WORDS:
        raise ValueError("word must permute (2,3)")
    order = (1,) + tuple(word) + (4,)
    positions = {label: index for index, label in enumerate(order)}
    powers: dict[GapDivisor, complex] = {}
    for index, first in enumerate(order):
        for second in order[index + 1 :]:
            _add_power(
                powers,
                _between(positions[first], positions[second]),
                _kappa(exponents, first, second),
            )
    return TriangleTerm(
        complex(coefficient), tuple(sorted(powers.items(), key=lambda item: item[0]))
    )


def negative_triangle_term(
    exponents: Mapping[tuple[int, int], complex],
    word: Word,
    *,
    coefficient: complex = 1.0,
) -> TriangleTerm:
    """Return the compactified scalar period for ``Y1<Y2<0``."""

    if tuple(word) not in WORDS:
        raise ValueError("word must permute (2,3)")
    q_order = (1, word[1], word[0], 0)
    positions = {label: index for index, label in enumerate(q_order[:-1])}

    def zero(label: int) -> GapDivisor:
        return tuple(range(0, positions[label]))

    def one(label: int) -> GapDivisor:
        return tuple(range(positions[label], 3))

    powers: dict[GapDivisor, complex] = {}
    for label in word:
        _add_power(powers, zero(label), _kappa(exponents, 1, label))
        _add_power(
            powers,
            one(label),
            -_kappa(exponents, 1, label)
            - _kappa(exponents, label, 4)
            - 2,
        )
    first, second = word
    pair = _kappa(exponents, first, second)
    _add_power(powers, _between(positions[first], positions[second]), pair)
    _add_power(powers, one(first), -pair)
    _add_power(powers, one(second), -pair)
    return TriangleTerm(
        complex(coefficient), tuple(sorted(powers.items(), key=lambda item: item[0]))
    )


def _hepp_factorization(
    term: TriangleTerm,
    permutation: tuple[int, int, int],
    variables: tuple[sp.Symbol, sp.Symbol],
) -> tuple[tuple[complex, complex], sp.Expr]:
    """Return Mellin parameters and smooth part for one triangle sector."""

    if set(permutation) != {0, 1, 2}:
        raise ValueError("a triangle Hepp sector must permute (0,1,2)")
    u1, u2 = variables
    total = 1 + u1 + u1 * u2
    parameters: list[complex] = [2.0 + 0.0j, 1.0 + 0.0j]
    total_power = sum((power for _divisor, power in term.powers), 0.0j)
    smooth: sp.Expr = sp.sympify(term.coefficient) * total ** (-3 - total_power)
    rank_of = {gap: rank for rank, gap in enumerate(permutation)}
    for divisor, power in term.powers:
        ranks = tuple(sorted(rank_of[index] for index in divisor))
        leading = ranks[0]
        for axis in range(2):
            if leading >= axis + 1:
                parameters[axis] += power
        polynomial = sp.Integer(0)
        for rank in ranks:
            monomial = sp.Integer(1)
            for axis in range(leading, rank):
                monomial *= variables[axis]
            polynomial += monomial
        smooth *= polynomial ** sp.sympify(power)
    return (complex(parameters[0]), complex(parameters[1])), sp.factor(smooth)


def integrate_triangle_term(
    term: TriangleTerm,
    *,
    rtol: float = 2.0e-8,
    atol: float = 2.0e-10,
    rule: str = "gauss-jacobi-10",
    max_subdivisions: int = 10_000,
) -> TrianglePeriod:
    """Meromorphically integrate one ordered-triangle period."""

    variables = sp.symbols("u1 u2", positive=True)
    values: list[complex] = []
    errors: list[float] = []
    converged = True
    for permutation in permutations((0, 1, 2)):
        parameters, smooth = _hepp_factorization(term, permutation, variables)
        result = meromorphic_cube_integral(
            smooth,
            variables,
            parameters,
            rtol=rtol,
            atol=atol / 6.0,
            rule=rule,
            max_subdivisions=max_subdivisions,
        )
        values.append(complex(result.value))
        errors.append(float(result.absolute_error))
        converged = converged and result.converged
    return TrianglePeriod(
        value=sum(values, 0.0j),
        absolute_error=sum(errors),
        sector_values=tuple(values),
        sector_errors=tuple(errors),
        converged=converged,
    )


def scalar_two_moving_klt(
    exponents: Mapping[tuple[int, int], complex],
    *,
    right_coefficient: complex = 1.0,
    rtol: float = 2.0e-8,
    atol: float = 2.0e-10,
    rule: str = "gauss-jacobi-10",
) -> tuple[complex, tuple[TrianglePeriod, ...], tuple[TrianglePeriod, ...]]:
    """Evaluate a scalar two-moving-point raw-area KLT integral."""

    left = tuple(
        integrate_triangle_term(
            positive_triangle_term(exponents, word),
            rtol=rtol,
            atol=atol,
            rule=rule,
        )
        for word in WORDS
    )
    right = tuple(
        integrate_triangle_term(
            negative_triangle_term(
                exponents, word, coefficient=right_coefficient
            ),
            rtol=rtol,
            atol=atol,
            rule=rule,
        )
        for word in WORDS
    )
    # The raw-area orientation is (-1)**m for m moving complex points.  The
    # existing six-point backend has m=3 and hence a minus sign; here m=2.
    value = 0.0j
    for sigma, left_word in enumerate(WORDS):
        for gamma, right_word in enumerate(WORDS):
            value += (
                left[sigma].value
                * complex(
                    sine_momentum_kernel(
                        right_word, left_word, exponents, root=1
                    )
                )
                * right[gamma].value
            )
    return value, left, right


def triple_contact_exponents(
    resonant_outgoing_momenta: Sequence[complex],
    *,
    twist_regulator: complex = 0.0,
) -> tuple[dict[tuple[int, int], complex], complex, tuple]:
    """Return KLT exponents, the remaining anti-fermion factor, and states."""

    projection = FivePointChannelProjection("SSSSS", ())
    family = soft_s3_family(
        projection, resonant_outgoing_momenta, (0.0j, 0.0j, 0.0j)
    )
    out1, out2, out3, out4, incoming = family.physical_states
    states = {1: out1, 2: out2, 3: out3, 4: out4, 5: incoming}
    contact_labels = {2, 3, 4}
    charges = {
        label: liouville_charge(state.liouville_momentum)
        + (1.0 if label in contact_labels else 0.0)
        for label, state in states.items()
    }
    times = {label: complex(state.time_momentum) for label, state in states.items()}
    exponents: dict[tuple[int, int], complex] = {}
    direction_values = (0.173 + 0.091j, 0.257 + 0.137j, 0.311 + 0.193j,
                        0.419 + 0.229j, 0.487 + 0.283j, 0.563 + 0.347j)
    direction_index = 0
    for first in range(1, 5):
        for second in range(first + 1, 5):
            exponent = -(
                charges[first] * charges[second]
                + times[first] * times[second]
            )
            exponent += complex(twist_regulator) * direction_values[direction_index]
            exponents[(first, second)] = exponent
            direction_index += 1
    anti_factor = charges[5] * charges[1]
    return exponents, anti_factor, family.physical_states


def evaluate_all_singlet_triple_contact_klt(
    resonant_outgoing_momenta: Sequence[complex],
    *,
    twist_regulator: complex = 0.0,
    mu: complex = 1.0,
    rtol: float = 2.0e-8,
    atol: float = 2.0e-10,
    rule: str = "gauss-jacobi-10",
) -> TripleContactKLTResult:
    """Evaluate the all-singlet ``c=3,m=l=0`` resonance component."""

    exponents, anti_factor, states = triple_contact_exponents(
        resonant_outgoing_momenta, twist_regulator=twist_regulator
    )
    raw, left, right = scalar_two_moving_klt(
        exponents,
        right_coefficient=anti_factor,
        rtol=rtol,
        atol=atol,
        rule=rule,
    )
    component = S3Component((2, 3, 7), 0, 0)
    coefficient = s3_component_action_coefficient(component, states, mu=mu)
    picture = 2.0 ** -3
    kernel_scale = max(
        abs(
            complex(sine_momentum_kernel(gamma, sigma, exponents, root=1))
        )
        for sigma in WORDS
        for gamma in WORDS
    )
    error = kernel_scale * (
        sum(item.absolute_error for item in left)
        * sum(abs(item.value) + item.absolute_error for item in right)
        + sum(item.absolute_error for item in right)
        * sum(abs(item.value) for item in left)
    )
    return TripleContactKLTResult(
        raw_area_value=raw,
        component_value=coefficient * picture * raw,
        action_coefficient=coefficient,
        picture_factor=picture,
        left_periods=(left[0].value, left[1].value),
        right_periods=(right[0].value, right[1].value),
        absolute_error_bound=abs(coefficient) * picture * error,
        periods_converged=all(item.converged for item in left + right),
    )


__all__ = [
    "TrianglePeriod",
    "TriangleTerm",
    "TripleContactKLTResult",
    "WORDS",
    "evaluate_all_singlet_triple_contact_klt",
    "integrate_triangle_term",
    "negative_triangle_term",
    "positive_triangle_term",
    "scalar_two_moving_klt",
    "triple_contact_exponents",
]
