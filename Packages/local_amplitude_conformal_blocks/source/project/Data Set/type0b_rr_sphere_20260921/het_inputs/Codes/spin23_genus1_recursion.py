#!/usr/bin/env python3
r"""Recursive genus-one two-point super-Virasoro blocks.

This module contains the recursion layer used by the genus-one Spin(23)
calculation.  The direct inverse-Gram implementation in
:mod:`spin23_genus1_blocks` remains the definition and the low-level oracle.

The NS implementation is the two-edge internal-weight recursion in the
necklace channel.  Write the two internal weights as

.. math::

   h_1=h, \qquad h_2=h+a.

At an NS Kac pole on either edge, the adjacent weight is evaluated along
this fixed-difference line.  An odd singular vector changes the fermion
routing between the two trinions, so the recursion retains a two-valued
routing label.  At :math:`b=1`, colliding Kac poles are combined before the
limit is taken by projecting the constant Laurent coefficient in
:math:`t=\log b`.

The Ramond implementation is in :mod:`spin23_two_virasoro_ramond`.  It uses
the exact branching of an auxiliary Majorana module tensored with a Ramond
super-Virasoro module into two commuting ordinary Virasoro modules.  The
mixed branching coefficients are computed from finite branch states; the
conditional closed product proposed in the supplied note is not used as a
definition.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Hashable, Mapping, TypeVar


LevelPair = tuple[int, int]
CoefficientKey = TypeVar("CoefficientKey", bound=Hashable)


def super_liouville_central_charge(b: complex) -> complex:
    r"""Return :math:`c=3/2+3(b+b^{-1})^2`."""

    b = complex(b)
    if b == 0:
        raise ValueError("b must be nonzero")
    q_background = b + 1.0 / b
    return 1.5 + 3.0 * q_background * q_background


def ns_liouville_weight(momentum: complex, b: complex) -> complex:
    r"""Return :math:`h_{\rm NS}=(Q^2+4P^2)/8`."""

    b = complex(b)
    momentum = complex(momentum)
    q_background = b + 1.0 / b
    return (q_background * q_background + 4.0 * momentum * momentum) / 8.0


def _ns_degenerate_weight(b: complex, r: int, s: int) -> complex:
    if r < 1 or s < 1 or (r + s) % 2:
        raise ValueError("NS Kac labels require r,s >= 1 and r+s even")
    return (
        -(r * s - 1.0) / 4.0
        + (1.0 - r * r) * b * b / 8.0
        + (1.0 - s * s) / (8.0 * b * b)
    )


def _ns_inverse_norm_residue(
    b: complex,
    r: int,
    s: int,
    pole_tolerance: float,
) -> complex:
    r"""Return the NS inverse-null-norm factor :math:`A_{r,s}`."""

    result = 0.5 + 0.0j
    for p in range(1 - r, r + 1):
        for q in range(1 - s, s + 1):
            if (p + q) % 2 or (p, q) in ((0, 0), (r, s)):
                continue
            denominator = (p * b + q / b) / math.sqrt(2.0)
            if abs(denominator) <= pole_tolerance:
                raise ZeroDivisionError(
                    "resonant NS inverse-norm factor; evaluate away from "
                    "rational b and take the assembled finite part"
                )
            result /= denominator
    return result


def _ns_fusion_polynomial(
    *,
    b: complex,
    r: int,
    s: int,
    lower_weight: complex,
    upper_weight: complex,
    starred: bool,
) -> complex:
    r"""Return the normalized NS--NS fusion polynomial at a Kac pole."""

    q_background = b + 1.0 / b
    lower_lambda = cmath.sqrt(
        q_background * q_background - 8.0 * lower_weight
    )
    upper_lambda = cmath.sqrt(
        q_background * q_background - 8.0 * upper_weight
    )
    wanted_parity = 1 if starred else 0
    denominator = 2.0 * math.sqrt(2.0)
    result = 1.0 + 0.0j
    for k in range(r):
        for ell in range(s):
            if (k + ell) % 2 != wanted_parity:
                continue
            p = 1 - r + 2 * k
            q = 1 - s + 2 * ell
            linear = p * b + q / b
            result *= (lower_lambda + upper_lambda - linear) / denominator
            result *= (lower_lambda - upper_lambda - linear) / denominator
    return result


@lru_cache(maxsize=None)
def _ns_character_coefficient(twice_level: int) -> int:
    r"""Return the generic NS Verma character coefficient at one level."""

    if not isinstance(twice_level, int) or twice_level < 0:
        raise ValueError("twice_level must be a nonnegative integer")
    # Product_{n>=1} (1+x^(2n-1))/(1-x^(2n)), truncated in x.
    coefficients = [0] * (twice_level + 1)
    coefficients[0] = 1
    for oscillator in range(1, twice_level + 1, 2):
        for level in range(twice_level, oscillator - 1, -1):
            coefficients[level] += coefficients[level - oscillator]
    for oscillator in range(2, twice_level + 1, 2):
        for level in range(oscillator, twice_level + 1):
            coefficients[level] += coefficients[level - oscillator]
    return coefficients[twice_level]


@dataclass(frozen=True)
class FinitePartDiagnostics:
    """Two-radius diagnostic for a coefficientwise Cauchy finite part."""

    value: complex
    check_value: complex
    radius: float
    check_radius: float
    samples: int

    @property
    def absolute_error(self) -> float:
        """Return the difference between the two contour projections."""

        return abs(self.value - self.check_value)

    @property
    def relative_error(self) -> float:
        """Return the radius-change error relative to the projected value."""

        return self.absolute_error / max(
            abs(self.value), abs(self.check_value), 1.0e-300
        )


def _validate_contour(radius: float, check_radius: float, samples: int) -> None:
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if not math.isfinite(check_radius) or check_radius <= 0:
        raise ValueError("check_radius must be finite and positive")
    if radius == check_radius:
        raise ValueError("radius and check_radius must differ")
    if not isinstance(samples, int) or samples < 8:
        raise ValueError("samples must be an integer of at least eight")


def dictionary_finite_part(
    evaluator: Callable[[complex], Mapping[CoefficientKey, complex]],
    *,
    keys: tuple[CoefficientKey, ...],
    radius: float,
    check_radius: float,
    samples: int,
    inversion_symmetric: bool = False,
) -> tuple[
    dict[CoefficientKey, complex],
    dict[CoefficientKey, FinitePartDiagnostics],
]:
    r"""Project the constant Laurent coefficient of an assembled table.

    The local uniformizer is :math:`b=e^t`.  Sampling at half-integer angular
    positions avoids the real and imaginary axes, where individual resonant
    terms are especially ill-conditioned.  If the *assembled* evaluator is
    known to obey ``evaluator(b) == evaluator(1 / b)``, antipodal contour
    samples are equal and ``inversion_symmetric=True`` evaluates one member
    of each pair.  This option must not be applied to individual singular
    branches before their complete symmetry-invariant sum is formed.
    """

    _validate_contour(radius, check_radius, samples)
    if not isinstance(inversion_symmetric, bool):
        raise TypeError("inversion_symmetric must be a boolean")
    if inversion_symmetric and samples % 2:
        raise ValueError(
            "inversion-symmetric finite-part projection requires an even "
            "sample count"
        )

    def average(current_radius: float) -> dict[CoefficientKey, complex]:
        totals = {key: 0.0j for key in keys}
        evaluated_samples = samples // 2 if inversion_symmetric else samples
        multiplicity = 2 if inversion_symmetric else 1
        for index in range(evaluated_samples):
            angle = 2.0 * math.pi * (index + 0.5) / samples
            t = current_radius * cmath.exp(1j * angle)
            values = evaluator(cmath.exp(t))
            for key in keys:
                totals[key] += multiplicity * complex(values[key])
        return {key: value / samples for key, value in totals.items()}

    primary = average(radius)
    check = average(check_radius)
    diagnostics = {
        key: FinitePartDiagnostics(
            value=primary[key],
            check_value=check[key],
            radius=radius,
            check_radius=check_radius,
            samples=samples,
        )
        for key in keys
    }
    return primary, diagnostics


class NSTorusTwoPointHRecursion:
    r"""Generic-:math:`b` NS two-edge necklace recursion.

    This block has two bottom-component external NS primaries.  The returned
    coefficients multiply :math:`q_1^{N_1}q_2^{N_2}`, with levels stored in
    half-units as ``twice_level_1`` and ``twice_level_2``.
    """

    def __init__(
        self,
        *,
        b: complex,
        internal_weight_1: complex,
        internal_weight_2: complex,
        external_weight_1: complex,
        external_weight_2: complex,
        form_parity: int = 0,
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.b = complex(b)
        self.c = super_liouville_central_charge(self.b)
        self.internal_weight_1 = complex(internal_weight_1)
        self.internal_weight_2 = complex(internal_weight_2)
        self.external_weight_1 = complex(external_weight_1)
        self.external_weight_2 = complex(external_weight_2)
        if form_parity not in (0, 1):
            raise ValueError("form_parity must be zero (even) or one (odd)")
        self.form_parity = int(form_parity)
        self.pole_tolerance = float(pole_tolerance)
        if not math.isfinite(self.pole_tolerance) or self.pole_tolerance <= 0:
            raise ValueError("pole_tolerance must be finite and positive")

    @staticmethod
    def _validate_twice_level(value: int, name: str) -> int:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return value

    @lru_cache(maxsize=None)
    def _coefficient(
        self,
        twice_level_1: int,
        twice_level_2: int,
        base_weight: complex,
        weight_difference: complex,
        routing: int,
    ) -> complex:
        if twice_level_1 < 0 or twice_level_2 < 0:
            return 0.0j
        if routing not in (0, 1):
            raise ValueError("routing must be zero or one")

        weight_1 = complex(base_weight)
        difference = complex(weight_difference)
        weight_2 = weight_1 + difference
        result = 0.0j
        if routing == 0 and twice_level_1 == twice_level_2:
            result = complex(_ns_character_coefficient(twice_level_1))

        for edge, available_level in (
            (1, twice_level_1),
            (2, twice_level_2),
        ):
            for r in range(1, available_level + 1):
                for s in range(1, available_level // r + 1):
                    product = r * s
                    if product > available_level or (r + s) % 2:
                        continue
                    degenerate = _ns_degenerate_weight(self.b, r, s)
                    if edge == 1:
                        denominator = weight_1 - degenerate
                        adjacent_at_pole = degenerate + difference
                    else:
                        denominator = weight_2 - degenerate
                        adjacent_at_pole = degenerate - difference
                    scale = max(
                        1.0, abs(weight_1), abs(weight_2), abs(degenerate)
                    )
                    if abs(denominator) <= self.pole_tolerance * scale:
                        raise ZeroDivisionError(
                            "NS h-recursion hit a Kac pole; use the "
                            "coefficientwise finite-part wrapper"
                        )

                    residue = _ns_inverse_norm_residue(
                        self.b, r, s, self.pole_tolerance
                    )
                    for external_weight in (
                        self.external_weight_1,
                        self.external_weight_2,
                    ):
                        residue *= _ns_fusion_polynomial(
                            b=self.b,
                            r=r,
                            s=s,
                            lower_weight=external_weight,
                            upper_weight=adjacent_at_pole,
                            starred=bool(routing),
                        )
                    next_routing = routing ^ (product % 2)
                    if edge == 1:
                        tail = self._coefficient(
                            twice_level_1 - product,
                            twice_level_2,
                            complex(degenerate + product / 2.0),
                            complex(difference - product / 2.0),
                            next_routing,
                        )
                    else:
                        tail = self._coefficient(
                            twice_level_1,
                            twice_level_2 - product,
                            complex(degenerate - difference),
                            complex(difference + product / 2.0),
                            next_routing,
                        )
                    result += residue * tail / denominator
        return complex(result)

    def coefficient(self, twice_level_1: int, twice_level_2: int) -> complex:
        """Return one stripped bivariate coefficient."""

        self._validate_twice_level(twice_level_1, "twice_level_1")
        self._validate_twice_level(twice_level_2, "twice_level_2")
        return self._coefficient(
            twice_level_1,
            twice_level_2,
            self.internal_weight_1,
            self.internal_weight_2 - self.internal_weight_1,
            self.form_parity,
        )

    def coefficients(
        self,
        max_twice_level_1: int,
        max_twice_level_2: int,
    ) -> dict[LevelPair, complex]:
        """Return all coefficients in the requested rectangular cutoff."""

        self._validate_twice_level(max_twice_level_1, "max_twice_level_1")
        self._validate_twice_level(max_twice_level_2, "max_twice_level_2")
        return {
            (level_1, level_2): self.coefficient(level_1, level_2)
            for level_1 in range(max_twice_level_1 + 1)
            for level_2 in range(max_twice_level_2 + 1)
        }

    def descendant_value(
        self,
        q_1: complex,
        q_2: complex,
        max_twice_level_1: int,
        max_twice_level_2: int,
    ) -> complex:
        """Evaluate the stripped descendant series."""

        q_1 = complex(q_1)
        q_2 = complex(q_2)
        return sum(
            coefficient * q_1 ** (level_1 / 2.0) * q_2 ** (level_2 / 2.0)
            for (level_1, level_2), coefficient in self.coefficients(
                max_twice_level_1, max_twice_level_2
            ).items()
        )


class SelfDualNSTorusTwoPointHRecursion:
    r"""Coefficientwise :math:`b=1` finite part of the NS recursion."""

    def __init__(
        self,
        *,
        internal_momentum_1: complex,
        internal_momentum_2: complex,
        external_momentum_1: complex,
        external_momentum_2: complex,
        form_parity: int = 0,
        radius: float = 0.04,
        check_radius: float = 0.05,
        samples: int = 24,
        difference_radius: float = 0.03,
        difference_samples: int = 16,
        collision_tolerance: float = 1.0e-12,
    ) -> None:
        self.internal_momentum_1 = complex(internal_momentum_1)
        self.internal_momentum_2 = complex(internal_momentum_2)
        self.external_momentum_1 = complex(external_momentum_1)
        self.external_momentum_2 = complex(external_momentum_2)
        if form_parity not in (0, 1):
            raise ValueError("form_parity must be zero (even) or one (odd)")
        self.form_parity = int(form_parity)
        self.radius = float(radius)
        self.check_radius = float(check_radius)
        self.samples = int(samples)
        self.difference_radius = float(difference_radius)
        self.difference_samples = int(difference_samples)
        self.collision_tolerance = float(collision_tolerance)
        _validate_contour(self.radius, self.check_radius, self.samples)
        if not math.isfinite(self.difference_radius) or self.difference_radius <= 0:
            raise ValueError("difference_radius must be finite and positive")
        if self.difference_samples < 8:
            raise ValueError("difference_samples must be at least eight")
        self._cache: dict[
            tuple[int, int],
            tuple[
                dict[LevelPair, complex],
                dict[LevelPair, FinitePartDiagnostics],
            ],
        ] = {}

    def _block_at(
        self,
        b: complex,
        *,
        weight_difference_shift: complex = 0.0j,
    ) -> NSTorusTwoPointHRecursion:
        return NSTorusTwoPointHRecursion(
            b=b,
            internal_weight_1=ns_liouville_weight(
                self.internal_momentum_1, b
            ),
            internal_weight_2=(
                ns_liouville_weight(self.internal_momentum_2, b)
                + weight_difference_shift
            ),
            external_weight_1=ns_liouville_weight(
                self.external_momentum_1, b
            ),
            external_weight_2=ns_liouville_weight(
                self.external_momentum_2, b
            ),
            form_parity=self.form_parity,
        )

    def _table_at(
        self,
        b: complex,
        max_twice_level_1: int,
        max_twice_level_2: int,
    ) -> dict[LevelPair, complex]:
        momentum_difference = (
            self.internal_momentum_2 * self.internal_momentum_2
            - self.internal_momentum_1 * self.internal_momentum_1
        ) / 2.0
        if abs(momentum_difference) > self.collision_tolerance:
            return self._block_at(b).coefficients(
                max_twice_level_1, max_twice_level_2
            )

        totals = {
            (level_1, level_2): 0.0j
            for level_1 in range(max_twice_level_1 + 1)
            for level_2 in range(max_twice_level_2 + 1)
        }
        for index in range(self.difference_samples):
            angle = (
                2.0 * math.pi * (index + 0.5) / self.difference_samples
            )
            displacement = self.difference_radius * cmath.exp(1j * angle)
            values = self._block_at(
                b, weight_difference_shift=displacement
            ).coefficients(max_twice_level_1, max_twice_level_2)
            for key, value in values.items():
                totals[key] += value
        return {
            key: value / self.difference_samples
            for key, value in totals.items()
        }

    def _data(
        self,
        max_twice_level_1: int,
        max_twice_level_2: int,
    ) -> tuple[
        dict[LevelPair, complex],
        dict[LevelPair, FinitePartDiagnostics],
    ]:
        NSTorusTwoPointHRecursion._validate_twice_level(
            max_twice_level_1, "max_twice_level_1"
        )
        NSTorusTwoPointHRecursion._validate_twice_level(
            max_twice_level_2, "max_twice_level_2"
        )
        cache_key = (max_twice_level_1, max_twice_level_2)
        if cache_key not in self._cache:
            keys = tuple(
                (level_1, level_2)
                for level_1 in range(max_twice_level_1 + 1)
                for level_2 in range(max_twice_level_2 + 1)
            )
            self._cache[cache_key] = dictionary_finite_part(
                lambda b: self._table_at(
                    b, max_twice_level_1, max_twice_level_2
                ),
                keys=keys,
                radius=self.radius,
                check_radius=self.check_radius,
                samples=self.samples,
            )
        return self._cache[cache_key]

    def coefficients(
        self,
        max_twice_level_1: int,
        max_twice_level_2: int,
    ) -> dict[LevelPair, complex]:
        """Return the self-dual coefficient table."""

        return dict(self._data(max_twice_level_1, max_twice_level_2)[0])

    def diagnostics(
        self,
        max_twice_level_1: int,
        max_twice_level_2: int,
    ) -> dict[LevelPair, FinitePartDiagnostics]:
        """Return the two-radius finite-part diagnostic for each coefficient."""

        return dict(self._data(max_twice_level_1, max_twice_level_2)[1])

    def descendant_value(
        self,
        q_1: complex,
        q_2: complex,
        max_twice_level_1: int,
        max_twice_level_2: int,
    ) -> complex:
        """Evaluate the self-dual stripped descendant series."""

        q_1 = complex(q_1)
        q_2 = complex(q_2)
        return sum(
            coefficient * q_1 ** (level_1 / 2.0) * q_2 ** (level_2 / 2.0)
            for (level_1, level_2), coefficient in self.coefficients(
                max_twice_level_1, max_twice_level_2
            ).items()
        )


__all__ = [
    "FinitePartDiagnostics",
    "NSTorusTwoPointHRecursion",
    "SelfDualNSTorusTwoPointHRecursion",
    "dictionary_finite_part",
    "ns_liouville_weight",
    "super_liouville_central_charge",
]
