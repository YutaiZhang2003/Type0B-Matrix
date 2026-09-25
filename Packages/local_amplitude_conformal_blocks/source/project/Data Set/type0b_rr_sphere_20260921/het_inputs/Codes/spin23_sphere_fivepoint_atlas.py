#!/usr/bin/env python3
r"""A nonoverlapping radial-order atlas for the ordered sphere five-point moduli.

The two minus-one-picture punctures keep labels 1 and 5 and remain at zero
and infinity.  The three zero-picture punctures are radially ordered.  If
their original labels, from smallest to largest modulus, are ``(a,b,c)``, the
active comb frame is ``(1,a,b,c,5)``.  Scaling puncture ``c`` to one gives

``q1=x_a/x_b`` and ``q2=x_b/x_c``, hence ``|q1|<=1`` and ``|q2|<=1``.

The six strict radial orderings partition the collision-free moduli space up
to equal-radius boundaries of measure zero.  Keeping the picture assignment
fixed avoids invoking pointwise picture-changing independence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


ChartPermutation = tuple[int, int, int, int, int]
RADIAL_CHARTS: tuple[ChartPermutation, ...] = (
    (0, 1, 2, 3, 4),
    (0, 1, 3, 2, 4),
    (0, 2, 1, 3, 4),
    (0, 2, 3, 1, 4),
    (0, 3, 1, 2, 4),
    (0, 3, 2, 1, 4),
)


@dataclass(frozen=True)
class RadialChartPoint:
    """One global moduli point expressed in its unique radial comb chart."""

    permutation: ChartPermutation
    z_2: complex
    z_3: complex
    chart_z_2: complex
    chart_z_3: complex
    q_1: complex
    q_2: complex
    holomorphic_jacobian: complex

    @property
    def measure_jacobian(self) -> float:
        return abs(self.holomorphic_jacobian) ** 2


@dataclass(frozen=True)
class _Jet2:
    value: complex
    derivative_2: complex = 0.0j
    derivative_3: complex = 0.0j

    def __add__(self, other: object) -> "_Jet2":
        right = _as_jet(other)
        return _Jet2(
            self.value + right.value,
            self.derivative_2 + right.derivative_2,
            self.derivative_3 + right.derivative_3,
        )

    __radd__ = __add__

    def __neg__(self) -> "_Jet2":
        return _Jet2(-self.value, -self.derivative_2, -self.derivative_3)

    def __sub__(self, other: object) -> "_Jet2":
        return self + (-_as_jet(other))

    def __rsub__(self, other: object) -> "_Jet2":
        return _as_jet(other) - self

    def __mul__(self, other: object) -> "_Jet2":
        right = _as_jet(other)
        return _Jet2(
            self.value * right.value,
            self.derivative_2 * right.value
            + self.value * right.derivative_2,
            self.derivative_3 * right.value
            + self.value * right.derivative_3,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> "_Jet2":
        right = _as_jet(other)
        if right.value == 0:
            raise ZeroDivisionError("colliding projective points")
        inverse = 1.0 / right.value
        return _Jet2(
            self.value * inverse,
            (self.derivative_2 * right.value
             - self.value * right.derivative_2)
            * inverse**2,
            (self.derivative_3 * right.value
             - self.value * right.derivative_3)
            * inverse**2,
        )

    def __rtruediv__(self, other: object) -> "_Jet2":
        return _as_jet(other) / self


def _as_jet(value: object) -> _Jet2:
    if isinstance(value, _Jet2):
        return value
    return _Jet2(complex(value))


HomogeneousJet = tuple[_Jet2, _Jet2]


def _determinant(first: HomogeneousJet, second: HomogeneousJet) -> _Jet2:
    return first[0] * second[1] - first[1] * second[0]


def _mobius_coordinate(
    point: HomogeneousJet,
    zero: HomogeneousJet,
    one: HomogeneousJet,
    infinity: HomogeneousJet,
) -> _Jet2:
    return (
        _determinant(point, zero)
        * _determinant(one, infinity)
        / (_determinant(point, infinity) * _determinant(one, zero))
    )


def _validate_global_moduli(z_2: complex, z_3: complex) -> tuple[complex, complex]:
    values = complex(z_2), complex(z_3)
    if any(
        not math.isfinite(part)
        for value in values
        for part in (value.real, value.imag)
    ):
        raise ValueError("global moduli must be finite")
    if values[0] in (0, 1) or values[1] in (0, 1) or values[0] == values[1]:
        raise ValueError("five sphere punctures must not collide")
    return values


def radial_chart_permutation(z_2: complex, z_3: complex) -> ChartPermutation:
    """Return the deterministic radial ordering of the zero-picture legs."""

    z_2, z_3 = _validate_global_moduli(z_2, z_3)
    finite = {1: z_2, 2: z_3, 3: 1.0 + 0.0j}
    ordered = tuple(sorted(finite, key=lambda leg: (abs(finite[leg]), leg)))
    return (0, ordered[0], ordered[1], ordered[2], 4)


def global_to_radial_chart(z_2: complex, z_3: complex) -> RadialChartPoint:
    """Map global ``(0,z2,z3,1,infinity)`` coordinates to their owner chart."""

    z_2, z_3 = _validate_global_moduli(z_2, z_3)
    permutation = radial_chart_permutation(z_2, z_3)
    zero = (_Jet2(0.0j), _Jet2(1.0 + 0.0j))
    infinity = (_Jet2(1.0 + 0.0j), _Jet2(0.0j))
    points: tuple[HomogeneousJet, ...] = (
        zero,
        (_Jet2(z_2, 1.0 + 0.0j, 0.0j), _Jet2(1.0 + 0.0j)),
        (_Jet2(z_3, 0.0j, 1.0 + 0.0j), _Jet2(1.0 + 0.0j)),
        (_Jet2(1.0 + 0.0j), _Jet2(1.0 + 0.0j)),
        infinity,
    )
    first, second, third, fourth, fifth = (
        points[index] for index in permutation
    )
    chart_z_2 = _mobius_coordinate(second, first, fourth, fifth)
    chart_z_3 = _mobius_coordinate(third, first, fourth, fifth)
    jacobian = (
        chart_z_2.derivative_2 * chart_z_3.derivative_3
        - chart_z_2.derivative_3 * chart_z_3.derivative_2
    )
    q_1 = chart_z_2.value / chart_z_3.value
    q_2 = chart_z_3.value
    tolerance = 5.0e-13
    if abs(q_1) > 1.0 + tolerance or abs(q_2) > 1.0 + tolerance:
        raise AssertionError("radial ordering failed to select a convergent comb")
    return RadialChartPoint(
        permutation=permutation,
        z_2=z_2,
        z_3=z_3,
        chart_z_2=chart_z_2.value,
        chart_z_3=chart_z_3.value,
        q_1=q_1,
        q_2=q_2,
        holomorphic_jacobian=jacobian,
    )


def radial_chart_to_global(
    permutation: Sequence[int],
    chart_z_2: complex,
    chart_z_3: complex,
) -> tuple[complex, complex]:
    """Invert one radial chart and restore original leg 4 to coordinate one."""

    normalized = tuple(int(value) for value in permutation)
    if normalized not in RADIAL_CHARTS:
        raise ValueError("permutation is not one of the six radial charts")
    standard = (0.0j, complex(chart_z_2), complex(chart_z_3), 1.0 + 0.0j)
    original: dict[int, complex] = {
        normalized[index]: standard[index] for index in range(4)
    }
    scale = original[3]
    if scale == 0:
        raise ValueError("the original fixed puncture collides with zero")
    z_2, z_3 = original[1] / scale, original[2] / scale
    _validate_global_moduli(z_2, z_3)
    return z_2, z_3


__all__ = [
    "ChartPermutation",
    "RADIAL_CHARTS",
    "RadialChartPoint",
    "global_to_radial_chart",
    "radial_chart_permutation",
    "radial_chart_to_global",
]
