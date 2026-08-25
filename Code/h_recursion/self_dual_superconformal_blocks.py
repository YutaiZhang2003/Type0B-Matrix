"""Collision-aware superconformal blocks at the Type-0B point ``b=1``.

The fixed-``b`` elliptic recursions are meromorphic term by term.  At
``b=1`` several Kac labels coalesce, so separate recursive terms contain
spurious poles although their sum is regular.  This module takes the finite
part of the *assembled coefficient* in the local uniformizer

    b = exp(t),  t -> 0.

For a Laurent series ``f(t)``, its finite part is the Cauchy projection

    FP[f] = (2 pi i)^(-1) integral f(t) dt/t.

The trapezoidal rule on a small complex circle is exponentially accurate for
analytic Laurent data and combines all resonant Kac terms before the value at
``b=1`` is formed.  This is not a one-sided ``c=27/2+epsilon`` evaluation or
an epsilon extrapolation.  Two radii are retained as an internal numerical
diagnostic.

The wrappers below cover the three sphere blocks currently used by the BRY
crossing layer:

* four Ramond external fields with NS exchange;
* RRNN with NS exchange;
* RRNN with a generic long-R exchange.

The last case is the one which genuinely exercises the internal ``G_0``
ground-state fiber.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
import math
from typing import Callable, Dict, Literal, Mapping, Sequence

from mixed_ramond_sphere_blocks import (
    MixedNSExchangeSphereFourPointBlock,
    MixedRExchangeSphereFourPointBlock,
)
from ramond_sphere_blocks import RamondExternalSphereFourPointBlock
from superconformal_blocks import elliptic_nome


Parity = Literal["even", "odd"]


@dataclass(frozen=True)
class FinitePartDiagnostics:
    """Value and radius-stability data for a Cauchy finite-part projection."""

    value: complex
    check_value: complex
    radius: float
    check_radius: float
    samples: int

    @property
    def absolute_error(self) -> float:
        return abs(self.value - self.check_value)

    @property
    def relative_error(self) -> float:
        return self.absolute_error / max(
            abs(self.value), abs(self.check_value), 1.0e-300
        )


def _validate_contour(radius: float, check_radius: float, samples: int) -> None:
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if not math.isfinite(check_radius) or check_radius <= 0.0:
        raise ValueError("check_radius must be finite and positive")
    if radius == check_radius:
        raise ValueError("radius and check_radius must be distinct")
    if not isinstance(samples, int) or samples < 8:
        raise ValueError("samples must be an integer of at least 8")


def _circle_average(
    evaluator: Callable[[complex], complex],
    *,
    radius: float,
    samples: int,
) -> complex:
    """Project the constant Laurent coefficient in ``t=log(b)``."""

    total = 0.0j
    # The half-step avoids sampling the real and imaginary axes, where
    # individual resonant factors can be particularly ill-conditioned.
    for index in range(samples):
        angle = 2.0 * math.pi * (index + 0.5) / samples
        t = radius * cmath.exp(1j * angle)
        total += complex(evaluator(cmath.exp(t)))
    return total / samples


def self_dual_finite_part(
    evaluator: Callable[[complex], complex],
    *,
    radius: float = 0.04,
    check_radius: float = 0.05,
    samples: int = 24,
) -> FinitePartDiagnostics:
    """Return the finite part at ``b=1`` and an independent radius check.

    ``evaluator(b)`` may contain poles in its individual recursion terms, but
    it must return the fully assembled coefficient or block at generic
    complex ``b``.  If the physical result is regular, the two projected
    values agree up to quadrature and floating-point error.
    """

    _validate_contour(radius, check_radius, samples)
    value = _circle_average(evaluator, radius=radius, samples=samples)
    check_value = _circle_average(
        evaluator, radius=check_radius, samples=samples
    )
    return FinitePartDiagnostics(
        value=value,
        check_value=check_value,
        radius=radius,
        check_radius=check_radius,
        samples=samples,
    )


def _dictionary_finite_part(
    evaluator: Callable[[complex], Mapping[int, complex]],
    *,
    keys: Sequence[int],
    radius: float,
    check_radius: float,
    samples: int,
) -> tuple[Dict[int, complex], Dict[int, FinitePartDiagnostics]]:
    _validate_contour(radius, check_radius, samples)

    def average(current_radius: float) -> Dict[int, complex]:
        totals = {key: 0.0j for key in keys}
        for index in range(samples):
            angle = 2.0 * math.pi * (index + 0.5) / samples
            t = current_radius * cmath.exp(1j * angle)
            values = evaluator(cmath.exp(t))
            for key in keys:
                totals[key] += complex(values[key])
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


class SelfDualRamondExternalSphereFourPointBlock:
    """Exact-self-dual four-R block with an internal NS representation."""

    def __init__(
        self,
        *,
        p1: complex,
        p2: complex,
        p3: complex,
        p4: complex,
        internal_momentum: complex,
        sign3: int = 1,
        sign2: int = 1,
        radius: float = 0.04,
        check_radius: float = 0.05,
        samples: int = 24,
    ) -> None:
        self.p1 = complex(p1)
        self.p2 = complex(p2)
        self.p3 = complex(p3)
        self.p4 = complex(p4)
        self.internal_momentum = complex(internal_momentum)
        self.sign3 = int(sign3)
        self.sign2 = int(sign2)
        self.radius = float(radius)
        self.check_radius = float(check_radius)
        self.samples = int(samples)
        _validate_contour(self.radius, self.check_radius, self.samples)
        self._cache: dict[
            tuple[int, Parity],
            tuple[Dict[int, complex], Dict[int, FinitePartDiagnostics]],
        ] = {}
        self._base = self._block_at(1.0)

    def _block_at(self, b: complex) -> RamondExternalSphereFourPointBlock:
        return RamondExternalSphereFourPointBlock.from_liouville_momenta(
            p1=self.p1,
            p2=self.p2,
            p3=self.p3,
            p4=self.p4,
            internal_momentum=self.internal_momentum,
            b=b,
            sign3=self.sign3,
            sign2=self.sign2,
        )

    def _data(
        self, order: int, parity: Parity
    ) -> tuple[Dict[int, complex], Dict[int, FinitePartDiagnostics]]:
        if order < 1:
            raise ValueError("order must be positive")
        if parity not in ("even", "odd"):
            raise ValueError("parity must be 'even' or 'odd'")
        key = (order, parity)
        if key not in self._cache:
            powers = [
                2 * level + (1 if parity == "odd" else 0)
                for level in range(order)
            ]
            self._cache[key] = _dictionary_finite_part(
                lambda b: self._block_at(b).elliptic_coefficients(
                    order, parity
                ),
                keys=powers,
                radius=self.radius,
                check_radius=self.check_radius,
                samples=self.samples,
            )
        return self._cache[key]

    def elliptic_coefficients(
        self, order: int, parity: Parity
    ) -> Dict[int, complex]:
        return dict(self._data(order, parity)[0])

    def coefficient_diagnostics(
        self, order: int, parity: Parity
    ) -> Dict[int, FinitePartDiagnostics]:
        return dict(self._data(order, parity)[1])

    def elliptic_block(self, z: complex, order: int, parity: Parity) -> complex:
        z = complex(z)
        q = elliptic_nome(z)
        series = sum(
            coefficient * q ** (power / 2.0)
            for power, coefficient in self.elliptic_coefficients(
                order, parity
            ).items()
        )
        return self._base._elliptic_prefactor(z) * series


class SelfDualMixedNSExchangeSphereFourPointBlock:
    """Exact-self-dual RRNN block in the NS exchange channel."""

    def __init__(
        self,
        *,
        p1_r: complex,
        p2_r: complex,
        p3_ns: complex,
        p4_ns: complex,
        internal_momentum: complex,
        sign2: int = 1,
        radius: float = 0.04,
        check_radius: float = 0.05,
        samples: int = 24,
    ) -> None:
        self.parameters = dict(
            p1_r=complex(p1_r),
            p2_r=complex(p2_r),
            p3_ns=complex(p3_ns),
            p4_ns=complex(p4_ns),
            internal_momentum=complex(internal_momentum),
            sign2=int(sign2),
        )
        self.radius = float(radius)
        self.check_radius = float(check_radius)
        self.samples = int(samples)
        _validate_contour(self.radius, self.check_radius, self.samples)
        self._cache: dict[
            tuple[int, Parity],
            tuple[Dict[int, complex], Dict[int, FinitePartDiagnostics]],
        ] = {}
        self._base = self._block_at(1.0)

    def _block_at(self, b: complex) -> MixedNSExchangeSphereFourPointBlock:
        return MixedNSExchangeSphereFourPointBlock(b=b, **self.parameters)

    def _data(
        self, order: int, parity: Parity
    ) -> tuple[Dict[int, complex], Dict[int, FinitePartDiagnostics]]:
        if order < 1:
            raise ValueError("order must be positive")
        if parity not in ("even", "odd"):
            raise ValueError("parity must be 'even' or 'odd'")
        key = (order, parity)
        if key not in self._cache:
            powers = [
                2 * level + (1 if parity == "odd" else 0)
                for level in range(order)
            ]
            self._cache[key] = _dictionary_finite_part(
                lambda b: self._block_at(b).elliptic_coefficients(
                    order, parity
                ),
                keys=powers,
                radius=self.radius,
                check_radius=self.check_radius,
                samples=self.samples,
            )
        return self._cache[key]

    def elliptic_coefficients(
        self, order: int, parity: Parity
    ) -> Dict[int, complex]:
        return dict(self._data(order, parity)[0])

    def coefficient_diagnostics(
        self, order: int, parity: Parity
    ) -> Dict[int, FinitePartDiagnostics]:
        return dict(self._data(order, parity)[1])

    def elliptic_block(self, z: complex, order: int, parity: Parity) -> complex:
        z = complex(z)
        q = elliptic_nome(z)
        series = sum(
            coefficient * q ** (power / 2.0)
            for power, coefficient in self.elliptic_coefficients(
                order, parity
            ).items()
        )
        return self._base._elliptic_prefactor(z) * series


class SelfDualMixedRExchangeSphereFourPointBlock:
    """Exact-self-dual RRNN block with a generic long-R internal module."""

    def __init__(
        self,
        *,
        p1_ns: complex,
        p2_r: complex,
        p3_r: complex,
        p4_ns: complex,
        internal_momentum: complex,
        sign3: int = 1,
        sign2: int = 1,
        radius: float = 0.04,
        check_radius: float = 0.05,
        samples: int = 24,
    ) -> None:
        self.parameters = dict(
            p1_ns=complex(p1_ns),
            p2_r=complex(p2_r),
            p3_r=complex(p3_r),
            p4_ns=complex(p4_ns),
            internal_momentum=complex(internal_momentum),
            sign3=int(sign3),
            sign2=int(sign2),
        )
        self.radius = float(radius)
        self.check_radius = float(check_radius)
        self.samples = int(samples)
        _validate_contour(self.radius, self.check_radius, self.samples)
        self._cache: dict[
            int,
            tuple[Dict[int, complex], Dict[int, FinitePartDiagnostics]],
        ] = {}
        self._base = self._block_at(1.0)

    def _block_at(self, b: complex) -> MixedRExchangeSphereFourPointBlock:
        return MixedRExchangeSphereFourPointBlock(b=b, **self.parameters)

    def _data(
        self, order: int
    ) -> tuple[Dict[int, complex], Dict[int, FinitePartDiagnostics]]:
        if order < 1:
            raise ValueError("order must be positive")
        if order not in self._cache:
            powers = list(range(order))
            self._cache[order] = _dictionary_finite_part(
                lambda b: self._block_at(b).elliptic_coefficients(order),
                keys=powers,
                radius=self.radius,
                check_radius=self.check_radius,
                samples=self.samples,
            )
        return self._cache[order]

    def elliptic_coefficients(self, order: int) -> Dict[int, complex]:
        return dict(self._data(order)[0])

    def coefficient_diagnostics(
        self, order: int
    ) -> Dict[int, FinitePartDiagnostics]:
        return dict(self._data(order)[1])

    def elliptic_block(self, z: complex, order: int) -> complex:
        z = complex(z)
        q = elliptic_nome(z)
        series = sum(
            coefficient * q**power
            for power, coefficient in self.elliptic_coefficients(order).items()
        )
        return self._base._elliptic_prefactor(z) * series

    def direct_level_one_coefficient(self) -> complex:
        """Return the independent fixed-``c`` Gram/Ward coefficient."""

        return self._base.direct_level_one_coefficient()

    def recursion_level_one_coefficient(self) -> complex:
        """Extract the local level-one coefficient from the finite-part block."""

        h = self._base.internal_weight
        vacuum_shift = (self._base.c - 1.5) / 24.0
        theta_exponent = (
            (self._base.c - 1.5) / 2.0
            - 4.0
            * (
                self._base.h1
                + self._base.h2
                + self._base.h3
                + self._base.h4
            )
            + 0.5
        )
        prefactor_coefficient = (
            0.5 * (h - vacuum_shift - 1.0 / 16.0)
            - (vacuum_shift - self._base.h2 - self._base.h3)
            + theta_exponent / 8.0
        )
        return (
            prefactor_coefficient
            + self.elliptic_coefficients(2)[1] / 16.0
        )


__all__ = [
    "FinitePartDiagnostics",
    "SelfDualMixedNSExchangeSphereFourPointBlock",
    "SelfDualMixedRExchangeSphereFourPointBlock",
    "SelfDualRamondExternalSphereFourPointBlock",
    "self_dual_finite_part",
]
