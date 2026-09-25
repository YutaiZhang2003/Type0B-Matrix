#!/usr/bin/env python3
r"""Local half-level series used by Ramond sphere degeneration patches.

The direct Ramond and crossed NS blocks store coefficients with keys equal to
twice the descendant level.  A fixed component therefore has support on one
parity class only: ``0,2,4,...`` or ``1,3,5,...``.  This module keeps that
information explicit while presenting the integer-spaced polynomial required
by the analytically continued disk and folded-lens integrators.

No region of moduli space is removed by these helpers.  They evaluate the
retained local OPE series over the excised patch, using the logarithmic finite
part when a radial exponent is exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from spin23_singlet_amplitudes import _integrate_series_lens
from heterotic_so23_1to3_vvvv_fit_bundle import (
    heterotic_so23_1to3_fast as fast,
)


@dataclass(frozen=True)
class LocalHalfLevelSeries:
    r"""Represent ``z**base_exponent * sum c_L z**(L/2)``.

    All nonzero twice-levels must have the same parity.  Missing levels in
    that parity class are allowed and are filled with zero on conversion to a
    dense polynomial.
    """

    base_exponent: complex
    coefficients: Mapping[int, complex]

    def __post_init__(self) -> None:
        normalized: dict[int, complex] = {}
        for level, coefficient in self.coefficients.items():
            if not isinstance(level, int):
                raise TypeError("twice-level keys must be integers")
            if level < 0:
                raise ValueError("twice-level keys must be nonnegative")
            value = complex(coefficient)
            if value != 0:
                normalized[level] = value
        if normalized:
            parity = next(iter(normalized)) % 2
            if any(level % 2 != parity for level in normalized):
                raise ValueError(
                    "one local component must occupy a single twice-level parity"
                )
        object.__setattr__(self, "base_exponent", complex(self.base_exponent))
        object.__setattr__(self, "coefficients", normalized)

    @property
    def twice_level_parity(self) -> int | None:
        """Return the occupied parity, or ``None`` for the zero series."""

        if not self.coefficients:
            return None
        return next(iter(self.coefficients)) % 2

    def dense_data(self) -> tuple[complex, np.ndarray]:
        r"""Return ``(effective_exponent, integer-spaced coefficients)``.

        If the first occupied twice-level is ``L0``, its factor
        ``z**(L0/2)`` is absorbed into the returned exponent.  Successive
        entries of the returned array then multiply successive integer powers
        of ``z``.
        """

        if not self.coefficients:
            return self.base_exponent, np.zeros(0, dtype=np.complex128)
        first = min(self.coefficients)
        last = max(self.coefficients)
        parity = first % 2
        if last % 2 != parity:
            raise AssertionError("validated parity changed unexpectedly")
        dense = np.zeros((last - first) // 2 + 1, dtype=np.complex128)
        for level, coefficient in self.coefficients.items():
            dense[(level - first) // 2] = coefficient
        return self.base_exponent + first / 2.0, dense

    def scaled(self, coefficient: complex) -> "LocalHalfLevelSeries":
        """Return the series multiplied by a scalar."""

        scale = complex(coefficient)
        return LocalHalfLevelSeries(
            self.base_exponent,
            {level: scale * value for level, value in self.coefficients.items()},
        )

    def shifted(self, power: complex) -> "LocalHalfLevelSeries":
        """Return the series multiplied by ``z**power``."""

        return LocalHalfLevelSeries(
            self.base_exponent + complex(power),
            self.coefficients,
        )

    def multiply_binomial(
        self,
        exponent: complex,
        *,
        maximum_integer_order: int,
    ) -> "LocalHalfLevelSeries":
        r"""Multiply by ``(1-z)**(-exponent)`` to the requested order."""

        order = int(maximum_integer_order)
        if order < 0:
            raise ValueError("maximum_integer_order must be nonnegative")
        if not self.coefficients:
            return self
        first = min(self.coefficients)
        effective, dense = self.dense_data()
        if abs(effective - (self.base_exponent + first / 2.0)) > 1.0e-14:
            raise AssertionError("inconsistent dense-series exponent")
        binomial = fast._binomial_minus_power(complex(exponent), order)
        product = np.convolve(dense, binomial)[: dense.size + order]
        return LocalHalfLevelSeries(
            self.base_exponent,
            {
                first + 2 * index: coefficient
                for index, coefficient in enumerate(product)
                if coefficient != 0
            },
        )

    def value(self, z: complex) -> complex:
        """Evaluate the retained local series on the principal branch."""

        z_value = complex(z)
        if z_value == 0:
            raise ValueError("a local series with a general leading power needs z != 0")
        exponent, dense = self.dense_data()
        if not dense.size:
            return 0.0j
        return z_value**exponent * np.polynomial.polynomial.polyval(z_value, dense)


def integrate_full_disk(
    holomorphic: LocalHalfLevelSeries,
    antiholomorphic: LocalHalfLevelSeries,
    epsilon: float,
) -> complex:
    r"""Meromorphically integrate a fixed component over ``|z|<epsilon``."""

    if not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must lie in (0, 1)")
    hol_power, hol = holomorphic.dense_data()
    anti_power, anti = antiholomorphic.dense_data()
    if not hol.size or not anti.size:
        return 0.0j
    return complex(
        fast._integrate_series_disk(
            hol,
            anti,
            hol_power,
            anti_power,
            float(epsilon),
        )
    )


def integrate_folded_lens(
    holomorphic: LocalHalfLevelSeries,
    antiholomorphic: LocalHalfLevelSeries,
    epsilon: float,
    *,
    angular_order: int | None = None,
) -> complex:
    r"""Meromorphically integrate a fixed component over the unit-disk lens.

    The lens is ``|w|<epsilon`` with ``|1-w|<1``.  Single-valuedness is
    checked by the underlying angular-series integrator after the fixed
    half-level offsets have been absorbed into the leading powers.
    """

    if not math.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must lie in (0, 1)")
    hol_power, hol = holomorphic.dense_data()
    anti_power, anti = antiholomorphic.dense_data()
    if not hol.size or not anti.size:
        return 0.0j
    return complex(
        _integrate_series_lens(
            hol,
            anti,
            hol_power,
            anti_power,
            float(epsilon),
            angular_order=angular_order,
        )
    )


__all__ = [
    "LocalHalfLevelSeries",
    "integrate_folded_lens",
    "integrate_full_disk",
]
