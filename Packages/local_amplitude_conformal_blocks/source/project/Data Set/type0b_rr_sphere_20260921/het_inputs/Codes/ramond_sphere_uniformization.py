r"""Typed nome-uniformization adapters for repository Ramond sphere blocks.

The direct block builders intentionally know nothing about elliptic
uniformization.  This module supplies the thin bridge from each of their
series dataclasses to :class:`NomeUniformizedBlockSeries`, while preserving
the leading OPE power used by the source class.

No Ramond elliptic recursion is assumed.  The adapters only perform the
algebraic change of variable from the already-computed direct ``z`` series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from ns_algebra.ns_sca import twice_level as ns_twice_level
from ramond_algebra.ramond_sca import twice_level as ramond_twice_level
from so7e8_ramond_fourpoint import (
    FourRamondNSBlockSeries,
    TwoRamondInternalNSBlockSeries,
)
from sphere_block_uniformization import (
    CutSide,
    NomeUniformizedBlockSeries,
    _cut_lip_argument,
    elliptic_nome,
    uniformize_direct_block,
)
from heterotic_so23_1to3_vvvv_fit_bundle.heterotic_so23_1to3 import (
    _modular_series,
    _series_multiply,
    _series_power,
)
from spin23_ramond_blocks import RamondSphereBlockSeries


RamondSphereSeries: TypeAlias = (
    RamondSphereBlockSeries
    | TwoRamondInternalNSBlockSeries
    | FourRamondNSBlockSeries
)
RamondSphereSeriesKind: TypeAlias = Literal[
    "internal_ramond",
    "two_ramond_internal_ns",
    "four_ramond_internal_ns",
]
ChannelCoordinate: TypeAlias = Literal["z", "w"]


@dataclass(frozen=True)
class UniformizedRamondSphereSeries:
    """One repository Ramond sphere series in ``t=sqrt(q)`` coordinates."""

    source: RamondSphereSeries
    nome_series: NomeUniformizedBlockSeries
    primary_exponent: complex
    series_kind: RamondSphereSeriesKind
    channel_coordinate: ChannelCoordinate

    def value_from_t(self, t: complex | np.ndarray) -> complex | np.ndarray:
        """Evaluate only the descendant polynomial at a supplied ``sqrt(q)``."""

        return self.nome_series.value_from_t(t)

    def descendant_value(
        self,
        coordinate: complex | np.ndarray,
        *,
        cut_side: CutSide = "upper",
    ) -> complex | np.ndarray:
        """Evaluate the uniformized descendant series at ``z`` or crossed ``w``."""

        return self.nome_series.descendant_value(coordinate, cut_side=cut_side)

    def value(
        self,
        coordinate: complex | np.ndarray,
        *,
        include_primary_power: bool = True,
        cut_side: CutSide = "upper",
    ) -> complex | np.ndarray:
        """Evaluate the block, preserving the source class's OPE exponent."""

        if not include_primary_power:
            return self.descendant_value(coordinate, cut_side=cut_side)
        points = np.asarray(coordinate, dtype=np.complex128)
        if np.any(points == 0.0):
            raise ValueError("the primary sphere-block power is singular at zero")
        return self.nome_series.value(
            coordinate,
            primary_exponent=self.primary_exponent,
            cut_side=cut_side,
        )


@dataclass(frozen=True)
class EllipticPrefactoredRamondSphereSeries:
    r"""Finite elliptic series with the appropriate NS-channel prefactor.

    For four external Ramond fields, HJS factor the chiral block as

    ``(16q)**(h-c0) z**(c0-h0-hz) (1-z)**(c0-hz-h1)``
    ``* theta3(q)**A * H(sqrt(q))``,

    where ``c0=(c-3/2)/24`` and
    ``A=(c-3/2)/2-4*sum(h_i)``.  For two NS and two Ramond fields with an
    internal NS module, Suchanek's mixed-sector recursion instead has the
    additional factor ``(1-z)**(1/16)*theta3(q)**(1/2)``.  The two shifts
    below keep that distinction explicit.

    Dividing the appropriate prefactor out before truncating is the same
    acceleration used by the repository's SO(23) NS integrator.  It is an
    algebraic re-expansion of direct block data and does not assume that a
    residue recursion has been implemented locally.
    """

    source: (
        RamondSphereBlockSeries
        | TwoRamondInternalNSBlockSeries
        | FourRamondNSBlockSeries
    )
    h_coefficients: tuple[complex, ...]
    effective_external_weights: tuple[complex, complex, complex, complex]
    internal_weight: complex
    central_charge: complex
    channel_coordinate: ChannelCoordinate
    known_through_twice_level: int
    zero_coordinate_shift: complex
    one_minus_coordinate_shift: complex
    theta_shift: complex

    @property
    def c0(self) -> complex:
        return (self.central_charge - 1.5) / 24.0

    @property
    def theta_exponent(self) -> complex:
        return (
            (self.central_charge - 1.5) / 2.0
            - 4.0 * sum(self.effective_external_weights, 0.0j)
            + self.theta_shift
        )

    def value_from_t(self, t: complex | np.ndarray) -> complex | np.ndarray:
        argument = np.asarray(t, dtype=np.complex128)
        scalar = argument.ndim == 0
        result = np.zeros_like(argument)
        for coefficient in reversed(self.h_coefficients):
            result = result * argument + coefficient
        if scalar:
            return complex(result.item())
        return result

    @staticmethod
    def _theta3(q: np.ndarray) -> np.ndarray:
        if np.any(np.abs(q) >= 1.0):
            raise ArithmeticError("elliptic nome left the open unit disk")
        theta = np.ones_like(q)
        # The slowest quadrature points can approach a degeneration.  Use an
        # adaptive theta sum instead of the old fixed sixty-term cutoff.
        for index in range(1, 10000):
            term = 2.0 * np.power(q, index * index)
            theta += term
            if np.max(np.abs(term), initial=0.0) < 2.0e-15:
                break
        else:
            raise ArithmeticError("theta3 sum did not converge")
        return theta

    def value(
        self,
        coordinate: complex | np.ndarray,
        *,
        cut_side: CutSide = "upper",
    ) -> complex | np.ndarray:
        points = np.asarray(coordinate, dtype=np.complex128)
        scalar = points.ndim == 0
        if np.any(points == 0.0) or np.any(points == 1.0):
            raise ValueError("the elliptic block is singular at zero or one")
        q = np.asarray(elliptic_nome(points, cut_side=cut_side), dtype=np.complex128)
        theta = self._theta3(q)
        h_series = np.asarray(self.value_from_t(np.sqrt(q)), dtype=np.complex128)

        # Use the same lip for all explicit powers as for the nome.  If z is
        # on (1,infinity), 1-z consequently lands on the opposite lip of the
        # negative real axis, as analytic continuation requires.
        branch_points = _cut_lip_argument(points.reshape(-1), cut_side).reshape(
            points.shape
        )
        h0, hz, h1, hinfinity = self.effective_external_weights
        exponent = (
            (self.internal_weight - self.c0 - self.zero_coordinate_shift)
            * np.log(16.0 * q)
            + (self.c0 - h0 - hz + self.zero_coordinate_shift)
            * np.log(branch_points)
            + (self.c0 - hz - h1 + self.one_minus_coordinate_shift)
            * np.log(1.0 - branch_points)
            + self.theta_exponent * np.log(theta)
        )
        result = np.exp(exponent) * h_series
        if np.any(~np.isfinite(result)):
            raise ArithmeticError("non-finite prefactored elliptic block value")
        if scalar:
            return complex(result.item())
        return result


def _ramond_internal_primary_exponent(series: RamondSphereBlockSeries) -> complex:
    h_zero, h_moving, _, _ = series.external_weights
    state_zero, _ = series.ramond_states
    word_moving, _ = series.ns_words
    effective_zero = h_zero + 0.5 * ramond_twice_level(state_zero.word)
    effective_moving = h_moving + (
        0.5 * ns_twice_level(word_moving) if word_moving else 0.0
    )
    return complex(series.h_internal - effective_zero - effective_moving)


def _two_ramond_internal_ns_primary_exponent(
    series: TwoRamondInternalNSBlockSeries,
) -> complex:
    h_zero, h_moving, _, _ = series.external_weights
    word_zero, word_moving = series.ns_words
    effective_zero = h_zero + (
        0.5 * ns_twice_level(word_zero) if word_zero else 0.0
    )
    effective_moving = h_moving + (
        0.5 * ns_twice_level(word_moving) if word_moving else 0.0
    )
    return complex(series.h_internal - effective_zero - effective_moving)


def _four_ramond_primary_exponent(series: FourRamondNSBlockSeries) -> complex:
    h_zero, h_moving, _, _ = series.external_weights
    return complex(series.h_internal - h_zero - h_moving)


def _two_ramond_internal_ns_effective_weights(
    series: TwoRamondInternalNSBlockSeries,
) -> tuple[complex, complex, complex, complex]:
    weights = [complex(value) for value in series.external_weights]
    word_zero, word_moving = series.ns_words
    if word_zero:
        weights[0] += 0.5 * ns_twice_level(word_zero)
    if word_moving:
        weights[1] += 0.5 * ns_twice_level(word_moving)
    return tuple(weights)  # type: ignore[return-value]


def _internal_ramond_effective_weights(
    series: RamondSphereBlockSeries,
) -> tuple[complex, complex, complex, complex]:
    weights = [complex(value) for value in series.external_weights]
    state_zero, state_infinity = series.ramond_states
    word_moving, word_one = series.ns_words
    weights[0] += 0.5 * ramond_twice_level(state_zero.word)
    weights[1] += 0.5 * ns_twice_level(word_moving) if word_moving else 0.0
    weights[2] += 0.5 * ns_twice_level(word_one) if word_one else 0.0
    weights[3] += 0.5 * ramond_twice_level(state_infinity.word)
    return tuple(weights)  # type: ignore[return-value]


def _prefactored_internal_ns_series(
    series: RamondSphereBlockSeries | TwoRamondInternalNSBlockSeries | FourRamondNSBlockSeries,
    *,
    channel_coordinate: ChannelCoordinate,
) -> EllipticPrefactoredRamondSphereSeries:
    if isinstance(series, RamondSphereBlockSeries):
        effective = _internal_ramond_effective_weights(series)
        # Suchanek's RNNR internal-R prefactor (arXiv:1012.2974) has
        # (16q)^(h-c0-1/16) z^(c0-h0-hz+1/16).
        zero_coordinate_shift = 1.0 / 16.0
        one_minus_coordinate_shift = 0.0
        theta_shift = 0.5
    elif isinstance(series, TwoRamondInternalNSBlockSeries):
        effective = _two_ramond_internal_ns_effective_weights(series)
        zero_coordinate_shift = 0.0
        one_minus_coordinate_shift = 1.0 / 16.0
        theta_shift = 0.5
    else:
        effective = tuple(complex(value) for value in series.external_weights)
        zero_coordinate_shift = 0.0
        one_minus_coordinate_shift = 0.0
        theta_shift = 0.0

    offset = (
        0
        if isinstance(series, RamondSphereBlockSeries) or series.component == "even"
        else 1
    )
    direct = uniformize_direct_block(
        series.coefficients,
        level_step=2,
        level_offset=offset,
    )
    length = len(direct.t_coefficients)
    z_series, reduced_lambda, theta3 = _modular_series(length)
    one_minus_z = -z_series.copy()
    one_minus_z[0] += 1.0

    c = complex(series.c)
    h = complex(series.h_internal)
    c0 = (c - 1.5) / 24.0
    h_zero, h_moving, h_one, h_infinity = effective
    theta_exponent = (
        (c - 1.5) / 2.0
        - 4.0 * sum(effective, 0.0j)
        + theta_shift
    )

    # H = F / universal_prefactor after the common primary z power is
    # stripped.  Since z=16*t**2*u, the reduced inverse prefactor is
    # u**(h-c0) (1-z)**(h_moving+h_one-c0) theta3**(-A).
    inverse_reduced_prefactor = _series_multiply(
        _series_power(reduced_lambda, h - c0 - zero_coordinate_shift, length),
        _series_power(
            one_minus_z,
            h_moving + h_one - c0 - one_minus_coordinate_shift,
            length,
        ),
        length,
    )
    inverse_reduced_prefactor = _series_multiply(
        inverse_reduced_prefactor,
        _series_power(theta3, -theta_exponent, length),
        length,
    )
    h_coefficients = _series_multiply(
        inverse_reduced_prefactor,
        np.asarray(direct.t_coefficients, dtype=np.complex128),
        length,
    )
    return EllipticPrefactoredRamondSphereSeries(
        source=series,
        h_coefficients=tuple(complex(value) for value in h_coefficients),
        effective_external_weights=effective,
        internal_weight=h,
        central_charge=c,
        channel_coordinate=channel_coordinate,
        known_through_twice_level=direct.known_through_twice_level,
        zero_coordinate_shift=complex(zero_coordinate_shift),
        one_minus_coordinate_shift=complex(one_minus_coordinate_shift),
        theta_shift=complex(theta_shift),
    )


def elliptically_prefactor_internal_ramond_series(
    series: RamondSphereBlockSeries,
) -> EllipticPrefactoredRamondSphereSeries:
    """Factor Suchanek's mixed ``R--NS | NS--R`` internal-R asymptotic."""

    if not isinstance(series, RamondSphereBlockSeries):
        raise TypeError("series must be a RamondSphereBlockSeries")
    return _prefactored_internal_ns_series(series, channel_coordinate="z")


def elliptically_prefactor_two_ramond_internal_ns_series(
    series: TwoRamondInternalNSBlockSeries,
) -> EllipticPrefactoredRamondSphereSeries:
    """Factor the universal NS-channel elliptic asymptotic from a mixed block."""

    if not isinstance(series, TwoRamondInternalNSBlockSeries):
        raise TypeError("series must be a TwoRamondInternalNSBlockSeries")
    return _prefactored_internal_ns_series(series, channel_coordinate="w")


def elliptically_prefactor_four_ramond_ns_series(
    series: FourRamondNSBlockSeries,
) -> EllipticPrefactoredRamondSphereSeries:
    """Factor the universal NS-channel elliptic asymptotic from a four-R block."""

    if not isinstance(series, FourRamondNSBlockSeries):
        raise TypeError("series must be a FourRamondNSBlockSeries")
    return _prefactored_internal_ns_series(series, channel_coordinate="z")


def uniformize_internal_ramond_series(
    series: RamondSphereBlockSeries,
) -> UniformizedRamondSphereSeries:
    """Uniformize an ``R--NS | NS--R`` block with an internal R module."""

    if not isinstance(series, RamondSphereBlockSeries):
        raise TypeError("series must be a RamondSphereBlockSeries")
    nome = uniformize_direct_block(
        series.coefficients,
        level_step=2,
        level_offset=0,
    )
    return UniformizedRamondSphereSeries(
        source=series,
        nome_series=nome,
        primary_exponent=_ramond_internal_primary_exponent(series),
        series_kind="internal_ramond",
        channel_coordinate="z",
    )


def uniformize_two_ramond_internal_ns_series(
    series: TwoRamondInternalNSBlockSeries,
) -> UniformizedRamondSphereSeries:
    """Uniformize a crossed ``NS--NS | R--R`` block in coordinate ``w``."""

    if not isinstance(series, TwoRamondInternalNSBlockSeries):
        raise TypeError("series must be a TwoRamondInternalNSBlockSeries")
    offset = 0 if series.component == "even" else 1
    nome = uniformize_direct_block(
        series.coefficients,
        level_step=2,
        level_offset=offset,
    )
    return UniformizedRamondSphereSeries(
        source=series,
        nome_series=nome,
        primary_exponent=_two_ramond_internal_ns_primary_exponent(series),
        series_kind="two_ramond_internal_ns",
        channel_coordinate="w",
    )


def uniformize_four_ramond_ns_series(
    series: FourRamondNSBlockSeries,
) -> UniformizedRamondSphereSeries:
    """Uniformize one even or odd four-external-R/internal-NS component."""

    if not isinstance(series, FourRamondNSBlockSeries):
        raise TypeError("series must be a FourRamondNSBlockSeries")
    offset = 0 if series.component == "even" else 1
    nome = uniformize_direct_block(
        series.coefficients,
        level_step=2,
        level_offset=offset,
    )
    return UniformizedRamondSphereSeries(
        source=series,
        nome_series=nome,
        primary_exponent=_four_ramond_primary_exponent(series),
        series_kind="four_ramond_internal_ns",
        channel_coordinate="z",
    )


def uniformize_ramond_sphere_series(
    series: RamondSphereSeries,
) -> UniformizedRamondSphereSeries:
    """Dispatch to the adapter for any supported Ramond sphere dataclass."""

    if isinstance(series, RamondSphereBlockSeries):
        return uniformize_internal_ramond_series(series)
    if isinstance(series, TwoRamondInternalNSBlockSeries):
        return uniformize_two_ramond_internal_ns_series(series)
    if isinstance(series, FourRamondNSBlockSeries):
        return uniformize_four_ramond_ns_series(series)
    raise TypeError(
        "series must be RamondSphereBlockSeries, "
        "TwoRamondInternalNSBlockSeries, or FourRamondNSBlockSeries"
    )


__all__ = [
    "ChannelCoordinate",
    "EllipticPrefactoredRamondSphereSeries",
    "RamondSphereSeries",
    "RamondSphereSeriesKind",
    "UniformizedRamondSphereSeries",
    "elliptically_prefactor_four_ramond_ns_series",
    "elliptically_prefactor_internal_ramond_series",
    "elliptically_prefactor_two_ramond_internal_ns_series",
    "uniformize_four_ramond_ns_series",
    "uniformize_internal_ramond_series",
    "uniformize_ramond_sphere_series",
    "uniformize_two_ramond_internal_ns_series",
]
