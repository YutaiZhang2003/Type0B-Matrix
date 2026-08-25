#!/usr/bin/env python3
"""Type-0B NS Liouville sphere multipoint correlators from c-recursion.

This module contracts :class:`NSSphereLinearCRecursion` blocks with the
self-dual super-Liouville structure constants and integrates every internal
NS momentum with the BRY ``dP/pi`` measure.  It computes the Euclidean
bottom-component matter correlator.  It is not by itself a complete Type-0B
string amplitude: the timelike matter, ghosts, picture-changing insertions,
and supermoduli measure are deliberately outside this layer.

All punctures supplied to :class:`BRYNSSphereMultipointCorrelator` are finite.
A channel is a permutation of their labels.  The first, penultimate, and
last punctures in that order are mapped to ``0``, ``1``, and ``infinity``.
For standard-frame points

``(0, z_2, ..., z_(N-2), 1, infinity)``,

the linear plumbing coordinates are

``q_i = z_(i+1)/z_(i+2)`` and ``q_(N-3) = z_(N-2)``.

The code restores both the leading comb-block powers and the primary-field
Jacobian needed to compare different standard frames at the same physical
puncture configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math
from typing import Iterable, Sequence, Union

import mpmath

from ns_multipoint_c_recursion import NSSphereLinearCRecursion
from super_liouville_structure_constants import (
    ns_structure_constant,
    ns_tilde_structure_constant,
)


Number = Union[complex, float]


def _finite_complex(name: str, value: Number) -> complex:
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    return result


def _real_nonnegative(name: str, value: Number) -> float:
    result = _finite_complex(name, value)
    if abs(result.imag) > 1.0e-14 or result.real < 0:
        raise ValueError(f"{name} must be a non-negative real number")
    return result.real


def _validate_order(order: Sequence[int], point_count: int) -> tuple[int, ...]:
    result = tuple(order)
    if len(result) != point_count or set(result) != set(range(point_count)):
        raise ValueError(
            f"order must be a permutation of 0,...,{point_count - 1}"
        )
    return result


def _even_sector_assignments(vertex_count: int) -> Iterable[tuple[int, ...]]:
    return (
        sectors
        for sectors in product((0, 1), repeat=vertex_count)
        if sum(sectors) % 2 == 0
    )


@lru_cache(maxsize=None)
def _legendre_interval(
    order: int, upper_limit: float
) -> tuple[tuple[float, float], ...]:
    if not isinstance(order, int) or order < 2:
        raise ValueError("quadrature_order must be an integer at least 2")
    if upper_limit <= 0 or not math.isfinite(upper_limit):
        raise ValueError("p_max must be positive and finite")
    nodes, weights = mpmath.gauss_quadrature(order, "legendre")
    scale = upper_limit / 2.0
    return tuple(
        (scale * (float(node) + 1.0), scale * float(weight))
        for node, weight in zip(nodes, weights)
    )


@dataclass(frozen=True)
class SphereCombFrame:
    """One standard comb frame for a fixed physical puncture configuration."""

    order: tuple[int, ...]
    q_values: tuple[complex, ...]
    standard_finite_points: tuple[complex, ...]
    covariance_factor: float

    @property
    def maximum_plumbing_modulus(self) -> float:
        return max(map(abs, self.q_values))


@dataclass(frozen=True)
class SphereChannelValue:
    """Integrated standard-frame and physical-frame channel values."""

    frame: SphereCombFrame
    standard_value: complex
    physical_value: complex


@dataclass(frozen=True)
class SphereCrossingResult:
    """Comparison of two independently sewn sphere channels."""

    left: SphereChannelValue
    right: SphereChannelValue

    @property
    def absolute_residual(self) -> float:
        return abs(self.left.physical_value - self.right.physical_value)

    @property
    def relative_residual(self) -> float:
        return self.absolute_residual / max(
            abs(self.left.physical_value),
            abs(self.right.physical_value),
            1.0e-300,
        )


def sphere_comb_frame(
    *,
    points: Sequence[Number],
    weights: Sequence[Number],
    order: Sequence[int],
) -> SphereCombFrame:
    r"""Map a finite puncture configuration to one standard comb frame.

    If ``a``, ``b``, and ``c`` are the first, penultimate, and last ordered
    punctures, the map is

    ``f(z) = (b-c)/(b-a) * (z-a)/(z-c)``.

    The covariance factor ``J`` obeys

    ``G_physical(points) = J * G_standard(0,...,1,infinity)``

    for scalar primaries of left/right weights ``(d_i,d_i)``.
    """

    point_tuple = tuple(
        _finite_complex(f"points[{index}]", value)
        for index, value in enumerate(points)
    )
    weight_tuple = tuple(
        _real_nonnegative(f"weights[{index}]", value)
        for index, value in enumerate(weights)
    )
    if len(point_tuple) < 4:
        raise ValueError("a sphere comb frame requires at least four points")
    if len(weight_tuple) != len(point_tuple):
        raise ValueError("weights and points must have the same length")
    for first in range(len(point_tuple)):
        for second in range(first):
            scale = max(1.0, abs(point_tuple[first]), abs(point_tuple[second]))
            if abs(point_tuple[first] - point_tuple[second]) <= 1.0e-14 * scale:
                raise ValueError("sphere punctures must be pairwise distinct")

    permutation = _validate_order(order, len(point_tuple))
    a = point_tuple[permutation[0]]
    b = point_tuple[permutation[-2]]
    c = point_tuple[permutation[-1]]
    normalization = (b - c) / (b - a)

    def transform(z: complex) -> complex:
        return normalization * (z - a) / (z - c)

    standard_finite = tuple(
        transform(point_tuple[index]) for index in permutation[:-1]
    )
    variable_points = standard_finite[1:-1]
    q_values = tuple(
        variable_points[index] / variable_points[index + 1]
        for index in range(len(variable_points) - 1)
    ) + (variable_points[-1],)

    # Near c, f(z)=K/(z-c)+O(1).  Combining the primary Jacobian
    # |f'(z)|^(2d_c) with the definition of the operator at infinity gives
    # the finite factor |K|^(-2d_c).
    pole_coefficient = normalization * (c - a)
    log_covariance = -2.0 * weight_tuple[permutation[-1]] * math.log(
        abs(pole_coefficient)
    )
    derivative_numerator = normalization * (a - c)
    for index in permutation[:-1]:
        derivative = derivative_numerator / (point_tuple[index] - c) ** 2
        log_covariance += 2.0 * weight_tuple[index] * math.log(abs(derivative))
    covariance = math.exp(log_covariance)

    return SphereCombFrame(
        order=permutation,
        q_values=q_values,
        standard_finite_points=standard_finite,
        covariance_factor=covariance,
    )


class BRYNSSphereMultipointCorrelator:
    r"""Evaluate bottom-component NS Liouville sphere ``N``-point functions.

    A channel contribution is

    ``integral prod_e(dP_e/pi) sum_alpha prod_v C_v(alpha_v) |F_alpha|^2``,

    where ``C_v(0)=C`` and ``C_v(1)=tilde C``.  Only even-total vertex
    sectors occur for bottom external components.  The chiral block is the
    finite local-plumbing series produced by the multipoint c-recursion.
    """

    def __init__(
        self,
        *,
        momenta: Sequence[Number],
        points: Sequence[Number],
        max_twice_levels: Sequence[int] | None = None,
        max_total_twice_level: int | None = None,
        recursion_max_twice_level: int | None = None,
        structure_precision: int = 30,
        central_charge_shift: float = 1.0e-5,
        block_working_precision: int = 60,
        pole_tolerance: float = 1.0e-30,
    ) -> None:
        if len(momenta) < 4:
            raise ValueError("a sphere correlator requires at least four momenta")
        self.momenta = tuple(
            _real_nonnegative(f"momenta[{index}]", value)
            for index, value in enumerate(momenta)
        )
        self.points = tuple(
            _finite_complex(f"points[{index}]", value)
            for index, value in enumerate(points)
        )
        if len(self.points) != len(self.momenta):
            raise ValueError("momenta and points must have the same length")
        self.edge_count = len(self.momenta) - 3
        self.vertex_count = self.edge_count + 1

        if max_twice_levels is None:
            max_twice_levels = (12,) * self.edge_count
        maxima = tuple(max_twice_levels)
        if len(maxima) != self.edge_count or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in maxima
        ):
            raise ValueError(
                f"max_twice_levels must contain {self.edge_count} "
                "non-negative integers"
            )
        if max_total_twice_level is not None and (
            not isinstance(max_total_twice_level, int)
            or isinstance(max_total_twice_level, bool)
            or max_total_twice_level < 0
        ):
            raise ValueError(
                "max_total_twice_level must be a non-negative integer or None"
            )
        if recursion_max_twice_level is not None and (
            not isinstance(recursion_max_twice_level, int)
            or isinstance(recursion_max_twice_level, bool)
            or recursion_max_twice_level < 0
        ):
            raise ValueError(
                "recursion_max_twice_level must be a non-negative integer or None"
            )
        if structure_precision < 15:
            raise ValueError("structure_precision must be at least 15 digits")
        if central_charge_shift < 0 or not math.isfinite(central_charge_shift):
            raise ValueError("central_charge_shift must be finite and non-negative")
        if block_working_precision < 30:
            raise ValueError("block_working_precision must be at least 30 digits")
        if pole_tolerance <= 0 or not math.isfinite(pole_tolerance):
            raise ValueError("pole_tolerance must be finite and positive")

        self.max_twice_levels = maxima
        self.max_total_twice_level = max_total_twice_level
        self.recursion_max_twice_level = recursion_max_twice_level
        self.structure_precision = int(structure_precision)
        self.central_charge_shift = float(central_charge_shift)
        self.block_working_precision = int(block_working_precision)
        self.pole_tolerance = float(pole_tolerance)
        self._block_cache: dict[tuple[object, ...], NSSphereLinearCRecursion] = {}
        self._structure_cache: dict[tuple[object, ...], complex] = {}

    @property
    def block_central_charge(self) -> float:
        return 13.5 + self.central_charge_shift

    def block_weight(self, momentum: Number) -> float:
        momentum_value = _real_nonnegative("momentum", momentum)
        q_squared = self.block_central_charge / 3.0 - 0.5
        return 0.5 * (q_squared / 4.0 + momentum_value * momentum_value)

    @property
    def external_weights(self) -> tuple[float, ...]:
        return tuple(self.block_weight(momentum) for momentum in self.momenta)

    def frame(self, order: Sequence[int]) -> SphereCombFrame:
        return sphere_comb_frame(
            points=self.points,
            weights=self.external_weights,
            order=order,
        )

    def _structure_constant(
        self, first: float, second: float, third: float, sector: int
    ) -> complex:
        key = (sector, tuple(sorted((first, second, third))))
        if key not in self._structure_cache:
            function = (
                ns_structure_constant
                if sector == 0
                else ns_tilde_structure_constant
            )
            self._structure_cache[key] = function(
                first, second, third, self.structure_precision
            )
        return self._structure_cache[key]

    def _structure_product(
        self,
        ordered_external_momenta: tuple[float, ...],
        internal_momenta: tuple[float, ...],
        sectors: tuple[int, ...],
    ) -> complex:
        result = self._structure_constant(
            ordered_external_momenta[0],
            ordered_external_momenta[1],
            internal_momenta[0],
            sectors[0],
        )
        for vertex in range(1, self.edge_count):
            result *= self._structure_constant(
                internal_momenta[vertex - 1],
                ordered_external_momenta[vertex + 1],
                internal_momenta[vertex],
                sectors[vertex],
            )
        result *= self._structure_constant(
            internal_momenta[-1],
            ordered_external_momenta[-2],
            ordered_external_momenta[-1],
            sectors[-1],
        )
        return result

    def _block(
        self,
        frame: SphereCombFrame,
        internal_momenta: tuple[float, ...],
        sectors: tuple[int, ...],
    ) -> NSSphereLinearCRecursion:
        key = (frame.order, internal_momenta, sectors)
        if key not in self._block_cache:
            ordered_weights = tuple(
                self.external_weights[index] for index in frame.order
            )
            self._block_cache[key] = NSSphereLinearCRecursion(
                central_charge=self.block_central_charge,
                external_weights=ordered_weights,
                internal_weights=tuple(
                    self.block_weight(momentum) for momentum in internal_momenta
                ),
                vertex_sectors=sectors,
                working_precision=self.block_working_precision,
                pole_tolerance=self.pole_tolerance,
            )
        return self._block_cache[key]

    def chiral_block(
        self,
        frame: SphereCombFrame,
        internal_momenta: Sequence[Number],
        sectors: Sequence[int],
        *,
        max_twice_levels: Sequence[int] | None = None,
        max_total_twice_level: int | None = None,
        recursion_max_twice_level: int | None = None,
    ) -> complex:
        """Return one full chiral comb block, including leading powers."""

        momenta = tuple(
            _real_nonnegative(f"internal_momenta[{index}]", value)
            for index, value in enumerate(internal_momenta)
        )
        if len(momenta) != self.edge_count:
            raise ValueError(
                f"internal_momenta must contain {self.edge_count} entries"
            )
        sector_tuple = tuple(sectors)
        if (
            len(sector_tuple) != self.vertex_count
            or any(value not in (0, 1) for value in sector_tuple)
            or sum(sector_tuple) % 2
        ):
            raise ValueError(
                "sectors must be an even-total zero/one assignment with one "
                "entry per trivalent vertex"
            )
        maxima = self.max_twice_levels if max_twice_levels is None else tuple(
            max_twice_levels
        )
        total_cutoff = (
            self.max_total_twice_level
            if max_total_twice_level is None
            else max_total_twice_level
        )
        recursion_cutoff = (
            self.recursion_max_twice_level
            if recursion_max_twice_level is None
            else recursion_max_twice_level
        )
        block = self._block(frame, momenta, sector_tuple)
        with mpmath.workdps(self.block_working_precision):
            if recursion_cutoff is None:
                reduced = block.series_value(
                    frame.q_values,
                    maxima,
                    max_total_twice_level=total_cutoff,
                )
            else:
                reduced = block.recursive_series_value(
                    frame.q_values,
                    recursion_cutoff,
                    maxima,
                    global_max_total_twice_level=total_cutoff,
                )
            ordered_external_weights = tuple(
                mpmath.mpf(self.external_weights[index]) for index in frame.order
            )
            internal_weights = tuple(
                mpmath.mpf(self.block_weight(momentum)) for momentum in momenta
            )
            leading = mpmath.mpc(1)
            cumulative_external_weight = mpmath.mpf(ordered_external_weights[0])
            for edge, (q_value, internal_weight) in enumerate(
                zip(frame.q_values, internal_weights)
            ):
                cumulative_external_weight += ordered_external_weights[edge + 1]
                leading *= mpmath.mpc(q_value) ** (
                    internal_weight - cumulative_external_weight
                )
            return complex(leading * reduced)

    def momentum_integrand(
        self,
        frame: SphereCombFrame,
        internal_momenta: Sequence[Number],
        *,
        max_twice_levels: Sequence[int] | None = None,
        max_total_twice_level: int | None = None,
    ) -> complex:
        """Return the complete ``prod_e dP_e`` integrand in one channel."""

        momenta = tuple(
            _real_nonnegative(f"internal_momenta[{index}]", value)
            for index, value in enumerate(internal_momenta)
        )
        if len(momenta) != self.edge_count:
            raise ValueError(
                f"internal_momenta must contain {self.edge_count} entries"
            )
        ordered_external_momenta = tuple(
            self.momenta[index] for index in frame.order
        )
        total = 0.0j
        for sectors in _even_sector_assignments(self.vertex_count):
            chiral = self.chiral_block(
                frame,
                momenta,
                sectors,
                max_twice_levels=max_twice_levels,
                max_total_twice_level=max_total_twice_level,
            )
            total += self._structure_product(
                ordered_external_momenta, momenta, sectors
            ) * abs(chiral) ** 2
        return total / math.pi**self.edge_count

    def evaluate_channel(
        self,
        order: Sequence[int],
        *,
        p_max: float = 5.0,
        quadrature_order: int | Sequence[int] = 12,
        max_twice_levels: Sequence[int] | None = None,
        max_total_twice_level: int | None = None,
    ) -> SphereChannelValue:
        """Integrate one independently assembled comb channel."""

        frame = self.frame(order)
        if frame.maximum_plumbing_modulus >= 1.0:
            raise ValueError(
                "the requested channel is outside the local plumbing-series "
                f"domain: max |q_i|={frame.maximum_plumbing_modulus:.8g}"
            )
        if isinstance(quadrature_order, int):
            # Equal internal weights create coincident fixed-c poles on
            # different edges.  The diagonal has zero continuum measure but
            # would be sampled by a tensor product of identical 1D rules.
            # Staggering the Gauss orders removes that artificial collision.
            quadrature_orders = tuple(
                quadrature_order + edge for edge in range(self.edge_count)
            )
        else:
            quadrature_orders = tuple(quadrature_order)
            if len(quadrature_orders) != self.edge_count:
                raise ValueError(
                    f"quadrature_order must contain {self.edge_count} entries"
                )
        node_sets = tuple(
            _legendre_interval(rule_order, float(p_max))
            for rule_order in quadrature_orders
        )
        total = 0.0j
        for node_tuple in product(*node_sets):
            internal_momenta = tuple(node for node, _ in node_tuple)
            quadrature_weight = math.prod(weight for _, weight in node_tuple)
            total += quadrature_weight * self.momentum_integrand(
                frame,
                internal_momenta,
                max_twice_levels=max_twice_levels,
                max_total_twice_level=max_total_twice_level,
            )
        return SphereChannelValue(
            frame=frame,
            standard_value=complex(total),
            physical_value=complex(frame.covariance_factor * total),
        )

    def compare_channels(
        self,
        left_order: Sequence[int],
        right_order: Sequence[int],
        **evaluation_options,
    ) -> SphereCrossingResult:
        """Evaluate two channels independently at the same physical points."""

        return SphereCrossingResult(
            left=self.evaluate_channel(left_order, **evaluation_options),
            right=self.evaluate_channel(right_order, **evaluation_options),
        )


__all__ = [
    "BRYNSSphereMultipointCorrelator",
    "SphereChannelValue",
    "SphereCombFrame",
    "SphereCrossingResult",
    "sphere_comb_frame",
]
