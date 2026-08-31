"""Nonchiral RRRR and RRNSNS sphere correlators and crossing diagnostics.

The pants coefficients use the delta-normalized b=1 BRY convention.  The
production RRRR class takes coefficient-wise finite parts directly at
c=27/2; the generic and symmetric displaced-c classes are retained as
diagnostics for the coincident Kac poles.  Because BRY's two-point function
is pi*delta(P-P'), every public continuum integration uses the spectral
measure dP/pi.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Tuple

import mpmath

from mixed_ramond_sphere_blocks import (
    MixedNSExchangeSphereFourPointBlock,
    MixedRExchangeSphereFourPointBlock,
)
from ramond_sphere_blocks import RamondExternalSphereFourPointBlock, b_from_c
from self_dual_superconformal_blocks import (
    SelfDualRamondExternalSphereFourPointBlock,
)
from super_liouville_structure_constants import (
    ns_structure_constant,
    ns_tilde_structure_constant,
    rr_ns_chiral_structure_constant,
)


@lru_cache(maxsize=None)
def _spectral_legendre_interval(
    order: int, upper_limit: float
) -> Tuple[Tuple[float, float], ...]:
    """Return nodes and weights for the BRY spectral measure dP/pi."""

    if order < 2:
        raise ValueError("quadrature order must be at least 2")
    if upper_limit <= 0 or not math.isfinite(upper_limit):
        raise ValueError("p_max must be finite and positive")
    nodes, weights = mpmath.gauss_quadrature(order, "legendre")
    scale = upper_limit / 2.0
    return tuple(
        (
            scale * (float(node) + 1.0),
            scale * float(weight) / math.pi,
        )
        for node, weight in zip(nodes, weights)
    )


def _block_product(block, z: complex, order: int, parity: str | None = None) -> complex:
    if parity is None:
        value = block.elliptic_block(z, order)
    else:
        value = block.elliptic_block(z, order, parity)
    # The antiholomorphic Ward phases are the complex conjugates of the
    # holomorphic phases.  This matters for mixed NS/R odd blocks, whose
    # coefficients carry explicit exp(+/- i*pi/4).
    return value * value.conjugate()


class RRRRSphereCorrelator:
    """Four R^+ primaries, factorized on a continuum of NS modules."""

    def __init__(
        self,
        *,
        p1: float,
        p2: float,
        p3: float,
        p4: float,
        block_order: int = 8,
        structure_precision: int = 35,
        central_charge_shift: float = 1.0e-5,
    ) -> None:
        self.p1, self.p2, self.p3, self.p4 = map(float, (p1, p2, p3, p4))
        self.block_order = int(block_order)
        self.structure_precision = int(structure_precision)
        self.c = 13.5 + float(central_charge_shift)
        self.b = b_from_c(self.c)
        self._structure_cache = {}
        self._block_cache = {}

    def _structure_product(self, momentum: float, sign3: int, sign2: int) -> complex:
        key = (momentum, sign3, sign2)
        if key not in self._structure_cache:
            left = rr_ns_chiral_structure_constant(
                self.p4,
                self.p3,
                momentum,
                sign3,
                self.structure_precision,
            )
            right = rr_ns_chiral_structure_constant(
                self.p2,
                self.p1,
                momentum,
                sign2,
                self.structure_precision,
            )
            self._structure_cache[key] = left * right
        return self._structure_cache[key]

    def _block(
        self, momentum: float, sign3: int, sign2: int
    ) -> RamondExternalSphereFourPointBlock:
        key = (momentum, sign3, sign2)
        if key not in self._block_cache:
            self._block_cache[key] = (
                RamondExternalSphereFourPointBlock.from_liouville_momenta(
                    p1=self.p1,
                    p2=self.p2,
                    p3=self.p3,
                    p4=self.p4,
                    internal_momentum=momentum,
                    b=self.b,
                    sign3=sign3,
                    sign2=sign2,
                )
            )
        return self._block_cache[key]

    def momentum_integrand(self, momentum: float, z: complex) -> complex:
        if momentum == 0:
            return 0.0j
        total = 0.0j
        for sign3 in (1, -1):
            for sign2 in (1, -1):
                block = self._block(momentum, sign3, sign2)
                blocks = _block_product(
                    block, z, self.block_order, "even"
                ) + _block_product(block, z, self.block_order, "odd")
                total += self._structure_product(
                    momentum, sign3, sign2
                ) * blocks
        return 0.5 * total

    def evaluate(
        self, z: complex, *, p_max: float = 5.0, quadrature_order: int = 24
    ) -> complex:
        z = complex(z)
        return sum(
            weight * self.momentum_integrand(momentum, z)
            for momentum, weight in _spectral_legendre_interval(
                quadrature_order, p_max
            )
        )

    def crossed(self) -> "RRRRSphereCorrelator":
        """Return the 4,1,2,3 ordering used at 1-z."""

        return RRRRSphereCorrelator(
            p1=self.p3,
            p2=self.p2,
            p3=self.p1,
            p4=self.p4,
            block_order=self.block_order,
            structure_precision=self.structure_precision,
            central_charge_shift=self.c - 13.5,
        )


class SymmetricRRRRSphereCorrelator:
    """Even-in-epsilon RRRR diagnostic at the Type-0B central charge."""

    def __init__(
        self,
        *,
        p1: float,
        p2: float,
        p3: float,
        p4: float,
        block_order: int = 8,
        structure_precision: int = 35,
        central_charge_shift: float = 1.0e-4,
    ) -> None:
        shift = float(central_charge_shift)
        if not math.isfinite(shift) or shift <= 0.0:
            raise ValueError("central_charge_shift must be finite and positive")
        common = dict(
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            block_order=block_order,
            structure_precision=structure_precision,
        )
        self.p1, self.p2, self.p3, self.p4 = map(
            float, (p1, p2, p3, p4)
        )
        self.block_order = int(block_order)
        self.structure_precision = int(structure_precision)
        self.central_charge_shift = shift
        self.plus = RRRRSphereCorrelator(
            **common, central_charge_shift=shift
        )
        self.minus = RRRRSphereCorrelator(
            **common, central_charge_shift=-shift
        )

    def evaluate(
        self, z: complex, *, p_max: float = 5.0, quadrature_order: int = 24
    ) -> complex:
        plus = self.plus.evaluate(
            z, p_max=p_max, quadrature_order=quadrature_order
        )
        minus = self.minus.evaluate(
            z, p_max=p_max, quadrature_order=quadrature_order
        )
        return 0.5 * (plus + minus)

    def crossed(self) -> "SymmetricRRRRSphereCorrelator":
        """Return the 4,1,2,3 ordering used at 1-z."""

        return SymmetricRRRRSphereCorrelator(
            p1=self.p3,
            p2=self.p2,
            p3=self.p1,
            p4=self.p4,
            block_order=self.block_order,
            structure_precision=self.structure_precision,
            central_charge_shift=self.central_charge_shift,
        )


class SelfDualRRRRSphereCorrelator(RRRRSphereCorrelator):
    """RRRR correlator using coefficient-wise finite parts directly at b=1."""

    def __init__(
        self,
        *,
        p1: float,
        p2: float,
        p3: float,
        p4: float,
        block_order: int = 8,
        structure_precision: int = 35,
        finite_part_radius: float = 0.04,
        finite_part_check_radius: float = 0.05,
        finite_part_samples: int = 24,
    ) -> None:
        super().__init__(
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            block_order=block_order,
            structure_precision=structure_precision,
        )
        self.finite_part_radius = float(finite_part_radius)
        self.finite_part_check_radius = float(finite_part_check_radius)
        self.finite_part_samples = int(finite_part_samples)

    def _block(
        self, momentum: float, sign3: int, sign2: int
    ) -> SelfDualRamondExternalSphereFourPointBlock:
        key = (momentum, sign3, sign2)
        if key not in self._block_cache:
            self._block_cache[key] = (
                SelfDualRamondExternalSphereFourPointBlock(
                    p1=self.p1,
                    p2=self.p2,
                    p3=self.p3,
                    p4=self.p4,
                    internal_momentum=momentum,
                    sign3=sign3,
                    sign2=sign2,
                    radius=self.finite_part_radius,
                    check_radius=self.finite_part_check_radius,
                    samples=self.finite_part_samples,
                )
            )
        return self._block_cache[key]

    def crossed(self) -> "SelfDualRRRRSphereCorrelator":
        """Return the exact-self-dual 4,1,2,3 ordering used at 1-z."""

        return SelfDualRRRRSphereCorrelator(
            p1=self.p3,
            p2=self.p2,
            p3=self.p1,
            p4=self.p4,
            block_order=self.block_order,
            structure_precision=self.structure_precision,
            finite_part_radius=self.finite_part_radius,
            finite_part_check_radius=self.finite_part_check_radius,
            finite_part_samples=self.finite_part_samples,
        )


class RRNNMixedChannelCorrelator:
    """Compare <NS_4 NS_3 R_2 R_1> in its NS and crossed R channels."""

    def __init__(
        self,
        *,
        p1_r: float,
        p2_r: float,
        p3_ns: float,
        p4_ns: float,
        block_order: int = 8,
        structure_precision: int = 35,
        central_charge_shift: float = 1.0e-5,
    ) -> None:
        self.p1_r = float(p1_r)
        self.p2_r = float(p2_r)
        self.p3_ns = float(p3_ns)
        self.p4_ns = float(p4_ns)
        self.block_order = int(block_order)
        self.structure_precision = int(structure_precision)
        self.c = 13.5 + float(central_charge_shift)
        self.b = b_from_c(self.c)
        self._cache = {}

    def _ns_channel_integrand(self, momentum: float, z: complex) -> complex:
        if momentum == 0:
            return 0.0j
        key = ("ns-structure", momentum)
        if key not in self._cache:
            self._cache[key] = (
                ns_structure_constant(
                    self.p4_ns,
                    self.p3_ns,
                    momentum,
                    self.structure_precision,
                ),
                ns_tilde_structure_constant(
                    self.p4_ns,
                    self.p3_ns,
                    momentum,
                    self.structure_precision,
                ),
            )
        c_left, ct_left = self._cache[key]
        total = 0.0j
        for sign2 in (1, -1):
            structure_key = ("ns-right", momentum, sign2)
            if structure_key not in self._cache:
                self._cache[structure_key] = rr_ns_chiral_structure_constant(
                    self.p2_r,
                    self.p1_r,
                    momentum,
                    sign2,
                    self.structure_precision,
                )
            block_key = ("ns-block", momentum, sign2)
            if block_key not in self._cache:
                self._cache[block_key] = MixedNSExchangeSphereFourPointBlock(
                    b=self.b,
                    p1_r=self.p1_r,
                    p2_r=self.p2_r,
                    p3_ns=self.p3_ns,
                    p4_ns=self.p4_ns,
                    internal_momentum=momentum,
                    sign2=sign2,
                )
            block = self._cache[block_key]
            right = self._cache[structure_key]
            # Suchanek writes this coefficient as -i * C_tilde_HJS.  BRY's
            # real component convention absorbs that phase:
            # C_tilde_BRY = -i * C_tilde_HJS.
            total += right * (
                c_left * _block_product(block, z, self.block_order, "even")
                + ct_left * _block_product(block, z, self.block_order, "odd")
            )
        return total

    def _r_channel_integrand(self, momentum: float, z: complex) -> complex:
        if momentum == 0:
            return 0.0j
        total = 0.0j
        for sign3 in (1, -1):
            for sign2 in (1, -1):
                left_key = ("r-left", momentum, sign3)
                if left_key not in self._cache:
                    self._cache[left_key] = rr_ns_chiral_structure_constant(
                        self.p1_r,
                        momentum,
                        self.p4_ns,
                        sign3,
                        self.structure_precision,
                    )
                right_key = ("r-right", momentum, sign2)
                if right_key not in self._cache:
                    self._cache[right_key] = rr_ns_chiral_structure_constant(
                        momentum,
                        self.p2_r,
                        self.p3_ns,
                        sign2,
                        self.structure_precision,
                    )
                block_key = ("r-block", momentum, sign3, sign2)
                if block_key not in self._cache:
                    self._cache[block_key] = MixedRExchangeSphereFourPointBlock(
                        b=self.b,
                        p1_ns=self.p3_ns,
                        p2_r=self.p2_r,
                        p3_r=self.p1_r,
                        p4_ns=self.p4_ns,
                        internal_momentum=momentum,
                        sign3=sign3,
                        sign2=sign2,
                    )
                block = self._cache[block_key]
                total += (
                    self._cache[left_key]
                    * self._cache[right_key]
                    * _block_product(block, z, self.block_order)
                )
        return total

    def evaluate_ns_channel(
        self, z: complex, *, p_max: float = 5.0, quadrature_order: int = 24
    ) -> complex:
        z = complex(z)
        return sum(
            weight * self._ns_channel_integrand(momentum, z)
            for momentum, weight in _spectral_legendre_interval(
                quadrature_order, p_max
            )
        )

    def evaluate_crossed_r_channel(
        self, z: complex, *, p_max: float = 5.0, quadrature_order: int = 24
    ) -> complex:
        crossed_z = 1.0 - complex(z)
        return sum(
            weight * self._r_channel_integrand(momentum, crossed_z)
            for momentum, weight in _spectral_legendre_interval(
                quadrature_order, p_max
            )
        )


class SymmetricRRNNMixedChannelCorrelator:
    """Even-in-epsilon extrapolation of the mixed correlator to b=1.

    At the rational point c=27/2, individual Kac-recursion terms contain
    resonant poles although their sum is finite.  Averaging the evaluations
    at c=27/2+epsilon and c=27/2-epsilon cancels the leading regulator
    dependence and is substantially better conditioned at high recursion
    order than taking epsilon too small in double precision.
    """

    def __init__(
        self,
        *,
        p1_r: float,
        p2_r: float,
        p3_ns: float,
        p4_ns: float,
        block_order: int = 8,
        structure_precision: int = 35,
        central_charge_shift: float = 1.0e-4,
    ) -> None:
        shift = float(central_charge_shift)
        if not math.isfinite(shift) or shift <= 0.0:
            raise ValueError("central_charge_shift must be finite and positive")
        common = dict(
            p1_r=p1_r,
            p2_r=p2_r,
            p3_ns=p3_ns,
            p4_ns=p4_ns,
            block_order=block_order,
            structure_precision=structure_precision,
        )
        self.central_charge_shift = shift
        self.plus = RRNNMixedChannelCorrelator(
            **common, central_charge_shift=shift
        )
        self.minus = RRNNMixedChannelCorrelator(
            **common, central_charge_shift=-shift
        )

    def evaluate_ns_channel(
        self, z: complex, *, p_max: float = 5.0, quadrature_order: int = 24
    ) -> complex:
        plus = self.plus.evaluate_ns_channel(
            z, p_max=p_max, quadrature_order=quadrature_order
        )
        minus = self.minus.evaluate_ns_channel(
            z, p_max=p_max, quadrature_order=quadrature_order
        )
        return 0.5 * (plus + minus)

    def evaluate_crossed_r_channel(
        self, z: complex, *, p_max: float = 5.0, quadrature_order: int = 24
    ) -> complex:
        plus = self.plus.evaluate_crossed_r_channel(
            z, p_max=p_max, quadrature_order=quadrature_order
        )
        minus = self.minus.evaluate_crossed_r_channel(
            z, p_max=p_max, quadrature_order=quadrature_order
        )
        return 0.5 * (plus + minus)


def relative_crossing_error(left: complex, right: complex) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)


__all__ = [
    "RRRRSphereCorrelator",
    "SelfDualRRRRSphereCorrelator",
    "SymmetricRRRRSphereCorrelator",
    "RRNNMixedChannelCorrelator",
    "SymmetricRRNNMixedChannelCorrelator",
    "relative_crossing_error",
]
