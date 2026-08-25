"""Type-0B NS torus two-point contributions in the necklace channel.

The NS class assembles the coefficient-wise finite part of the two-edge
internal-weight recursion.  The R-handle class sums the compatible HJS sign
sectors of the paired beta-pole recursion.  The older direct leading NS
block remains available as a cheap degeneration and normalization benchmark.

Together these classes assemble the bottom-component NS external
contributions with either NS or R internal edges.  A complete physical
genus-one observable still requires the intended spin-structure sum; two
external R insertions lead to the separate mixed NS--R necklace.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy

from super_liouville_structure_constants import (
    ns_structure_constant,
    rr_ns_chiral_structure_constant,
)
from superconformal_blocks import central_charge, ns_liouville_weight
from superconformal_torus_blocks import (
    NSPlumbingParameter,
    RamondPlumbingParameter,
)
from superconformal_torus_two_point import (
    NSTorusTwoPointLeadingBlock,
    SelfDualNSTorusTwoPointHRecursionBlock,
    SelfDualRamondTorusTwoPointBetaRecursionBlock,
)


class Type0BNSTorusTwoPointLeadingCorrelator:
    """Double ``dP/pi`` integral for two bottom-component NS primaries."""

    def __init__(
        self,
        *,
        external_momentum_1: float,
        external_momentum_2: float,
        p_max: float = 3.5,
        quadrature_order: int = 12,
        structure_precision: int = 25,
    ) -> None:
        if p_max <= 0.0 or not math.isfinite(p_max):
            raise ValueError("p_max must be finite and positive")
        if quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two")
        if structure_precision < 15:
            raise ValueError("structure_precision must be at least 15")
        self.external_momentum_1 = float(external_momentum_1)
        self.external_momentum_2 = float(external_momentum_2)
        self.p_max = float(p_max)
        self.quadrature_order = int(quadrature_order)
        self.structure_precision = int(structure_precision)
        self.central_charge = central_charge(1.0)
        self.external_weight_1 = ns_liouville_weight(
            self.external_momentum_1, 1.0
        )
        self.external_weight_2 = ns_liouville_weight(
            self.external_momentum_2, 1.0
        )
        self._structure_cache: Dict[Tuple[float, float], complex] = {}

    def _structure_product(
        self, momentum_1: float, momentum_2: float
    ) -> complex:
        key = (momentum_1, momentum_2)
        if key not in self._structure_cache:
            self._structure_cache[key] = ns_structure_constant(
                momentum_1,
                self.external_momentum_1,
                momentum_2,
                self.structure_precision,
            ) * ns_structure_constant(
                momentum_2,
                self.external_momentum_2,
                momentum_1,
                self.structure_precision,
            )
        return self._structure_cache[key]

    def momentum_integrand(
        self,
        momentum_1: float,
        momentum_2: float,
        plumbing_1: NSPlumbingParameter,
        plumbing_2: NSPlumbingParameter,
    ) -> complex:
        """Return the integrand before the two ``dP/pi`` measures."""

        block = NSTorusTwoPointLeadingBlock(
            central_charge=self.central_charge,
            internal_weight_1=ns_liouville_weight(momentum_1, 1.0),
            internal_weight_2=ns_liouville_weight(momentum_2, 1.0),
            external_weight_1=self.external_weight_1,
            external_weight_2=self.external_weight_2,
        )
        return self._structure_product(momentum_1, momentum_2) * abs(
            block.chiral_block(plumbing_1, plumbing_2)
        ) ** 2

    def evaluate(
        self,
        plumbing_1: NSPlumbingParameter,
        plumbing_2: NSPlumbingParameter,
    ) -> complex:
        """Evaluate the leading double spectral integral."""

        nodes, weights = numpy.polynomial.legendre.leggauss(
            self.quadrature_order
        )
        midpoint = 0.5 * self.p_max
        channels = tuple(
            (
                midpoint * (float(node) + 1.0),
                midpoint * float(weight) / math.pi,
            )
            for node, weight in zip(nodes, weights)
        )
        return sum(
            weight_1
            * weight_2
            * self.momentum_integrand(
                momentum_1,
                momentum_2,
                plumbing_1,
                plumbing_2,
            )
            for momentum_1, weight_1 in channels
            for momentum_2, weight_2 in channels
        )


class Type0BNSTorusTwoPointHRecursionCorrelator:
    """The NS ``C*C`` double integral using the self-dual h-recursion.

    ``max_twice_level_i`` truncates the descendant level on edge ``i`` in
    units of one half.  For example, a value of four keeps levels through
    two on that edge.  The finite part at ``b=1`` is taken coefficient by
    coefficient before the bivariate series is evaluated.
    """

    def __init__(
        self,
        *,
        external_momentum_1: float,
        external_momentum_2: float,
        max_twice_level_1: int = 4,
        max_twice_level_2: int = 4,
        p_max: float = 3.5,
        quadrature_order: int = 8,
        structure_precision: int = 25,
        finite_part_radius: float = 0.04,
        finite_part_check_radius: float = 0.05,
        finite_part_samples: int = 24,
        difference_radius: float = 0.03,
        difference_samples: int = 16,
    ) -> None:
        if (
            not isinstance(max_twice_level_1, int)
            or max_twice_level_1 < 0
        ):
            raise ValueError(
                "max_twice_level_1 must be a nonnegative integer"
            )
        if (
            not isinstance(max_twice_level_2, int)
            or max_twice_level_2 < 0
        ):
            raise ValueError(
                "max_twice_level_2 must be a nonnegative integer"
            )
        if p_max <= 0.0 or not math.isfinite(p_max):
            raise ValueError("p_max must be finite and positive")
        if quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two")
        if structure_precision < 15:
            raise ValueError("structure_precision must be at least 15")

        self.external_momentum_1 = float(external_momentum_1)
        self.external_momentum_2 = float(external_momentum_2)
        self.max_twice_level_1 = max_twice_level_1
        self.max_twice_level_2 = max_twice_level_2
        self.p_max = float(p_max)
        self.quadrature_order = int(quadrature_order)
        self.structure_precision = int(structure_precision)
        self.finite_part_radius = float(finite_part_radius)
        self.finite_part_check_radius = float(finite_part_check_radius)
        self.finite_part_samples = int(finite_part_samples)
        self.difference_radius = float(difference_radius)
        self.difference_samples = int(difference_samples)
        self.central_charge = central_charge(1.0)
        self._structure_cache: Dict[Tuple[float, float], complex] = {}
        self._block_cache: Dict[
            Tuple[float, float], SelfDualNSTorusTwoPointHRecursionBlock
        ] = {}

    def _structure_product(
        self, momentum_1: float, momentum_2: float
    ) -> complex:
        key = (momentum_1, momentum_2)
        if key not in self._structure_cache:
            self._structure_cache[key] = ns_structure_constant(
                momentum_1,
                self.external_momentum_1,
                momentum_2,
                self.structure_precision,
            ) * ns_structure_constant(
                momentum_2,
                self.external_momentum_2,
                momentum_1,
                self.structure_precision,
            )
        return self._structure_cache[key]

    def _block(
        self, momentum_1: float, momentum_2: float
    ) -> SelfDualNSTorusTwoPointHRecursionBlock:
        key = (momentum_1, momentum_2)
        if key not in self._block_cache:
            self._block_cache[key] = (
                SelfDualNSTorusTwoPointHRecursionBlock(
                    internal_momentum_1=momentum_1,
                    internal_momentum_2=momentum_2,
                    external_momentum_1=self.external_momentum_1,
                    external_momentum_2=self.external_momentum_2,
                    radius=self.finite_part_radius,
                    check_radius=self.finite_part_check_radius,
                    samples=self.finite_part_samples,
                    difference_radius=self.difference_radius,
                    difference_samples=self.difference_samples,
                )
            )
        return self._block_cache[key]

    def momentum_integrand(
        self,
        momentum_1: float,
        momentum_2: float,
        plumbing_1: NSPlumbingParameter,
        plumbing_2: NSPlumbingParameter,
    ) -> complex:
        """Return the truncated recursive integrand before ``dP_i/pi``."""

        block = self._block(float(momentum_1), float(momentum_2))
        return self._structure_product(momentum_1, momentum_2) * abs(
            block.chiral_block(
                plumbing_1,
                plumbing_2,
                self.max_twice_level_1,
                self.max_twice_level_2,
            )
        ) ** 2

    def evaluate(
        self,
        plumbing_1: NSPlumbingParameter,
        plumbing_2: NSPlumbingParameter,
    ) -> complex:
        """Evaluate the truncated double spectral integral."""

        nodes, weights = numpy.polynomial.legendre.leggauss(
            self.quadrature_order
        )
        midpoint = 0.5 * self.p_max
        channels = tuple(
            (
                midpoint * (float(node) + 1.0),
                midpoint * float(weight) / math.pi,
            )
            for node, weight in zip(nodes, weights)
        )
        return sum(
            weight_1
            * weight_2
            * self.momentum_integrand(
                momentum_1,
                momentum_2,
                plumbing_1,
                plumbing_2,
            )
            for momentum_1, weight_1 in channels
            for momentum_2, weight_2 in channels
        )


class Type0BRamondTorusTwoPointBetaRecursionCorrelator:
    """Two-NS contribution with both necklace edges in the R sector.

    The two compatible HJS sign branches are summed.  Each term carries the
    two RRNS structure constants and one factor of two from the closed
    long-R ground fiber.  The chiral block used inside the absolute square
    is normalized to ground coefficient one, so this multiplicity is not
    double-counted.

    The current direct regular-seed oracle is validated through level one on
    each edge; higher cutoffs are intentionally rejected.
    """

    def __init__(
        self,
        *,
        external_momentum_1: float,
        external_momentum_2: float,
        max_level_1: int = 1,
        max_level_2: int = 1,
        cycle_insertion_1: str = "identity",
        cycle_insertion_2: str = "identity",
        p_max: float = 3.5,
        quadrature_order: int = 6,
        structure_precision: int = 25,
        finite_part_radius: float = 0.04,
        finite_part_check_radius: float = 0.05,
        finite_part_samples: int = 16,
    ) -> None:
        if (
            not isinstance(max_level_1, int)
            or not 0 <= max_level_1 <= 1
        ):
            raise ValueError(
                "max_level_1 must be zero or one with the current R seed"
            )
        if (
            not isinstance(max_level_2, int)
            or not 0 <= max_level_2 <= 1
        ):
            raise ValueError(
                "max_level_2 must be zero or one with the current R seed"
            )
        if cycle_insertion_1 not in ("identity", "parity"):
            raise ValueError(
                "cycle_insertion_1 must be 'identity' or 'parity'"
            )
        if cycle_insertion_2 not in ("identity", "parity"):
            raise ValueError(
                "cycle_insertion_2 must be 'identity' or 'parity'"
            )
        if p_max <= 0.0 or not math.isfinite(p_max):
            raise ValueError("p_max must be finite and positive")
        if quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two")
        if structure_precision < 15:
            raise ValueError("structure_precision must be at least 15")

        self.external_momentum_1 = float(external_momentum_1)
        self.external_momentum_2 = float(external_momentum_2)
        self.max_level_1 = max_level_1
        self.max_level_2 = max_level_2
        self.cycle_insertion_1 = cycle_insertion_1
        self.cycle_insertion_2 = cycle_insertion_2
        self.p_max = float(p_max)
        self.quadrature_order = int(quadrature_order)
        self.structure_precision = int(structure_precision)
        self.finite_part_radius = float(finite_part_radius)
        self.finite_part_check_radius = float(
            finite_part_check_radius
        )
        self.finite_part_samples = int(finite_part_samples)
        self._structure_cache: Dict[
            Tuple[float, float, int, int], complex
        ] = {}
        self._block_cache: Dict[
            Tuple[float, float, int, int],
            SelfDualRamondTorusTwoPointBetaRecursionBlock,
        ] = {}

    @property
    def sign_sectors(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        odd_cycle = (
            (self.cycle_insertion_1 == "parity")
            ^ (self.cycle_insertion_2 == "parity")
        )
        product = -1 if odd_cycle else 1
        return ((1, product), (-1, -product))

    def _structure_product(
        self,
        momentum_1: float,
        momentum_2: float,
        sign_1: int,
        sign_2: int,
    ) -> complex:
        key = (momentum_1, momentum_2, sign_1, sign_2)
        if key not in self._structure_cache:
            self._structure_cache[key] = (
                rr_ns_chiral_structure_constant(
                    momentum_1,
                    momentum_2,
                    self.external_momentum_1,
                    sign_1,
                    self.structure_precision,
                )
                * rr_ns_chiral_structure_constant(
                    momentum_2,
                    momentum_1,
                    self.external_momentum_2,
                    sign_2,
                    self.structure_precision,
                )
            )
        return self._structure_cache[key]

    def _block(
        self,
        momentum_1: float,
        momentum_2: float,
        sign_1: int,
        sign_2: int,
    ) -> SelfDualRamondTorusTwoPointBetaRecursionBlock:
        key = (momentum_1, momentum_2, sign_1, sign_2)
        if key not in self._block_cache:
            self._block_cache[key] = (
                SelfDualRamondTorusTwoPointBetaRecursionBlock(
                    internal_momentum_1=momentum_1,
                    internal_momentum_2=momentum_2,
                    external_momentum_1=self.external_momentum_1,
                    external_momentum_2=self.external_momentum_2,
                    vertex_sign_1=sign_1,
                    vertex_sign_2=sign_2,
                    cycle_insertion_1=self.cycle_insertion_1,
                    cycle_insertion_2=self.cycle_insertion_2,
                    radius=self.finite_part_radius,
                    check_radius=self.finite_part_check_radius,
                    samples=self.finite_part_samples,
                )
            )
        return self._block_cache[key]

    def momentum_integrand(
        self,
        momentum_1: float,
        momentum_2: float,
        plumbing_1: RamondPlumbingParameter,
        plumbing_2: RamondPlumbingParameter,
    ) -> complex:
        """Return the sign-summed integrand before both ``dP/pi`` measures."""

        if plumbing_1.cycle_insertion != self.cycle_insertion_1:
            raise ValueError("plumbing_1 cycle insertion does not match")
        if plumbing_2.cycle_insertion != self.cycle_insertion_2:
            raise ValueError("plumbing_2 cycle insertion does not match")
        result = 0.0j
        for sign_1, sign_2 in self.sign_sectors:
            block_value = self._block(
                momentum_1, momentum_2, sign_1, sign_2
            ).normalized_chiral_block(
                plumbing_1,
                plumbing_2,
                self.max_level_1,
                self.max_level_2,
            )
            result += (
                2.0
                * self._structure_product(
                    momentum_1,
                    momentum_2,
                    sign_1,
                    sign_2,
                )
                * abs(block_value) ** 2
            )
        return result

    def evaluate(
        self,
        plumbing_1: RamondPlumbingParameter,
        plumbing_2: RamondPlumbingParameter,
    ) -> complex:
        nodes, weights = numpy.polynomial.legendre.leggauss(
            self.quadrature_order
        )
        midpoint = 0.5 * self.p_max
        channels = tuple(
            (
                midpoint * (float(node) + 1.0),
                midpoint * float(weight) / math.pi,
            )
            for node, weight in zip(nodes, weights)
        )
        return sum(
            weight_1
            * weight_2
            * self.momentum_integrand(
                momentum_1,
                momentum_2,
                plumbing_1,
                plumbing_2,
            )
            for momentum_1, weight_1 in channels
            for momentum_2, weight_2 in channels
        )


__all__ = [
    "Type0BNSTorusTwoPointHRecursionCorrelator",
    "Type0BNSTorusTwoPointLeadingCorrelator",
    "Type0BRamondTorusTwoPointBetaRecursionCorrelator",
]
