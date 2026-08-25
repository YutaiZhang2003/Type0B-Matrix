"""Type-0B torus two-point contributions in the sphere--torus OPE channel.

The two external bottom-component NS fields fuse through an NS bridge with
momentum ``Ps``.  The bridge is inserted into a torus one-point block whose
handle momentum is ``Ph``.  In BRY normalization the even-form NS-handle
contribution is

    integral dPs dPh / pi^2
        C(P1,P2,Ps) C(Ph,Ps,Ph) |F_NS(Ps,Ph;x,q)|^2.

For an ordinary R handle, the second factor is ``2 C_even(Ph,Ph,Ps)`` and
the normalized positive HJS torus block is used.

The handle direction uses the all-level torus one-point h-recursion.  The
bridge is currently sewn directly through level one; its all-level
h-recursion requires the coupled even/odd toric three-form families.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

from super_liouville_structure_constants import (
    ns_structure_constant,
    rr_ns_structure_constants,
)
from super_liouville_torus_one_point import type0b_ns_gauss_legendre_rule
from superconformal_blocks import central_charge
from superconformal_torus_blocks import (
    NSPlumbingParameter,
    RamondPlumbingParameter,
)
from superconformal_torus_two_point_ope import (
    SelfDualNSTorusTwoPointOPEBlock,
    SelfDualRamondHandleTorusTwoPointOPEBlock,
)


class Type0BNSHandleTorusTwoPointOPECorrelator:
    """Even NS-form OPE contribution with an NS torus handle."""

    def __init__(
        self,
        *,
        external_momentum_1: float,
        external_momentum_2: float,
        max_bridge_twice_level: int = 2,
        max_handle_twice_level: int = 8,
        p_max: float = 3.5,
        quadrature_order: int = 8,
        structure_precision: int = 25,
        finite_part_radius: float = 0.04,
        finite_part_check_radius: float = 0.05,
        finite_part_samples: int = 24,
    ) -> None:
        if (
            not isinstance(max_bridge_twice_level, int)
            or not 0 <= max_bridge_twice_level <= 2
        ):
            raise ValueError(
                "max_bridge_twice_level must be 0, 1, or 2"
            )
        if (
            not isinstance(max_handle_twice_level, int)
            or max_handle_twice_level < 0
        ):
            raise ValueError(
                "max_handle_twice_level must be a nonnegative integer"
            )
        if p_max <= 0.0 or not math.isfinite(p_max):
            raise ValueError("p_max must be finite and positive")
        if quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two")
        if structure_precision < 15:
            raise ValueError("structure_precision must be at least 15")
        if finite_part_samples < 8:
            raise ValueError("finite_part_samples must be at least eight")

        self.external_momentum_1 = float(external_momentum_1)
        self.external_momentum_2 = float(external_momentum_2)
        self.max_bridge_twice_level = max_bridge_twice_level
        self.max_handle_twice_level = max_handle_twice_level
        self.p_max = float(p_max)
        self.quadrature_order = int(quadrature_order)
        self.structure_precision = int(structure_precision)
        self.finite_part_radius = float(finite_part_radius)
        self.finite_part_check_radius = float(
            finite_part_check_radius
        )
        self.finite_part_samples = int(finite_part_samples)
        self.c = central_charge(1.0)
        self._structure_cache: Dict[Tuple[float, float], complex] = {}
        self._block_cache: Dict[
            Tuple[float, float], SelfDualNSTorusTwoPointOPEBlock
        ] = {}

    def _structure_product(
        self, bridge_momentum: float, handle_momentum: float
    ) -> complex:
        key = (bridge_momentum, handle_momentum)
        if key not in self._structure_cache:
            self._structure_cache[key] = ns_structure_constant(
                self.external_momentum_1,
                self.external_momentum_2,
                bridge_momentum,
                self.structure_precision,
            ) * ns_structure_constant(
                handle_momentum,
                bridge_momentum,
                handle_momentum,
                self.structure_precision,
            )
        return self._structure_cache[key]

    def _block(
        self, bridge_momentum: float, handle_momentum: float
    ) -> SelfDualNSTorusTwoPointOPEBlock:
        key = (bridge_momentum, handle_momentum)
        if key not in self._block_cache:
            self._block_cache[key] = SelfDualNSTorusTwoPointOPEBlock(
                bridge_momentum=bridge_momentum,
                handle_momentum=handle_momentum,
                external_momentum_1=self.external_momentum_1,
                external_momentum_2=self.external_momentum_2,
                radius=self.finite_part_radius,
                check_radius=self.finite_part_check_radius,
                samples=self.finite_part_samples,
            )
        return self._block_cache[key]

    def momentum_integrand(
        self,
        bridge_momentum: float,
        handle_momentum: float,
        collision_parameter: complex,
        plumbing: NSPlumbingParameter,
    ) -> complex:
        block = self._block(bridge_momentum, handle_momentum)
        return self._structure_product(
            bridge_momentum, handle_momentum
        ) * abs(
            block.chiral_block(
                collision_parameter,
                plumbing,
                self.max_bridge_twice_level,
                self.max_handle_twice_level,
            )
        ) ** 2

    def evaluate(
        self,
        collision_parameter: complex,
        plumbing: NSPlumbingParameter,
    ) -> complex:
        channels = type0b_ns_gauss_legendre_rule(
            self.p_max, self.quadrature_order
        )
        return sum(
            bridge_weight
            * handle_weight
            / (math.pi * math.pi)
            * self.momentum_integrand(
                bridge_momentum,
                handle_momentum,
                collision_parameter,
                plumbing,
            )
            for bridge_momentum, bridge_weight in channels
            for handle_momentum, handle_weight in channels
        )


class Type0BRamondHandleTorusTwoPointOPECorrelator:
    """Even NS-form OPE contribution with an ordinary R handle."""

    def __init__(
        self,
        *,
        external_momentum_1: float,
        external_momentum_2: float,
        max_bridge_twice_level: int = 2,
        max_handle_level: int = 4,
        cycle_insertion: str = "identity",
        p_max: float = 3.5,
        quadrature_order: int = 8,
        structure_precision: int = 25,
        finite_part_radius: float = 0.04,
        finite_part_check_radius: float = 0.05,
        finite_part_samples: int = 24,
    ) -> None:
        if (
            not isinstance(max_bridge_twice_level, int)
            or not 0 <= max_bridge_twice_level <= 2
        ):
            raise ValueError(
                "max_bridge_twice_level must be 0, 1, or 2"
            )
        if not isinstance(max_handle_level, int) or max_handle_level < 0:
            raise ValueError(
                "max_handle_level must be a nonnegative integer"
            )
        if cycle_insertion != "identity":
            raise ValueError(
                "the current R-handle OPE contribution is the ordinary "
                "identity-cycle trace"
            )
        if p_max <= 0.0 or not math.isfinite(p_max):
            raise ValueError("p_max must be finite and positive")
        if quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two")
        if structure_precision < 15:
            raise ValueError("structure_precision must be at least 15")
        if finite_part_samples < 8:
            raise ValueError("finite_part_samples must be at least eight")

        self.external_momentum_1 = float(external_momentum_1)
        self.external_momentum_2 = float(external_momentum_2)
        self.max_bridge_twice_level = max_bridge_twice_level
        self.max_handle_level = max_handle_level
        self.cycle_insertion = cycle_insertion
        self.p_max = float(p_max)
        self.quadrature_order = int(quadrature_order)
        self.structure_precision = int(structure_precision)
        self.finite_part_radius = float(finite_part_radius)
        self.finite_part_check_radius = float(
            finite_part_check_radius
        )
        self.finite_part_samples = int(finite_part_samples)
        self.c = central_charge(1.0)
        self._structure_cache: Dict[Tuple[float, float], complex] = {}
        self._block_cache: Dict[
            Tuple[float, float],
            SelfDualRamondHandleTorusTwoPointOPEBlock,
        ] = {}

    def _structure_product(
        self, bridge_momentum: float, handle_momentum: float
    ) -> complex:
        key = (bridge_momentum, handle_momentum)
        if key not in self._structure_cache:
            c_even, _ = rr_ns_structure_constants(
                handle_momentum,
                handle_momentum,
                bridge_momentum,
                self.structure_precision,
            )
            self._structure_cache[key] = ns_structure_constant(
                self.external_momentum_1,
                self.external_momentum_2,
                bridge_momentum,
                self.structure_precision,
            ) * (2.0 * c_even)
        return self._structure_cache[key]

    def _block(
        self, bridge_momentum: float, handle_momentum: float
    ) -> SelfDualRamondHandleTorusTwoPointOPEBlock:
        key = (bridge_momentum, handle_momentum)
        if key not in self._block_cache:
            self._block_cache[
                key
            ] = SelfDualRamondHandleTorusTwoPointOPEBlock(
                bridge_momentum=bridge_momentum,
                handle_momentum=handle_momentum,
                external_momentum_1=self.external_momentum_1,
                external_momentum_2=self.external_momentum_2,
                sign=1,
                radius=self.finite_part_radius,
                check_radius=self.finite_part_check_radius,
                samples=self.finite_part_samples,
            )
        return self._block_cache[key]

    def momentum_integrand(
        self,
        bridge_momentum: float,
        handle_momentum: float,
        collision_parameter: complex,
        plumbing: RamondPlumbingParameter,
    ) -> complex:
        if plumbing.cycle_insertion != self.cycle_insertion:
            raise ValueError("plumbing cycle insertion does not match")
        block = self._block(bridge_momentum, handle_momentum)
        return self._structure_product(
            bridge_momentum, handle_momentum
        ) * abs(
            block.normalized_chiral_block(
                collision_parameter,
                plumbing,
                self.max_bridge_twice_level,
                self.max_handle_level,
            )
        ) ** 2

    def evaluate(
        self,
        collision_parameter: complex,
        plumbing: RamondPlumbingParameter,
    ) -> complex:
        channels = type0b_ns_gauss_legendre_rule(
            self.p_max, self.quadrature_order
        )
        return sum(
            bridge_weight
            * handle_weight
            / (math.pi * math.pi)
            * self.momentum_integrand(
                bridge_momentum,
                handle_momentum,
                collision_parameter,
                plumbing,
            )
            for bridge_momentum, bridge_weight in channels
            for handle_momentum, handle_weight in channels
        )


__all__ = [
    "Type0BNSHandleTorusTwoPointOPECorrelator",
    "Type0BRamondHandleTorusTwoPointOPECorrelator",
]
