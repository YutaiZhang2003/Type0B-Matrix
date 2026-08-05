"""BRY genus-zero four-point functions as functions of the cross ratio z.

The module contracts the NS sphere blocks with the b=1 super-Liouville
structure constants and performs the continuum integral over the exchanged
Liouville momentum.  It implements BRY Appendix A correlators G, H, and J,
and the unregularized four-tachyon moduli integrand in their equation (4.10).

The external momenta may be complex, as required by BRY's analytic
continuation.  The z integration, its power-divergence counterterms, and
internal-momentum contour deformations are implemented in the companion
``bry_one_to_three.py`` layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Dict, Literal, Optional, Sequence, Tuple, Union

import mpmath

from super_liouville_structure_constants import (
    ns_structure_constant,
    ns_tilde_structure_constant,
)
from superconformal_blocks import HighPrecisionNSSphereFourPointBlock


Number = Union[complex, float]
CorrelatorKind = Literal["G", "H", "J"]


@dataclass(frozen=True)
class FourPointCorrelators:
    """The three BRY nonchiral correlators at a fixed cross ratio."""

    G: complex
    H: complex
    J: complex

    def by_name(self, kind: CorrelatorKind) -> complex:
        if kind == "G":
            return self.G
        if kind == "H":
            return self.H
        if kind == "J":
            return self.J
        raise ValueError("kind must be 'G', 'H', or 'J'")


def _real_nonnegative(name: str, value: Number) -> float:
    value = complex(value)
    if abs(value.imag) > 1.0e-14 or value.real < 0 or not math.isfinite(value.real):
        raise ValueError(f"{name} must be a finite nonnegative real momentum")
    return value.real


def _finite_complex(name: str, value: Number) -> complex:
    value = complex(value)
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError(f"{name} must be finite")
    return value


@lru_cache(maxsize=None)
def _legendre_interval(order: int, upper_limit: float) -> Tuple[Tuple[float, float], ...]:
    if order < 2:
        raise ValueError("quadrature_order must be at least 2")
    if upper_limit <= 0 or not math.isfinite(upper_limit):
        raise ValueError("p_max must be positive and finite")
    nodes, weights = mpmath.gauss_quadrature(order, "legendre")
    scale = upper_limit / 2.0
    return tuple(
        (scale * (float(node) + 1.0), scale * float(weight))
        for node, weight in zip(nodes, weights)
    )


class BRYNSFourPointCorrelator:
    """Evaluate BRY's G, H, and J correlators for NS momenta.

    Punctures are ordered as

        V_{P4}(infinity) O_{P3}(1) O_{P2}(z,zbar) V_{P1}(0).

    The default finite momentum cutoff is a numerical truncation, not a
    physical regulator.  Convergence should be checked by increasing both
    ``p_max`` and ``quadrature_order``.  When selected,
    ``c_recursion_order=N`` is a maximum accumulated *physical* null level,
    i.e. twice-level ``2*N`` in the general graph-recursion convention.
    """

    def __init__(
        self,
        *,
        p1: Number,
        p2: Number,
        p3: Number,
        p4: Number,
        block_order: int = 8,
        bry_q_order: Optional[int] = None,
        c_recursion_order: Optional[int] = None,
        structure_precision: int = 30,
        central_charge_shift: float = 1.0e-5,
        block_working_precision: int = 60,
    ) -> None:
        if block_order < 1:
            raise ValueError("block_order must be positive")
        if bry_q_order is not None and bry_q_order < 1:
            raise ValueError("bry_q_order must be positive when specified")
        if c_recursion_order is not None and (
            not isinstance(c_recursion_order, int) or c_recursion_order < 0
        ):
            raise ValueError(
                "c_recursion_order must be a nonnegative integer when specified"
            )
        if bry_q_order is not None and c_recursion_order is not None:
            raise ValueError(
                "choose either direct c-recursion or elliptic-q truncation"
            )
        if structure_precision < 15:
            raise ValueError("structure_precision must be at least 15 digits")
        if central_charge_shift < 0 or not math.isfinite(central_charge_shift):
            raise ValueError("central_charge_shift must be finite and nonnegative")
        if block_working_precision < 30:
            raise ValueError("block_working_precision must be at least 30 digits")
        self.p1 = _finite_complex("p1", p1)
        self.p2 = _finite_complex("p2", p2)
        self.p3 = _finite_complex("p3", p3)
        self.p4 = _finite_complex("p4", p4)
        self.block_order = int(block_order)
        self.bry_q_order = None if bry_q_order is None else int(bry_q_order)
        self.c_recursion_order = (
            None if c_recursion_order is None else int(c_recursion_order)
        )
        self.structure_precision = int(structure_precision)
        self.central_charge_shift = float(central_charge_shift)
        self.block_working_precision = int(block_working_precision)
        self._structure_cache: Dict[float, Tuple[complex, complex]] = {}
        self._block_cache: Dict[
            float,
            Tuple[
                HighPrecisionNSSphereFourPointBlock,
                HighPrecisionNSSphereFourPointBlock,
            ],
        ] = {}

    @property
    def block_central_charge(self) -> float:
        """Central charge used by the regulated numerical block recursion."""

        return 13.5 + self.central_charge_shift

    def block_weight(self, momentum: Number) -> Number:
        """Liouville weight consistent with ``block_central_charge``."""

        momentum = _finite_complex("momentum", momentum)
        q_squared = self.block_central_charge / 3.0 - 0.5
        weight = 0.5 * (q_squared / 4.0 + momentum * momentum)
        return weight.real if abs(weight.imag) <= 1.0e-15 else weight

    def _block_value(
        self,
        block: HighPrecisionNSSphereFourPointBlock,
        z: Number,
        parity: Literal["even", "odd"],
    ) -> complex:
        if self.c_recursion_order is not None:
            return block.recursive_z_block(
                z, self.c_recursion_order, parity
            )
        if self.bry_q_order is not None:
            return block.bry_elliptic_block(z, self.bry_q_order, parity)
        return block.elliptic_block(z, self.block_order, parity)

    def _block_values(
        self,
        block: HighPrecisionNSSphereFourPointBlock,
        z_values: Sequence[Number],
        parity: Literal["even", "odd"],
    ) -> Tuple[complex, ...]:
        if self.c_recursion_order is not None:
            return block.recursive_z_blocks(
                z_values, self.c_recursion_order, parity
            )
        return tuple(self._block_value(block, z, parity) for z in z_values)

    def _structure_products(self, internal_momentum: float) -> Tuple[complex, complex]:
        if internal_momentum not in self._structure_cache:
            c12 = ns_structure_constant(
                self.p1, self.p2, internal_momentum, self.structure_precision
            )
            c34 = ns_structure_constant(
                self.p3, self.p4, internal_momentum, self.structure_precision
            )
            ct12 = ns_tilde_structure_constant(
                self.p1, self.p2, internal_momentum, self.structure_precision
            )
            ct34 = ns_tilde_structure_constant(
                self.p3, self.p4, internal_momentum, self.structure_precision
            )
            self._structure_cache[internal_momentum] = (c12 * c34, ct12 * ct34)
        return self._structure_cache[internal_momentum]

    def _blocks(
        self, internal_momentum: float
    ) -> Tuple[HighPrecisionNSSphereFourPointBlock, HighPrecisionNSSphereFourPointBlock]:
        if internal_momentum not in self._block_cache:
            # BRY's b -> 1 crossing benchmark evaluates the recursion at
            # c=27/2+10^-5.  The displacement avoids coincident-pole
            # cancellations that are ill-conditioned in floating arithmetic.
            c_block = self.block_central_charge

            common = dict(
                c=c_block,
                h1=self.block_weight(self.p1),
                h2=self.block_weight(self.p2),
                h3=self.block_weight(self.p3),
                h4=self.block_weight(self.p4),
                internal_weight=self.block_weight(internal_momentum),
                working_precision=self.block_working_precision,
            )
            primary = HighPrecisionNSSphereFourPointBlock(**common)
            double_descendant = HighPrecisionNSSphereFourPointBlock(
                **common, star2=True, star3=True
            )
            self._block_cache[internal_momentum] = (primary, double_descendant)
        return self._block_cache[internal_momentum]

    @staticmethod
    def _validate_z(z: Number) -> complex:
        z = complex(z)
        if not math.isfinite(z.real) or not math.isfinite(z.imag):
            raise ValueError("z must be finite")
        if z == 0 or z == 1:
            raise ValueError("the unregularized correlator is singular at z=0 and z=1")
        return z

    def momentum_integrands(self, internal_momentum: Number, z: Number) -> FourPointCorrelators:
        """Return the dP integrands of G, H, and J, including the factor 1/pi."""

        internal_momentum = _real_nonnegative("internal_momentum", internal_momentum)
        z = self._validate_z(z)
        if internal_momentum == 0:
            return FourPointCorrelators(0.0j, 0.0j, 0.0j)

        c_product, ct_product = self._structure_products(internal_momentum)
        primary, starred = self._blocks(internal_momentum)
        zbar = z.conjugate()

        pe = self._block_value(primary, z, "even")
        po = self._block_value(primary, z, "odd")
        se = self._block_value(starred, z, "even")
        # The low-level block module follows the signed ancillary-notebook
        # phase.  BRY's scalar-correlator convention has one additional minus
        # sign for the doubly-starred odd block.  It cancels in H but fixes the
        # relative sign of the mixed even/odd products in J.
        so = -self._block_value(starred, z, "odd")
        pe_bar = self._block_value(primary, zbar, "even")
        po_bar = self._block_value(primary, zbar, "odd")
        se_bar = self._block_value(starred, zbar, "even")
        so_bar = -self._block_value(starred, zbar, "odd")

        inverse_one_minus_z = 1.0 / (1.0 - z)
        inverse_one_minus_zbar = 1.0 / (1.0 - zbar)
        g_value = c_product * pe * pe_bar + ct_product * po * po_bar
        h_value = -(ct_product * se * se_bar + c_product * so * so_bar)
        j_value = -(
            c_product
            * (
                inverse_one_minus_zbar * so * pe_bar
                + inverse_one_minus_z * pe * so_bar
            )
            + ct_product
            * (
                inverse_one_minus_zbar * se * po_bar
                + inverse_one_minus_z * po * se_bar
            )
        )
        normalization = 1.0 / math.pi
        return FourPointCorrelators(
            normalization * g_value,
            normalization * h_value,
            normalization * j_value,
        )

    def h_momentum_integrand(self, internal_momentum: Number, z: Number) -> complex:
        """Return only the dP integrand of BRY's H correlator.

        This avoids evaluating the unstarred blocks needed by G and J and is
        the efficient path for reproducing BRY's Figure 4 crossing test.
        """

        internal_momentum = _real_nonnegative("internal_momentum", internal_momentum)
        z = self._validate_z(z)
        if internal_momentum == 0:
            return 0.0j

        c_product, ct_product = self._structure_products(internal_momentum)
        _, starred = self._blocks(internal_momentum)
        zbar = z.conjugate()
        se = self._block_value(starred, z, "even")
        so = -self._block_value(starred, z, "odd")
        se_bar = self._block_value(starred, zbar, "even")
        so_bar = -self._block_value(starred, zbar, "odd")
        return -(
            ct_product * se * se_bar + c_product * so * so_bar
        ) / math.pi

    def g_momentum_integrand(self, internal_momentum: Number, z: Number) -> complex:
        """Return only the dP integrand of BRY's G correlator.

        This is the efficient path for the four-bottom-component NS crossing
        equation.  It avoids constructing the doubly-starred blocks needed
        only by H and J.
        """

        internal_momentum = _real_nonnegative("internal_momentum", internal_momentum)
        z = self._validate_z(z)
        if internal_momentum == 0:
            return 0.0j

        c_product, ct_product = self._structure_products(internal_momentum)
        primary, _ = self._blocks(internal_momentum)
        zbar = z.conjugate()
        pe = self._block_value(primary, z, "even")
        po = self._block_value(primary, z, "odd")
        pe_bar = self._block_value(primary, zbar, "even")
        po_bar = self._block_value(primary, zbar, "odd")
        return (c_product * pe * pe_bar + ct_product * po * po_bar) / math.pi

    def g_momentum_integrands_grid(
        self,
        internal_momentum: Number,
        z_values: Sequence[Number],
    ) -> Tuple[complex, ...]:
        """Vectorized dP integrands for a direct-recursion z grid."""

        internal_momentum = _real_nonnegative(
            "internal_momentum", internal_momentum
        )
        points = tuple(self._validate_z(z) for z in z_values)
        if not points:
            raise ValueError("z_values must not be empty")
        if internal_momentum == 0:
            return tuple(0.0j for _ in points)

        c_product, ct_product = self._structure_products(internal_momentum)
        primary, _ = self._blocks(internal_momentum)
        pe = self._block_values(primary, points, "even")
        po = self._block_values(primary, points, "odd")
        if all(point.imag == 0 for point in points):
            pe_bar = pe
            po_bar = po
        else:
            conjugates = tuple(point.conjugate() for point in points)
            pe_bar = self._block_values(primary, conjugates, "even")
            po_bar = self._block_values(primary, conjugates, "odd")
        return tuple(
            (c_product * even * even_bar + ct_product * odd * odd_bar)
            / math.pi
            for even, even_bar, odd, odd_bar in zip(pe, pe_bar, po, po_bar)
        )

    def evaluate(
        self,
        z: Number,
        *,
        p_max: float = 6.0,
        quadrature_order: int = 48,
    ) -> FourPointCorrelators:
        """Integrate G, H, and J over 0 <= P <= p_max."""

        z = self._validate_z(z)
        total_g = 0.0j
        total_h = 0.0j
        total_j = 0.0j
        for momentum, weight in _legendre_interval(quadrature_order, float(p_max)):
            values = self.momentum_integrands(momentum, z)
            total_g += weight * values.G
            total_h += weight * values.H
            total_j += weight * values.J
        return FourPointCorrelators(total_g, total_h, total_j)

    def evaluate_h(
        self,
        z: Number,
        *,
        p_max: float = 6.0,
        quadrature_order: int = 48,
    ) -> complex:
        """Integrate only H over 0 <= P <= p_max."""

        z = self._validate_z(z)
        total = 0.0j
        for momentum, weight in _legendre_interval(quadrature_order, float(p_max)):
            total += weight * self.h_momentum_integrand(momentum, z)
        return total

    def evaluate_g(
        self,
        z: Number,
        *,
        p_max: float = 6.0,
        quadrature_order: int = 48,
    ) -> complex:
        """Integrate only G over 0 <= P <= p_max."""

        z = self._validate_z(z)
        total = 0.0j
        for momentum, weight in _legendre_interval(quadrature_order, float(p_max)):
            total += weight * self.g_momentum_integrand(momentum, z)
        return total

    def evaluate_g_grid(
        self,
        z_values: Sequence[Number],
        *,
        p_max: float = 6.0,
        quadrature_order: int = 48,
    ) -> Tuple[complex, ...]:
        """Integrate G on a grid while sharing each direct recursion tree."""

        points = tuple(self._validate_z(z) for z in z_values)
        if not points:
            raise ValueError("z_values must not be empty")
        totals = [0.0j] * len(points)
        for momentum, weight in _legendre_interval(
            quadrature_order, float(p_max)
        ):
            values = self.g_momentum_integrands_grid(momentum, points)
            for index, value in enumerate(values):
                totals[index] += weight * value
        return tuple(totals)

    def correlator(
        self,
        kind: CorrelatorKind,
        z: Number,
        *,
        p_max: float = 6.0,
        quadrature_order: int = 48,
    ) -> complex:
        """Evaluate one named correlator; useful for plotting a z-grid."""

        if kind == "G":
            return self.evaluate_g(
                z, p_max=p_max, quadrature_order=quadrature_order
            )
        if kind == "H":
            return self.evaluate_h(
                z, p_max=p_max, quadrature_order=quadrature_order
            )
        return self.evaluate(
            z, p_max=p_max, quadrature_order=quadrature_order
        ).by_name(kind)


class BRYFourTachyonSphere:
    """The unregularized z-dependent four-tachyon string integrand.

    The assignment is P4=omega, P3=omega3, P2=omega2, P1=omega1.  The
    returned reduced integrand is the expression inside braces in BRY (4.10),
    before the overall (i/2) delta(...) g_s^4 C_{S^2} and before d^2z
    integration.
    """

    def __init__(
        self,
        *,
        omega: Number,
        omega1: Number,
        omega2: Number,
        omega3: Number,
        block_order: int = 8,
        bry_q_order: Optional[int] = None,
        c_recursion_order: Optional[int] = None,
        structure_precision: int = 30,
        central_charge_shift: float = 1.0e-5,
        block_working_precision: int = 60,
    ) -> None:
        self.omega = _finite_complex("omega", omega)
        self.omega1 = _finite_complex("omega1", omega1)
        self.omega2 = _finite_complex("omega2", omega2)
        self.omega3 = _finite_complex("omega3", omega3)
        self.liouville = BRYNSFourPointCorrelator(
            p1=self.omega1,
            p2=self.omega2,
            p3=self.omega3,
            p4=self.omega,
            block_order=block_order,
            bry_q_order=bry_q_order,
            c_recursion_order=c_recursion_order,
            structure_precision=structure_precision,
            central_charge_shift=central_charge_shift,
            block_working_precision=block_working_precision,
        )

    def reduced_integrand(
        self,
        z: Number,
        *,
        p_max: float = 6.0,
        quadrature_order: int = 48,
    ) -> complex:
        """Return the BRY four-tachyon density as a function of z."""

        z = self.liouville._validate_z(z)
        values = self.liouville.evaluate(
            z, p_max=p_max, quadrature_order=quadrature_order
        )
        return self.combine_correlators(z, values)

    def combine_correlators(
        self, z: Number, values: FourPointCorrelators
    ) -> complex:
        """Combine precomputed G, H, and J into the BRY moduli density."""

        z = self.liouville._validate_z(z)
        kinematic = (
            abs(z) ** (-2.0 * self.omega1 * self.omega2)
            * abs(1.0 - z) ** (-2.0 * self.omega2 * self.omega3)
        )
        return kinematic * (
            self.omega2**2 * self.omega3**2 / abs(1.0 - z) ** 2 * values.G
            - values.H
            - self.omega2 * self.omega3 * values.J
        )


__all__ = [
    "BRYFourTachyonSphere",
    "BRYNSFourPointCorrelator",
    "FourPointCorrelators",
]
