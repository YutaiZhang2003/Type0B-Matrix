"""First regulated BRY tree-level 1->3 sphere-amplitude benchmark.

This module implements the particularly economical complex-energy family in
BRY (4.15),

    omega = 1/3 + i a,       omega1 = omega2 = omega3 = omega/3.

For this family the s- and u-channel power counterterms vanish.  The only
subtraction is the leading NS-C term in the t channel.  The three terms in
the picture-raised four-tachyon density combine into the single OPE
coefficient

    [c0(P) + omega2 omega3]^2
      = [1 + (omega2 + omega3)^2 - P^2]^2 / 4.

The full z plane is folded to the unit disk by z -> 1/z.  Equal outgoing
energies make the folded contribution identical to the direct one.  The
small disks around z=0 and z=1 are evaluated with BRY's direct- and
crossed-channel coefficients; the rest of the disk uses the elliptic
recursion blocks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from typing import Callable, Literal, Sequence

import mpmath

from sphere_four_point import BRYFourTachyonSphere, FourPointCorrelators
from superconformal_blocks import elliptic_nome
from super_liouville_structure_constants import (
    ns_structure_constant,
    ns_tilde_structure_constant,
)
from type0b_sphere_four_point_hybrid import (
    Type0BSphereFourPointHybrid,
    best_four_point_channel,
    canonical_chart_channel,
)


ComplexFunction = Callable[[float], complex]


@lru_cache(maxsize=None)
def _legendre_rule(order: int) -> tuple[tuple[float, float], ...]:
    if order < 2:
        raise ValueError("quadrature orders must be at least 2")
    nodes, weights = mpmath.gauss_quadrature(order, "legendre")
    return tuple((float(node), float(weight)) for node, weight in zip(nodes, weights))


def _integrate_interval(
    function: ComplexFunction, lower: float, upper: float, order: int
) -> complex:
    if upper <= lower:
        return 0.0j
    midpoint = 0.5 * (lower + upper)
    scale = 0.5 * (upper - lower)
    return scale * sum(
        weight * function(midpoint + scale * node)
        for node, weight in _legendre_rule(order)
    )


def _physical_weight(momentum: complex) -> complex:
    return 0.5 * (1.0 + momentum * momentum)


@dataclass(frozen=True)
class OneToThreeWorldsheetResult:
    """Target-free regulated worldsheet result in the native BRY normalization."""

    omega: complex
    omega_out: complex
    epsilon: float
    p_max: float
    p_quadrature_order: int
    angular_order: int
    radial_order: int
    cap_angular_order: int
    cap_radial_order: int
    block_q_order: int
    block_backend: str
    hybrid_corner_radius: float
    hybrid_elliptic_nome_threshold: float
    hybrid_asymptotic_radius: float
    low_z_region_integral: complex
    bulk_region_integral: complex
    t_cap_region_integral: complex
    reduced_moduli_integral: complex
    worldsheet_amplitude_coefficient: complex

    def json_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key, value in tuple(data.items()):
            if isinstance(value, complex):
                data[key] = {"real": value.real, "imag": value.imag}
        return data


@dataclass(frozen=True)
class OneToThreeResult(OneToThreeWorldsheetResult):
    """Worldsheet result followed by the independently supplied MQM target."""

    reduced_moduli_target: complex
    matrix_amplitude_coefficient: complex
    relative_error: float


class BRYOneToThreeBenchmark:
    """Regulated BRY (4.15) benchmark at a fixed complex incoming energy."""

    def __init__(
        self,
        *,
        incoming_imaginary: float = 0.6,
        epsilon: float = 1.0e-2,
        p_max: float = 4.0,
        p_quadrature_order: int = 16,
        angular_order: int = 10,
        radial_order: int = 10,
        cap_angular_order: int = 10,
        cap_radial_order: int = 8,
        block_q_order: int = 8,
        block_backend: Literal["hybrid", "h", "c"] = "h",
        hybrid_corner_radius: float = 0.15,
        hybrid_elliptic_nome_threshold: float = 0.3,
        hybrid_asymptotic_radius: float = 1.0e-4,
        structure_precision: int = 30,
        block_working_precision: int = 60,
    ) -> None:
        if not math.isfinite(incoming_imaginary):
            raise ValueError("incoming_imaginary must be finite")
        if epsilon <= 0.0 or epsilon >= 0.25:
            raise ValueError("epsilon must lie between 0 and 1/4")
        if p_max <= 0.0 or not math.isfinite(p_max):
            raise ValueError("p_max must be positive and finite")
        if block_backend not in ("hybrid", "h", "c"):
            raise ValueError("block_backend must be 'hybrid', 'h', or 'c'")
        if not 0.0 < hybrid_corner_radius < 1.0:
            raise ValueError("hybrid_corner_radius must lie in (0,1)")
        if not 0.0 < hybrid_elliptic_nome_threshold < 1.0:
            raise ValueError(
                "hybrid_elliptic_nome_threshold must lie in (0,1)"
            )
        if not 0.0 < hybrid_asymptotic_radius < epsilon:
            raise ValueError(
                "hybrid_asymptotic_radius must lie between zero and epsilon"
            )
        self.omega = 1.0 / 3.0 + 1j * float(incoming_imaginary)
        self.omega1 = self.omega2 = self.omega3 = self.omega / 3.0
        self.epsilon = float(epsilon)
        self.p_max = float(p_max)
        self.p_quadrature_order = int(p_quadrature_order)
        self.angular_order = int(angular_order)
        self.radial_order = int(radial_order)
        self.cap_angular_order = int(cap_angular_order)
        self.cap_radial_order = int(cap_radial_order)
        self.block_q_order = int(block_q_order)
        self.block_backend = block_backend
        self.hybrid_corner_radius = float(hybrid_corner_radius)
        self.hybrid_elliptic_nome_threshold = float(
            hybrid_elliptic_nome_threshold
        )
        self.hybrid_asymptotic_radius = float(hybrid_asymptotic_radius)
        self.structure_precision = int(structure_precision)
        self._t_structure_cache: dict[float, tuple[complex, complex]] = {}
        self.sphere = BRYFourTachyonSphere(
            omega=self.omega,
            omega1=self.omega1,
            omega2=self.omega2,
            omega3=self.omega3,
            bry_q_order=self.block_q_order,
            # In hybrid mode this object is the fast, verified linear-channel
            # h chart.  The separate six-frame atlas below supplies c charts.
            block_backend=("h" if self.block_backend == "hybrid" else self.block_backend),
            hybrid_corner_radius=self.hybrid_corner_radius,
            structure_precision=self.structure_precision,
            block_working_precision=block_working_precision,
        )
        self.hybrid_atlas = (
            Type0BSphereFourPointHybrid(
                outgoing_energies=(self.omega1, self.omega2, self.omega3),
                contour_prescription="fixed",
                block_backend="hybrid",
                hybrid_corner_radius=self.hybrid_corner_radius,
                hybrid_elliptic_nome_threshold=(
                    self.hybrid_elliptic_nome_threshold
                ),
                recursion_max_twice_level=2 * self.block_q_order,
                momentum_order=2,
                momentum_maximum=self.p_max,
                structure_precision=self.structure_precision,
                # The c-recursive corner has no confluent h-pole problem and
                # can be evaluated at the physical central charge.  Keeping
                # the 1e-5 h-chart detuning here would leave a mismatched OPE
                # exponent after subtracting BRY's c=13.5 polynomial.
                central_charge_shift=0.0,
                block_working_precision=block_working_precision,
                allow_finite_part=True,
            )
            if self.block_backend == "hybrid"
            else None
        )
        t_threshold_squared = 1.0 + ((self.omega2 + self.omega3) ** 2).real
        if t_threshold_squared <= 0.0:
            raise ValueError("the selected energies have no BRY leading t counterterm")
        self.t_threshold = math.sqrt(t_threshold_squared)
        if ((self.omega2 + self.omega3) ** 2).real >= 0.0:
            raise ValueError(
                "this first benchmark assumes BRY's family with no tilde-C t counterterm"
            )

    @property
    def reduced_moduli_target(self) -> complex:
        """Dimensionless moduli integral implied by BRY (2.13) and (4.14)."""

        return (
            math.pi
            * self.omega
            * self.omega1
            * self.omega2
            * self.omega3
            * (1.0 + 2j * self.omega)
        )

    @property
    def matrix_amplitude_coefficient(self) -> complex:
        """BRY (2.13), with the energy delta function and mu^-2 stripped."""

        return (
            8j
            * self.omega
            * self.omega1
            * self.omega2
            * self.omega3
            * (1.0 + 2j * self.omega)
        )

    def _t_structure_products(self, momentum: float) -> tuple[complex, complex]:
        momentum = float(momentum)
        if momentum not in self._t_structure_cache:
            c_product = ns_structure_constant(
                self.omega1, self.omega, momentum, self.structure_precision
            ) * ns_structure_constant(
                self.omega2, self.omega3, momentum, self.structure_precision
            )
            ct_product = ns_tilde_structure_constant(
                self.omega1, self.omega, momentum, self.structure_precision
            ) * ns_tilde_structure_constant(
                self.omega2, self.omega3, momentum, self.structure_precision
            )
            self._t_structure_cache[momentum] = (c_product, ct_product)
        return self._t_structure_cache[momentum]

    def leading_t_coefficient(self, momentum: float) -> complex:
        """Combined BRY b_00 coefficient of VVVV, VWWV, and VLambdaLambdaV."""

        momentum = float(momentum)
        return 0.25 * (
            1.0 + (self.omega2 + self.omega3) ** 2 - momentum * momentum
        ) ** 2

    def t_counterterm_momentum_density(self, momentum: float, z: complex) -> complex:
        """Return the leading t-channel counterterm at fixed P, including dP/pi."""

        if momentum >= self.t_threshold:
            return 0.0j
        distance = abs(1.0 - z)
        if distance == 0.0:
            raise ValueError("the t counterterm is singular at z=1")
        c_product, _ = self._t_structure_products(momentum)
        exponent = -3.0 + momentum * momentum - (self.omega2 + self.omega3) ** 2
        return (
            c_product
            * self.leading_t_coefficient(momentum)
            * distance**exponent
            / math.pi
        )

    def folded_t_counterterm_momentum_density(
        self, momentum: float, z: complex
    ) -> complex:
        """Sum the t counterterm and its z->1/z image on the unit disk."""

        if momentum >= self.t_threshold:
            return 0.0j
        modulus = abs(z)
        if modulus == 0.0:
            raise ValueError("the folded counterterm is singular at z=0")
        exponent = -3.0 + momentum * momentum - (self.omega2 + self.omega3) ** 2
        direct = self.t_counterterm_momentum_density(momentum, z)
        return direct * (1.0 + modulus ** (-4.0 - exponent))

    def direct_momentum_density(self, momentum: float, z: complex) -> complex:
        """Full picture-raised density at fixed P in the direct channel."""

        correlators = self.sphere.liouville.momentum_integrands(momentum, z)
        return self.sphere.combine_correlators(z, correlators)

    def _hybrid_channel(self, z: complex, preferred_chart: int):
        """Choose h in the declared elliptic patch and c in a local chart."""

        if self.hybrid_atlas is None:
            raise RuntimeError("the channel atlas is available only in hybrid mode")
        positions = self.hybrid_atlas.fixed_positions(z)
        preferred = canonical_chart_channel(positions, preferred_chart)
        if (
            abs(elliptic_nome(preferred.q))
            < self.hybrid_elliptic_nome_threshold
            and self.hybrid_atlas._selected_backend(preferred, "auto") == "h"
        ):
            return preferred
        return best_four_point_channel(positions)

    def hybrid_momentum_density(
        self, momentum: float, z: complex, *, preferred_chart: int = 0
    ) -> complex:
        """Full channel-adapted density at fixed spectral momentum."""

        if self.hybrid_atlas is None:
            raise RuntimeError("hybrid_momentum_density requires hybrid mode")
        channel = self._hybrid_channel(complex(z), preferred_chart)
        return self.hybrid_atlas.fixed_momentum_density(
            z,
            momentum,
            channel=channel,
            block_region="auto",
        )

    def folded_hybrid_momentum_density(
        self, momentum: float, z: complex
    ) -> complex:
        """Fold the full channel-adapted plane density to ``0<|z|<=1``."""

        z = complex(z)
        modulus = abs(z)
        if modulus == 0.0 or modulus > 1.0 + 1.0e-13:
            raise ValueError("the folded point must satisfy 0<|z|<=1")
        if self.hybrid_atlas is None:
            raise RuntimeError("folded_hybrid_momentum_density requires hybrid mode")
        direct_channel = canonical_chart_channel(
            self.hybrid_atlas.fixed_positions(z), 0
        )
        if (
            abs(elliptic_nome(direct_channel.q))
            < self.hybrid_elliptic_nome_threshold
        ):
            # Equal outgoing energies make the inversion image identical.
            # Retaining the specialized BRY linear-channel evaluator here is
            # substantially faster than rebuilding the same h block twice in
            # the general channel atlas.
            return complex(2.0 * self.direct_momentum_density(momentum, z))
        direct = self.hybrid_momentum_density(
            momentum, z, preferred_chart=0
        )
        inverse = 1.0 / z
        image = self.hybrid_momentum_density(
            momentum, inverse, preferred_chart=2
        )
        return complex(direct + modulus**-4 * image)

    def regulated_folded_momentum_density(
        self, momentum: float, z: complex
    ) -> complex:
        """Full folded density with BRY's explicit OPE polynomial removed."""

        if self.block_backend == "hybrid":
            value = self.folded_hybrid_momentum_density(momentum, z)
        else:
            value = 2.0 * self.direct_momentum_density(momentum, z)
        return complex(
            value - self.folded_t_counterterm_momentum_density(momentum, z)
        )

    def t_local_momentum_correlators(
        self, momentum: float, z: complex
    ) -> FourPointCorrelators:
        """BRY crossed-channel correlators through the first nontrivial local orders."""

        w = 1.0 - complex(z)
        if w == 0.0:
            raise ValueError("the local t-channel blocks are singular at z=1")
        wbar = w.conjugate()
        h1 = _physical_weight(self.omega1)
        h2 = _physical_weight(self.omega2)
        h3 = _physical_weight(self.omega3)
        h4 = _physical_weight(self.omega)
        h = _physical_weight(complex(momentum))
        exponent = h - h2 - h3

        primary_1 = (h - h3 + h2) * (h + h1 - h4) / (2.0 * h)
        primary_half = 1.0 / (2.0 * h)
        starred_0 = h2 + h3 - h
        starred_1 = -(
            (h - h3 + h2)
            * (h + h1 - h4)
            * (h - h2 - h3)
            / (2.0 * h)
        )
        starred_half = (h + h2 + h3 - 0.5) / (2.0 * h)

        def blocks(value: complex) -> tuple[complex, complex, complex, complex]:
            primary_even = value**exponent * (1.0 + primary_1 * value)
            primary_odd = value ** (exponent + 0.5) * primary_half
            starred_even = value ** (exponent - 1.0) * (
                starred_0 + starred_1 * value
            )
            starred_odd = value ** (exponent - 0.5) * starred_half
            return primary_even, primary_odd, starred_even, starred_odd

        pe, po, se, so = blocks(w)
        pe_bar, po_bar, se_bar, so_bar = blocks(wbar)
        c_product, ct_product = self._t_structure_products(momentum)
        g_value = c_product * pe * pe_bar + ct_product * po * po_bar
        h_value = -(c_product * se * se_bar + ct_product * so * so_bar)
        j_value = -(
            c_product * (se * pe_bar / wbar + pe * se_bar / w)
            + ct_product * (so * po_bar / wbar + po * so_bar / w)
        )
        return FourPointCorrelators(
            g_value / math.pi,
            h_value / math.pi,
            j_value / math.pi,
        )

    def s_local_momentum_correlators(
        self, momentum: float, z: complex
    ) -> FourPointCorrelators:
        """BRY direct-channel correlators through NLO near ``z=0``.

        These are the coefficients displayed below BRY (A.12).  Keeping the
        first two terms of each block is more than sufficient in the disk
        ``|z| < epsilon`` used in their numerical prescription.
        """

        z = complex(z)
        if z == 0.0:
            raise ValueError("the local s-channel blocks are singular at z=0")
        zbar = z.conjugate()
        h1 = _physical_weight(self.omega1)
        h2 = _physical_weight(self.omega2)
        h3 = _physical_weight(self.omega3)
        h4 = _physical_weight(self.omega)
        h = _physical_weight(complex(momentum))
        exponent = h - h1 - h2
        central_charge = 13.5

        primary_1 = (h - h1 + h2) * (h + h3 - h4) / (2.0 * h)
        primary_half = 1.0 / (2.0 * h)
        primary_three_half = (
            (0.5 + h - h1 + h2)
            * (0.5 + h + h3 - h4)
            / (2.0 * h * (1.0 + 2.0 * h))
            - 6.0
            * (h1 - h2)
            * (h3 - h4)
            / (
                (1.0 + 2.0 * h)
                * (central_charge + 2.0 * central_charge * h + 3.0 * h * (-3.0 + 2.0 * h))
            )
        )
        starred_1 = (
            (0.5 + h - h1 + h2) * (0.5 + h + h3 - h4) / (2.0 * h)
        )
        starred_half = (h - h1 + h2) * (h + h3 - h4) / (2.0 * h)
        starred_three_half = (
            (h - h1 + h2)
            * (1.0 + h - h1 + h2)
            * (h + h3 - h4)
            * (1.0 + h + h3 - h4)
            / (2.0 * h * (1.0 + 2.0 * h))
            + 3.0
            * (
                h1
                - 2.0 * h1 * h1
                + h2
                + 4.0 * h1 * h2
                - 2.0 * h2 * h2
                + h * (2.0 * h1 + 2.0 * h2 - 1.0)
            )
            * (
                h3
                - 2.0 * h3 * h3
                + h4
                + 4.0 * h3 * h4
                - 2.0 * h4 * h4
                + h * (2.0 * h3 + 2.0 * h4 - 1.0)
            )
            / (
                2.0
                * (1.0 + 2.0 * h)
                * (central_charge + 2.0 * central_charge * h + 3.0 * h * (-3.0 + 2.0 * h))
            )
        )

        def blocks(value: complex) -> tuple[complex, complex, complex, complex]:
            primary_even = value**exponent * (1.0 + primary_1 * value)
            primary_odd = value ** (exponent + 0.5) * (
                primary_half + primary_three_half * value
            )
            starred_even = value ** (exponent - 0.5) * (
                1.0 + starred_1 * value
            )
            # This is the scalar-correlator phase used by momentum_integrands.
            starred_odd = value**exponent * (
                starred_half + starred_three_half * value
            )
            return primary_even, primary_odd, starred_even, starred_odd

        pe, po, se, so = blocks(z)
        pe_bar, po_bar, se_bar, so_bar = blocks(zbar)
        c_product, ct_product = self.sphere.liouville._structure_products(momentum)
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
        return FourPointCorrelators(
            g_value / math.pi,
            h_value / math.pi,
            j_value / math.pi,
        )

    def s_local_momentum_density(self, momentum: float, z: complex) -> complex:
        correlators = self.s_local_momentum_correlators(momentum, z)
        return self.sphere.combine_correlators(z, correlators)

    def t_local_momentum_density(self, momentum: float, z: complex) -> complex:
        correlators = self.t_local_momentum_correlators(momentum, z)
        return self.sphere.combine_correlators(z, correlators)

    def stable_t_local_regulated_folded_momentum_density(
        self, momentum: float, z: complex
    ) -> complex:
        """Evaluate ``2 I_t-R_t^fold`` before losing endpoint precision."""

        with mpmath.workdps(max(50, self.sphere.liouville.block_working_precision)):
            z_mp = mpmath.mpc(z)
            w = 1 - z_mp
            if w == 0:
                raise ValueError("the regulated local density excludes z=1")
            wbar = w.conjugate()
            omega1 = mpmath.mpc(self.omega1)
            omega2 = mpmath.mpc(self.omega2)
            omega3 = mpmath.mpc(self.omega3)
            omega4 = mpmath.mpc(self.omega)
            momentum_mp = mpmath.mpf(momentum)

            def weight(value):
                return (1 + value * value) / 2

            h1 = weight(omega1)
            h2 = weight(omega2)
            h3 = weight(omega3)
            h4 = weight(omega4)
            h = weight(momentum_mp)
            exponent = h - h2 - h3
            primary_1 = (h - h3 + h2) * (h + h1 - h4) / (2 * h)
            primary_half = 1 / (2 * h)
            starred_0 = h2 + h3 - h
            starred_1 = -(
                (h - h3 + h2)
                * (h + h1 - h4)
                * (h - h2 - h3)
                / (2 * h)
            )
            starred_half = (h + h2 + h3 - mpmath.mpf("0.5")) / (2 * h)

            def blocks(value):
                return (
                    value**exponent * (1 + primary_1 * value),
                    value ** (exponent + mpmath.mpf("0.5")) * primary_half,
                    value ** (exponent - 1) * (starred_0 + starred_1 * value),
                    value ** (exponent - mpmath.mpf("0.5")) * starred_half,
                )

            pe, po, se, so = blocks(w)
            pe_bar, po_bar, se_bar, so_bar = blocks(wbar)
            c_product_raw, ct_product_raw = self._t_structure_products(momentum)
            c_product = mpmath.mpc(c_product_raw)
            ct_product = mpmath.mpc(ct_product_raw)
            g_value = c_product * pe * pe_bar + ct_product * po * po_bar
            h_value = -(c_product * se * se_bar + ct_product * so * so_bar)
            j_value = -(
                c_product * (se * pe_bar / wbar + pe * se_bar / w)
                + ct_product * (so * po_bar / wbar + po * so_bar / w)
            )
            normalization = 1 / mpmath.pi
            kinematic = (
                abs(z_mp) ** (-2 * omega1 * omega2)
                * abs(w) ** (-2 * omega2 * omega3)
            )
            density = kinematic * normalization * (
                omega2**2 * omega3**2 / abs(w) ** 2 * g_value
                - h_value
                - omega2 * omega3 * j_value
            )

            if momentum >= self.t_threshold:
                folded_counterterm = 0
            else:
                counterterm_exponent = (
                    -3 + momentum_mp**2 - (omega2 + omega3) ** 2
                )
                leading = (
                    1 + (omega2 + omega3) ** 2 - momentum_mp**2
                ) ** 2 / 4
                direct_counterterm = (
                    c_product
                    * leading
                    * abs(w) ** counterterm_exponent
                    / mpmath.pi
                )
                folded_counterterm = direct_counterterm * (
                    1 + abs(z_mp) ** (-4 - counterterm_exponent)
                )
            return complex(2 * density - folded_counterterm)

    def integrated_t_asymptotic_remainder(
        self, momentum: float, radius: float
    ) -> complex:
        r"""Integrate the explicit BRY local polynomial on a half disk.

        At ``r <= 1e-4`` the folded lens differs from a half disk, and the
        omitted ``|z|`` factors differ from one, only by relative ``O(r)``.
        The retained local blocks are finite sums of ``w^a wbar^b``.  Their
        radial and angular integrals are evaluated meromorphically, so the
        complex endpoint powers and the leading BRY subtraction cancel
        before the result is converted to double precision.
        """

        if radius <= 0.0:
            return 0.0j
        with mpmath.workdps(max(60, self.sphere.liouville.block_working_precision)):
            omega1 = mpmath.mpc(self.omega1)
            omega2 = mpmath.mpc(self.omega2)
            omega3 = mpmath.mpc(self.omega3)
            omega4 = mpmath.mpc(self.omega)
            momentum_mp = mpmath.mpf(momentum)

            def weight(value):
                return (1 + value * value) / 2

            h1 = weight(omega1)
            h2 = weight(omega2)
            h3 = weight(omega3)
            h4 = weight(omega4)
            h = weight(momentum_mp)
            exponent = h - h2 - h3
            primary_1 = (h - h3 + h2) * (h + h1 - h4) / (2 * h)
            primary_half = 1 / (2 * h)
            starred_0 = h2 + h3 - h
            starred_1 = -(
                (h - h3 + h2)
                * (h + h1 - h4)
                * (h - h2 - h3)
                / (2 * h)
            )
            starred_half = (h + h2 + h3 - mpmath.mpf("0.5")) / (2 * h)
            pe = ((1, exponent), (primary_1, exponent + 1))
            po = ((primary_half, exponent + mpmath.mpf("0.5")),)
            se = ((starred_0, exponent - 1), (starred_1, exponent))
            so = ((starred_half, exponent - mpmath.mpf("0.5")),)
            c_raw, ct_raw = self._t_structure_products(momentum)
            c_product = mpmath.mpc(c_raw)
            ct_product = mpmath.mpc(ct_raw)

            def products(left, right, scale=1):
                return [
                    (scale * first * second, first_power, second_power)
                    for first, first_power in left
                    for second, second_power in right
                ]

            g_terms = products(pe, pe, c_product) + products(
                po, po, ct_product
            )
            h_terms = products(se, se, -c_product) + products(
                so, so, -ct_product
            )
            j_terms = []
            j_terms.extend(
                (coefficient, left_power, right_power - 1)
                for coefficient, left_power, right_power in products(
                    se, pe, -c_product
                )
            )
            j_terms.extend(
                (coefficient, left_power - 1, right_power)
                for coefficient, left_power, right_power in products(
                    pe, se, -c_product
                )
            )
            j_terms.extend(
                (coefficient, left_power, right_power - 1)
                for coefficient, left_power, right_power in products(
                    so, po, -ct_product
                )
            )
            j_terms.extend(
                (coefficient, left_power - 1, right_power)
                for coefficient, left_power, right_power in products(
                    po, so, -ct_product
                )
            )
            density_terms = []
            density_terms.extend(
                (omega2**2 * omega3**2 * coefficient, left - 1, right - 1)
                for coefficient, left, right in g_terms
            )
            density_terms.extend(
                (-coefficient, left, right)
                for coefficient, left, right in h_terms
            )
            density_terms.extend(
                (-omega2 * omega3 * coefficient, left, right)
                for coefficient, left, right in j_terms
            )
            timelike_shift = -omega2 * omega3
            density_terms = [
                (
                    2 * coefficient / mpmath.pi,
                    left + timelike_shift,
                    right + timelike_shift,
                )
                for coefficient, left, right in density_terms
            ]

            def z_factor_terms(chiral_power, order=3):
                chiral = tuple(
                    (
                        mpmath.binomial(chiral_power, degree)
                        * (-1) ** degree,
                        degree,
                    )
                    for degree in range(order + 1)
                )
                return tuple(
                    (left_coefficient * right_coefficient, left, right)
                    for left_coefficient, left in chiral
                    for right_coefficient, right in chiral
                )

            # Restore |z|^{-2 omega1 omega2}; its linear term contributes at
            # the very first integrable power after the leading subtraction.
            z_kinematic = z_factor_terms(-omega1 * omega2)
            density_terms = [
                (
                    coefficient * z_coefficient,
                    left + z_left,
                    right + z_right,
                )
                for coefficient, left, right in density_terms
                for z_coefficient, z_left, z_right in z_kinematic
            ]
            if momentum < self.t_threshold:
                counterterm_exponent = (
                    -3 + momentum_mp**2 - (omega2 + omega3) ** 2
                )
                leading = (
                    1 + (omega2 + omega3) ** 2 - momentum_mp**2
                ) ** 2 / 4
                counterterm_coefficient = -c_product * leading / mpmath.pi
                density_terms.append(
                    (
                        counterterm_coefficient,
                        counterterm_exponent / 2,
                        counterterm_exponent / 2,
                    )
                )
                image_power = (-4 - counterterm_exponent) / 2
                density_terms.extend(
                    (
                        counterterm_coefficient * z_coefficient,
                        counterterm_exponent / 2 + z_left,
                        counterterm_exponent / 2 + z_right,
                    )
                    for z_coefficient, z_left, z_right in z_factor_terms(
                        image_power
                    )
                )

            total = mpmath.mpc(0)
            radius_mp = mpmath.mpf(radius)
            for coefficient, left, right in density_terms:
                spin = left - right
                angular = (
                    mpmath.pi
                    if abs(spin) < mpmath.mpf("1e-50")
                    else 2 * mpmath.sin(mpmath.pi * spin / 2) / spin
                )
                radial_power = left + right + 2
                total += (
                    coefficient
                    * angular
                    * radius_mp**radial_power
                    / radial_power
                )
            return complex(total)

    def _radial_intervals_outside_cap(self, theta: float) -> tuple[tuple[float, float], ...]:
        sine = math.sin(theta)
        cosine = math.cos(theta)
        discriminant = self.epsilon * self.epsilon - sine * sine
        if cosine <= 0.0 or discriminant <= 0.0:
            return ((self.epsilon, 1.0),)
        root = math.sqrt(discriminant)
        cap_lower = max(0.0, cosine - root)
        cap_upper = min(1.0, cosine + root)
        if cap_lower >= cap_upper:
            return ((self.epsilon, 1.0),)
        intervals = []
        if cap_lower > self.epsilon:
            intervals.append((self.epsilon, cap_lower))
        if cap_upper < 1.0:
            intervals.append((max(self.epsilon, cap_upper), 1.0))
        return tuple(intervals)

    def _s_disk_integral(self, momentum: float) -> complex:
        """Integrate BRY's low-z expansion with a cusp-resolving radius map."""

        radial_power = 8

        def angular_integrand(theta: float) -> complex:
            phase = complex(math.cos(theta), math.sin(theta))

            def mapped_radial_integrand(unit_radius: float) -> complex:
                radius = self.epsilon * unit_radius**radial_power
                radial_jacobian = (
                    self.epsilon
                    * radial_power
                    * unit_radius ** (radial_power - 1)
                )
                z = radius * phase
                if self.block_backend == "hybrid":
                    value = self.regulated_folded_momentum_density(
                        momentum, z
                    )
                else:
                    value = 2.0 * self.s_local_momentum_density(momentum, z)
                    value -= self.folded_t_counterterm_momentum_density(
                        momentum, z
                    )
                return radius * radial_jacobian * value

            return _integrate_interval(
                mapped_radial_integrand, 0.0, 1.0, self.cap_radial_order
            )

        return _integrate_interval(
            angular_integrand, -math.pi, math.pi, self.cap_angular_order
        )

    def _rest_of_disk_integral(self, momentum: float) -> complex:
        def angular_integrand(theta: float) -> complex:
            phase = complex(math.cos(theta), math.sin(theta))

            def radial_integrand(radius: float) -> complex:
                z = radius * phase
                value = self.regulated_folded_momentum_density(momentum, z)
                return radius * value

            return sum(
                _integrate_interval(
                    radial_integrand, lower, upper, self.radial_order
                )
                for lower, upper in self._radial_intervals_outside_cap(theta)
            )

        # A single Gauss rule on [-pi, pi] misses the narrow boundary layer
        # adjacent to the excised z=1 cap.  The cap endpoints contain
        # sqrt(epsilon^2-sin(theta)^2), so direct theta quadrature also sees a
        # square-root cusp at |theta|=asin(epsilon).  Resolve the central
        # slice with sin(theta)=epsilon*sin(u), which makes that boundary
        # smooth, and use ordinary theta rules only on the shoulders.
        shoulder = max(0.2, 8.0 * self.epsilon)
        cap_angle = math.asin(self.epsilon)
        angular_breaks = (
            -math.pi,
            -shoulder,
            -cap_angle,
            cap_angle,
            shoulder,
            math.pi,
        )
        outer = sum(
            _integrate_interval(
                angular_integrand,
                lower,
                upper,
                self.angular_order,
            )
            for lower, upper in zip(angular_breaks, angular_breaks[1:])
            if not (lower == -cap_angle and upper == cap_angle)
        )

        def mapped_cap_integrand(unit_angle: float) -> complex:
            sine = math.sin(unit_angle)
            cosine = math.cos(unit_angle)
            theta = math.asin(self.epsilon * sine)
            jacobian = (
                self.epsilon
                * cosine
                / math.sqrt(1.0 - self.epsilon**2 * sine**2)
            )
            return jacobian * angular_integrand(theta)

        central = _integrate_interval(
            mapped_cap_integrand,
            -0.5 * math.pi,
            0.5 * math.pi,
            self.angular_order,
        )
        return outer + central

    def _t_cap_integral(self, momentum: float) -> complex:
        if self.block_backend == "hybrid":
            return self._hybrid_t_cap_integral(momentum)

        def radial_integrand(radius: float) -> complex:
            angular_limit = math.acos(0.5 * radius)

            def angular_integrand(phi: float) -> complex:
                w = radius * complex(math.cos(phi), math.sin(phi))
                z = 1.0 - w
                if self.block_backend == "hybrid":
                    value = self.regulated_folded_momentum_density(
                        momentum, z
                    )
                else:
                    value = 2.0 * self.t_local_momentum_density(momentum, z)
                    value -= self.folded_t_counterterm_momentum_density(
                        momentum, z
                    )
                return value

            return radius * _integrate_interval(
                angular_integrand,
                -angular_limit,
                angular_limit,
                self.cap_angular_order,
            )

        return _integrate_interval(
            radial_integrand, 0.0, self.epsilon, self.cap_radial_order
        )

    def _t_cap_switch_radius(
        self, phase: complex, maximum_radius: float
    ) -> float:
        """Locate ``|q_ell(1-r*phase)|=q_threshold`` on one cap ray."""

        if maximum_radius <= 0.0:
            return 0.0

        def nome_magnitude(radius: float) -> float:
            return float(abs(elliptic_nome(1.0 - radius * phase)))

        # At r=0 the s-channel nome tends to one, hence the local c chart is
        # selected.  If the outer endpoint has not entered the h patch there
        # is no interface on this ray.
        if nome_magnitude(maximum_radius) >= self.hybrid_elliptic_nome_threshold:
            return maximum_radius
        lower = 0.0
        upper = maximum_radius
        for _ in range(52):
            midpoint = 0.5 * (lower + upper)
            if nome_magnitude(midpoint) >= self.hybrid_elliptic_nome_threshold:
                lower = midpoint
            else:
                upper = midpoint
        return 0.5 * (lower + upper)

    def _hybrid_t_cap_integral(self, momentum: float) -> complex:
        """Integrate the t cap with the h/c nome interface split exactly."""
        asymptotic_radius = self.hybrid_asymptotic_radius

        def angular_integrand(phi: float) -> complex:
            phase = complex(math.cos(phi), math.sin(phi))
            maximum_radius = min(self.epsilon, max(0.0, 2.0 * math.cos(phi)))
            if maximum_radius <= 0.0:
                return 0.0j
            switch = self._t_cap_switch_radius(phase, maximum_radius)

            def full_radial_integrand(radius: float) -> complex:
                z = 1.0 - radius * phase
                return radius * self.regulated_folded_momentum_density(
                    momentum, z
                )

            local_upper = min(asymptotic_radius, maximum_radius)
            c_upper = min(switch, maximum_radius)
            return (
                _integrate_interval(
                    full_radial_integrand,
                    local_upper,
                    c_upper,
                    self.cap_radial_order,
                )
                + _integrate_interval(
                    full_radial_integrand,
                    max(switch, local_upper),
                    maximum_radius,
                    self.cap_radial_order,
                )
            )

        shoulder = math.acos(0.5 * self.epsilon)
        intervals = (
            (-0.5 * math.pi, -shoulder),
            (-shoulder, shoulder),
            (shoulder, 0.5 * math.pi),
        )
        numerical_outer = sum(
            _integrate_interval(
                angular_integrand, lower, upper, self.cap_angular_order
            )
            for lower, upper in intervals
        )
        return numerical_outer + self.integrated_t_asymptotic_remainder(
            momentum, asymptotic_radius
        )

    def z_integral_at_momentum(self, momentum: float) -> complex:
        """Folded full-plane z integral at fixed internal momentum P."""

        return (
            self._s_disk_integral(momentum)
            + self._rest_of_disk_integral(momentum)
            + self._t_cap_integral(momentum)
        )

    def evaluate_worldsheet(self) -> OneToThreeWorldsheetResult:
        """Evaluate the regulated moduli integral without constructing a target."""

        region_functions = (
            self._s_disk_integral,
            self._rest_of_disk_integral,
            self._t_cap_integral,
        )
        region_totals = [0.0j, 0.0j, 0.0j]
        threshold = min(self.t_threshold, self.p_max)
        for lower, upper in ((0.0, threshold), (threshold, self.p_max)):
            if upper <= lower:
                continue
            midpoint = 0.5 * (lower + upper)
            scale = 0.5 * (upper - lower)
            for node, weight in _legendre_rule(self.p_quadrature_order):
                momentum = midpoint + scale * node
                measured = tuple(function(momentum) for function in region_functions)
                for index, value in enumerate(measured):
                    region_totals[index] += scale * weight * value
        low_z, bulk, t_cap = (complex(value) for value in region_totals)
        reduced = low_z + bulk + t_cap
        worldsheet_amplitude = 8j * reduced / math.pi
        return OneToThreeWorldsheetResult(
            omega=self.omega,
            omega_out=self.omega1,
            epsilon=self.epsilon,
            p_max=self.p_max,
            p_quadrature_order=self.p_quadrature_order,
            angular_order=self.angular_order,
            radial_order=self.radial_order,
            cap_angular_order=self.cap_angular_order,
            cap_radial_order=self.cap_radial_order,
            block_q_order=self.block_q_order,
            block_backend=self.block_backend,
            hybrid_corner_radius=self.hybrid_corner_radius,
            hybrid_elliptic_nome_threshold=(
                self.hybrid_elliptic_nome_threshold
            ),
            hybrid_asymptotic_radius=self.hybrid_asymptotic_radius,
            low_z_region_integral=low_z,
            bulk_region_integral=bulk,
            t_cap_region_integral=t_cap,
            reduced_moduli_integral=reduced,
            worldsheet_amplitude_coefficient=worldsheet_amplitude,
        )

    def evaluate(self) -> OneToThreeResult:
        """Evaluate the worldsheet first and only then compare with BRY (2.13)."""

        worldsheet = self.evaluate_worldsheet()
        target = self.reduced_moduli_target
        relative_error = abs(worldsheet.reduced_moduli_integral - target) / abs(target)
        return OneToThreeResult(
            **asdict(worldsheet),
            reduced_moduli_target=target,
            matrix_amplitude_coefficient=self.matrix_amplitude_coefficient,
            relative_error=relative_error,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incoming-imaginary", type=float, default=0.6)
    parser.add_argument("--epsilon", type=float, default=1.0e-2)
    parser.add_argument("--p-max", type=float, default=4.0)
    parser.add_argument("--p-order", type=int, default=16)
    parser.add_argument("--angular-order", type=int, default=10)
    parser.add_argument("--radial-order", type=int, default=10)
    parser.add_argument("--cap-angular-order", type=int, default=10)
    parser.add_argument("--cap-radial-order", type=int, default=8)
    parser.add_argument("--block-q-order", type=int, default=8)
    parser.add_argument(
        "--block-backend",
        choices=("hybrid", "h", "c"),
        default="h",
    )
    parser.add_argument("--hybrid-corner-radius", type=float, default=0.15)
    parser.add_argument(
        "--hybrid-elliptic-nome-threshold", type=float, default=0.3
    )
    parser.add_argument("--hybrid-asymptotic-radius", type=float, default=1.0e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark = BRYOneToThreeBenchmark(
        incoming_imaginary=args.incoming_imaginary,
        epsilon=args.epsilon,
        p_max=args.p_max,
        p_quadrature_order=args.p_order,
        angular_order=args.angular_order,
        radial_order=args.radial_order,
        cap_angular_order=args.cap_angular_order,
        cap_radial_order=args.cap_radial_order,
        block_q_order=args.block_q_order,
        block_backend=args.block_backend,
        hybrid_corner_radius=args.hybrid_corner_radius,
        hybrid_elliptic_nome_threshold=(
            args.hybrid_elliptic_nome_threshold
        ),
        hybrid_asymptotic_radius=args.hybrid_asymptotic_radius,
    )
    result = benchmark.evaluate()
    print(json.dumps(result.json_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BRYOneToThreeBenchmark",
    "OneToThreeResult",
    "OneToThreeWorldsheetResult",
]
