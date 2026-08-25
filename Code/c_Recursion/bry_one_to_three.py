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
from typing import Callable, Sequence

import mpmath

from sphere_four_point import BRYFourTachyonSphere, FourPointCorrelators
from super_liouville_structure_constants import (
    ns_structure_constant,
    ns_tilde_structure_constant,
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
class OneToThreeResult:
    """Numerical worldsheet result and the BRY/MQM comparison target."""

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
    reduced_moduli_integral: complex
    reduced_moduli_target: complex
    worldsheet_amplitude_coefficient: complex
    matrix_amplitude_coefficient: complex
    relative_error: float

    def json_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key, value in tuple(data.items()):
            if isinstance(value, complex):
                data[key] = {"real": value.real, "imag": value.imag}
        return data


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
        structure_precision: int = 30,
        block_working_precision: int = 60,
    ) -> None:
        if not math.isfinite(incoming_imaginary):
            raise ValueError("incoming_imaginary must be finite")
        if epsilon <= 0.0 or epsilon >= 0.25:
            raise ValueError("epsilon must lie between 0 and 1/4")
        if p_max <= 0.0 or not math.isfinite(p_max):
            raise ValueError("p_max must be positive and finite")
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
        self.structure_precision = int(structure_precision)
        self._t_structure_cache: dict[float, tuple[complex, complex]] = {}
        self.sphere = BRYFourTachyonSphere(
            omega=self.omega,
            omega1=self.omega1,
            omega2=self.omega2,
            omega3=self.omega3,
            bry_q_order=self.block_q_order,
            structure_precision=self.structure_precision,
            block_working_precision=block_working_precision,
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
                value = 2.0 * self.s_local_momentum_density(momentum, z)
                value -= self.folded_t_counterterm_momentum_density(momentum, z)
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
                value = 2.0 * self.direct_momentum_density(momentum, z)
                value -= self.folded_t_counterterm_momentum_density(momentum, z)
                return radius * value

            return sum(
                _integrate_interval(
                    radial_integrand, lower, upper, self.radial_order
                )
                for lower, upper in self._radial_intervals_outside_cap(theta)
            )

        # A single Gauss rule on [-pi, pi] misses the narrow boundary layer
        # |theta| = O(epsilon) adjacent to the excised z=1 cap.  Resolve that
        # layer explicitly, with two wider shoulders for the remaining rapid
        # angular variation.
        shoulder = max(0.2, 8.0 * self.epsilon)
        angular_breaks = (
            -math.pi,
            -shoulder,
            -self.epsilon,
            self.epsilon,
            shoulder,
            math.pi,
        )
        return sum(
            _integrate_interval(
                angular_integrand,
                lower,
                upper,
                self.angular_order,
            )
            for lower, upper in zip(angular_breaks, angular_breaks[1:])
        )

    def _t_cap_integral(self, momentum: float) -> complex:
        def radial_integrand(radius: float) -> complex:
            angular_limit = math.acos(0.5 * radius)

            def angular_integrand(phi: float) -> complex:
                w = radius * complex(math.cos(phi), math.sin(phi))
                z = 1.0 - w
                value = 2.0 * self.t_local_momentum_density(momentum, z)
                value -= self.folded_t_counterterm_momentum_density(momentum, z)
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

    def z_integral_at_momentum(self, momentum: float) -> complex:
        """Folded full-plane z integral at fixed internal momentum P."""

        return (
            self._s_disk_integral(momentum)
            + self._rest_of_disk_integral(momentum)
            + self._t_cap_integral(momentum)
        )

    def evaluate(self) -> OneToThreeResult:
        """Evaluate the regulated moduli integral and compare with BRY (2.13)."""

        below = _integrate_interval(
            self.z_integral_at_momentum,
            0.0,
            min(self.t_threshold, self.p_max),
            self.p_quadrature_order,
        )
        above = _integrate_interval(
            self.z_integral_at_momentum,
            min(self.t_threshold, self.p_max),
            self.p_max,
            self.p_quadrature_order,
        )
        reduced = below + above
        worldsheet_amplitude = 8j * reduced / math.pi
        target = self.reduced_moduli_target
        relative_error = abs(reduced - target) / abs(target)
        return OneToThreeResult(
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
            reduced_moduli_integral=reduced,
            reduced_moduli_target=target,
            worldsheet_amplitude_coefficient=worldsheet_amplitude,
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
    )
    result = benchmark.evaluate()
    print(json.dumps(result.json_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BRYOneToThreeBenchmark", "OneToThreeResult"]
