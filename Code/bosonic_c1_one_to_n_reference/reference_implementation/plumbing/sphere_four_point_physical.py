#!/usr/bin/env python3
"""Direct equal-energy sphere-four amplitude with the BRY subtraction.

This is the executable baby step for the sphere-five physical integrator. The
complex frequencies obey energy conservation before the epsilon limit. The
raw correlator uses the fastest of six elliptic crossing sectors.  The default
integrator excises three OPE collars and restores their spin-zero terms with
the analytic BRY radial finite part; the equivalent global-counterterm form
is retained as an independent diagnostic.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

try:
    from ccy_sphere_four_point import (
        sphere_four_point_ccy_coefficients,
        sphere_four_point_elliptic_descendant_block,
        sphere_four_point_elliptic_h_coefficients,
    )
    from liouville_torus import UpsilonB, log_yin_structure_constant_momentum
    from sphere_four_point_subtraction import (
        bry_divergent_levels,
        bry_regular_chiral_coefficients,
        bry_s_channel_projector,
        evaluate_series,
        equal_one_to_three_frequencies,
    )
    from sphere_five_point_equal_energy import _gauss_legendre_grid
    from torus_two_point_blocks import elliptic_nome
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.ccy_sphere_four_point import (
        sphere_four_point_ccy_coefficients,
        sphere_four_point_elliptic_descendant_block,
        sphere_four_point_elliptic_h_coefficients,
    )
    from plumbing.liouville_torus import (
        UpsilonB,
        log_yin_structure_constant_momentum,
    )
    from plumbing.sphere_four_point_subtraction import (
        bry_divergent_levels,
        bry_regular_chiral_coefficients,
        bry_s_channel_projector,
        evaluate_series,
        equal_one_to_three_frequencies,
    )
    from plumbing.sphere_five_point_equal_energy import _gauss_legendre_grid
    from plumbing.torus_two_point_blocks import elliptic_nome


@dataclass(frozen=True)
class FourPointMomentumEntry:
    momentum: float
    weighted_structure_constant: complex
    block_coefficients: tuple[complex, ...]
    elliptic_block_coefficients: tuple[complex, ...]
    regular_coefficients: tuple[complex, ...]


@dataclass(frozen=True)
class FourPointPhysicalResult:
    outgoing_energy: float
    epsilon: float
    estimates: tuple[complex, ...]
    mean: complex
    standard_error_real: float
    standard_error_imag: float
    collar_radius: float | None = None
    analytic_collar_contribution: complex = 0.0 + 0.0j


def _crossing_frames(z: complex) -> tuple[tuple[complex, complex], ...]:
    """Return the six anharmonic cross ratios and their derivatives."""

    z = complex(z)
    if z == 0.0 or z == 1.0:
        raise ZeroDivisionError("a crossing frame hit a puncture")
    return (
        (z, 1.0 + 0.0j),
        (1.0 - z, -1.0 + 0.0j),
        (1.0 / z, -1.0 / z**2),
        (1.0 / (1.0 - z), 1.0 / (1.0 - z) ** 2),
        (z / (z - 1.0), -1.0 / (z - 1.0) ** 2),
        ((z - 1.0) / z, 1.0 / z**2),
    )


class EqualOneToThreeBRYKernel:
    """Precomputed one-momentum four-point BRY kernel."""

    def __init__(
        self,
        outgoing_energy: float,
        epsilon: float,
        *,
        block_order: int = 8,
        momentum_order: int = 12,
        momentum_maximum: float = 6.0,
        momentum_panels: int = 1,
        special_dps: int = 35,
    ) -> None:
        frequencies = equal_one_to_three_frequencies(outgoing_energy, epsilon)
        self.outgoing_energy = float(outgoing_energy)
        self.epsilon = float(epsilon)
        self.incoming = frequencies[0]
        self.outgoing = frequencies[1]
        self.channel_energy = self.incoming - self.outgoing
        self._initialize_entries(
            block_order=block_order,
            momentum_order=momentum_order,
            momentum_maximum=momentum_maximum,
            momentum_panels=momentum_panels,
            special_dps=special_dps,
        )

    def _initialize_entries(
        self,
        *,
        block_order: int,
        momentum_order: int,
        momentum_maximum: float,
        momentum_panels: int,
        special_dps: int,
    ) -> None:
        """Build data shared by the physical and convergent-ray kernels."""

        self.block_order = int(block_order)
        incoming_momentum = 0.5 * self.incoming
        outgoing_momentum = 0.5 * self.outgoing
        external_weights = (
            1.0 + outgoing_momentum**2,
            1.0 + incoming_momentum**2,
            1.0 + outgoing_momentum**2,
            1.0 + outgoing_momentum**2,
        )
        self.external_weights = tuple(external_weights)
        threshold = 0.25 * (self.channel_energy**2).real
        endpoints = [
            math.sqrt(threshold - level)
            for level in range(self.block_order + 1)
            if threshold > level
        ]
        # The physical imaginary part is concentrated near these endpoints.
        # Add symmetric panels on the scale set by Im(kappa^2), in addition
        # to splitting exactly at every endpoint.
        refined_breakpoints = list(endpoints)
        endpoint_width = abs((self.channel_energy**2).imag)
        for endpoint in endpoints:
            local_width = endpoint_width / max(4.0 * endpoint, 1.0e-12)
            for multiple in (0.25, 0.5, 1.0, 2.0, 4.0):
                refined_breakpoints.extend(
                    (endpoint - multiple * local_width, endpoint + multiple * local_width)
                )
        nodes, weights = _gauss_legendre_grid(
            int(momentum_order),
            float(momentum_maximum),
            int(momentum_panels),
            refined_breakpoints,
        )
        special = UpsilonB(1.0, dps=int(special_dps))
        entries = []
        for momentum, weight in zip(nodes, weights):
            block = sphere_four_point_ccy_coefficients(
                central_charge=25.0,
                external_weights=external_weights,
                internal_weight=1.0 + float(momentum) ** 2,
                order=self.block_order,
            )
            regular = bry_regular_chiral_coefficients(
                block,
                0.5 * self.incoming * self.outgoing,
                self.block_order,
            )
            elliptic_block = sphere_four_point_elliptic_h_coefficients(
                block,
                central_charge=25.0,
                external_weights=self.external_weights,
                internal_weight=1.0 + float(momentum) ** 2,
            )
            logarithm = (
                math.log(float(weight) / math.pi)
                + complex(
                    log_yin_structure_constant_momentum(
                        special,
                        incoming_momentum,
                        outgoing_momentum,
                        float(momentum),
                    )
                )
                + complex(
                    log_yin_structure_constant_momentum(
                        special,
                        outgoing_momentum,
                        outgoing_momentum,
                        float(momentum),
                    )
                )
            )
            entries.append(
                FourPointMomentumEntry(
                    momentum=float(momentum),
                    weighted_structure_constant=cmath.exp(logarithm),
                    block_coefficients=tuple(block),
                    elliptic_block_coefficients=tuple(elliptic_block),
                    regular_coefficients=tuple(regular),
                )
            )
        self.entries = tuple(entries)

    def analytic_local_collar_integral(self, collar_radius: float) -> complex:
        """Integrate ``raw-R_s`` over one OPE disk analytically.

        Angular integration retains only equal holomorphic and
        antiholomorphic levels.  Divergent levels cancel the BRY projector;
        all remaining levels give ``2*pi*a_n^2*rho^x/x``.  This is the
        finite-radius version of equations (3.9)--(3.13) and captures the
        threshold contribution that is poorly sampled in Cartesian moduli.
        """

        collar_radius = float(collar_radius)
        if not 0.0 < collar_radius < 0.25:
            raise ValueError("collar_radius must lie between zero and 1/4")
        logarithmic_radius = math.log(collar_radius)
        total = 0.0 + 0.0j
        for entry in self.entries:
            divergent = set(
                bry_divergent_levels(
                    self.channel_energy,
                    entry.momentum,
                    len(entry.regular_coefficients) - 1,
                )
            )
            for level, coefficient in enumerate(entry.regular_coefficients):
                if level in divergent:
                    continue
                radial_power = (
                    2.0 * (entry.momentum**2 + level)
                    - 0.5 * self.channel_energy**2
                )
                total += (
                    entry.weighted_structure_constant
                    * 2.0
                    * math.pi
                    * coefficient**2
                    * cmath.exp(radial_power * logarithmic_radius)
                    / radial_power
                )
        return complex(total)

    def analytic_local_finite_part(self, collar_radius: float) -> complex:
        """Analytically continue the complete spin-zero OPE collar.

        Unlike :meth:`analytic_local_collar_integral`, this includes the
        power-divergent levels through ``2*pi*rho^x/x``.  Combining it with
        the raw integral outside three disjoint collars is algebraically
        equivalent to the global BRY counterterms, but avoids numerically
        integrating their long-range extensions.
        """

        collar_radius = float(collar_radius)
        if not 0.0 < collar_radius < 0.25:
            raise ValueError("collar_radius must lie between zero and 1/4")
        logarithmic_radius = math.log(collar_radius)
        total = 0.0 + 0.0j
        for entry in self.entries:
            for level, coefficient in enumerate(entry.regular_coefficients):
                radial_power = (
                    2.0 * (entry.momentum**2 + level)
                    - 0.5 * self.channel_energy**2
                )
                total += (
                    entry.weighted_structure_constant
                    * 2.0
                    * math.pi
                    * coefficient**2
                    * cmath.exp(radial_power * logarithmic_radius)
                    / radial_power
                )
        return complex(total)

    def raw_density(self, z: complex) -> complex:
        """Evaluate the raw correlator in the fastest of six crossing frames."""

        local, derivative = min(_crossing_frames(z), key=lambda item: abs(item[0]))
        return complex(self.local_raw_density(local) * abs(derivative) ** 2)

    def raw_outside_collars_density(self, z: complex, collar_radius: float) -> complex:
        """Return the raw density with three disjoint OPE collars excised."""

        z = complex(z)
        collar_radius = float(collar_radius)
        if (
            abs(z) < collar_radius
            or abs(z - 1.0) < collar_radius
            or abs(z) > 1.0 / collar_radius
        ):
            return 0.0 + 0.0j
        return self.raw_density(z)

    def collar_remainder_density(self, z: complex, collar_radius: float) -> complex:
        """Return the globally subtracted density with local collars removed."""

        z = complex(z)
        collar_radius = float(collar_radius)
        regulator_s = self.local_counterterm_density(z)
        regulator_t = self.local_counterterm_density(z - 1.0)
        regulator_u = abs(z) ** -4 * self.local_counterterm_density(1.0 / z)
        if abs(z) < collar_radius:
            return complex(-regulator_t - regulator_u)
        if abs(z - 1.0) < collar_radius:
            return complex(-regulator_s - regulator_u)
        if abs(z) > 1.0 / collar_radius:
            return complex(-regulator_s - regulator_t)
        return self.subtracted_density(z)

    def local_raw_density(self, z: complex) -> complex:
        """Return the unregularized density in one local channel."""

        z = complex(z)
        if z == 0.0:
            raise ZeroDivisionError("the local coordinate lies on the boundary")
        log_radius = math.log(abs(z))
        nome = elliptic_nome(z)
        regular_timelike_holomorphic = (1.0 - z) ** (
            0.5 * self.incoming * self.outgoing
        )
        regular_timelike_antiholomorphic = (1.0 - z.conjugate()) ** (
            0.5 * self.incoming * self.outgoing
        )
        total = 0.0 + 0.0j
        for entry in self.entries:
            base_power = (
                -2.0
                - 0.5 * self.channel_energy**2
                + 2.0 * entry.momentum**2
            )
            primary = cmath.exp(base_power * log_radius)
            block_holomorphic = sphere_four_point_elliptic_descendant_block(
                z,
                entry.elliptic_block_coefficients,
                central_charge=25.0,
                external_weights=self.external_weights,
                internal_weight=1.0 + entry.momentum**2,
                nome=nome,
            )
            block_antiholomorphic = sphere_four_point_elliptic_descendant_block(
                z.conjugate(),
                entry.elliptic_block_coefficients,
                central_charge=25.0,
                external_weights=self.external_weights,
                internal_weight=1.0 + entry.momentum**2,
                nome=elliptic_nome(z.conjugate()),
            )
            original = (
                primary
                * regular_timelike_holomorphic
                * regular_timelike_antiholomorphic
                * block_holomorphic
                * block_antiholomorphic
            )
            total += entry.weighted_structure_constant * original
        return complex(total)

    def local_counterterm_density(self, z: complex) -> complex:
        """Return R_s of BRY equation (3.13) in its local coordinate."""

        z = complex(z)
        if z == 0.0:
            raise ZeroDivisionError("the local coordinate lies on the boundary")
        return complex(
            sum(
                entry.weighted_structure_constant
                * bry_s_channel_projector(
                    z,
                    channel_energy=self.channel_energy,
                    momentum=entry.momentum,
                    regular_chiral_coefficients=entry.regular_coefficients,
                )
                for entry in self.entries
            )
        )

    def subtracted_density(self, z: complex) -> complex:
        """Evaluate the exact equal-energy form of BRY equation (3.15)."""

        z = complex(z)
        raw = self.raw_density(z)
        regulator_s = self.local_counterterm_density(z)
        regulator_t = self.local_counterterm_density(z - 1.0)
        regulator_u = (
            abs(z) ** -4 * self.local_counterterm_density(1.0 / z)
        )
        return complex(raw - regulator_s - regulator_t - regulator_u)


def _plane_sample(first: float, second: float) -> tuple[complex, float]:
    radius = math.tan(0.5 * math.pi * float(first))
    angle = 2.0 * math.pi * float(second)
    jacobian = math.pi**2 * radius * (1.0 + radius**2)
    return complex(radius * cmath.exp(1.0j * angle)), jacobian


def integrate_equal_one_to_three_bry(
    kernel: EqualOneToThreeBRYKernel,
    *,
    sobol_power: int = 8,
    replicates: int = 4,
    seed: int = 7103,
    collar_radius: float | None = 0.08,
    subtraction_scheme: str = "local_finite_part",
) -> FourPointPhysicalResult:
    """Integrate one physical four-point amplitude without an omega fit."""

    if subtraction_scheme not in {"local_finite_part", "global_counterterm"}:
        raise ValueError(
            "subtraction_scheme must be 'local_finite_part' or 'global_counterterm'"
        )
    analytic_collar = 0.0 + 0.0j
    if collar_radius is not None:
        collar_radius = float(collar_radius)
        if subtraction_scheme == "local_finite_part":
            analytic_collar = 3.0 * kernel.analytic_local_finite_part(collar_radius)
        else:
            analytic_collar = 3.0 * kernel.analytic_local_collar_integral(collar_radius)
    estimates = []
    for replicate in range(int(replicates)):
        sampler = qmc.Sobol(d=2, scramble=True, seed=int(seed) + replicate)
        values = []
        for point in sampler.random_base2(int(sobol_power)):
            z, jacobian = _plane_sample(point[0], point[1])
            if collar_radius is None:
                density = kernel.subtracted_density(z)
            elif subtraction_scheme == "local_finite_part":
                density = kernel.raw_outside_collars_density(z, collar_radius)
            else:
                density = kernel.collar_remainder_density(z, collar_radius)
            values.append(density * jacobian)
        estimates.append(
            complex(np.mean(np.asarray(values, dtype=complex))) + analytic_collar
        )
    array = np.asarray(estimates, dtype=complex)
    return FourPointPhysicalResult(
        outgoing_energy=kernel.outgoing_energy,
        epsilon=kernel.epsilon,
        estimates=tuple(estimates),
        mean=complex(np.mean(array)),
        standard_error_real=float(np.std(array.real, ddof=1) / math.sqrt(replicates)),
        standard_error_imag=float(np.std(array.imag, ddof=1) / math.sqrt(replicates)),
        collar_radius=collar_radius,
        analytic_collar_contribution=complex(analytic_collar),
    )
