#!/usr/bin/env python3
"""Blind sphere ``1->3`` worldsheet integral on ``omega=i*t``.

The only input is the gauge-fixed worldsheet correlator.  In the chamber
``0<t<1/2`` the Liouville momenta remain on ``P>=0`` and every Deligne--
Mumford boundary is locally integrable, so neither a BRY finite-part
subtraction nor a DOZZ residue is used.

The moduli integral is sampled with an exact three-chart mixture resolving
the degenerations at ``z=0,1,infinity`` symmetrically.  This module contains
no matrix-model function or fitted target coefficients.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import qmc

try:
    from sphere_four_point_physical import EqualOneToThreeBRYKernel
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.sphere_four_point_physical import EqualOneToThreeBRYKernel


FIRST_RESIDUE_WALL = 0.5


@dataclass(frozen=True)
class FourPointImaginaryResult:
    t: float
    omega: complex
    estimates: tuple[complex, ...]
    mean: complex
    standard_error_real: float
    standard_error_imag: float
    samples_per_replicate: int
    replicates: int
    block_order: int
    momentum_order: int
    radial_power: float


class ImaginaryOneToThreeKernel(EqualOneToThreeBRYKernel):
    """Precomputed subtraction-free kernel for equal outgoing energies."""

    def __init__(
        self,
        t: float,
        *,
        block_order: int = 8,
        momentum_order: int = 20,
        momentum_maximum: float = 6.0,
        momentum_panels: int = 1,
        special_dps: int = 35,
    ) -> None:
        t = float(t)
        if not math.isfinite(t) or not 0.0 < t < FIRST_RESIDUE_WALL:
            raise ValueError("the residue-free kernel requires 0<t<1/2")
        self.t = t
        self.omega = 1.0j * t
        self.outgoing_energy = self.omega
        self.epsilon = 0.0
        self.outgoing = self.omega
        self.incoming = 3.0 * self.omega
        self.channel_energy = self.incoming - self.outgoing
        self.momentum_order = int(momentum_order)
        self.momentum_maximum = float(momentum_maximum)
        self.momentum_panels = int(momentum_panels)
        self._initialize_entries(
            block_order=block_order,
            momentum_order=momentum_order,
            momentum_maximum=momentum_maximum,
            momentum_panels=momentum_panels,
            special_dps=special_dps,
        )

    @property
    def leading_boundary_radial_power(self) -> float:
        """Return the positive ``r``-integration power at internal ``P=0``."""

        return float((-0.5 * self.channel_energy**2).real)


def _logsumexp(values: list[float]) -> float:
    largest = max(values)
    return largest + math.log(sum(math.exp(value - largest) for value in values))


def _power_disk_sample(
    radial_uniform: float,
    angular_uniform: float,
    radial_power: float,
) -> complex:
    u = min(
        max(float(radial_uniform), np.nextafter(0.0, 1.0)),
        np.nextafter(1.0, 0.0),
    )
    radius = math.exp(math.log(u) / float(radial_power))
    angle = 2.0 * math.pi * float(angular_uniform)
    return complex(radius * cmath.exp(1.0j * angle))


def three_chart_log_mixture_density(z: complex, radial_power: float) -> float:
    """Return the exact proposal log-density with respect to ``d^2 z``."""

    z = complex(z)
    radial_power = float(radial_power)
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    base = math.log(radial_power / (2.0 * math.pi))
    terms: list[float] = []
    radius_zero = abs(z)
    radius_one = abs(1.0 - z)
    if 0.0 < radius_zero < 1.0:
        terms.append(base + (radial_power - 2.0) * math.log(radius_zero))
    if 0.0 < radius_one < 1.0:
        terms.append(base + (radial_power - 2.0) * math.log(radius_one))
    if radius_zero > 1.0:
        terms.append(base + (-radial_power - 2.0) * math.log(radius_zero))
    if not terms:
        raise ArithmeticError("the three-chart proposal failed to cover a point")
    return _logsumexp(terms) - math.log(3.0)


def integrate_convergent_one_to_three_atlas_qmc(
    kernel: ImaginaryOneToThreeKernel,
    *,
    sobol_power: int = 11,
    replicates: int = 8,
    radial_power: float | None = None,
    seed: int = 20260824,
) -> FourPointImaginaryResult:
    """Integrate the raw labelled four-point correlator over ``Mbar_0,4``."""

    sobol_power = int(sobol_power)
    replicates = int(replicates)
    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    if radial_power is None:
        radial_power = kernel.leading_boundary_radial_power
    radial_power = float(radial_power)
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")

    estimates: list[complex] = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=3, scramble=True, seed=int(seed) + replicate)
        values: list[complex] = []
        for point in sampler.random_base2(sobol_power):
            local = _power_disk_sample(point[0], point[1], radial_power)
            chart = min(int(point[2] * 3), 2)
            if chart == 0:
                z = local
            elif chart == 1:
                z = 1.0 - local
            else:
                z = 1.0 / local
            log_proposal = three_chart_log_mixture_density(z, radial_power)
            values.append(kernel.raw_density(z) * math.exp(-log_proposal))
        estimates.append(complex(np.mean(np.asarray(values, dtype=complex))))

    estimate_array = np.asarray(estimates, dtype=complex)
    return FourPointImaginaryResult(
        t=kernel.t,
        omega=kernel.omega,
        estimates=tuple(estimates),
        mean=complex(np.mean(estimate_array)),
        standard_error_real=float(
            np.std(estimate_array.real, ddof=1) / math.sqrt(replicates)
        ),
        standard_error_imag=float(
            np.std(estimate_array.imag, ddof=1) / math.sqrt(replicates)
        ),
        samples_per_replicate=2**sobol_power,
        replicates=replicates,
        block_order=kernel.block_order,
        momentum_order=kernel.momentum_order,
        radial_power=radial_power,
    )

