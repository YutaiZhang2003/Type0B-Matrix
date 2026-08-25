#!/usr/bin/env python3
"""Reusable equal-energy kernel for the sphere five-point worldsheet integral.

For ``omega_2=...=omega_5=omega`` the incoming energy is ``4*omega``.
Only the slot occupied by the incoming primary distinguishes the external
weights in an oriented linear channel.  This module therefore precomputes the
three DOZZ factors and the five-point block coefficient table on a fixed
``(P1,P2)`` quadrature grid for each of the five possible incoming slots.
Moduli evaluations then require only bivariate polynomial evaluation and
elementary powers, rather than a fresh h-recursion and Upsilon evaluation.

The QMC driver at the bottom is a diagnostic for convergent complex-energy
points.  The physical real-energy calculation must use the boundary forest
in ``sphere_five_point_subtraction.py``; the raw driver deliberately rejects
energies for which a power subtraction is required.
"""

from __future__ import annotations

import cmath
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.special import roots_jacobi
from scipy.stats import qmc

try:
    from ccy_sphere_five_point import (
        CoefficientTable,
        evaluate_sphere_five_point_series,
        sphere_five_point_c_coefficients,
        sphere_five_point_h_c25_limit,
    )
    from liouville_torus import UpsilonB, log_yin_structure_constant_momentum
    from sphere_five_point_liouville import (
        INFINITY,
        LinearChannel,
        best_linear_channels,
        linear_channel_complex_jacobian_to_chart,
        linear_channel_from_ordering,
        linear_channel_positions_by_label,
        linear_channel_to_original_chart,
        liouville_primary_covariance,
        liouville_primary_covariance_log,
        oriented_tree_orderings,
        timelike_free_boson_factor,
        timelike_free_boson_log_factor,
    )
    from sphere_five_point_subtraction import (
        BoundaryCorner,
        BoundaryDivisor,
        canonical_corner_ordering,
        canonical_divisor_ordering,
        divergent_spin_zero_levels,
        equal_outgoing_signed_energies,
        five_point_boundary_corners,
        five_point_face_sector_orderings,
        five_point_plumbing_channel_energies,
        five_point_plumbing_radial_exponents,
        five_point_regular_factor_coefficients,
        five_point_boundary_divisors,
        signed_channel_energy,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_sphere_five_point import (
        CoefficientTable,
        evaluate_sphere_five_point_series,
        sphere_five_point_c_coefficients,
        sphere_five_point_h_c25_limit,
    )
    from plumbing.liouville_torus import UpsilonB, log_yin_structure_constant_momentum
    from plumbing.sphere_five_point_liouville import (
        INFINITY,
        LinearChannel,
        best_linear_channels,
        linear_channel_complex_jacobian_to_chart,
        linear_channel_from_ordering,
        linear_channel_positions_by_label,
        linear_channel_to_original_chart,
        liouville_primary_covariance,
        liouville_primary_covariance_log,
        oriented_tree_orderings,
        timelike_free_boson_factor,
        timelike_free_boson_log_factor,
    )
    from plumbing.sphere_five_point_subtraction import (
        BoundaryCorner,
        BoundaryDivisor,
        canonical_corner_ordering,
        canonical_divisor_ordering,
        divergent_spin_zero_levels,
        equal_outgoing_signed_energies,
        five_point_boundary_corners,
        five_point_face_sector_orderings,
        five_point_plumbing_channel_energies,
        five_point_plumbing_radial_exponents,
        five_point_regular_factor_coefficients,
        five_point_boundary_divisors,
        signed_channel_energy,
    )


@dataclass(frozen=True)
class MomentumGridEntry:
    p1: complex | float
    p2: complex | float
    h1: complex | float
    h2: complex | float
    log_weighted_structure_constant: complex
    weighted_structure_constant: complex
    coefficients: CoefficientTable
    coefficient_errors: CoefficientTable


@dataclass(frozen=True)
class MomentumGridArrays:
    """Vectorized form of one incoming-slot momentum quadrature."""

    log_weighted_structure_constants: np.ndarray
    primary_x_exponents: np.ndarray
    primary_y_exponents: np.ndarray
    coefficient_keys: tuple[tuple[int, int], ...]
    coefficient_matrix: np.ndarray
    p1_values: np.ndarray
    p2_values: np.ndarray
    regular_coefficient_keys: tuple[tuple[int, int], ...]
    regular_coefficient_matrix: np.ndarray
    plumbing_q1_exponents: np.ndarray
    plumbing_q2_exponents: np.ndarray
    channel_energy1: complex
    channel_energy2: complex
    regular_timelike_exponents: tuple[complex, complex, complex]


@dataclass(frozen=True)
class ContinuedMomentumGridArrays:
    """Vectorized one-dimensional residue contribution after a pole crossing."""

    log_weighted_structure_constants: np.ndarray
    primary_x_exponents: np.ndarray
    primary_y_exponents: np.ndarray
    coefficient_keys: tuple[tuple[int, int], ...]
    coefficient_matrix: np.ndarray


@dataclass(frozen=True)
class EqualEnergyQMCResult:
    omega: complex
    estimates: tuple[complex, ...]
    mean: complex
    standard_error_real: float
    standard_error_imag: float
    samples_per_replicate: int
    replicates: int
    block_order: int
    momentum_order: int
    block_scheme: str

    def to_json(self) -> dict[str, object]:
        return {
            "omega_real": self.omega.real,
            "omega_imag": self.omega.imag,
            "estimate_real": self.mean.real,
            "estimate_imag": self.mean.imag,
            "standard_error_real": self.standard_error_real,
            "standard_error_imag": self.standard_error_imag,
            "replicate_estimates": [
                {"real": value.real, "imag": value.imag} for value in self.estimates
            ],
            "samples_per_replicate": self.samples_per_replicate,
            "replicates": self.replicates,
            "block_order": self.block_order,
            "momentum_order": self.momentum_order,
            "block_scheme": self.block_scheme,
        }


@dataclass(frozen=True)
class FivePointFinitePartResult:
    """Bulk, face, and corner strata of the physical five-point integral."""

    omega: complex
    collar_radius: float
    estimates: tuple[complex, ...]
    bulk_estimates: tuple[complex, ...]
    face_estimates: tuple[complex, ...]
    corner_contribution: complex
    mean: complex
    standard_error_real: float
    standard_error_imag: float
    bulk_sobol_power: int
    face_sobol_power: int
    replicates: int
    block_order: int
    momentum_order: int
    block_scheme: str

    def to_json(self) -> dict[str, object]:
        return {
            "omega_real": self.omega.real,
            "omega_imag": self.omega.imag,
            "collar_radius": self.collar_radius,
            "estimate_real": self.mean.real,
            "estimate_imag": self.mean.imag,
            "standard_error_real": self.standard_error_real,
            "standard_error_imag": self.standard_error_imag,
            "bulk_mean_real": complex(np.mean(self.bulk_estimates)).real,
            "bulk_mean_imag": complex(np.mean(self.bulk_estimates)).imag,
            "face_mean_real": complex(np.mean(self.face_estimates)).real,
            "face_mean_imag": complex(np.mean(self.face_estimates)).imag,
            "corner_real": self.corner_contribution.real,
            "corner_imag": self.corner_contribution.imag,
            "replicate_estimates": [
                {"real": value.real, "imag": value.imag} for value in self.estimates
            ],
            "bulk_sobol_power": self.bulk_sobol_power,
            "face_sobol_power": self.face_sobol_power,
            "replicates": self.replicates,
            "block_order": self.block_order,
            "momentum_order": self.momentum_order,
            "block_scheme": self.block_scheme,
            "subtraction": "direct i-epsilon local finite part on 10 faces and 15 corners",
        }


def _gauss_legendre_grid(
    order: int,
    maximum: float,
    panels: int = 1,
    breakpoints: Sequence[float] = (),
    power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a composite rule split at requested physical-P endpoints.

    ``power>1`` applies ``P=maximum*u**power`` before quadrature.  This
    concentrates nodes near the Liouville threshold while retaining every
    requested breakpoint exactly.
    """

    order = int(order)
    panels = int(panels)
    maximum = float(maximum)
    power = float(power)
    if order <= 0 or panels <= 0 or not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("order, panels, and maximum must be positive")
    if not math.isfinite(power) or power < 1.0:
        raise ValueError("power must be finite and at least one")
    canonical_nodes, canonical_weights = np.polynomial.legendre.leggauss(order)
    boundaries = {0.0, maximum}
    boundaries.update(panel * maximum / panels for panel in range(1, panels))
    boundaries.update(
        float(value)
        for value in breakpoints
        if math.isfinite(float(value)) and 0.0 < float(value) < maximum
    )
    ordered_boundaries = sorted(boundaries)
    if len(ordered_boundaries) == 2 and power != 1.0:
        # Integrate the Jacobian weight u**(power-1) with its native
        # Gauss--Jacobi rule.  Besides resolving the threshold, this keeps
        # normalization moments exact and places the first node less
        # aggressively than an ordinary Gauss rule applied after the map.
        jacobi_nodes, jacobi_weights = roots_jacobi(
            order,
            0.0,
            power - 1.0,
        )
        nodes_u = 0.5 * (jacobi_nodes + 1.0)
        nodes_p = maximum * nodes_u**power
        weights_p = (
            maximum
            * power
            * 2.0 ** (-power)
            * np.asarray(jacobi_weights, dtype=float)
        )
        return np.asarray(nodes_p, dtype=float), weights_p
    all_nodes: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    for left, right in zip(ordered_boundaries[:-1], ordered_boundaries[1:]):
        left_u = (left / maximum) ** (1.0 / power)
        right_u = (right / maximum) ** (1.0 / power)
        panel_width_u = right_u - left_u
        nodes_u = left_u + 0.5 * panel_width_u * (canonical_nodes + 1.0)
        nodes_p = maximum * nodes_u**power
        jacobian = maximum * power * nodes_u ** (power - 1.0)
        all_nodes.append(nodes_p)
        all_weights.append(0.5 * panel_width_u * canonical_weights * jacobian)
    return np.concatenate(all_nodes), np.concatenate(all_weights)


class EqualEnergyFivePointKernel:
    """Precomputed double-momentum kernel for one complex outgoing energy."""

    def __init__(
        self,
        omega: complex,
        *,
        block_order: int = 3,
        momentum_order: int = 8,
        momentum_maximum: float = 6.0,
        momentum_panels: int = 1,
        momentum_power: float = 1.0,
        endpoint_refinement: int = 0,
        block_scheme: str = "h",
        liouville_contour: str = "real",
        special_dps: int = 35,
        h_regulator_etas: Sequence[float] = (0.16, 0.13, 0.10, 0.075, 0.055),
    ) -> None:
        self.omega = complex(omega)
        self.block_order = int(block_order)
        self.momentum_order = int(momentum_order)
        self.momentum_maximum = float(momentum_maximum)
        self.momentum_panels = int(momentum_panels)
        self.momentum_power = float(momentum_power)
        self.endpoint_refinement = int(endpoint_refinement)
        self.block_scheme = str(block_scheme)
        self.liouville_contour = str(liouville_contour)
        if (
            self.block_order < 0
            or self.momentum_order <= 0
            or self.momentum_panels <= 0
            or self.endpoint_refinement < 0
        ):
            raise ValueError("block_order must be non-negative and momentum_order positive")
        if not math.isfinite(self.momentum_maximum) or self.momentum_maximum <= 0.0:
            raise ValueError("momentum_maximum must be positive and finite")
        if not math.isfinite(self.momentum_power) or self.momentum_power < 1.0:
            raise ValueError("momentum_power must be finite and at least one")
        if self.block_scheme not in {"h", "c"}:
            raise ValueError("block_scheme must be 'h' or 'c'")
        if self.liouville_contour not in {"real", "continued"}:
            raise ValueError("liouville_contour must be 'real' or 'continued'")

        self.crossed_pole: complex | None = None
        self.crossed_cherry_residue = 0.0 + 0.0j
        if self.liouville_contour == "continued":
            if abs(self.omega.real) > 1.0e-13 or not 0.0 < self.omega.imag < 0.5:
                raise ValueError(
                    "the continued Liouville contour currently supports only "
                    "omega=i*t with 0<t<1/2"
                )
            if abs(self.omega.imag - 0.4) < 1.0e-13:
                raise ValueError(
                    "t=2/5 is a Liouville contour pinch; evaluate its limit "
                    "from either adjacent chamber"
                )
            # At an incoming--outgoing cherry, C(2it,it/2,P) has the two
            # nearest denominator poles
            #
            #   P_+=i(5t/2-1),       P_-=-P_+.
            #
            # They pinch the quotient-contour endpoint at t=2/5.  Continuing
            # through the pinch keeps the real-contour integral and adds the
            # crossed-pole residue with the BRY coefficient -2i.  A second
            # family pinches this representation at t=1/2, which is why this
            # first continuation is deliberately restricted to t<1/2.
            if self.omega.imag > 0.4:
                self.crossed_pole = 2.5 * self.omega - 1.0j

        self.signed_energies = equal_outgoing_signed_energies(self.omega)
        self.external_momenta = (
            2.0 * self.omega,
            0.5 * self.omega,
            0.5 * self.omega,
            0.5 * self.omega,
            0.5 * self.omega,
        )
        self.external_weights = tuple(
            1.0 + momentum * momentum for momentum in self.external_momenta
        )
        self.special = UpsilonB(1.0, dps=int(special_dps))
        self.h_regulator_etas = tuple(float(value) for value in h_regulator_etas)
        # Adjacent quadrature orders keep each tensor grid away from the exact
        # P1=P2 diagonal, where the common-h1 form of the CCY h-recursion has
        # coincident internal-weight poles.  Averaging N x (N+1) with its
        # transpose restores exact P1<->P2 symmetry of the quadrature rule.
        power_breakpoints: set[float] = set()
        for channel_energy in (3.0 * self.omega, -2.0 * self.omega):
            threshold = 0.25 * (channel_energy**2).real
            for level in range(self.block_order + 1):
                if threshold > level:
                    endpoint = math.sqrt(threshold - level)
                    power_breakpoints.add(endpoint)
                    local_width = abs((channel_energy**2).imag) / max(
                        4.0 * endpoint,
                        1.0e-12,
                    )
                    multiples = (0.5, 1.0, 2.0, 4.0)[: self.endpoint_refinement]
                    for multiple in multiples:
                        power_breakpoints.update(
                            (
                                endpoint - multiple * local_width,
                                endpoint + multiple * local_width,
                            )
                        )
        p_low_nodes, p_low_weights = _gauss_legendre_grid(
            self.momentum_order,
            self.momentum_maximum,
            self.momentum_panels,
            sorted(power_breakpoints),
            self.momentum_power,
        )
        p_high_nodes, p_high_weights = _gauss_legendre_grid(
            self.momentum_order + 1,
            self.momentum_maximum,
            self.momentum_panels,
            sorted(power_breakpoints),
            self.momentum_power,
        )
        self.momentum_grids = (
            (p_low_nodes, p_low_weights, p_high_nodes, p_high_weights, 0.5),
            (p_high_nodes, p_high_weights, p_low_nodes, p_low_weights, 0.5),
        )
        self.entries_by_incoming_slot = self._build_entries()
        self.arrays_by_incoming_slot = self._build_arrays()
        self.discrete_entries_by_incoming_slot = self._build_discrete_entries()
        self.discrete_arrays_by_incoming_slot = self._build_discrete_arrays()

    def requires_power_subtraction(self) -> bool:
        """Return whether any of the ten boundary projectors is nonempty."""

        return any(
            divergent_spin_zero_levels(
                signed_channel_energy(divisor, self.signed_energies)
            )
            for divisor in five_point_boundary_divisors()
        )

    def _build_entries(self) -> tuple[tuple[MomentumGridEntry, ...], ...]:
        entries_by_slot: list[tuple[MomentumGridEntry, ...]] = []
        incoming_momentum = self.external_momenta[0]
        outgoing_momentum = self.external_momenta[1]
        for incoming_slot in range(5):
            ordered_momenta = tuple(
                incoming_momentum if slot == incoming_slot else outgoing_momentum
                for slot in range(5)
            )
            ordered_weights = tuple(1.0 + value * value for value in ordered_momenta)
            pa, pb, pc, pd, pe = ordered_momenta
            slot_entries: list[MomentumGridEntry] = []
            for p1_nodes, p1_weights, p2_nodes, p2_weights, grid_scale in self.momentum_grids:
                first_logs = tuple(
                    complex(
                        log_yin_structure_constant_momentum(
                            self.special, pa, pb, float(p1)
                        )
                    )
                    for p1 in p1_nodes
                )
                last_logs = tuple(
                    complex(
                        log_yin_structure_constant_momentum(
                            self.special, float(p2), pd, pe
                        )
                    )
                    for p2 in p2_nodes
                )
                for p1_index, (p1, weight1) in enumerate(zip(p1_nodes, p1_weights)):
                    p1_value = float(p1)
                    h1 = 1.0 + p1_value**2
                    for p2_index, (p2, weight2) in enumerate(zip(p2_nodes, p2_weights)):
                        p2_value = float(p2)
                        middle_log = complex(
                            log_yin_structure_constant_momentum(
                                self.special, p1_value, pc, p2_value
                            )
                        )
                        log_weighted_structure_constant = complex(
                            math.log(
                                grid_scale
                                * float(weight1)
                                * float(weight2)
                                / (math.pi * math.pi)
                            )
                            + first_logs[p1_index]
                            + middle_log
                            + last_logs[p2_index]
                        )
                        h2 = 1.0 + p2_value**2
                        if self.block_scheme == "h":
                            coefficients, coefficient_errors = sphere_five_point_h_c25_limit(
                                external_weights=ordered_weights,
                                internal_weights=(h1, h2),
                                order1=self.block_order,
                                order2=self.block_order,
                                max_total_order=self.block_order,
                                regulator_etas=self.h_regulator_etas,
                                polynomial_degree=min(3, len(self.h_regulator_etas) - 1),
                            )
                        else:
                            coefficients = sphere_five_point_c_coefficients(
                                central_charge=25.0,
                                external_weights=ordered_weights,
                                internal_weights=(h1, h2),
                                order1=self.block_order,
                                order2=self.block_order,
                                max_total_order=self.block_order,
                            )
                            coefficient_errors = {key: 0.0 + 0.0j for key in coefficients}
                        slot_entries.append(
                            MomentumGridEntry(
                                p1=p1_value,
                                p2=p2_value,
                                h1=h1,
                                h2=h2,
                                log_weighted_structure_constant=log_weighted_structure_constant,
                                weighted_structure_constant=cmath.exp(
                                    log_weighted_structure_constant
                                ),
                                coefficients=coefficients,
                                coefficient_errors=coefficient_errors,
                            )
                        )
            entries_by_slot.append(tuple(slot_entries))
        return tuple(entries_by_slot)

    def _crossed_structure_constant_residue(
        self,
        first: complex,
        second: complex,
        pole: complex,
    ) -> complex:
        """Numerically extract ``Res_P C(first,second,P)`` symmetrically."""

        pole = complex(pole)
        step = min(1.0e-5, max(1.0e-7, 1.0e-3 * abs(pole)))

        def estimate(offset: float) -> complex:
            plus = cmath.exp(
                complex(
                    log_yin_structure_constant_momentum(
                        self.special, first, second, pole + offset
                    )
                )
            )
            minus = cmath.exp(
                complex(
                    log_yin_structure_constant_momentum(
                        self.special, first, second, pole - offset
                    )
                )
            )
            return 0.5 * (offset * plus + (-offset) * minus)

        coarse = estimate(step)
        fine = estimate(0.5 * step)
        return complex((4.0 * fine - coarse) / 3.0)

    def _build_discrete_entries(self) -> tuple[tuple[MomentumGridEntry, ...], ...]:
        """Build the one-dimensional terms generated at the t=2/5 pinch."""

        if self.crossed_pole is None:
            return tuple(tuple() for _ in range(5))

        pole = complex(self.crossed_pole)
        incoming_momentum = self.external_momenta[0]
        outgoing_momentum = self.external_momenta[1]
        residue = self._crossed_structure_constant_residue(
            incoming_momentum,
            outgoing_momentum,
            pole,
        )
        self.crossed_cherry_residue = residue
        contour_coefficient = -2.0j * residue
        entries_by_slot: list[tuple[MomentumGridEntry, ...]] = []
        for incoming_slot in range(5):
            if incoming_slot == 2:
                entries_by_slot.append(tuple())
                continue
            ordered_momenta = tuple(
                incoming_momentum if slot == incoming_slot else outgoing_momentum
                for slot in range(5)
            )
            ordered_weights = tuple(1.0 + value * value for value in ordered_momenta)
            pa, pb, pc, pd, pe = ordered_momenta
            slot_entries: list[MomentumGridEntry] = []
            incoming_on_left = incoming_slot in (0, 1)
            for p1_nodes, p1_weights, p2_nodes, p2_weights, grid_scale in self.momentum_grids:
                variable_nodes = p2_nodes if incoming_on_left else p1_nodes
                variable_weights = p2_weights if incoming_on_left else p1_weights
                for variable, weight in zip(variable_nodes, variable_weights):
                    variable = float(variable)
                    if incoming_on_left:
                        p1_value: complex | float = pole
                        p2_value: complex | float = variable
                        remaining_logs = (
                            log_yin_structure_constant_momentum(
                                self.special, pole, pc, variable
                            ),
                            log_yin_structure_constant_momentum(
                                self.special, variable, pd, pe
                            ),
                        )
                    else:
                        p1_value = variable
                        p2_value = pole
                        remaining_logs = (
                            log_yin_structure_constant_momentum(
                                self.special, pa, pb, variable
                            ),
                            log_yin_structure_constant_momentum(
                                self.special, variable, pc, pole
                            ),
                        )
                    h1 = 1.0 + p1_value**2
                    h2 = 1.0 + p2_value**2
                    logarithm = complex(
                        cmath.log(
                            contour_coefficient
                            * grid_scale
                            * float(weight)
                            / math.pi
                        )
                        + complex(remaining_logs[0])
                        + complex(remaining_logs[1])
                    )
                    if self.block_scheme == "h":
                        coefficients, coefficient_errors = sphere_five_point_h_c25_limit(
                            external_weights=ordered_weights,
                            internal_weights=(h1, h2),
                            order1=self.block_order,
                            order2=self.block_order,
                            max_total_order=self.block_order,
                            regulator_etas=self.h_regulator_etas,
                            polynomial_degree=min(3, len(self.h_regulator_etas) - 1),
                        )
                    else:
                        coefficients = sphere_five_point_c_coefficients(
                            central_charge=25.0,
                            external_weights=ordered_weights,
                            internal_weights=(h1, h2),
                            order1=self.block_order,
                            order2=self.block_order,
                            max_total_order=self.block_order,
                        )
                        coefficient_errors = {
                            key: 0.0 + 0.0j for key in coefficients
                        }
                    slot_entries.append(
                        MomentumGridEntry(
                            p1=p1_value,
                            p2=p2_value,
                            h1=h1,
                            h2=h2,
                            log_weighted_structure_constant=logarithm,
                            weighted_structure_constant=cmath.exp(logarithm),
                            coefficients=coefficients,
                            coefficient_errors=coefficient_errors,
                        )
                    )
            entries_by_slot.append(tuple(slot_entries))
        return tuple(entries_by_slot)

    def _build_discrete_arrays(
        self,
    ) -> tuple[ContinuedMomentumGridArrays | None, ...]:
        arrays: list[ContinuedMomentumGridArrays | None] = []
        incoming_momentum = self.external_momenta[0]
        outgoing_momentum = self.external_momenta[1]
        for incoming_slot, entries in enumerate(self.discrete_entries_by_incoming_slot):
            if not entries:
                arrays.append(None)
                continue
            ordered_momenta = tuple(
                incoming_momentum if slot == incoming_slot else outgoing_momentum
                for slot in range(5)
            )
            ordered_weights = tuple(1.0 + value * value for value in ordered_momenta)
            coefficient_keys = tuple(
                sorted(
                    {key for entry in entries for key in entry.coefficients},
                    key=lambda key: (key[0] + key[1], key[0], key[1]),
                )
            )
            arrays.append(
                ContinuedMomentumGridArrays(
                    log_weighted_structure_constants=np.asarray(
                        [entry.log_weighted_structure_constant for entry in entries],
                        dtype=complex,
                    ),
                    primary_x_exponents=np.asarray(
                        [
                            2.0
                            * (entry.h1 - ordered_weights[0] - ordered_weights[1])
                            for entry in entries
                        ],
                        dtype=complex,
                    ),
                    primary_y_exponents=np.asarray(
                        [
                            2.0
                            * (entry.h2 - ordered_weights[2] - entry.h1)
                            for entry in entries
                        ],
                        dtype=complex,
                    ),
                    coefficient_keys=coefficient_keys,
                    coefficient_matrix=np.asarray(
                        [
                            [
                                entry.coefficients.get(key, 0.0 + 0.0j)
                                for key in coefficient_keys
                            ]
                            for entry in entries
                        ],
                        dtype=complex,
                    ),
                )
            )
        return tuple(arrays)

    def _build_arrays(self) -> tuple[MomentumGridArrays, ...]:
        arrays: list[MomentumGridArrays] = []
        incoming_momentum = self.external_momenta[0]
        outgoing_momentum = self.external_momenta[1]
        incoming_signed_energy = self.signed_energies[0]
        outgoing_signed_energy = self.signed_energies[1]
        for incoming_slot, entries in enumerate(self.entries_by_incoming_slot):
            ordered_momenta = tuple(
                incoming_momentum if slot == incoming_slot else outgoing_momentum
                for slot in range(5)
            )
            ordered_weights = tuple(1.0 + value * value for value in ordered_momenta)
            ordered_signed_energies = tuple(
                incoming_signed_energy if slot == incoming_slot else outgoing_signed_energy
                for slot in range(5)
            )
            coefficient_keys = tuple(
                sorted(
                    {key for entry in entries for key in entry.coefficients},
                    key=lambda key: (key[0] + key[1], key[0], key[1]),
                )
            )
            regular_by_entry = tuple(
                five_point_regular_factor_coefficients(
                    entry.coefficients,
                    ordered_signed_energies,
                    order1=self.block_order,
                    order2=self.block_order,
                )
                for entry in entries
            )
            regular_coefficient_keys = tuple(
                sorted(
                    {key for coefficients in regular_by_entry for key in coefficients},
                    key=lambda key: (key[0] + key[1], key[0], key[1]),
                )
            )
            plumbing_exponents = tuple(
                five_point_plumbing_radial_exponents(
                    ordered_signed_energies,
                    entry.p1,
                    entry.p2,
                )
                for entry in entries
            )
            channel_energy1, channel_energy2 = five_point_plumbing_channel_energies(
                ordered_signed_energies
            )
            momenta_are_real = all(
                abs(complex(entry.p1).imag) < 1.0e-15
                and abs(complex(entry.p2).imag) < 1.0e-15
                for entry in entries
            )
            momentum_dtype = float if momenta_are_real else complex
            _, signed_b, signed_c, signed_d, _ = ordered_signed_energies
            arrays.append(
                MomentumGridArrays(
                    log_weighted_structure_constants=np.asarray(
                        [entry.log_weighted_structure_constant for entry in entries],
                        dtype=complex,
                    ),
                    primary_x_exponents=np.asarray(
                        [
                            2.0
                            * (entry.h1 - ordered_weights[0] - ordered_weights[1])
                            for entry in entries
                        ],
                        dtype=complex,
                    ),
                    primary_y_exponents=np.asarray(
                        [
                            2.0
                            * (entry.h2 - ordered_weights[2] - entry.h1)
                            for entry in entries
                        ],
                        dtype=complex,
                    ),
                    coefficient_keys=coefficient_keys,
                    coefficient_matrix=np.asarray(
                        [
                            [entry.coefficients.get(key, 0.0 + 0.0j) for key in coefficient_keys]
                            for entry in entries
                        ],
                        dtype=complex,
                    ),
                    p1_values=np.asarray(
                        [complex(entry.p1).real if momenta_are_real else entry.p1 for entry in entries],
                        dtype=momentum_dtype,
                    ),
                    p2_values=np.asarray(
                        [complex(entry.p2).real if momenta_are_real else entry.p2 for entry in entries],
                        dtype=momentum_dtype,
                    ),
                    regular_coefficient_keys=regular_coefficient_keys,
                    regular_coefficient_matrix=np.asarray(
                        [
                            [coefficients.get(key, 0.0 + 0.0j) for key in regular_coefficient_keys]
                            for coefficients in regular_by_entry
                        ],
                        dtype=complex,
                    ),
                    plumbing_q1_exponents=np.asarray(
                        [value[0] for value in plumbing_exponents],
                        dtype=complex,
                    ),
                    plumbing_q2_exponents=np.asarray(
                        [value[1] for value in plumbing_exponents],
                        dtype=complex,
                    ),
                    channel_energy1=complex(channel_energy1),
                    channel_energy2=complex(channel_energy2),
                    regular_timelike_exponents=(
                        complex(-0.5 * signed_b * signed_c),
                        complex(-0.5 * signed_b * signed_d),
                        complex(-0.5 * signed_c * signed_d),
                    ),
                )
            )
        return tuple(arrays)

    @staticmethod
    def _series_monomials(
        q1: complex,
        q2: complex,
        keys: Sequence[tuple[int, int]],
    ) -> np.ndarray:
        return np.asarray(
            [complex(q1) ** first * complex(q2) ** second for first, second in keys],
            dtype=complex,
        )

    def _momentum_sum(
        self,
        incoming_slot: int,
        q1: complex,
        q2: complex,
        log_x: float,
        log_y: float,
        base_logarithm: complex,
    ) -> complex:
        def evaluate_arrays(arrays: MomentumGridArrays | ContinuedMomentumGridArrays) -> complex:
            holomorphic = np.sum(
                arrays.coefficient_matrix
                * self._series_monomials(q1, q2, arrays.coefficient_keys)[None, :],
                axis=1,
            )
            antiholomorphic = np.sum(
                arrays.coefficient_matrix
                * self._series_monomials(
                    complex(q1).conjugate(),
                    complex(q2).conjugate(),
                    arrays.coefficient_keys,
                )[None, :],
                axis=1,
            )
            exponentials = np.exp(
                complex(base_logarithm)
                + arrays.log_weighted_structure_constants
                + arrays.primary_x_exponents * float(log_x)
                + arrays.primary_y_exponents * float(log_y)
            )
            return complex(np.sum(exponentials * holomorphic * antiholomorphic))

        incoming_slot = int(incoming_slot)
        total = evaluate_arrays(self.arrays_by_incoming_slot[incoming_slot])
        discrete = self.discrete_arrays_by_incoming_slot[incoming_slot]
        if discrete is not None:
            total += evaluate_arrays(discrete)
        return complex(total)

    def validate_physical_subtraction_order(self) -> None:
        """Require the block table to contain every divergent OPE level."""

        if self.liouville_contour != "real":
            raise ValueError(
                "the physical subtraction forest requires the real Liouville contour"
            )

        required = set()
        for arrays in self.arrays_by_incoming_slot:
            required.update(divergent_spin_zero_levels(arrays.channel_energy1))
            required.update(divergent_spin_zero_levels(arrays.channel_energy2))
        if required and max(required) > self.block_order:
            raise ValueError(
                "block_order is too small for the physical BRY projector: "
                f"need at least {max(required)}, received {self.block_order}"
            )

    def _forest_momentum_sum(
        self,
        incoming_slot: int,
        q1: complex,
        q2: complex,
        *,
        base_logarithm: complex = 0.0 + 0.0j,
    ) -> complex:
        """Return the double-P sum of I-S1 I-S2 I+S1 S2 I.

        The density is with respect to d2q1 d2q2 in the best linear-channel
        gauge. The q2 Jacobian from x=q1*q2, y=q2 is included in the stored
        plumbing exponents.
        """

        arrays = self.arrays_by_incoming_slot[int(incoming_slot)]
        q1 = complex(q1)
        q2 = complex(q2)
        if q1 == 0.0 or q2 == 0.0:
            raise ZeroDivisionError("the physical forest is singular on its boundary")
        log_radius1 = math.log(abs(q1))
        log_radius2 = math.log(abs(q2))
        keys = arrays.regular_coefficient_keys
        matrix = arrays.regular_coefficient_matrix
        common_exponential = np.exp(
            complex(base_logarithm)
            + arrays.log_weighted_structure_constants
            + arrays.plumbing_q1_exponents * log_radius1
            + arrays.plumbing_q2_exponents * log_radius2
        )
        block_holomorphic = np.sum(
            arrays.coefficient_matrix
            * self._series_monomials(q1, q2, arrays.coefficient_keys)[None, :],
            axis=1,
        )
        block_antiholomorphic = np.sum(
            arrays.coefficient_matrix
            * self._series_monomials(
                q1.conjugate(),
                q2.conjugate(),
                arrays.coefficient_keys,
            )[None, :],
            axis=1,
        )
        exponent_first, exponent_diagonal, exponent_second = (
            arrays.regular_timelike_exponents
        )
        regular_timelike_holomorphic = (
            (1.0 - q1) ** exponent_first
            * (1.0 - q1 * q2) ** exponent_diagonal
            * (1.0 - q2) ** exponent_second
        )
        regular_timelike_antiholomorphic = (
            (1.0 - q1.conjugate()) ** exponent_first
            * (1.0 - q1.conjugate() * q2.conjugate()) ** exponent_diagonal
            * (1.0 - q2.conjugate()) ** exponent_second
        )
        original = np.sum(
            common_exponential
            * block_holomorphic
            * block_antiholomorphic
            * regular_timelike_holomorphic
            * regular_timelike_antiholomorphic
        )

        threshold1 = 0.25 * (arrays.channel_energy1**2).real
        threshold2 = 0.25 * (arrays.channel_energy2**2).real
        face1 = 0.0 + 0.0j
        face2 = 0.0 + 0.0j
        corner = 0.0 + 0.0j
        for level1 in range(self.block_order + 1):
            mask1 = arrays.p1_values**2 + level1 < threshold1
            if not np.any(mask1):
                continue
            key_mask1 = np.asarray([key[0] == level1 for key in keys], dtype=bool)
            row_holomorphic = np.sum(
                matrix[:, key_mask1]
                * np.asarray(
                    [q2 ** keys[index][1] for index in np.flatnonzero(key_mask1)],
                    dtype=complex,
                )[None, :],
                axis=1,
            )
            row_antiholomorphic = np.sum(
                matrix[:, key_mask1]
                * np.asarray(
                    [
                        q2.conjugate() ** keys[index][1]
                        for index in np.flatnonzero(key_mask1)
                    ],
                    dtype=complex,
                )[None, :],
                axis=1,
            )
            face1 += np.sum(
                common_exponential
                * cmath.exp(2.0 * level1 * log_radius1)
                * row_holomorphic
                * row_antiholomorphic
                * mask1
            )
            for level2 in range(self.block_order + 1):
                mask2 = arrays.p2_values**2 + level2 < threshold2
                combined_mask = mask1 & mask2
                if not np.any(combined_mask):
                    continue
                try:
                    coefficient_index = keys.index((level1, level2))
                except ValueError:
                    continue
                coefficients = matrix[:, coefficient_index]
                corner += np.sum(
                    common_exponential
                    * cmath.exp(
                        2.0 * level1 * log_radius1
                        + 2.0 * level2 * log_radius2
                    )
                    * coefficients**2
                    * combined_mask
                )

        for level2 in range(self.block_order + 1):
            mask2 = arrays.p2_values**2 + level2 < threshold2
            if not np.any(mask2):
                continue
            key_mask2 = np.asarray([key[1] == level2 for key in keys], dtype=bool)
            column_holomorphic = np.sum(
                matrix[:, key_mask2]
                * np.asarray(
                    [q1 ** keys[index][0] for index in np.flatnonzero(key_mask2)],
                    dtype=complex,
                )[None, :],
                axis=1,
            )
            column_antiholomorphic = np.sum(
                matrix[:, key_mask2]
                * np.asarray(
                    [
                        q1.conjugate() ** keys[index][0]
                        for index in np.flatnonzero(key_mask2)
                    ],
                    dtype=complex,
                )[None, :],
                axis=1,
            )
            face2 += np.sum(
                common_exponential
                * cmath.exp(2.0 * level2 * log_radius2)
                * column_holomorphic
                * column_antiholomorphic
                * mask2
            )
        return complex(original - face1 - face2 + corner)

    def _face_finite_part_momentum_sum(
        self,
        incoming_slot: int,
        remaining_modulus: complex,
        collar_radius: float,
        *,
        logarithmic_weight: float = 0.0,
    ) -> complex:
        """Integrate the left plumbing radius by analytic continuation."""

        arrays = self.arrays_by_incoming_slot[int(incoming_slot)]
        remaining_modulus = complex(remaining_modulus)
        if remaining_modulus == 0.0:
            raise ZeroDivisionError("the face coordinate lies on a corner")
        logarithmic_radius = math.log(abs(remaining_modulus))
        logarithmic_collar = math.log(float(collar_radius))
        common = np.exp(
            float(logarithmic_weight)
            + arrays.log_weighted_structure_constants
            + arrays.plumbing_q2_exponents * logarithmic_radius
        )
        total = 0.0 + 0.0j
        keys = arrays.regular_coefficient_keys
        matrix = arrays.regular_coefficient_matrix
        for level in range(self.block_order + 1):
            key_mask = np.asarray([key[0] == level for key in keys], dtype=bool)
            if not np.any(key_mask):
                continue
            selected_indices = np.flatnonzero(key_mask)
            row = np.sum(
                matrix[:, key_mask]
                * np.asarray(
                    [remaining_modulus ** keys[index][1] for index in selected_indices],
                    dtype=complex,
                )[None, :],
                axis=1,
            )
            row_bar = np.sum(
                matrix[:, key_mask]
                * np.asarray(
                    [
                        remaining_modulus.conjugate() ** keys[index][1]
                        for index in selected_indices
                    ],
                    dtype=complex,
                )[None, :],
                axis=1,
            )
            alpha = 0.5 * (arrays.plumbing_q1_exponents + 2.0) + level
            radial = math.pi * np.exp(2.0 * alpha * logarithmic_collar) / alpha
            total += np.sum(common * radial * row * row_bar)
        return complex(total)

    def _corner_finite_part_momentum_sum(
        self,
        incoming_slot: int,
        collar_radius1: float,
        collar_radius2: float,
    ) -> complex:
        """Analytically continue both commuting plumbing radii."""

        arrays = self.arrays_by_incoming_slot[int(incoming_slot)]
        log_radius1 = math.log(float(collar_radius1))
        log_radius2 = math.log(float(collar_radius2))
        total = 0.0 + 0.0j
        for coefficient_index, (level1, level2) in enumerate(
            arrays.regular_coefficient_keys
        ):
            alpha1 = 0.5 * (arrays.plumbing_q1_exponents + 2.0) + level1
            alpha2 = 0.5 * (arrays.plumbing_q2_exponents + 2.0) + level2
            radial1 = math.pi * np.exp(2.0 * alpha1 * log_radius1) / alpha1
            radial2 = math.pi * np.exp(2.0 * alpha2 * log_radius2) / alpha2
            coefficients = arrays.regular_coefficient_matrix[:, coefficient_index]
            total += np.sum(
                np.exp(arrays.log_weighted_structure_constants)
                * coefficients**2
                * radial1
                * radial2
            )
        return complex(total)

    def face_finite_part_density(
        self,
        remaining_modulus: complex,
        divisor: BoundaryDivisor,
        collar_radius: float,
        *,
        ordering: Sequence[int] | None = None,
    ) -> complex:
        """Return the canonical one-dimensional finite part for one divisor."""

        if ordering is None:
            selected = canonical_divisor_ordering(divisor)
        else:
            selected = tuple(int(label) for label in ordering)
            if len(selected) != 5 or set(selected) != set(range(5)):
                raise ValueError("ordering must permute labels 0,...,4")
            if set(selected[:2]) != set(divisor.cherry):
                raise ValueError("the first two ordering labels must be the divisor cherry")
        return self._face_finite_part_momentum_sum(
            selected.index(0),
            remaining_modulus,
            collar_radius,
        )

    def corner_finite_part(
        self,
        corner: BoundaryCorner,
        collar_radius: float,
    ) -> complex:
        """Return the canonical double finite part at one boundary corner."""

        ordering = canonical_corner_ordering(corner)
        return self._corner_finite_part_momentum_sum(
            ordering.index(0),
            collar_radius,
            collar_radius,
        )

    def liouville_correlator(self, channel: LinearChannel) -> complex:
        """Evaluate the precomputed double-momentum sum in one channel."""

        incoming_slot = channel.ordering.index(0)
        ordered_weights = tuple(self.external_weights[label] for label in channel.ordering)
        x = complex(channel.positions[1])
        y = complex(channel.positions[2])
        q1 = channel.q1
        q2 = channel.q2
        q1_bar = q1.conjugate()
        q2_bar = q2.conjugate()
        return self._momentum_sum(
            incoming_slot,
            q1,
            q2,
            math.log(abs(x)),
            math.log(abs(y)),
            0.0 + 0.0j,
        )

    def integrand(self, z_incoming: complex, z_outgoing: complex) -> complex:
        """Return the raw time-like times Liouville matter integrand."""

        positions = (
            complex(z_incoming),
            complex(z_outgoing),
            0.0 + 0.0j,
            1.0 + 0.0j,
            INFINITY,
        )
        channel = best_linear_channels(positions, limit=1)[0]
        incoming_slot = channel.ordering.index(0)
        ordered_weights = tuple(self.external_weights[label] for label in channel.ordering)
        x = complex(channel.positions[1])
        y = complex(channel.positions[2])
        log_x = math.log(abs(x))
        log_y = math.log(abs(y))
        base_logarithm = (
            liouville_primary_covariance_log(channel, self.external_weights)
            + timelike_free_boson_log_factor(positions, self.signed_energies)
        )
        return self._momentum_sum(
            incoming_slot,
            channel.q1,
            channel.q2,
            log_x,
            log_y,
            base_logarithm,
        )

    def integrand_linear_gauge_weighted(
        self,
        q1: complex,
        q2: complex,
        ordering: Sequence[int],
        *,
        logarithmic_weight: float = 0.0,
    ) -> complex:
        r"""Evaluate the density in an oriented ``(0,q1*q2,q2,1,inf)`` gauge.

        ``logarithmic_weight`` multiplies the answer by its exponential inside
        every momentum summand.  The atlas integrator uses this to divide by
        a possibly enormous mixture density without first overflowing either
        the raw OPE integrand or that density.
        """

        ordering_tuple = tuple(int(label) for label in ordering)
        if len(ordering_tuple) != 5 or set(ordering_tuple) != set(range(5)):
            raise ValueError("ordering must be a permutation of labels 0,...,4")
        q1 = complex(q1)
        q2 = complex(q2)
        if q1 == 0.0 or q2 == 0.0:
            raise ZeroDivisionError("the linear gauge lies on a plumbing boundary")
        positions = linear_channel_positions_by_label(q1, q2, ordering_tuple)
        # The proposal chart resolves its chosen boundary, but need not be the
        # fastest conformal-block chart for a point near another edge of its
        # bidisc.  Re-expand in the best of the fifteen trees while retaining
        # the sampled gauge for the moduli measure and time-like correlator.
        channel = best_linear_channels(positions, limit=1)[0]
        incoming_slot = channel.ordering.index(0)
        ordered_weights = tuple(
            self.external_weights[label] for label in channel.ordering
        )
        log_x = math.log(abs(complex(channel.positions[1])))
        log_y = math.log(abs(complex(channel.positions[2])))
        base_logarithm = (
            timelike_free_boson_log_factor(positions, self.signed_energies)
            + liouville_primary_covariance_log(channel, self.external_weights)
            + float(logarithmic_weight)
        )
        return self._momentum_sum(
            incoming_slot,
            channel.q1,
            channel.q2,
            log_x,
            log_y,
            base_logarithm,
        )

    def integrand_linear_gauge(
        self,
        q1: complex,
        q2: complex,
        ordering: Sequence[int],
    ) -> complex:
        """Return the raw density in an oriented linear-channel gauge."""

        return self.integrand_linear_gauge_weighted(q1, q2, ordering)

    def forest_subtracted_integrand_linear_gauge_weighted(
        self,
        q1: complex,
        q2: complex,
        ordering: Sequence[int],
        *,
        logarithmic_weight: float = 0.0,
    ) -> complex:
        """Evaluate the physical BRY forest density in a sampled gauge.

        The best crossing sector supplies the two local OPE projectors. Its
        q-density is transformed back to the sampled fixed-label gauge before
        the proposal weight is applied.
        """

        self.validate_physical_subtraction_order()
        selected = tuple(int(label) for label in ordering)
        if len(selected) != 5 or set(selected) != set(range(5)):
            raise ValueError("ordering must be a permutation of labels 0,...,4")
        positions = linear_channel_positions_by_label(q1, q2, selected)
        channel = best_linear_channels(positions, limit=1)[0]
        incoming_slot = channel.ordering.index(0)
        fixed_zero, fixed_one, fixed_infinity = (
            selected[0],
            selected[3],
            selected[4],
        )
        jacobian = linear_channel_complex_jacobian_to_chart(
            channel.q1,
            channel.q2,
            channel.ordering,
            fixed_zero=fixed_zero,
            fixed_one=fixed_one,
            fixed_infinity=fixed_infinity,
            moving_labels=(selected[1], selected[2]),
        )
        absolute_jacobian = abs(jacobian)
        if not math.isfinite(absolute_jacobian) or absolute_jacobian <= 0.0:
            raise ArithmeticError("the crossing-sector Jacobian is non-positive")
        return self._forest_momentum_sum(
            incoming_slot,
            channel.q1,
            channel.q2,
            base_logarithm=complex(
                float(logarithmic_weight) - 2.0 * math.log(absolute_jacobian)
            ),
        )


def _plane_map(radial_coordinate: float, angular_coordinate: float) -> tuple[complex, float]:
    """Map the unit square to the complex plane and return its area Jacobian."""

    radial_coordinate = float(radial_coordinate)
    angular_coordinate = float(angular_coordinate)
    radius = math.tan(0.5 * math.pi * radial_coordinate)
    angle = 2.0 * math.pi * angular_coordinate
    secant_squared = 1.0 + radius * radius
    jacobian = math.pi * math.pi * radius * secant_squared
    return complex(radius * cmath.exp(1.0j * angle)), jacobian


def integrate_convergent_equal_energy_qmc(
    kernel: EqualEnergyFivePointKernel,
    *,
    sobol_power: int = 7,
    replicates: int = 4,
    seed: int = 20260823,
) -> EqualEnergyQMCResult:
    """Integrate a subtraction-free complex-energy point over ``C^2``.

    This diagnostic is intentionally unavailable when any boundary has a
    power divergence; the physical calculation must go through the complete
    forest rather than silently using the raw integral.
    """

    if kernel.requires_power_subtraction():
        raise ValueError(
            "this energy requires the sphere-five-point boundary subtraction forest"
        )
    sobol_power = int(sobol_power)
    replicates = int(replicates)
    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    estimates: list[complex] = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=4, scramble=True, seed=int(seed) + replicate)
        points = sampler.random_base2(sobol_power)
        values: list[complex] = []
        for point in points:
            z_in, jacobian_in = _plane_map(point[0], point[1])
            z_out, jacobian_out = _plane_map(point[2], point[3])
            try:
                value = kernel.integrand(z_in, z_out)
            except (ArithmeticError, RuntimeError, ValueError, ZeroDivisionError):
                # Sobol points never sit exactly on a boundary.  A failed
                # interior point is therefore a genuine numerical error.
                raise
            values.append(value * jacobian_in * jacobian_out)
        estimates.append(complex(np.mean(np.asarray(values, dtype=complex))))
    estimate_array = np.asarray(estimates, dtype=complex)
    mean = complex(np.mean(estimate_array))
    return EqualEnergyQMCResult(
        omega=kernel.omega,
        estimates=tuple(estimates),
        mean=mean,
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
        block_scheme=kernel.block_scheme,
    )


def _power_disk_sample(
    radial_uniform: float,
    angular_uniform: float,
    radial_power: float,
) -> tuple[complex, float]:
    r"""Sample the unit disc with radial CDF ``r**radial_power``.

    Returns the complex point and its probability density with respect to
    plane area.
    """

    radial_power = float(radial_power)
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    u = min(max(float(radial_uniform), np.nextafter(0.0, 1.0)), np.nextafter(1.0, 0.0))
    radius = u ** (1.0 / radial_power)
    angle = 2.0 * math.pi * float(angular_uniform)
    density = radial_power / (2.0 * math.pi) * radius ** (radial_power - 2.0)
    return complex(radius * cmath.exp(1.0j * angle)), float(density)


def oriented_bidisc_mixture_density(
    positions: Sequence[complex | None],
    *,
    radial_power: float,
) -> float:
    """Return the 120-chart bidisc mixture density in the original chart."""

    orderings = oriented_tree_orderings()
    total = 0.0
    for ordering in orderings:
        try:
            channel = linear_channel_from_ordering(positions, ordering)
        except (ValueError, ZeroDivisionError):
            continue
        radius1 = abs(channel.q1)
        radius2 = abs(channel.q2)
        if not (0.0 < radius1 < 1.0 and 0.0 < radius2 < 1.0):
            continue
        forward = linear_channel_to_original_chart(
            channel.q1, channel.q2, ordering
        )
        if forward.area_jacobian <= 0.0:
            continue
        density1 = float(radial_power) / (2.0 * math.pi) * radius1 ** (
            float(radial_power) - 2.0
        )
        density2 = float(radial_power) / (2.0 * math.pi) * radius2 ** (
            float(radial_power) - 2.0
        )
        total += density1 * density2 / forward.area_jacobian
    mixture = total / len(orderings)
    if not math.isfinite(mixture) or mixture <= 0.0:
        raise ArithmeticError("the oriented-bidisc mixture density is non-positive")
    return float(mixture)


def oriented_bidisc_log_mixture_density_in_frame(
    positions: Sequence[complex | None],
    selected_ordering: Sequence[int],
    *,
    radial_power: float,
) -> float:
    r"""Return the log mixture density in a selected linear-channel gauge.

    The selected ordering fixes its labels ``(a,d,e)`` at ``(0,1,inf)`` and
    integrates the labels ``(b,c)``.  Each of the 120 proposal charts is
    transformed directly into that same gauge.  A log-sum-exp keeps the
    density finite in arbitrarily deep OPE collars.
    """

    selected = tuple(int(label) for label in selected_ordering)
    if len(selected) != 5 or set(selected) != set(range(5)):
        raise ValueError("selected_ordering must permute labels 0,...,4")
    if len(positions) != 5:
        raise ValueError("positions must contain five label-ordered punctures")
    radial_power = float(radial_power)
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    fixed_zero, fixed_one, fixed_infinity = selected[0], selected[3], selected[4]
    moving_labels = (selected[1], selected[2])
    base_log_density = math.log(radial_power / (2.0 * math.pi))
    logarithmic_terms: list[float] = []
    orderings = oriented_tree_orderings()
    for ordering in orderings:
        try:
            channel = linear_channel_from_ordering(positions, ordering)
            radius1 = abs(channel.q1)
            radius2 = abs(channel.q2)
            if not (0.0 < radius1 < 1.0 and 0.0 < radius2 < 1.0):
                continue
            jacobian = linear_channel_complex_jacobian_to_chart(
                channel.q1,
                channel.q2,
                ordering,
                fixed_zero=fixed_zero,
                fixed_one=fixed_one,
                fixed_infinity=fixed_infinity,
                moving_labels=moving_labels,
            )
            absolute_jacobian = abs(jacobian)
            if not math.isfinite(absolute_jacobian) or absolute_jacobian <= 0.0:
                continue
            logarithmic_terms.append(
                2.0 * base_log_density
                + (radial_power - 2.0) * (math.log(radius1) + math.log(radius2))
                - 2.0 * math.log(absolute_jacobian)
            )
        except (ArithmeticError, OverflowError, ValueError, ZeroDivisionError):
            continue
    if not logarithmic_terms:
        raise ArithmeticError("no oriented bidisc contributes to the mixture density")
    maximum = max(logarithmic_terms)
    return float(
        maximum
        + math.log(sum(math.exp(value - maximum) for value in logarithmic_terms))
        - math.log(len(orderings))
    )


def integrate_convergent_equal_energy_atlas_qmc(
    kernel: EqualEnergyFivePointKernel,
    *,
    sobol_power: int = 7,
    replicates: int = 4,
    radial_power: float = 0.08,
    seed: int = 20260823,
) -> EqualEnergyQMCResult:
    r"""Integrate a convergent point using the 120-chart plumbing mixture.

    Sampling in plumbing coordinates resolves all ten boundary divisors and
    all fifteen corners.  The mixture-density denominator removes multiple
    coverage exactly.  A small ``radial_power`` importance-samples the
    integrable but sharp OPE collars of imaginary-energy kinematics.
    """

    if kernel.requires_power_subtraction():
        raise ValueError(
            "this energy requires the sphere-five-point boundary subtraction forest"
        )
    sobol_power = int(sobol_power)
    replicates = int(replicates)
    radial_power = float(radial_power)
    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    orderings = oriented_tree_orderings()
    estimates: list[complex] = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=5, scramble=True, seed=int(seed) + replicate)
        points = sampler.random_base2(sobol_power)
        values: list[complex] = []
        for point in points:
            q1, _density1 = _power_disk_sample(point[0], point[1], radial_power)
            q2, _density2 = _power_disk_sample(point[2], point[3], radial_power)
            ordering_index = min(int(point[4] * len(orderings)), len(orderings) - 1)
            selected_ordering = orderings[ordering_index]
            positions = linear_channel_positions_by_label(q1, q2, selected_ordering)
            log_mixture_density = oriented_bidisc_log_mixture_density_in_frame(
                positions,
                selected_ordering,
                radial_power=radial_power,
            )
            values.append(
                kernel.integrand_linear_gauge_weighted(
                    q1,
                    q2,
                    selected_ordering,
                    logarithmic_weight=-log_mixture_density,
                )
            )
        estimates.append(complex(np.mean(np.asarray(values, dtype=complex))))
    estimate_array = np.asarray(estimates, dtype=complex)
    mean = complex(np.mean(estimate_array))
    return EqualEnergyQMCResult(
        omega=kernel.omega,
        estimates=tuple(estimates),
        mean=mean,
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
        block_scheme=kernel.block_scheme,
    )


def integrate_physical_equal_energy_atlas_qmc(
    kernel: EqualEnergyFivePointKernel,
    *,
    sobol_power: int = 7,
    replicates: int = 4,
    radial_power: float = 0.5,
    seed: int = 20260823,
) -> EqualEnergyQMCResult:
    """Integrate the direct physical-domain BRY subtraction forest.

    Unlike the convergent-ray diagnostic, this routine requires nonempty
    power projectors. Each sampled moduli point is assigned to its best
    crossing sector, where the two-face inclusion-exclusion subtraction is
    evaluated before integration. No fit or continuation in omega occurs.
    """

    if not kernel.requires_power_subtraction():
        raise ValueError(
            "the physical forest driver expects at least one power subtraction"
        )
    kernel.validate_physical_subtraction_order()
    sobol_power = int(sobol_power)
    replicates = int(replicates)
    radial_power = float(radial_power)
    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    orderings = oriented_tree_orderings()
    estimates: list[complex] = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=5, scramble=True, seed=int(seed) + replicate)
        points = sampler.random_base2(sobol_power)
        values: list[complex] = []
        for point in points:
            q1, _density1 = _power_disk_sample(point[0], point[1], radial_power)
            q2, _density2 = _power_disk_sample(point[2], point[3], radial_power)
            ordering_index = min(int(point[4] * len(orderings)), len(orderings) - 1)
            selected_ordering = orderings[ordering_index]
            positions = linear_channel_positions_by_label(q1, q2, selected_ordering)
            log_mixture_density = oriented_bidisc_log_mixture_density_in_frame(
                positions,
                selected_ordering,
                radial_power=radial_power,
            )
            values.append(
                kernel.forest_subtracted_integrand_linear_gauge_weighted(
                    q1,
                    q2,
                    selected_ordering,
                    logarithmic_weight=-log_mixture_density,
                )
            )
        estimates.append(complex(np.mean(np.asarray(values, dtype=complex))))
    estimate_array = np.asarray(estimates, dtype=complex)
    mean = complex(np.mean(estimate_array))
    return EqualEnergyQMCResult(
        omega=kernel.omega,
        estimates=tuple(estimates),
        mean=mean,
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
        block_scheme=kernel.block_scheme,
    )


def _outside_best_channel_collars(
    positions: Sequence[complex | None],
    collar_radius: float,
) -> bool:
    """Return whether a moduli point belongs to the finite-part bulk."""

    channel = best_linear_channels(positions, limit=1)[0]
    return abs(channel.q1) >= float(collar_radius) and abs(channel.q2) >= float(
        collar_radius
    )


def _four_point_fundamental_cell(value: complex) -> bool:
    r"""Return the BRY six-sector cell ``|z-1|<1, 0<Re z<1/2``."""

    value = complex(value)
    return abs(value - 1.0) < 1.0 and 0.0 < value.real < 0.5


def integrate_physical_equal_energy_finite_part_qmc(
    kernel: EqualEnergyFivePointKernel,
    *,
    collar_radius: float = 0.10,
    bulk_sobol_power: int = 7,
    face_sobol_power: int = 8,
    replicates: int = 4,
    radial_power: float = 0.5,
    seed: int = 20260823,
) -> FivePointFinitePartResult:
    r"""Integrate the physical five-point amplitude by boundary strata.

    The moduli integral is decomposed into one bulk, all ten boundary faces,
    and all fifteen compatible corners.  The bulk has both plumbing collars
    excised.  On every face the normal radius is integrated with
    ``pi*rho^(2*alpha)/alpha`` and the remaining four-point modulus is split
    into the six BRY crossing cells.  At a corner both commuting radial
    finite parts are applied.  Thus no continuation or fit in ``omega`` is
    used anywhere.
    """

    if not kernel.requires_power_subtraction():
        raise ValueError("the physical finite-part driver expects a power subtraction")
    kernel.validate_physical_subtraction_order()
    collar_radius = float(collar_radius)
    bulk_sobol_power = int(bulk_sobol_power)
    face_sobol_power = int(face_sobol_power)
    replicates = int(replicates)
    radial_power = float(radial_power)
    if not 0.0 < collar_radius < 0.2:
        raise ValueError("collar_radius must lie between zero and 0.2")
    if bulk_sobol_power < 1 or face_sobol_power < 1 or replicates < 2:
        raise ValueError("Sobol powers must be positive and replicates at least two")
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")

    orderings = oriented_tree_orderings()
    divisors = five_point_boundary_divisors()
    face_orderings = five_point_face_sector_orderings()
    corners = five_point_boundary_corners()
    corner_contribution = complex(
        sum(kernel.corner_finite_part(corner, collar_radius) for corner in corners)
    )

    bulk_estimates: list[complex] = []
    face_estimates: list[complex] = []
    estimates: list[complex] = []
    for replicate in range(replicates):
        bulk_sampler = qmc.Sobol(d=5, scramble=True, seed=int(seed) + replicate)
        bulk_values: list[complex] = []
        for point in bulk_sampler.random_base2(bulk_sobol_power):
            q1, _density1 = _power_disk_sample(point[0], point[1], radial_power)
            q2, _density2 = _power_disk_sample(point[2], point[3], radial_power)
            ordering_index = min(int(point[4] * len(orderings)), len(orderings) - 1)
            selected_ordering = orderings[ordering_index]
            positions = linear_channel_positions_by_label(q1, q2, selected_ordering)
            if not _outside_best_channel_collars(positions, collar_radius):
                bulk_values.append(0.0 + 0.0j)
                continue
            log_mixture_density = oriented_bidisc_log_mixture_density_in_frame(
                positions,
                selected_ordering,
                radial_power=radial_power,
            )
            bulk_values.append(
                kernel.integrand_linear_gauge_weighted(
                    q1,
                    q2,
                    selected_ordering,
                    logarithmic_weight=-log_mixture_density,
                )
            )
        bulk_estimate = complex(np.mean(np.asarray(bulk_values, dtype=complex)))

        face_sampler = qmc.Sobol(
            d=2,
            scramble=True,
            seed=int(seed) + 10000 + replicate,
        )
        face_values: list[complex] = []
        for point in face_sampler.random_base2(face_sobol_power):
            modulus, jacobian = _plane_map(point[0], point[1])
            if (
                not _four_point_fundamental_cell(modulus)
                or abs(modulus) < collar_radius
            ):
                face_values.append(0.0 + 0.0j)
                continue
            density = sum(
                kernel.face_finite_part_density(
                    modulus,
                    divisor,
                    collar_radius,
                    ordering=ordering,
                )
                for divisor, ordering in face_orderings
            )
            face_values.append(complex(density * jacobian))
        face_estimate = complex(np.mean(np.asarray(face_values, dtype=complex)))

        bulk_estimates.append(bulk_estimate)
        face_estimates.append(face_estimate)
        estimates.append(bulk_estimate + face_estimate + corner_contribution)

    estimate_array = np.asarray(estimates, dtype=complex)
    return FivePointFinitePartResult(
        omega=kernel.omega,
        collar_radius=collar_radius,
        estimates=tuple(estimates),
        bulk_estimates=tuple(bulk_estimates),
        face_estimates=tuple(face_estimates),
        corner_contribution=corner_contribution,
        mean=complex(np.mean(estimate_array)),
        standard_error_real=float(
            np.std(estimate_array.real, ddof=1) / math.sqrt(replicates)
        ),
        standard_error_imag=float(
            np.std(estimate_array.imag, ddof=1) / math.sqrt(replicates)
        ),
        bulk_sobol_power=bulk_sobol_power,
        face_sobol_power=face_sobol_power,
        replicates=replicates,
        block_order=kernel.block_order,
        momentum_order=kernel.momentum_order,
        block_scheme=kernel.block_scheme,
    )


def write_qmc_result(
    result: EqualEnergyQMCResult | FivePointFinitePartResult,
    path: Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2) + "\n")
