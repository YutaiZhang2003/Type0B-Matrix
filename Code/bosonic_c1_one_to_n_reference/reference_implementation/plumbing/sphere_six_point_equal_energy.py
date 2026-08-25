#!/usr/bin/env python3
"""Residue-free equal-energy kernel for the sphere 1->5 worldsheet integral.

The module covers the analytic chamber ``omega=i*t`` with ``0<t<1/3``.  It
precomputes the three-Liouville-momentum quadrature and the six-point block for
each possible slot of the distinguished incoming operator.  Moduli points are
sampled from a mixed 720-comb plus 720-star plumbing atlas and re-expanded in
the channel with the smallest maximum plumbing radius.

No matrix-model formula is imported here.  The output is the labelled
worldsheet integral ``I6`` and its QMC uncertainty.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.stats import qmc

try:
    from ccy_sphere_six_point import (
        sphere_six_point_c_coefficients,
        sphere_six_point_h_c25_limit,
    )
    from ccy_sphere_six_point_star import (
        sphere_six_point_star_c_coefficients,
        sphere_six_point_star_direct_coefficients,
    )
    from liouville_torus import UpsilonB, log_yin_structure_constant_momentum
    from sphere_six_point_atlas import (
        SixPointLinearChannel,
        SixPointStarChannel,
        best_linear_channels,
        best_star_channels,
        liouville_primary_covariance_log,
        linear_channel_positions_by_label,
        mixed_atlas_log_density_in_frame,
        oriented_comb_orderings,
        oriented_tridisc_log_mixture_density_in_frame,
        star_channel_positions_by_label,
        timelike_free_boson_log_factor,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_sphere_six_point import (
        sphere_six_point_c_coefficients,
        sphere_six_point_h_c25_limit,
    )
    from plumbing.ccy_sphere_six_point_star import (
        sphere_six_point_star_c_coefficients,
        sphere_six_point_star_direct_coefficients,
    )
    from plumbing.liouville_torus import UpsilonB, log_yin_structure_constant_momentum
    from plumbing.sphere_six_point_atlas import (
        SixPointLinearChannel,
        SixPointStarChannel,
        best_linear_channels,
        best_star_channels,
        liouville_primary_covariance_log,
        linear_channel_positions_by_label,
        mixed_atlas_log_density_in_frame,
        oriented_comb_orderings,
        oriented_tridisc_log_mixture_density_in_frame,
        star_channel_positions_by_label,
        timelike_free_boson_log_factor,
    )


FIRST_RESIDUE_WALL = 1.0 / 3.0
Topology = Literal["comb", "star"]


@dataclass(frozen=True)
class SixPointMomentumArrays:
    """One topology and incoming-slot precomputation."""

    log_weighted_structure_constants: np.ndarray
    primary_exponents: np.ndarray
    coefficient_keys: tuple[tuple[int, int, int], ...]
    coefficient_matrix: np.ndarray
    p_values: np.ndarray


@dataclass(frozen=True)
class SixPointQMCResult:
    """Replicated QMC estimate of the labelled worldsheet integral I6."""

    t: float
    estimates: tuple[complex, ...]
    mean: complex
    standard_error_real: float
    standard_error_imag: float
    samples_per_replicate: int
    replicates: int
    block_order: int
    momentum_order: int
    momentum_maximum: float
    radial_power: float
    comb_selection_fraction: float
    star_selection_fraction: float
    maximum_selected_radius: float

    def to_json(self) -> dict[str, object]:
        return {
            "t": self.t,
            "I6": {"real": self.mean.real, "imag": self.mean.imag},
            "I6_standard_error": {
                "real": self.standard_error_real,
                "imag": self.standard_error_imag,
            },
            "replicate_I6": [
                {"real": value.real, "imag": value.imag} for value in self.estimates
            ],
            "samples_per_replicate": self.samples_per_replicate,
            "replicates": self.replicates,
            "block_order": self.block_order,
            "momentum_order": self.momentum_order,
            "momentum_maximum": self.momentum_maximum,
            "radial_power": self.radial_power,
            "comb_selection_fraction": self.comb_selection_fraction,
            "star_selection_fraction": self.star_selection_fraction,
            "maximum_selected_radius": self.maximum_selected_radius,
        }


def equal_outgoing_signed_energies(omega: complex) -> tuple[complex, ...]:
    """Return ``(+5 omega,-omega,...,-omega)`` for labelled 1->5."""

    omega = complex(omega)
    return (5.0 * omega,) + (-omega,) * 5


def _power_gauss_grid(
    order: int,
    maximum: float,
    power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre in u with P=Pmax*u**power."""

    order = int(order)
    maximum = float(maximum)
    power = float(power)
    if order <= 0 or maximum <= 0.0 or power < 1.0:
        raise ValueError("order and maximum must be positive and power at least one")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u = 0.5 * (nodes + 1.0)
    base_weights = 0.5 * weights
    p_values = maximum * u**power
    p_weights = base_weights * maximum * power * u ** (power - 1.0)
    return np.asarray(p_values, dtype=float), np.asarray(p_weights, dtype=float)


def _power_disk_sample(
    radial_uniform: float,
    angular_uniform: float,
    radial_power: float,
) -> complex:
    radial_power = float(radial_power)
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    u = min(
        max(float(radial_uniform), np.nextafter(0.0, 1.0)),
        np.nextafter(1.0, 0.0),
    )
    radius = u ** (1.0 / radial_power)
    angle = 2.0 * math.pi * float(angular_uniform)
    return complex(radius * cmath.exp(1.0j * angle))


class EqualEnergySixPointKernel:
    """Cached three-momentum kernel for one omega=i*t below the first wall."""

    def __init__(
        self,
        t: float,
        *,
        block_order: int = 4,
        momentum_order: int = 5,
        momentum_maximum: float = 6.0,
        momentum_power: float = 1.25,
        special_dps: int = 35,
    ) -> None:
        self.t = float(t)
        if not 0.0 < self.t < FIRST_RESIDUE_WALL:
            raise ValueError("the residue-free six-point kernel requires 0<t<1/3")
        self.omega = 1.0j * self.t
        self.block_order = int(block_order)
        self.momentum_order = int(momentum_order)
        self.momentum_maximum = float(momentum_maximum)
        self.momentum_power = float(momentum_power)
        if self.block_order < 0 or self.momentum_order <= 0:
            raise ValueError("block_order must be non-negative and momentum_order positive")
        if self.momentum_maximum <= 0.0 or self.momentum_power < 1.0:
            raise ValueError("invalid momentum grid")

        self.signed_energies = equal_outgoing_signed_energies(self.omega)
        self.external_momenta = (2.5 * self.omega,) + (0.5 * self.omega,) * 5
        self.external_weights = tuple(
            1.0 + momentum * momentum for momentum in self.external_momenta
        )
        self.special = UpsilonB(1.0, dps=int(special_dps))
        # Adjacent orders keep the three internal weights off exact diagonal
        # Kac collisions in both c- and common-weight h-recursion.  Each edge
        # still receives a convergent Gauss rule; adjacent base orders are the
        # production momentum-error ladder.
        self.momentum_grids = tuple(
            _power_gauss_grid(
                self.momentum_order + offset,
                self.momentum_maximum,
                self.momentum_power + 1.0e-6 * offset,
            )
            for offset in range(3)
        )
        self.fallback_counts: dict[str, int] = {
            "comb_regulated_h": 0,
            "star_direct": 0,
        }
        self.arrays: dict[tuple[Topology, int], SixPointMomentumArrays] = {}
        for topology in ("comb", "star"):
            for incoming_slot in range(6):
                self.arrays[(topology, incoming_slot)] = self._build_arrays(
                    topology, incoming_slot
                )

    @staticmethod
    def _coefficient_keys(order: int) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            sorted(
                (
                    (n1, n2, n3)
                    for n1 in range(order + 1)
                    for n2 in range(order + 1)
                    for n3 in range(order + 1)
                    if n1 + n2 + n3 <= order
                ),
                key=lambda key: (sum(key), key[0], key[1], key[2]),
            )
        )

    def _build_arrays(self, topology: Topology, incoming_slot: int) -> SixPointMomentumArrays:
        ordered_momenta = tuple(
            self.external_momenta[0] if slot == incoming_slot else self.external_momenta[1]
            for slot in range(6)
        )
        ordered_weights = tuple(1.0 + momentum * momentum for momentum in ordered_momenta)
        pa, pb, pc, pd, pe, pf = ordered_momenta
        keys = self._coefficient_keys(self.block_order)
        logs: list[complex] = []
        exponents: list[tuple[complex, complex, complex]] = []
        coefficient_rows: list[list[complex]] = []
        momentum_rows: list[tuple[float, float, float]] = []

        nodes1, weights1 = self.momentum_grids[0]
        nodes2, weights2 = self.momentum_grids[1]
        nodes3, weights3 = self.momentum_grids[2]
        outer1_logs = tuple(
            complex(log_yin_structure_constant_momentum(self.special, pa, pb, float(p1)))
            for p1 in nodes1
        )
        if topology == "star":
            outer2_logs = tuple(
                complex(log_yin_structure_constant_momentum(self.special, pc, pd, float(p2)))
                for p2 in nodes2
            )
            outer3_logs = tuple(
                complex(log_yin_structure_constant_momentum(self.special, float(p3), pe, pf))
                for p3 in nodes3
            )
        else:
            outer2_logs = ()
            outer3_logs = tuple(
                complex(log_yin_structure_constant_momentum(self.special, float(p3), pe, pf))
                for p3 in nodes3
            )

        for index1, (p1, weight1) in enumerate(
            zip(nodes1, weights1)
        ):
            p1_value = float(p1)
            h1 = 1.0 + p1_value**2
            for index2, (p2, weight2) in enumerate(
                zip(nodes2, weights2)
            ):
                p2_value = float(p2)
                h2 = 1.0 + p2_value**2
                if topology == "comb":
                    middle12_log = complex(
                        log_yin_structure_constant_momentum(
                            self.special, p1_value, pc, p2_value
                        )
                    )
                for index3, (p3, weight3) in enumerate(
                    zip(nodes3, weights3)
                ):
                    p3_value = float(p3)
                    h3 = 1.0 + p3_value**2
                    if topology == "comb":
                        remaining_logs = (
                            middle12_log,
                            complex(
                                log_yin_structure_constant_momentum(
                                    self.special, p2_value, pd, p3_value
                                )
                            ),
                            outer3_logs[index3],
                        )
                        try:
                            coefficients = sphere_six_point_c_coefficients(
                                central_charge=25.0,
                                external_weights=ordered_weights,
                                internal_weights=(h1, h2, h3),
                                order1=self.block_order,
                                order2=self.block_order,
                                order3=self.block_order,
                                max_total_order=self.block_order,
                            )
                        except ZeroDivisionError:
                            coefficients, _errors = sphere_six_point_h_c25_limit(
                                external_weights=ordered_weights,
                                internal_weights=(h1, h2, h3),
                                order1=self.block_order,
                                order2=self.block_order,
                                order3=self.block_order,
                                max_total_order=self.block_order,
                            )
                            self.fallback_counts["comb_regulated_h"] += 1
                        x_exponent = 2.0 * (
                            h1 - ordered_weights[0] - ordered_weights[1]
                        )
                        y_exponent = 2.0 * (
                            h2 - ordered_weights[2] - h1
                        )
                        z_exponent = 2.0 * (
                            h3 - ordered_weights[3] - h2
                        )
                        # x=q1*q2*q3, y=q2*q3, z=q3.
                        primary = (
                            x_exponent,
                            x_exponent + y_exponent,
                            x_exponent + y_exponent + z_exponent,
                        )
                    else:
                        remaining_logs = (
                            outer2_logs[index2],
                            outer3_logs[index3],
                            complex(
                                log_yin_structure_constant_momentum(
                                    self.special, p3_value, p2_value, p1_value
                                )
                            ),
                        )
                        try:
                            coefficients = sphere_six_point_star_c_coefficients(
                                central_charge=25.0,
                                external_weights=ordered_weights,
                                internal_weights=(h1, h2, h3),
                                order1=self.block_order,
                                order2=self.block_order,
                                order3=self.block_order,
                                max_total_order=self.block_order,
                            )
                        except ZeroDivisionError:
                            coefficients = sphere_six_point_star_direct_coefficients(
                                central_charge=25.0,
                                external_weights=ordered_weights,
                                internal_weights=(h1, h2, h3),
                                order1=self.block_order,
                                order2=self.block_order,
                                order3=self.block_order,
                                max_total_order=self.block_order,
                            )
                            self.fallback_counts["star_direct"] += 1
                        primary = (
                            2.0 * (h1 - ordered_weights[0] - ordered_weights[1]),
                            2.0 * (h2 - ordered_weights[2] - ordered_weights[3]),
                            # The third cherry is represented in the global
                            # z chart by (1/q3,infinity).  Transforming the
                            # finite w=1 field and the w=0 state at infinity
                            # gives q3^(h3+d5-d6).  This also supplies the
                            # |q3|^4 falloff needed to cancel the
                            # d^2(1/q3) Jacobian at the boundary.
                            2.0 * (h3 + ordered_weights[4] - ordered_weights[5]),
                        )
                    logarithm = complex(
                        math.log(
                            float(weight1)
                            * float(weight2)
                            * float(weight3)
                            / math.pi**3
                        )
                        + outer1_logs[index1]
                        + sum(remaining_logs)
                    )
                    logs.append(logarithm)
                    exponents.append(tuple(complex(value) for value in primary))
                    coefficient_rows.append([coefficients.get(key, 0.0j) for key in keys])
                    momentum_rows.append((p1_value, p2_value, p3_value))
        return SixPointMomentumArrays(
            log_weighted_structure_constants=np.asarray(logs, dtype=complex),
            primary_exponents=np.asarray(exponents, dtype=complex),
            coefficient_keys=keys,
            coefficient_matrix=np.asarray(coefficient_rows, dtype=complex),
            p_values=np.asarray(momentum_rows, dtype=float),
        )

    @staticmethod
    def _series_monomials(
        q_values: Sequence[complex],
        keys: Sequence[tuple[int, int, int]],
    ) -> np.ndarray:
        q1, q2, q3 = (complex(value) for value in q_values)
        return np.asarray(
            [q1**n1 * q2**n2 * q3**n3 for n1, n2, n3 in keys],
            dtype=complex,
        )

    def _momentum_sum(
        self,
        topology: Topology,
        incoming_slot: int,
        q_values: Sequence[complex],
        base_logarithm: complex,
    ) -> complex:
        arrays = self.arrays[(topology, int(incoming_slot))]
        radii = tuple(abs(complex(value)) for value in q_values)
        if any(radius <= 0.0 for radius in radii):
            raise ZeroDivisionError("the selected channel lies on a plumbing boundary")
        logarithmic_radii = np.log(np.asarray(radii, dtype=float))
        block_q_values = tuple(complex(value) for value in q_values)
        holomorphic = np.sum(
            arrays.coefficient_matrix
            * self._series_monomials(block_q_values, arrays.coefficient_keys)[None, :],
            axis=1,
        )
        antiholomorphic = np.sum(
            arrays.coefficient_matrix
            * self._series_monomials(
                tuple(value.conjugate() for value in block_q_values),
                arrays.coefficient_keys,
            )[None, :],
            axis=1,
        )
        exponentials = np.exp(
            complex(base_logarithm)
            + arrays.log_weighted_structure_constants
            + np.sum(
                arrays.primary_exponents * logarithmic_radii[None, :],
                axis=1,
            )
        )
        return complex(np.sum(exponentials * holomorphic * antiholomorphic))

    def select_channel(
        self,
        positions: Sequence[complex | None],
    ) -> tuple[Topology, SixPointLinearChannel | SixPointStarChannel]:
        """Select the topology with the smaller maximum plumbing radius."""

        comb = best_linear_channels(positions, limit=1)[0]
        try:
            star = best_star_channels(positions, limit=1)[0]
        except RuntimeError:
            return "comb", comb
        if star.score < comb.score:
            return "star", star
        return "comb", comb

    def integrand_at_positions(
        self,
        positions: Sequence[complex | None],
        *,
        logarithmic_weight: float = 0.0,
    ) -> tuple[complex, Topology, float]:
        """Evaluate the matter integrand in the supplied fixed-label gauge."""

        topology, channel = self.select_channel(positions)
        return self.integrand_in_channel(
            positions,
            topology,
            channel,
            logarithmic_weight=logarithmic_weight,
        )

    def integrand_in_channel(
        self,
        positions: Sequence[complex | None],
        topology: Topology,
        channel: SixPointLinearChannel | SixPointStarChannel,
        *,
        logarithmic_weight: float = 0.0,
    ) -> tuple[complex, Topology, float]:
        """Evaluate in an explicitly supplied comb or star channel."""

        if topology not in ("comb", "star"):
            raise ValueError("topology must be 'comb' or 'star'")
        if topology == "comb" and not isinstance(channel, SixPointLinearChannel):
            raise TypeError("a comb evaluation requires SixPointLinearChannel")
        if topology == "star" and not isinstance(channel, SixPointStarChannel):
            raise TypeError("a star evaluation requires SixPointStarChannel")
        q_values = (channel.q1, channel.q2, channel.q3)
        incoming_slot = channel.ordering.index(0)
        base_logarithm = (
            timelike_free_boson_log_factor(positions, self.signed_energies)
            + liouville_primary_covariance_log(channel, self.external_weights)
            + float(logarithmic_weight)
        )
        return (
            self._momentum_sum(
                topology,
                incoming_slot,
                q_values,
                base_logarithm,
            ),
            topology,
            channel.score,
        )


def integrate_convergent_equal_energy_atlas_qmc(
    kernel: EqualEnergySixPointKernel,
    *,
    sobol_power: int = 5,
    replicates: int = 4,
    radial_power: float = 0.08,
    seed: int = 20260823,
) -> SixPointQMCResult:
    """Integrate I6 with the mixed comb/star plumbing atlas."""

    sobol_power = int(sobol_power)
    replicates = int(replicates)
    radial_power = float(radial_power)
    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    orderings = oriented_comb_orderings()
    proposal_count = 2 * len(orderings)
    estimates: list[complex] = []
    comb_count = 0
    star_count = 0
    maximum_score = 0.0

    for replicate in range(replicates):
        sampler = qmc.Sobol(d=7, scramble=True, seed=int(seed) + replicate)
        points = sampler.random_base2(sobol_power)
        values: list[complex] = []
        for point in points:
            q_values = (
                _power_disk_sample(point[0], point[1], radial_power),
                _power_disk_sample(point[2], point[3], radial_power),
                _power_disk_sample(point[4], point[5], radial_power),
            )
            proposal_index = min(int(point[6] * proposal_count), proposal_count - 1)
            if proposal_index < len(orderings):
                sampled_topology: Topology = "comb"
                ordering = orderings[proposal_index]
                positions = linear_channel_positions_by_label(*q_values, ordering)
                fixed = (ordering[0], ordering[4], ordering[5])
                moving = (ordering[1], ordering[2], ordering[3])
            else:
                sampled_topology = "star"
                ordering = orderings[proposal_index - len(orderings)]
                positions = star_channel_positions_by_label(*q_values, ordering)
                fixed = (ordering[0], ordering[2], ordering[5])
                moving = (ordering[1], ordering[3], ordering[4])
            log_density = mixed_atlas_log_density_in_frame(
                positions,
                fixed_zero=fixed[0],
                fixed_one=fixed[1],
                fixed_infinity=fixed[2],
                moving_labels=moving,
                radial_power=radial_power,
            )
            value, selected_topology, score = kernel.integrand_at_positions(
                positions,
                logarithmic_weight=-log_density,
            )
            values.append(value)
            if selected_topology == "comb":
                comb_count += 1
            else:
                star_count += 1
            maximum_score = max(maximum_score, score)
        estimates.append(complex(np.mean(np.asarray(values, dtype=complex))))

    estimate_array = np.asarray(estimates, dtype=complex)
    mean = complex(np.mean(estimate_array))
    total_selections = comb_count + star_count
    return SixPointQMCResult(
        t=kernel.t,
        estimates=tuple(estimates),
        mean=mean,
        standard_error_real=float(np.std(estimate_array.real, ddof=1) / math.sqrt(replicates)),
        standard_error_imag=float(np.std(estimate_array.imag, ddof=1) / math.sqrt(replicates)),
        samples_per_replicate=2**sobol_power,
        replicates=replicates,
        block_order=kernel.block_order,
        momentum_order=kernel.momentum_order,
        momentum_maximum=kernel.momentum_maximum,
        radial_power=radial_power,
        comb_selection_fraction=comb_count / total_selections,
        star_selection_fraction=star_count / total_selections,
        maximum_selected_radius=maximum_score,
    )


def integrate_convergent_equal_energy_comb_qmc(
    kernel: EqualEnergySixPointKernel,
    *,
    sobol_power: int = 5,
    replicates: int = 4,
    radial_power: float = 0.08,
    seed: int = 20260823,
) -> SixPointQMCResult:
    """Comb-only 720-chart baseline used to audit the mixed atlas."""

    sobol_power = int(sobol_power)
    replicates = int(replicates)
    radial_power = float(radial_power)
    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    orderings = oriented_comb_orderings()
    estimates: list[complex] = []
    maximum_score = 0.0
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=7, scramble=True, seed=int(seed) + replicate)
        points = sampler.random_base2(sobol_power)
        values: list[complex] = []
        for point in points:
            q_values = (
                _power_disk_sample(point[0], point[1], radial_power),
                _power_disk_sample(point[2], point[3], radial_power),
                _power_disk_sample(point[4], point[5], radial_power),
            )
            ordering_index = min(int(point[6] * len(orderings)), len(orderings) - 1)
            ordering = orderings[ordering_index]
            positions = linear_channel_positions_by_label(*q_values, ordering)
            log_density = oriented_tridisc_log_mixture_density_in_frame(
                positions,
                ordering,
                radial_power=radial_power,
            )
            channel = best_linear_channels(positions, limit=1)[0]
            value, _topology, score = kernel.integrand_in_channel(
                positions,
                "comb",
                channel,
                logarithmic_weight=-log_density,
            )
            values.append(value)
            maximum_score = max(maximum_score, score)
        estimates.append(complex(np.mean(np.asarray(values, dtype=complex))))
    estimate_array = np.asarray(estimates, dtype=complex)
    mean = complex(np.mean(estimate_array))
    return SixPointQMCResult(
        t=kernel.t,
        estimates=tuple(estimates),
        mean=mean,
        standard_error_real=float(np.std(estimate_array.real, ddof=1) / math.sqrt(replicates)),
        standard_error_imag=float(np.std(estimate_array.imag, ddof=1) / math.sqrt(replicates)),
        samples_per_replicate=2**sobol_power,
        replicates=replicates,
        block_order=kernel.block_order,
        momentum_order=kernel.momentum_order,
        momentum_maximum=kernel.momentum_maximum,
        radial_power=radial_power,
        comb_selection_fraction=1.0,
        star_selection_fraction=0.0,
        maximum_selected_radius=maximum_score,
    )


__all__ = [
    "EqualEnergySixPointKernel",
    "FIRST_RESIDUE_WALL",
    "SixPointQMCResult",
    "equal_outgoing_signed_energies",
    "integrate_convergent_equal_energy_atlas_qmc",
    "integrate_convergent_equal_energy_comb_qmc",
]
