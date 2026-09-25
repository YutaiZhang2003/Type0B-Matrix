#!/usr/bin/env python3
r"""Randomized-QMC genus-one vector two-point moduli integral.

The integration variables are

.. math::

   (\tau_1,\tau_2,a,b,P_1,P_2),\qquad z=a+b\tau,

with ``tau`` in a truncated modular fundamental domain, ``a,b`` in the unit
square, and the two super-Liouville momenta on the positive half-line.  The
positive-momentum convention already includes the reflection doubling, so
each spectral resolution is ``dP / pi``.

Independent scrambled Sobol replicates provide a falsifiable sampling-error
diagnostic.  The external-energy-independent finite-level block response is
saved in stage 1.  Stage 2 evaluates that response for an energy grid and
assembles the free fields, ghosts, PCO components, and diagonal GSO sum.

The direct necklace block is not uniformly convergent as one plumbing
parameter approaches the unit circle.  Every artifact therefore records both
level-4 and level-3 results.  Their difference is a truncation diagnostic,
not a rigorous error bound.  Likewise, ``tau2_max`` defines an explicit cusp
regulator; no result from this module should be called the unregulated
amplitude until stability under increasing ``tau2_max`` is demonstrated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Sequence

import numpy as np
from scipy.special import gamma, gammaincinv
from scipy.stats import qmc

from spin23_genus1_amplitude import GenusOneFixedGeometryWorkspace
from spin23_genus1_free_fields import (
    even_superghost_chiral_partition,
    flavored_majorana_wick_factor,
    heterotic_fixed_puncture_measure_density,
    majorana_chiral_partition,
    majorana_wick_factor,
    noncompact_boson_partition_per_unit_volume,
    torus_scalar_green,
)
from spin23_genus1_gso import diagonal_gso_table
from spin23_genus1_two_point_sweep import (
    EMPTY_WORD,
    G_MINUS_HALF,
    SCHEMA_VERSION,
    _FAMILIES,
    _atomic_savez,
    _build_family_response,
    _clear_node_caches,
    _decode_complex,
    _encode_complex,
    _family_metadata,
    _load_npz,
    _pad_polynomials,
    _stable_complex_sum,
)
from spin23_super_liouville_data import (
    ns_structure_constants,
    ns_weight,
    rr_ns_structure_constants,
)


_SPIN_LABELS = ("NS", "NS_tilde", "R")
_FAMILY_BY_SPIN_COMPONENT = {
    ("NS", 0): 0,
    ("NS", 1): 1,
    ("NS_tilde", 0): 2,
    ("NS_tilde", 1): 3,
    ("R", 0): 4,
    ("R", 1): 5,
}


def _validate_power_of_two(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 2 or value & (value - 1):
        raise ValueError(f"{name} must be a power of two of at least two")
    return value


def _validate_partition(task_index: int, task_count: int) -> None:
    if isinstance(task_index, bool) or not isinstance(task_index, int):
        raise TypeError("task_index must be an integer")
    if isinstance(task_count, bool) or not isinstance(task_count, int):
        raise TypeError("task_count must be an integer")
    if task_count < 1 or not 0 <= task_index < task_count:
        raise ValueError("task partition must satisfy 0 <= index < count")


def scrambled_sobol_replicates(
    *,
    replicate_count: int,
    samples_per_replicate: int,
    base_seed: int,
) -> np.ndarray:
    """Return independent six-dimensional scrambled Sobol replicates."""

    if isinstance(replicate_count, bool) or not isinstance(replicate_count, int):
        raise TypeError("replicate_count must be an integer")
    if replicate_count < 2:
        raise ValueError("at least two replicates are required for an error estimate")
    count = _validate_power_of_two(samples_per_replicate, "samples_per_replicate")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    exponent = count.bit_length() - 1
    return np.stack(
        tuple(
            qmc.Sobol(d=6, scramble=True, seed=base_seed + replicate).random_base2(
                m=exponent
            )
            for replicate in range(replicate_count)
        )
    )


def fundamental_domain_point(
    u_real: float,
    u_imaginary: float,
    *,
    tau2_max: float,
) -> tuple[complex, float]:
    r"""Map a unit-square point to ``F`` truncated at ``tau2_max``.

    The proposal density is the normalized hyperbolic density

    .. math::

       p(\tau)=\frac{1}{C_Y\tau_2^2},\qquad
       C_Y=\frac{\pi}{3}-\frac{1}{Y},

    on ``|tau| >= 1``, ``|Re(tau)| <= 1/2``, and ``tau2 <= Y``.  The returned
    weight converts this proposal to the Euclidean measure ``d^2 tau``.
    """

    first = float(u_real)
    second = float(u_imaginary)
    upper = float(tau2_max)
    if not 0.0 < first < 1.0 or not 0.0 < second < 1.0:
        raise ValueError("scrambled coordinates must lie strictly inside (0,1)")
    if not math.isfinite(upper) or upper <= 1.0:
        raise ValueError("tau2_max must be finite and exceed one")
    normalization = math.pi / 3.0 - 1.0 / upper

    # Invert asin(x)+pi/6-(x+1/2)/Y = u*C by monotone Newton steps.
    x = math.sin(-math.pi / 6.0 + first * math.pi / 3.0)
    target = first * normalization
    for _ in range(12):
        residual = (
            math.asin(x)
            + math.pi / 6.0
            - (x + 0.5) / upper
            - target
        )
        derivative = 1.0 / math.sqrt(1.0 - x * x) - 1.0 / upper
        candidate = x - residual / derivative
        x = min(0.5, max(-0.5, candidate))
    residual = (
        math.asin(x)
        + math.pi / 6.0
        - (x + 0.5) / upper
        - target
    )
    if abs(residual) > 2.0e-14:
        raise ArithmeticError("fundamental-domain inverse CDF did not converge")

    lower = math.sqrt(1.0 - x * x)
    inverse_y = 1.0 / lower - second * (1.0 / lower - 1.0 / upper)
    y = 1.0 / inverse_y
    tau = complex(x, y)
    if abs(tau) < 1.0 - 2.0e-14 or y > upper * (1.0 + 2.0e-14):
        raise AssertionError("sample lies outside the truncated fundamental domain")
    return tau, normalization * y * y


def torus_point(
    tau: complex,
    u_a: float,
    u_b: float,
) -> tuple[complex, float]:
    r"""Return ``z=a+b*tau`` and the Jacobian ``d^2z=tau2 da db``."""

    a = float(u_a)
    b = float(u_b)
    if not 0.0 < a < 1.0 or not 0.0 < b < 1.0:
        raise ValueError("scrambled torus coordinates must lie inside (0,1)")
    modulus = complex(tau)
    return a + b * modulus, modulus.imag


def spectral_importance_point(
    unit_value: float,
    *,
    gaussian_scale: float,
    endpoint_power: int,
) -> tuple[float, float]:
    r"""Map one uniform coordinate to a Gaussian spectral proposal.

    For ``m=endpoint_power`` and ``s=gaussian_scale``, the map samples
    ``t=s*P**2`` from ``Gamma((m+1)/2,1)``.  The returned weight multiplies
    the Gaussian-stripped integrand after its endpoint factor ``P**m`` has
    also been removed.
    """

    unit = float(unit_value)
    scale = float(gaussian_scale)
    power = endpoint_power
    if not 0.0 < unit < 1.0:
        raise ValueError("the spectral unit coordinate must lie inside (0,1)")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("gaussian_scale must be finite and positive")
    if isinstance(power, bool) or not isinstance(power, int) or power < 0:
        raise ValueError("endpoint_power must be a nonnegative integer")
    shape = 0.5 * (power + 1.0)
    transformed = float(gammaincinv(shape, unit))
    momentum = math.sqrt(transformed / scale)
    if not math.isfinite(momentum) or momentum <= 0.0:
        raise ArithmeticError("spectral importance map returned an invalid momentum")
    weight = float(gamma(shape)) / (
        2.0 * math.pi * scale**shape * momentum**power
    )
    return momentum, weight


def _word_shift(word: tuple[object, ...]) -> float:
    if word == EMPTY_WORD:
        return 0.0
    if word == G_MINUS_HALF:
        return 0.5
    raise ValueError("the vector sweep supports only P and G_-1/2 words")


def _frame_exponential_data(
    workspace: GenusOneFixedGeometryWorkspace,
    holomorphic_words: Sequence[tuple[object, ...]],
    antiholomorphic_words: Sequence[tuple[object, ...]],
) -> tuple[complex, complex]:
    r"""Return ``(A,B)`` such that the frame factor is ``A exp(B omega^2)``."""

    coordinates = workspace.coordinates
    constant = 0.0j
    slope = 0.0j
    log_holomorphic_constant = np.log(2j * math.pi)
    log_antiholomorphic_constant = np.log(-2j * math.pi)
    for point, hol_word, anti_word in zip(
        coordinates.additive_points,
        holomorphic_words,
        antiholomorphic_words,
    ):
        holomorphic_log = log_holomorphic_constant + 2j * math.pi * point
        antiholomorphic_log = (
            log_antiholomorphic_constant - 2j * math.pi * point.conjugate()
        )
        constant += (0.5 + _word_shift(tuple(hol_word))) * holomorphic_log
        constant += (0.5 + _word_shift(tuple(anti_word))) * antiholomorphic_log
        slope += 0.5 * (holomorphic_log + antiholomorphic_log)
    return complex(np.exp(constant)), complex(slope)


def two_vector_geometry_data(
    tau: complex,
    z: complex,
    *,
    flavor: int,
    free_field_precision: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return reusable coefficients for the three even-spin contributions.

    ``coefficients[s,c]`` and ``exponents[c]`` assemble one fixed-spin value
    from its two Liouville PCO components as

    ``sum_c coefficients[s,c] * exp(exponents[c]*omega**2)`` times the
    component momentum polynomial and Liouville value.
    """

    if isinstance(flavor, bool) or not isinstance(flavor, int) or not 0 <= flavor < 23:
        raise ValueError("flavor must be an integer in [0,23)")
    workspace = GenusOneFixedGeometryWorkspace.from_points(tau, (0.0j, z))
    coordinates = workspace.coordinates
    green = torus_scalar_green(
        coordinates.additive_points[0],
        coordinates.additive_points[1],
        coordinates.tau,
        precision=free_field_precision,
    )
    scalar_partition = noncompact_boson_partition_per_unit_volume(
        coordinates.tau,
        alpha_prime=2.0,
        precision=free_field_precision,
    )
    frame_data = tuple(
        _frame_exponential_data(
            workspace,
            family.holomorphic_words,
            family.antiholomorphic_words,
        )
        for family in (_FAMILIES[0], _FAMILIES[1])
    )
    exponents = np.asarray(
        tuple(-green + frame_slope for _, frame_slope in frame_data),
        dtype=np.complex128,
    )
    coefficients = np.empty((len(_SPIN_LABELS), 2), dtype=np.complex128)
    even_entries = tuple(
        entry for entry in diagonal_gso_table() if not entry.spin_structure.arf_invariant
    )
    for spin_index, entry in enumerate(even_entries):
        spin = entry.spin_structure
        fixed_common = (
            0.25
            * scalar_partition
            * majorana_chiral_partition(
                coordinates.tau,
                spin,
                n_fermions=1,
                precision=free_field_precision,
            )
            * flavored_majorana_wick_factor(
                coordinates.additive_points,
                (flavor, flavor),
                coordinates.tau,
                spin,
                n_fermions=23,
                include_even_partition=True,
                precision=free_field_precision,
            ).conjugate()
            * even_superghost_chiral_partition(
                coordinates.tau,
                spin,
                precision=free_field_precision,
            )
            * heterotic_fixed_puncture_measure_density(
                coordinates.tau,
                precision=free_field_precision,
            )
        )
        wick_factors = (
            majorana_wick_factor((), coordinates.tau, spin, precision=free_field_precision),
            majorana_wick_factor(
                coordinates.additive_points,
                coordinates.tau,
                spin,
                precision=free_field_precision,
            ),
        )
        for component in range(2):
            coefficients[spin_index, component] = (
                fixed_common
                * wick_factors[component]
                * frame_data[component][0]
            )
    return coefficients, exponents


def assemble_two_vector_values(
    omega_values: Sequence[complex],
    liouville_values: np.ndarray,
    coefficients: np.ndarray,
    exponents: np.ndarray,
    *,
    include_string_phase: bool,
) -> np.ndarray:
    """Assemble total and three fixed-spin values for one joint sample."""

    energies = np.asarray(tuple(complex(value) for value in omega_values))
    if liouville_values.shape != (len(energies), len(_FAMILIES)):
        raise ValueError("liouville_values has the wrong shape")
    if coefficients.shape != (3, 2) or exponents.shape != (2,):
        raise ValueError("geometry assembly arrays have the wrong shape")
    omega_squared = energies * energies
    momentum_factors = np.stack(
        (np.ones(len(energies), dtype=np.complex128), -omega_squared),
        axis=1,
    )
    fixed_spin = np.zeros((len(energies), 3), dtype=np.complex128)
    for spin_index, spin_label in enumerate(_SPIN_LABELS):
        for component in range(2):
            family_index = _FAMILY_BY_SPIN_COMPONENT[(spin_label, component)]
            fixed_spin[:, spin_index] += (
                coefficients[spin_index, component]
                * np.exp(exponents[component] * omega_squared)
                * momentum_factors[:, component]
                * liouville_values[:, family_index]
            )
    projector = np.asarray(
        tuple(
            entry.projector_coefficient
            for entry in diagonal_gso_table()
            if not entry.spin_structure.arf_invariant
        ),
        dtype=np.float64,
    )
    total = fixed_spin @ projector
    if include_string_phase:
        total *= -1.0  # i**2 for two external states
    return np.concatenate((total[:, None], fixed_spin), axis=1)


def _cutoff_pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, bool):
        raise TypeError("maximum_twice_levels must be an integer or pair")
    result = (value, value) if isinstance(value, int) else tuple(int(v) for v in value)
    if len(result) != 2 or any(v < 0 or v % 2 for v in result):
        raise ValueError("twice-level cutoffs must be a nonnegative even pair")
    return result


def prepare_moduli_response_shard(
    *,
    output_path: Path,
    replicate_count: int,
    samples_per_replicate: int,
    base_seed: int,
    tau2_max: float,
    flavor: int,
    maximum_twice_levels: int | Sequence[int],
    lower_maximum_twice_levels: int | Sequence[int],
    condition_limit: float,
    free_field_precision: int,
    task_index: int,
    task_count: int,
) -> Path:
    """Prepare one shard of joint moduli, puncture, and momentum responses."""

    _validate_partition(task_index, task_count)
    high_cutoffs = _cutoff_pair(maximum_twice_levels)
    low_cutoffs = _cutoff_pair(lower_maximum_twice_levels)
    if any(low > high for low, high in zip(low_cutoffs, high_cutoffs)):
        raise ValueError("the lower rectangle cannot exceed the main rectangle")
    unit_replicates = scrambled_sobol_replicates(
        replicate_count=replicate_count,
        samples_per_replicate=samples_per_replicate,
        base_seed=base_seed,
    )
    total_samples = replicate_count * samples_per_replicate
    sample_indices = tuple(range(task_index, total_samples, task_count))
    count = len(sample_indices)
    unit_points = np.empty((count, 6), dtype=np.float64)
    replicate_indices = np.empty(count, dtype=np.int64)
    tau_values = np.empty(count, dtype=np.complex128)
    z_values = np.empty(count, dtype=np.complex128)
    plumbing_values = np.empty((count, 2), dtype=np.complex128)
    geometry_weights = np.empty(count, dtype=np.float64)
    momenta = np.empty((count, len(_FAMILIES), 2), dtype=np.float64)
    spectral_weights = np.empty((count, len(_FAMILIES)), dtype=np.float64)
    assembly_coefficients = np.empty((count, 3, 2), dtype=np.complex128)
    assembly_exponents = np.empty((count, 2), dtype=np.complex128)
    high_responses = []
    low_responses = []
    started = time.perf_counter()
    for local_index, sample_index in enumerate(sample_indices):
        replicate_index, within_replicate = divmod(
            sample_index, samples_per_replicate
        )
        unit = unit_replicates[replicate_index, within_replicate]
        tau, tau_weight = fundamental_domain_point(
            unit[0], unit[1], tau2_max=tau2_max
        )
        z, z_weight = torus_point(tau, unit[2], unit[3])
        workspace = GenusOneFixedGeometryWorkspace.from_points(tau, (0.0j, z))
        plumbing = workspace.coordinates.plumbing_parameters
        scales = tuple(-math.log(abs(value)) for value in plumbing)
        coefficients, exponents = two_vector_geometry_data(
            tau,
            z,
            flavor=flavor,
            free_field_precision=free_field_precision,
        )

        sample_high = []
        sample_low = []
        for family_index, family in enumerate(_FAMILIES):
            selected = tuple(
                spectral_importance_point(
                    unit[4 + edge],
                    gaussian_scale=scales[edge],
                    endpoint_power=family.endpoint_powers[edge],
                )
                for edge in range(2)
            )
            internal = tuple(value[0] for value in selected)
            momenta[local_index, family_index] = internal
            spectral_weights[local_index, family_index] = math.prod(
                value[1] for value in selected
            )
            sample_high.append(
                _build_family_response(
                    family,
                    internal,
                    plumbing,
                    maximum_twice_levels=high_cutoffs,
                    condition_limit=condition_limit,
                )
            )
            sample_low.append(
                _build_family_response(
                    family,
                    internal,
                    plumbing,
                    maximum_twice_levels=high_cutoffs,
                    condition_limit=condition_limit,
                    evaluation_twice_levels=low_cutoffs,
                )
            )

        unit_points[local_index] = unit
        replicate_indices[local_index] = replicate_index
        tau_values[local_index] = tau
        z_values[local_index] = z
        plumbing_values[local_index] = plumbing
        geometry_weights[local_index] = tau_weight * z_weight
        assembly_coefficients[local_index] = coefficients
        assembly_exponents[local_index] = exponents
        high_responses.append(sample_high)
        low_responses.append(sample_low)
        _clear_node_caches()

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "spin23_genus1_moduli_response",
        "integration_variables": ["tau_re", "tau_im", "torus_a", "torus_b", "P1", "P2"],
        "replicate_count": replicate_count,
        "samples_per_replicate": samples_per_replicate,
        "total_samples": total_samples,
        "base_seed": base_seed,
        "tau2_max": float(tau2_max),
        "flavor": flavor,
        "maximum_twice_levels": list(high_cutoffs),
        "lower_maximum_twice_levels": list(low_cutoffs),
        "condition_limit": float(condition_limit),
        "free_field_precision": free_field_precision,
        "task_index": task_index,
        "task_count": task_count,
        "families": [_family_metadata(family) for family in _FAMILIES],
        "runtime_seconds": time.perf_counter() - started,
        "scope": "tau2_truncated_moduli_integral_with_level_diagnostic",
    }
    _atomic_savez(
        output_path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        replicate_indices=replicate_indices,
        unit_points=unit_points,
        tau_values=tau_values,
        z_values=z_values,
        plumbing_values=plumbing_values,
        geometry_weights=geometry_weights,
        momenta=momenta,
        spectral_weights=spectral_weights,
        assembly_coefficients=assembly_coefficients,
        assembly_exponents=assembly_exponents,
        response_polynomials=_pad_polynomials(high_responses),
        lower_response_polynomials=_pad_polynomials(low_responses),
    )
    return output_path


def _liouville_value_pair_for_sample(
    *,
    omega_values: np.ndarray,
    momenta: np.ndarray,
    spectral_weights: np.ndarray,
    plumbing: np.ndarray,
    high_polynomials: np.ndarray,
    low_polynomials: np.ndarray,
    structure_precision: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate level-high/low families while sharing structure constants."""

    high_values = np.zeros((len(omega_values), len(_FAMILIES)), dtype=np.complex128)
    low_values = np.zeros((len(omega_values), len(_FAMILIES)), dtype=np.complex128)
    ns_primary = math.prod(abs(complex(value)) ** (-1.0 / 8.0) for value in plumbing)
    for energy_index, omega in enumerate(omega_values):
        external_weight = ns_weight(complex(omega))
        ns_cache: dict[tuple[float, float], tuple[complex, complex]] = {}
        rr_cache: dict[tuple[float, float], tuple[complex, complex]] = {}
        for family_index, family in enumerate(_FAMILIES):
            internal = tuple(float(value) for value in momenta[family_index])
            if family.sector == "NS":
                constants = ns_cache.get(internal)
                if constants is None:
                    constants = ns_structure_constants(
                        internal[1], complex(omega), internal[0], precision=structure_precision
                    )
                    ns_cache[internal] = constants
                branch_coefficients = (constants[0] ** 2, constants[1] ** 2)
                primary = ns_primary
            else:
                constants = rr_cache.get(internal)
                if constants is None:
                    constants = rr_ns_structure_constants(
                        internal[1], internal[0], complex(omega), precision=structure_precision
                    )
                    rr_cache[internal] = constants
                branch_coefficients = (
                    2.0 * constants[1] ** 2,
                    2.0 * constants[0] ** 2,
                )
                primary = 1.0
            high_node_value = 0.0j
            low_node_value = 0.0j
            for branch_index, branch_coefficient in enumerate(branch_coefficients):
                high_holomorphic = np.polynomial.polynomial.polyval(
                    external_weight,
                    high_polynomials[family_index, branch_index, 0],
                )
                high_antiholomorphic = np.polynomial.polynomial.polyval(
                    external_weight,
                    high_polynomials[family_index, branch_index, 1],
                )
                low_holomorphic = np.polynomial.polynomial.polyval(
                    external_weight,
                    low_polynomials[family_index, branch_index, 0],
                )
                low_antiholomorphic = np.polynomial.polynomial.polyval(
                    external_weight,
                    low_polynomials[family_index, branch_index, 1],
                )
                high_node_value += (
                    branch_coefficient * high_holomorphic * high_antiholomorphic
                )
                low_node_value += (
                    branch_coefficient * low_holomorphic * low_antiholomorphic
                )
            common_weight = spectral_weights[family_index] * primary
            high_values[energy_index, family_index] = common_weight * high_node_value
            low_values[energy_index, family_index] = common_weight * low_node_value
    return high_values, low_values


def evaluate_moduli_response_shard(
    *,
    response_path: Path,
    output_path: Path,
    omega_values: Sequence[complex],
    structure_precision: int,
    include_string_phase: bool,
    tau2_diagnostic_cutoffs: Sequence[float] | None = None,
) -> Path:
    """Evaluate one joint-sample shard and accumulate replicate sums."""

    metadata, arrays = _load_npz(response_path)
    if metadata.get("kind") != "spin23_genus1_moduli_response":
        raise ValueError("input is not a moduli-response artifact")
    energies = np.asarray(tuple(complex(value) for value in omega_values))
    if energies.ndim != 1 or not len(energies) or not np.all(np.isfinite(energies)):
        raise ValueError("omega_values must be a finite nonempty grid")
    replicate_count = int(metadata["replicate_count"])
    maximum_tau2 = float(metadata["tau2_max"])
    cutoffs = (
        (maximum_tau2,)
        if tau2_diagnostic_cutoffs is None
        else tuple(sorted(set(float(value) for value in tau2_diagnostic_cutoffs)))
    )
    if not cutoffs or any(
        not math.isfinite(value) or value <= 1.0 or value > maximum_tau2
        for value in cutoffs
    ):
        raise ValueError("tau2 diagnostic cutoffs must lie in (1,tau2_max]")
    if abs(cutoffs[-1] - maximum_tau2) > 1.0e-12:
        raise ValueError("the largest tau2 diagnostic cutoff must equal tau2_max")
    shape = (len(cutoffs), replicate_count, len(energies), 4)
    high_sums = np.zeros(shape, dtype=np.complex128)
    low_sums = np.zeros(shape, dtype=np.complex128)
    absolute_sums = np.zeros(shape, dtype=np.float64)
    sample_counts = np.zeros(replicate_count, dtype=np.int64)
    accepted_counts = np.zeros((len(cutoffs), replicate_count), dtype=np.int64)
    started = time.perf_counter()
    for local_index, replicate_index in enumerate(arrays["replicate_indices"]):
        high_liouville, low_liouville = _liouville_value_pair_for_sample(
            omega_values=energies,
            momenta=arrays["momenta"][local_index],
            spectral_weights=arrays["spectral_weights"][local_index],
            plumbing=arrays["plumbing_values"][local_index],
            high_polynomials=arrays["response_polynomials"][local_index],
            low_polynomials=arrays["lower_response_polynomials"][local_index],
            structure_precision=structure_precision,
        )
        common = {
            "omega_values": energies,
            "coefficients": arrays["assembly_coefficients"][local_index],
            "exponents": arrays["assembly_exponents"][local_index],
            "include_string_phase": include_string_phase,
        }
        high = assemble_two_vector_values(
            liouville_values=high_liouville,
            **common,
        )
        low = assemble_two_vector_values(
            liouville_values=low_liouville,
            **common,
        )
        weighted_high = arrays["geometry_weights"][local_index] * high
        weighted_low = arrays["geometry_weights"][local_index] * low
        sample_tau2 = float(arrays["tau_values"][local_index].imag)
        for cutoff_index, cutoff in enumerate(cutoffs):
            if sample_tau2 <= cutoff:
                high_sums[cutoff_index, replicate_index] += weighted_high
                low_sums[cutoff_index, replicate_index] += weighted_low
                absolute_sums[cutoff_index, replicate_index] += np.abs(weighted_high)
                accepted_counts[cutoff_index, replicate_index] += 1
        sample_counts[replicate_index] += 1

    output_metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "spin23_genus1_moduli_partial",
        "source_metadata": metadata,
        "structure_precision": structure_precision,
        "include_string_phase": bool(include_string_phase),
        "tau2_diagnostic_cutoffs": list(cutoffs),
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_savez(
        output_path,
        metadata_json=np.asarray(json.dumps(output_metadata, sort_keys=True)),
        sample_indices=arrays["sample_indices"],
        omega_values=energies,
        replicate_high_sums=high_sums,
        replicate_low_sums=low_sums,
        replicate_absolute_sums=absolute_sums,
        replicate_sample_counts=sample_counts,
        replicate_accepted_counts=accepted_counts,
    )
    return output_path


def _replicate_standard_errors(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return real, imaginary, and Euclidean standard errors across replicates."""

    replicate_count = values.shape[0]
    if replicate_count < 2:
        raise ValueError("at least two replicate estimates are required")
    real_error = np.std(values.real, axis=0, ddof=1) / math.sqrt(replicate_count)
    imaginary_error = np.std(values.imag, axis=0, ddof=1) / math.sqrt(replicate_count)
    return real_error, imaginary_error, np.hypot(real_error, imaginary_error)


def reduce_moduli_partials(
    *,
    partial_paths: Sequence[Path],
    output_csv: Path,
    output_npz: Path,
) -> tuple[Path, Path]:
    """Reduce every shard into replicate estimates and an energy table."""

    if not partial_paths:
        raise ValueError("at least one partial artifact is required")
    loaded = tuple(_load_npz(path) for path in partial_paths)
    reference = loaded[0][0]
    source = reference["source_metadata"]
    energies = loaded[0][1]["omega_values"]
    for metadata, arrays in loaded[1:]:
        candidate = metadata["source_metadata"]
        for key in (
            "replicate_count",
            "samples_per_replicate",
            "total_samples",
            "base_seed",
            "tau2_max",
            "flavor",
            "maximum_twice_levels",
            "lower_maximum_twice_levels",
            "condition_limit",
            "task_count",
            "families",
        ):
            if candidate[key] != source[key]:
                raise ValueError(f"partial artifacts disagree on {key}")
        if not np.array_equal(arrays["omega_values"], energies):
            raise ValueError("partial artifacts use different energy grids")
        for key in (
            "structure_precision",
            "include_string_phase",
            "tau2_diagnostic_cutoffs",
        ):
            if metadata[key] != reference[key]:
                raise ValueError(f"partial artifacts disagree on {key}")
    all_indices = np.concatenate([arrays["sample_indices"] for _, arrays in loaded])
    total_samples = int(source["total_samples"])
    if not np.array_equal(np.sort(all_indices), np.arange(total_samples)):
        raise ValueError("partial artifacts do not cover every joint sample exactly once")

    replicate_count = int(source["replicate_count"])
    samples_per_replicate = int(source["samples_per_replicate"])
    cutoffs = tuple(float(value) for value in reference["tau2_diagnostic_cutoffs"])
    shape = (len(cutoffs), replicate_count, len(energies), 4)
    high_sums = np.empty(shape, dtype=np.complex128)
    low_sums = np.empty(shape, dtype=np.complex128)
    absolute_sums = np.empty(shape, dtype=np.float64)
    counts = np.zeros(replicate_count, dtype=np.int64)
    accepted_counts = np.zeros((len(cutoffs), replicate_count), dtype=np.int64)
    for cutoff_index in range(len(cutoffs)):
        for replicate in range(replicate_count):
            for energy_index in range(len(energies)):
                for channel in range(4):
                    high_sums[cutoff_index, replicate, energy_index, channel] = _stable_complex_sum(
                        arrays["replicate_high_sums"][cutoff_index, replicate, energy_index, channel]
                        for _, arrays in loaded
                    )
                    low_sums[cutoff_index, replicate, energy_index, channel] = _stable_complex_sum(
                        arrays["replicate_low_sums"][cutoff_index, replicate, energy_index, channel]
                        for _, arrays in loaded
                    )
                    absolute_sums[cutoff_index, replicate, energy_index, channel] = math.fsum(
                        float(arrays["replicate_absolute_sums"][cutoff_index, replicate, energy_index, channel])
                        for _, arrays in loaded
                    )
            accepted_counts[cutoff_index, replicate] = sum(
                int(arrays["replicate_accepted_counts"][cutoff_index, replicate])
                for _, arrays in loaded
            )
    for replicate in range(replicate_count):
        counts[replicate] = sum(
            int(arrays["replicate_sample_counts"][replicate]) for _, arrays in loaded
        )
    if not np.array_equal(counts, np.full(replicate_count, samples_per_replicate)):
        raise ValueError("replicate sample counts are incomplete")

    replicate_high = high_sums / samples_per_replicate
    replicate_low = low_sums / samples_per_replicate
    replicate_absolute = absolute_sums / samples_per_replicate
    mean_high = np.mean(replicate_high, axis=1)
    mean_low = np.mean(replicate_low, axis=1)
    mean_absolute = np.mean(replicate_absolute, axis=1)
    error_data = tuple(
        _replicate_standard_errors(replicate_high[cutoff_index])
        for cutoff_index in range(len(cutoffs))
    )
    error_re = np.stack(tuple(value[0] for value in error_data))
    error_im = np.stack(tuple(value[1] for value in error_data))
    error_abs = np.stack(tuple(value[2] for value in error_data))
    level_difference = np.abs(mean_high - mean_low)
    level_error_data = tuple(
        _replicate_standard_errors(
            replicate_high[cutoff_index] - replicate_low[cutoff_index]
        )
        for cutoff_index in range(len(cutoffs))
    )
    level_difference_error_abs = np.stack(
        tuple(value[2] for value in level_error_data)
    )
    cancellation_factor = mean_absolute / np.maximum(np.abs(mean_high), np.finfo(float).tiny)
    if len(cutoffs) > 1:
        cusp_increment_error_abs = _replicate_standard_errors(
            replicate_high[-1] - replicate_high[-2]
        )[2]
    else:
        cusp_increment_error_abs = np.zeros_like(error_abs[-1])

    audit_metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "spin23_genus1_moduli_result",
        "source_metadata": source,
        "structure_precision": reference["structure_precision"],
        "include_string_phase": reference["include_string_phase"],
        "error_definition": "independent_scramble_standard_error",
        "level_diagnostic": "absolute_difference_of_replicate_means",
        "tau2_diagnostic_cutoffs": list(cutoffs),
    }
    _atomic_savez(
        output_npz,
        metadata_json=np.asarray(json.dumps(audit_metadata, sort_keys=True)),
        omega_values=energies,
        replicate_high=replicate_high,
        replicate_low=replicate_low,
        replicate_absolute=replicate_absolute,
        mean_high=mean_high,
        mean_low=mean_low,
        rqmc_error_re=error_re,
        rqmc_error_im=error_im,
        rqmc_error_abs=error_abs,
        level_difference_abs=level_difference,
        level_difference_rqmc_error_abs=level_difference_error_abs,
        cancellation_factor=cancellation_factor,
        tau2_diagnostic_cutoffs=np.asarray(cutoffs),
        replicate_accepted_counts=accepted_counts,
        cusp_increment_rqmc_error_abs=cusp_increment_error_abs,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "omega_re",
            "omega_im",
            "value_re",
            "value_im",
            "rqmc_error_re",
            "rqmc_error_im",
            "rqmc_error_abs",
            "rqmc_relative_error",
            "level3_value_re",
            "level3_value_im",
            "level_difference_abs",
            "level_difference_relative",
            "level_difference_rqmc_error_abs",
            "absolute_sample_scale",
            "cancellation_factor",
            "NS_re",
            "NS_im",
            "NS_tilde_re",
            "NS_tilde_im",
            "R_re",
            "R_im",
            "tau2_max",
            "replicate_count",
            "samples_per_replicate",
            "total_samples",
            "previous_tau2_max",
            "previous_value_re",
            "previous_value_im",
            "cusp_increment_abs",
            "cusp_increment_relative",
            "cusp_increment_rqmc_error_abs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        final_cutoff_index = len(cutoffs) - 1
        previous_cutoff_index = max(0, final_cutoff_index - 1)
        for energy_index, omega in enumerate(energies):
            value = mean_high[final_cutoff_index, energy_index, 0]
            lower = mean_low[final_cutoff_index, energy_index, 0]
            previous_value = mean_high[previous_cutoff_index, energy_index, 0]
            cusp_increment = abs(value - previous_value)
            scale = max(abs(value), np.finfo(float).tiny)
            row = {
                "omega_re": float(complex(omega).real),
                "omega_im": float(complex(omega).imag),
                "value_re": float(value.real),
                "value_im": float(value.imag),
                "rqmc_error_re": float(error_re[final_cutoff_index, energy_index, 0]),
                "rqmc_error_im": float(error_im[final_cutoff_index, energy_index, 0]),
                "rqmc_error_abs": float(error_abs[final_cutoff_index, energy_index, 0]),
                "rqmc_relative_error": float(error_abs[final_cutoff_index, energy_index, 0] / scale),
                "level3_value_re": float(lower.real),
                "level3_value_im": float(lower.imag),
                "level_difference_abs": float(level_difference[final_cutoff_index, energy_index, 0]),
                "level_difference_relative": float(level_difference[final_cutoff_index, energy_index, 0] / scale),
                "level_difference_rqmc_error_abs": float(
                    level_difference_error_abs[final_cutoff_index, energy_index, 0]
                ),
                "absolute_sample_scale": float(mean_absolute[final_cutoff_index, energy_index, 0]),
                "cancellation_factor": float(cancellation_factor[final_cutoff_index, energy_index, 0]),
                "tau2_max": float(source["tau2_max"]),
                "replicate_count": replicate_count,
                "samples_per_replicate": samples_per_replicate,
                "total_samples": total_samples,
                "previous_tau2_max": cutoffs[previous_cutoff_index],
                "previous_value_re": float(previous_value.real),
                "previous_value_im": float(previous_value.imag),
                "cusp_increment_abs": float(cusp_increment),
                "cusp_increment_relative": float(cusp_increment / scale),
                "cusp_increment_rqmc_error_abs": float(
                    cusp_increment_error_abs[energy_index, 0]
                ),
            }
            for spin_index, spin_label in enumerate(_SPIN_LABELS, start=1):
                spin_value = mean_high[final_cutoff_index, energy_index, spin_index]
                row[f"{spin_label}_re"] = float(spin_value.real)
                row[f"{spin_label}_im"] = float(spin_value.imag)
            writer.writerow(row)
    return output_csv, output_npz


def _omega_grid(count: int, minimum: float, maximum: float, imaginary: float) -> np.ndarray:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("omega_count must be a positive integer")
    if not all(math.isfinite(value) for value in (minimum, maximum, imaginary)):
        raise ValueError("omega-grid controls must be finite")
    if maximum < minimum:
        raise ValueError("omega_max must not be below omega_min")
    return np.linspace(minimum, maximum, count) + 1j * imaginary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--replicate-count", type=int, default=10)
    prepare.add_argument("--samples-per-replicate", type=int, default=1024)
    prepare.add_argument("--base-seed", type=int, default=230701)
    prepare.add_argument("--tau2-max", type=float, default=8.0)
    prepare.add_argument("--flavor", type=int, default=4)
    prepare.add_argument("--maximum-twice-level", type=int, default=8)
    prepare.add_argument("--lower-maximum-twice-level", type=int, default=6)
    prepare.add_argument("--condition-limit", type=float, default=1.0e11)
    prepare.add_argument("--free-field-precision", type=int, default=32)
    prepare.add_argument("--task-index", type=int, required=True)
    prepare.add_argument("--task-count", type=int, required=True)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--response", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--omega-count", type=int, default=1000)
    evaluate.add_argument("--omega-min", type=float, default=0.05)
    evaluate.add_argument("--omega-max", type=float, default=1.0)
    evaluate.add_argument("--omega-imag", type=float, default=0.0)
    evaluate.add_argument("--structure-precision", type=int, default=24)
    evaluate.add_argument(
        "--tau2-diagnostic-cutoff",
        type=float,
        action="append",
        default=None,
    )
    evaluate.add_argument("--include-string-phase", action="store_true")

    reduce = commands.add_parser("reduce")
    reduce.add_argument("--partials", type=Path, nargs="+", required=True)
    reduce.add_argument("--output-csv", type=Path, required=True)
    reduce.add_argument("--output-npz", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_moduli_response_shard(
            output_path=args.output,
            replicate_count=args.replicate_count,
            samples_per_replicate=args.samples_per_replicate,
            base_seed=args.base_seed,
            tau2_max=args.tau2_max,
            flavor=args.flavor,
            maximum_twice_levels=args.maximum_twice_level,
            lower_maximum_twice_levels=args.lower_maximum_twice_level,
            condition_limit=args.condition_limit,
            free_field_precision=args.free_field_precision,
            task_index=args.task_index,
            task_count=args.task_count,
        )
        return 0
    if args.command == "evaluate":
        evaluate_moduli_response_shard(
            response_path=args.response,
            output_path=args.output,
            omega_values=_omega_grid(
                args.omega_count, args.omega_min, args.omega_max, args.omega_imag
            ),
            structure_precision=args.structure_precision,
            include_string_phase=args.include_string_phase,
            tau2_diagnostic_cutoffs=args.tau2_diagnostic_cutoff,
        )
        return 0
    reduce_moduli_partials(
        partial_paths=args.partials,
        output_csv=args.output_csv,
        output_npz=args.output_npz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
