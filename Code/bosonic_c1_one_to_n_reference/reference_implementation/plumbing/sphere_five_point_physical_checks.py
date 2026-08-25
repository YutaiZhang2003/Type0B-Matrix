#!/usr/bin/env python3
"""Checks for the pointwise physical i-epsilon sphere-five integrand."""

from __future__ import annotations

import cmath
import math

import numpy as np

try:
    from sphere_five_point_equal_energy import EqualEnergyFivePointKernel
    from sphere_five_point_subtraction import (
        five_point_fixed_momenta_corner_finite_part,
        five_point_fixed_momenta_face_finite_part,
        five_point_fixed_momenta_forest,
        five_point_plumbing_radial_exponents,
        five_point_regular_factor_coefficients,
    )
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.sphere_five_point_equal_energy import EqualEnergyFivePointKernel
    from plumbing.sphere_five_point_subtraction import (
        five_point_fixed_momenta_corner_finite_part,
        five_point_fixed_momenta_face_finite_part,
        five_point_fixed_momenta_forest,
        five_point_plumbing_radial_exponents,
        five_point_regular_factor_coefficients,
    )


def relative_error(value: complex, target: complex) -> float:
    return abs(value - target) / max(abs(target), 1.0e-300)


def check_empty_forest_is_raw_integrand() -> None:
    kernel = EqualEnergyFivePointKernel(
        0.12j,
        block_order=2,
        momentum_order=3,
        momentum_maximum=4.0,
        block_scheme="c",
        special_dps=25,
    )
    q1 = 0.19 + 0.07j
    q2 = -0.23 + 0.11j
    ordering = (0, 1, 2, 3, 4)
    raw = kernel.integrand_linear_gauge(q1, q2, ordering)
    forest = kernel.forest_subtracted_integrand_linear_gauge_weighted(
        q1,
        q2,
        ordering,
    )
    error = relative_error(forest, raw)
    print("empty five-point forest")
    print(f"  raw/forest relative error={error:.3e}")
    if error > 2.0e-12:
        raise AssertionError("the forest changed a convergent integrand")


def check_vectorized_forest_against_scalar_definition() -> None:
    kernel = EqualEnergyFivePointKernel(
        0.35 + 0.01j,
        block_order=2,
        momentum_order=2,
        momentum_maximum=3.0,
        block_scheme="c",
        special_dps=25,
    )
    incoming_slot = 0
    q1 = 0.16 + 0.09j
    q2 = -0.21 + 0.08j
    ordered_signed = (
        kernel.signed_energies[0],
        kernel.signed_energies[1],
        kernel.signed_energies[1],
        kernel.signed_energies[1],
        kernel.signed_energies[1],
    )
    scalar = 0.0 + 0.0j
    scalar_face = 0.0 + 0.0j
    scalar_corner = 0.0 + 0.0j
    collar_radius = 0.11
    for entry in kernel.entries_by_incoming_slot[incoming_slot]:
        regular = five_point_regular_factor_coefficients(
            entry.coefficients,
            ordered_signed,
            order1=kernel.block_order,
            order2=kernel.block_order,
        )
        forest = five_point_fixed_momenta_forest(
            q1,
            q2,
            ordered_signed_energies=ordered_signed,
            momentum1=entry.p1,
            momentum2=entry.p2,
            regular_coefficients=regular,
        )
        exponent1, exponent2 = five_point_plumbing_radial_exponents(
            ordered_signed,
            entry.p1,
            entry.p2,
        )
        primary = cmath.exp(
            exponent1 * math.log(abs(q1)) + exponent2 * math.log(abs(q2))
        )
        block_holomorphic = sum(
            value * q1**level1 * q2**level2
            for (level1, level2), value in entry.coefficients.items()
        )
        block_antiholomorphic = sum(
            value * q1.conjugate() ** level1 * q2.conjugate() ** level2
            for (level1, level2), value in entry.coefficients.items()
        )
        _, k_b, k_c, k_d, _ = ordered_signed
        timelike_holomorphic = (
            (1.0 - q1) ** (-0.5 * k_b * k_c)
            * (1.0 - q1 * q2) ** (-0.5 * k_b * k_d)
            * (1.0 - q2) ** (-0.5 * k_c * k_d)
        )
        timelike_antiholomorphic = (
            (1.0 - q1.conjugate()) ** (-0.5 * k_b * k_c)
            * (1.0 - q1.conjugate() * q2.conjugate()) ** (-0.5 * k_b * k_d)
            * (1.0 - q2.conjugate()) ** (-0.5 * k_c * k_d)
        )
        exact_original = (
            primary
            * block_holomorphic
            * block_antiholomorphic
            * timelike_holomorphic
            * timelike_antiholomorphic
        )
        exact_remainder = exact_original - forest.face1 - forest.face2 + forest.corner
        scalar += entry.weighted_structure_constant * exact_remainder
        scalar_face += entry.weighted_structure_constant * five_point_fixed_momenta_face_finite_part(
            q2,
            ordered_signed_energies=ordered_signed,
            momentum1=entry.p1,
            momentum2=entry.p2,
            regular_coefficients=regular,
            collar_radius=collar_radius,
            edge=1,
        )
        scalar_corner += entry.weighted_structure_constant * five_point_fixed_momenta_corner_finite_part(
            ordered_signed_energies=ordered_signed,
            momentum1=entry.p1,
            momentum2=entry.p2,
            regular_coefficients=regular,
            collar_radius1=collar_radius,
            collar_radius2=collar_radius,
        )
    vectorized = kernel._forest_momentum_sum(incoming_slot, q1, q2)
    error = relative_error(vectorized, scalar)
    print("\nvectorized physical forest")
    print(f"  scalar/vectorized relative error={error:.3e}")
    if error > 3.0e-12:
        raise AssertionError("the vectorized forest differs from its definition")
    vectorized_face = kernel._face_finite_part_momentum_sum(
        incoming_slot,
        q2,
        collar_radius,
    )
    vectorized_corner = kernel._corner_finite_part_momentum_sum(
        incoming_slot,
        collar_radius,
        collar_radius,
    )
    face_error = relative_error(vectorized_face, scalar_face)
    corner_error = relative_error(vectorized_corner, scalar_corner)
    print(f"  face finite-part relative error={face_error:.3e}")
    print(f"  corner finite-part relative error={corner_error:.3e}")
    if max(face_error, corner_error) > 3.0e-12:
        raise AssertionError("the vectorized finite-part strata differ from their definitions")


def check_endpoint_split_and_iepsilon_sign() -> None:
    outgoing = 0.35 + 0.01j
    kernel = EqualEnergyFivePointKernel(
        outgoing,
        block_order=2,
        momentum_order=3,
        momentum_maximum=3.0,
        block_scheme="c",
        special_dps=25,
    )
    endpoints = (
        math.sqrt(0.25 * ((3.0 * outgoing) ** 2).real),
        math.sqrt(0.25 * ((-2.0 * outgoing) ** 2).real),
    )
    arrays = kernel.arrays_by_incoming_slot[0]
    straddled = []
    for endpoint in endpoints:
        distances_below = endpoint - arrays.p1_values[arrays.p1_values < endpoint]
        distances_above = arrays.p1_values[arrays.p1_values > endpoint] - endpoint
        straddled.append(distances_below.size > 0 and distances_above.size > 0)
    alpha_imaginary_parts = tuple(
        (-0.25 * channel**2).imag
        for channel in (3.0 * outgoing, -2.0 * outgoing)
    )
    print("\nphysical endpoints and i-epsilon")
    print(f"  endpoints={endpoints}, straddled={straddled}")
    print(f"  Im(alpha)={alpha_imaginary_parts}")
    if not all(straddled):
        raise AssertionError("momentum quadrature does not split an OPE endpoint")
    if not all(value < 0.0 for value in alpha_imaginary_parts):
        raise AssertionError("the propagator does not have the BRY i-epsilon sign")


def main() -> None:
    check_empty_forest_is_raw_integrand()
    check_vectorized_forest_against_scalar_definition()
    check_endpoint_split_and_iepsilon_sign()
    print("\nall physical sphere-five subtraction checks passed")


if __name__ == "__main__":
    main()
