#!/usr/bin/env python3
"""Checks for the complete ``Mbar_{0,5}`` subtraction forest."""

from __future__ import annotations

import cmath
import math

import numpy as np

try:
    from sphere_five_point_subtraction import (
        canonical_corner_ordering,
        canonical_divisor_ordering,
        divergent_momentum_endpoint,
        divergent_spin_zero_levels,
        equal_outgoing_signed_energies,
        five_point_fixed_momenta_forest,
        five_point_fixed_momenta_corner_finite_part,
        five_point_fixed_momenta_face_finite_part,
        five_point_boundary_corners,
        five_point_boundary_divisors,
        five_point_face_sector_orderings,
        five_point_plumbing_channel_energies,
        five_point_plumbing_radial_exponents,
        five_point_regular_factor_coefficients,
        forest_finite_part_terms,
        radial_finite_part,
        signed_channel_energy,
        visible_boundary_chart,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.sphere_five_point_subtraction import (
        canonical_corner_ordering,
        canonical_divisor_ordering,
        divergent_momentum_endpoint,
        divergent_spin_zero_levels,
        equal_outgoing_signed_energies,
        five_point_fixed_momenta_forest,
        five_point_fixed_momenta_corner_finite_part,
        five_point_fixed_momenta_face_finite_part,
        five_point_boundary_corners,
        five_point_boundary_divisors,
        five_point_face_sector_orderings,
        five_point_plumbing_channel_energies,
        five_point_plumbing_radial_exponents,
        five_point_regular_factor_coefficients,
        forest_finite_part_terms,
        radial_finite_part,
        signed_channel_energy,
        visible_boundary_chart,
    )


def check_boundary_combinatorics() -> None:
    divisors = five_point_boundary_divisors()
    corners = five_point_boundary_corners()
    print("Mbar_0,5 boundary combinatorics")
    print(f"  divisors={len(divisors)}, corners={len(corners)}")
    if len(divisors) != 10 or len(corners) != 15:
        raise AssertionError("the five-point boundary forest is incomplete")
    if any(not left.is_compatible_with(right) for left, right in (item.divisors for item in corners)):
        raise AssertionError("an incompatible divisor pair was included as a corner")
    divisor_orderings = tuple(canonical_divisor_ordering(value) for value in divisors)
    corner_orderings = tuple(canonical_corner_ordering(value) for value in corners)
    face_sectors = five_point_face_sector_orderings()
    if len(set(divisor_orderings)) != 10 or len(set(corner_orderings)) != 15:
        raise AssertionError("canonical face or corner frames are not unique")
    if len(face_sectors) != 60:
        raise AssertionError("the ten faces do not each contain six crossing cells")


def check_visible_loci_are_complete() -> None:
    chart = visible_boundary_chart()
    visible = set(chart.by_locus.values())
    complete = set(five_point_boundary_divisors())
    print("\nblown-up (z1,z2) chart")
    print(f"  named loci={len(chart.by_locus)}, distinct divisors={len(visible)}")
    if visible != complete:
        raise AssertionError("the seven visible and three exceptional loci miss a divisor")


def check_equal_energy_thresholds() -> None:
    omega = 0.9 + 0.01j
    signed = equal_outgoing_signed_energies(omega)
    divisors = five_point_boundary_divisors()
    energies = [signed_channel_energy(divisor, signed) for divisor in divisors]
    incoming_type = sum(abs(value - 3.0 * omega) < 1.0e-14 for value in energies)
    outgoing_type = sum(abs(value + 2.0 * omega) < 1.0e-14 for value in energies)
    print("\nequal-outgoing-energy channels")
    print(f"  incoming+outgoing divisors={incoming_type}")
    print(f"  outgoing+outgoing divisors={outgoing_type}")
    if incoming_type != 4 or outgoing_type != 6:
        raise AssertionError("equal-energy channel thresholds have the wrong multiplicities")


def check_divergent_level_rule() -> None:
    channel_energy = 3.7 + 0.02j
    levels = divergent_spin_zero_levels(channel_energy)
    endpoints = tuple(divergent_momentum_endpoint(channel_energy, level) for level in levels)
    print("\nspin-zero OPE projector")
    print(f"  levels={levels}, endpoints={endpoints}")
    if levels != (0, 1, 2, 3):
        raise AssertionError("power-divergent OPE levels were selected incorrectly")
    if any(endpoint <= 0.0 for endpoint in endpoints):
        raise AssertionError("a selected divergence has an empty momentum interval")


def check_radial_analytic_continuation() -> None:
    alpha = 0.37 + 0.11j
    rho = 0.23
    observed = radial_finite_part(alpha, rho)
    target = math.pi * rho ** (2.0 * alpha) / alpha
    log_value = radial_finite_part(0.0, rho)
    print("\nradial finite part")
    print(f"  meromorphic formula error={abs(observed-target):.3e}")
    print(f"  logarithmic finite term={log_value.real:.12e}")
    if abs(observed - target) > 1.0e-14:
        raise AssertionError("the radial analytic continuation is incorrect")
    if abs(log_value - 2.0 * math.pi * math.log(rho)) > 1.0e-14:
        raise AssertionError("the logarithmic finite part is incorrect")


def check_forest_algebra() -> None:
    terms = forest_finite_part_terms(17.0, 5.0, 7.0, 2.0)
    reconstructed = (
        terms.bulk_remainder
        + terms.face1_remainder
        + terms.face2_remainder
        + terms.corner_coefficient
    )
    print("\ntwo-boundary forest algebra")
    print(f"  bulk={terms.bulk_remainder}, reconstructed={reconstructed}")
    if reconstructed != 17.0:
        raise AssertionError("forest sectors do not reconstruct the integrand")
    if terms.bulk_remainder != 7.0 or terms.face1_remainder != 3.0 or terms.face2_remainder != 5.0:
        raise AssertionError("forest inclusion-exclusion signs are wrong")


def check_regular_timelike_bivariate_series() -> None:
    energies = (
        2.4 + 0.04j,
        -0.6 - 0.01j,
        -0.6 - 0.01j,
        -0.6 - 0.01j,
        -0.6 - 0.01j,
    )
    block = {
        (0, 0): 1.0 + 0.0j,
        (1, 0): 0.21 + 0.02j,
        (0, 1): -0.13 + 0.01j,
        (1, 1): 0.037 - 0.006j,
        (2, 0): 0.012,
        (0, 2): -0.008,
    }
    regular = five_point_regular_factor_coefficients(
        block,
        energies,
        order1=4,
        order2=4,
    )
    q1 = 0.008 + 0.006j
    q2 = -0.009 + 0.007j
    block_value = sum(
        value * q1**level1 * q2**level2
        for (level1, level2), value in block.items()
    )
    _, k_b, k_c, k_d, _ = energies
    direct = (
        (1.0 - q1) ** (-0.5 * k_b * k_c)
        * (1.0 - q1 * q2) ** (-0.5 * k_b * k_d)
        * (1.0 - q2) ** (-0.5 * k_c * k_d)
        * block_value
    )
    truncated = sum(
        value * q1**level1 * q2**level2
        for (level1, level2), value in regular.items()
    )
    error = abs(direct - truncated)
    print("\nfive-point regular timelike series")
    print(f"  truncation error={error:.3e}")
    if error > 2.0e-10:
        raise AssertionError("the bivariate timelike convolution is incorrect")


def check_plumbing_exponents_include_jacobian() -> None:
    energies = (
        2.0 + 0.05j,
        -0.4 - 0.01j,
        -0.5 - 0.01j,
        -0.6 - 0.015j,
        -0.5 - 0.015j,
    )
    momentum1 = 0.37
    momentum2 = 0.52
    observed1, observed2 = five_point_plumbing_radial_exponents(
        energies,
        momentum1,
        momentum2,
    )
    weights = tuple(1.0 + 0.25 * energy**2 for energy in energies)
    h1 = 1.0 + momentum1**2
    h2 = 1.0 + momentum2**2
    k_a, k_b, k_c, _, _ = energies
    liouville_x = 2.0 * (h1 - weights[0] - weights[1])
    liouville_y = 2.0 * (h2 - weights[2] - h1)
    expected1 = liouville_x - k_a * k_b
    expected2 = (
        liouville_x
        + liouville_y
        - k_a * k_b
        - k_a * k_c
        - k_b * k_c
        + 2.0
    )
    error = max(abs(observed1 - expected1), abs(observed2 - expected2))
    channels = five_point_plumbing_channel_energies(energies)
    print("\nplumbing radial exponents")
    print(f"  channels={channels}, identity error={error:.3e}")
    if error > 2.0e-15:
        raise AssertionError("the q2 moduli Jacobian is missing or incorrect")


def check_explicit_two_edge_power_cancellation() -> None:
    energies = (1.6 + 0.04j,) + 4 * (-0.4 - 0.01j,)
    block = {
        (0, 0): 1.0 + 0.0j,
        (1, 0): 0.24 - 0.02j,
        (0, 1): -0.17 + 0.03j,
        (1, 1): 0.051 - 0.007j,
        (2, 0): 0.018,
        (0, 2): -0.011,
        (2, 1): 0.004,
        (1, 2): -0.003,
        (2, 2): 0.001,
    }
    regular = five_point_regular_factor_coefficients(
        block,
        energies,
        order1=2,
        order2=2,
    )
    angles = 2.0 * math.pi * np.arange(48) / 48
    radial_values = (0.02, 0.01, 0.005)
    radial_remainders = []
    for radius in radial_values:
        values = []
        for angle1 in angles:
            q1 = radius * cmath.exp(1.0j * angle1)
            for angle2 in angles:
                q2 = radius * cmath.exp(1.0j * angle2)
                values.append(
                    five_point_fixed_momenta_forest(
                        q1,
                        q2,
                        ordered_signed_energies=energies,
                        momentum1=0.10,
                        momentum2=0.15,
                        regular_coefficients=regular,
                    ).remainder
                )
        # Include the two polar area factors.
        radial_remainders.append(radius**2 * abs(np.mean(values)))
    ratios = (
        radial_remainders[1] / radial_remainders[0],
        radial_remainders[2] / radial_remainders[1],
    )
    sample = five_point_fixed_momenta_forest(
        0.013 + 0.009j,
        -0.011 + 0.007j,
        ordered_signed_energies=energies,
        momentum1=0.10,
        momentum2=0.15,
        regular_coefficients=regular,
    )
    reconstruction_error = abs(
        sample.remainder
        - (sample.original - sample.face1 - sample.face2 + sample.corner)
    )
    print("\nexplicit five-point BRY forest")
    print(f"  selected levels={sample.levels1}, {sample.levels2}")
    print(f"  successive radial ratios={ratios[0]:.6f}, {ratios[1]:.6f}")
    print(f"  reconstruction error={reconstruction_error:.3e}")
    if sample.levels1 != (0,) or sample.levels2 != (0,):
        raise AssertionError("the physical face projectors selected wrong levels")
    if max(ratios) > 0.7:
        raise AssertionError("the two-edge forest did not remove the corner powers")
    if reconstruction_error > 1.0e-13:
        raise AssertionError("the explicit forest signs are inconsistent")


def check_face_and_corner_finite_parts() -> None:
    energies = (1.2 + 0.06j, -0.3 - 0.015j, -0.3 - 0.015j, -0.3 - 0.015j, -0.3 - 0.015j)
    coefficients = {
        (0, 0): 1.0 + 0.0j,
        (1, 0): 0.23 - 0.04j,
        (0, 1): -0.17 + 0.02j,
        (1, 1): 0.031 + 0.006j,
    }
    momentum1 = 0.71
    momentum2 = 0.64
    rho1 = 0.13
    rho2 = 0.09
    q2 = 0.21 + 0.08j
    face = five_point_fixed_momenta_face_finite_part(
        q2,
        ordered_signed_energies=energies,
        momentum1=momentum1,
        momentum2=momentum2,
        regular_coefficients=coefficients,
        collar_radius=rho1,
        edge=1,
    )
    channel1, channel2 = five_point_plumbing_channel_energies(energies)
    exponent2 = -2.0 - 0.5 * channel2**2 + 2.0 * momentum2**2
    manual_face = 0.0 + 0.0j
    for level1 in (0, 1):
        row = sum(
            value * q2**level2
            for (first, level2), value in coefficients.items()
            if first == level1
        )
        row_bar = sum(
            value * q2.conjugate() ** level2
            for (first, level2), value in coefficients.items()
            if first == level1
        )
        alpha1 = momentum1**2 + level1 - 0.25 * channel1**2
        manual_face += (
            radial_finite_part(alpha1, rho1)
            * cmath.exp(exponent2 * math.log(abs(q2)))
            * row
            * row_bar
        )
    corner = five_point_fixed_momenta_corner_finite_part(
        ordered_signed_energies=energies,
        momentum1=momentum1,
        momentum2=momentum2,
        regular_coefficients=coefficients,
        collar_radius1=rho1,
        collar_radius2=rho2,
    )
    manual_corner = sum(
        value**2
        * radial_finite_part(momentum1**2 + level1 - 0.25 * channel1**2, rho1)
        * radial_finite_part(momentum2**2 + level2 - 0.25 * channel2**2, rho2)
        for (level1, level2), value in coefficients.items()
    )
    face_error = abs(face - manual_face)
    corner_error = abs(corner - manual_corner)
    print("\nface and corner analytic finite parts")
    print(f"  face error={face_error:.3e}, corner error={corner_error:.3e}")
    if max(face_error, corner_error) > 2.0e-13:
        raise AssertionError("the stratified finite-part primitives are inconsistent")


def run() -> None:
    check_boundary_combinatorics()
    check_visible_loci_are_complete()
    check_equal_energy_thresholds()
    check_divergent_level_rule()
    check_radial_analytic_continuation()
    check_forest_algebra()
    check_regular_timelike_bivariate_series()
    check_plumbing_exponents_include_jacobian()
    check_explicit_two_edge_power_cancellation()
    check_face_and_corner_finite_parts()
    print("\nall sphere five-point subtraction checks passed")


if __name__ == "__main__":
    run()
