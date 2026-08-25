#!/usr/bin/env python3
"""Checks for the executable BRY sphere-four subtraction baby step."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad

try:
    from sphere_four_point_subtraction import (
        bry_analytic_disk_integral,
        bry_collar_finite_part,
        bry_divergent_levels,
        bry_regular_chiral_coefficients,
        bry_subtracted_fixed_momentum_density,
        equal_one_to_three_frequencies,
        generalized_binomial_series,
    )
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.sphere_four_point_subtraction import (
        bry_analytic_disk_integral,
        bry_collar_finite_part,
        bry_divergent_levels,
        bry_regular_chiral_coefficients,
        bry_subtracted_fixed_momentum_density,
        equal_one_to_three_frequencies,
        generalized_binomial_series,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_bry_equations_39_310() -> None:
    radial_power = -0.73 + 0.19j

    def integrand(radius: float) -> complex:
        return -2.0 * math.pi * radius ** (radial_power - 1.0)

    real = quad(lambda radius: integrand(radius).real, 1.0, np.inf, epsabs=1e-11)[0]
    imag = quad(lambda radius: integrand(radius).imag, 1.0, np.inf, epsabs=1e-11)[0]
    numerical_global_subtraction = complex(real, imag)
    analytic = bry_analytic_disk_integral(radial_power)
    error = abs(numerical_global_subtraction - analytic)
    print("BRY equations (3.9)--(3.10)")
    print(f"  global subtraction error={error:.3e}")
    require(error < 2.0e-10, "the global BRY subtraction is incorrect")


def check_collar_equivalence() -> None:
    radial_power = -1.17 + 0.08j
    target = bry_analytic_disk_integral(radial_power)
    values = tuple(
        bry_collar_finite_part(radial_power, radius)
        for radius in (0.41, 0.17, 0.053)
    )
    spread = max(abs(value - target) for value in values)
    print("\nglobal subtraction versus collar finite part")
    print(f"  maximum radius dependence={spread:.3e}")
    require(spread < 1.0e-13, "the collar and global prescriptions disagree")


def check_spin_zero_level_rule() -> None:
    channel_energy = 3.7 + 0.02j
    levels = bry_divergent_levels(channel_energy, 0.61, 8)
    print("\nBRY equation (3.12)")
    print(f"  selected levels={levels}")
    require(levels == (0, 1, 2, 3), "wrong spin-zero power projector")


def check_timelike_ope_convolution() -> None:
    exponent = 0.37 - 0.11j
    order = 5
    block = (1.0, 0.21 + 0.03j, -0.07, 0.014j, 0.003, -0.0008)
    regular = bry_regular_chiral_coefficients(block, exponent, order)
    z = 0.013 - 0.017j
    direct = (1.0 - z) ** exponent * sum(
        coefficient * z**level for level, coefficient in enumerate(block)
    )
    truncated = sum(
        coefficient * z**level for level, coefficient in enumerate(regular)
    )
    error = abs(direct - truncated)
    binomial = generalized_binomial_series(exponent, order)
    print("\nregular OPE coefficient convolution")
    print(f"  first binomial coefficient={binomial[1]!r}")
    print(f"  truncation error={error:.3e}")
    require(error < 2.0e-10, "the timelike OPE convolution is wrong")


def check_local_power_cancellation() -> None:
    channel_energy = 1.2 + 0.04j
    momentum = 0.37
    coefficients = (1.0, 0.23 - 0.02j, -0.04 + 0.01j)
    radii = (2.0e-2, 1.0e-2, 5.0e-3)
    angles = 2.0 * math.pi * np.arange(2048) / 2048
    scaled = []
    for radius in radii:
        angular_average = np.mean(
            [
                bry_subtracted_fixed_momentum_density(
                    radius * np.exp(1.0j * angle),
                    channel_energy=channel_energy,
                    momentum=momentum,
                    regular_chiral_coefficients=coefficients,
                )
                for angle in angles
            ]
        )
        # Include the polar area factor r.  The subtracted radial density must
        # now be locally integrable; nonzero-spin pointwise terms cancel only
        # after the angular average.
        scaled.append(radius * abs(angular_average))
    ratio1 = scaled[1] / scaled[0]
    ratio2 = scaled[2] / scaled[1]
    print("\nlocal BRY power cancellation")
    print(f"  successive remainder ratios={ratio1:.6f}, {ratio2:.6f}")
    require(
        ratio1 < 0.75 and ratio2 < 0.75,
        "subtracting the spin-zero power term did not improve the OPE behavior",
    )


def check_equal_iepsilon_assignment() -> None:
    incoming, *outgoing = equal_one_to_three_frequencies(0.7, 0.03)
    conservation = incoming - sum(outgoing)
    differences = tuple(incoming - value for value in outgoing)
    print("\nenergy-conserving i-epsilon assignment")
    print(f"  conservation error={abs(conservation):.3e}")
    print(f"  Im(in-out)={differences[0].imag:.6e}")
    require(abs(conservation) < 1.0e-15, "complex energy conservation failed")
    require(
        all(value.imag > 0.0 for value in differences),
        "the BRY 1-to-3 analyticity domain is not satisfied",
    )


def main() -> None:
    check_bry_equations_39_310()
    check_collar_equivalence()
    check_spin_zero_level_rule()
    check_timelike_ope_convolution()
    check_local_power_cancellation()
    check_equal_iepsilon_assignment()
    print("\nall sphere-four BRY subtraction checks passed")


if __name__ == "__main__":
    main()
