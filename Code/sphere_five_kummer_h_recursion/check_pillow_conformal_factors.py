#!/usr/bin/env python3
"""Check the two equivalent normalizations of sphere-n pillow blocks.

The E.103-style normalization uses central charge c and keeps the
large-weight pillow character explicit. The standard Zamolodchikov
normalization shifts c to c-1 in the universal prefactor and absorbs that
character. This script verifies the theta-function identity relating the two
conventions and checks complete prefactors with zero, one, and three mobile
insertions at generic complex moduli.
"""

from __future__ import annotations

import cmath


def _theta_constants(q: complex, order: int = 80) -> tuple[complex, complex]:
    theta2 = sum(q ** ((n + 0.5) ** 2) for n in range(-order, order + 1))
    theta3 = sum(q ** (n * n) for n in range(-order, order + 1))
    return theta2, theta3


def _pillow_character(q: complex, order: int = 200) -> complex:
    value = 1.0 + 0.0j
    for n in range(1, order + 1):
        value *= (1.0 - q ** (2 * n)) ** -0.5
    return value


def _power(base: complex, exponent: complex) -> complex:
    return cmath.exp(complex(exponent) * cmath.log(complex(base)))


def _lambda_n(
    *,
    kappa: complex,
    theta3: complex,
    z: complex,
    mobile_positions: tuple[complex, ...],
    corner_weights: tuple[complex, complex, complex, complex],
    mobile_weights: tuple[complex, ...],
) -> complex:
    if len(mobile_positions) != len(mobile_weights):
        raise ValueError("mobile positions and weights must have equal length")
    d1, d2, d3, d4 = corner_weights
    value = (
        _power(
            theta3,
            kappa / 2.0
            - 4.0 * sum(corner_weights)
            - 2.0 * sum(mobile_weights),
        )
        * _power(z, kappa / 24.0 - d1 - d2)
        * _power(1.0 - z, kappa / 24.0 - d2 - d3)
    )
    for t, mu in zip(mobile_positions, mobile_weights):
        value *= _power(t * (1.0 - t) * (z - t), -mu / 2.0)
    return value


def check_case(
    tau: complex,
    mobile_fractions: tuple[complex, ...],
) -> tuple[float, float]:
    q = cmath.exp(cmath.pi * 1.0j * tau)
    theta2, theta3 = _theta_constants(q)
    z = (theta2 / theta3) ** 4
    character = _pillow_character(q)

    identity_left = (
        _power(theta3, 0.5)
        * _power(z * (1.0 - z), 1.0 / 24.0)
        * _power(16.0 * q, -1.0 / 24.0)
    )
    identity_error = abs(identity_left * character - 1.0)

    mobile_positions = tuple(fraction * z for fraction in mobile_fractions)
    segment_count = len(mobile_positions) + 1
    raw_segments = [
        cmath.exp((0.09 + 0.07 * i) * 1.0j)
        * _power(q, (i + 1.0) / (segment_count + 1.0))
        for i in range(segment_count - 1)
    ]
    raw_segments.append(q / _product(raw_segments))
    if segment_count == 1:
        rhos = [16.0 * q]
    else:
        rhos = [
            4.0 * raw_segments[0],
            *raw_segments[1:-1],
            4.0 * raw_segments[-1],
        ]
    central_charge = 17.3
    internal_weights = tuple(0.91 + 0.23 * i for i in range(segment_count))
    corners = (0.23, 0.37, 0.41, 0.52)
    mobile_weights = tuple(0.29 + 0.11 * i for i in range(len(mobile_positions)))

    geometric = _lambda_n(
        kappa=central_charge,
        theta3=theta3,
        z=z,
        mobile_positions=mobile_positions,
        corner_weights=corners,
        mobile_weights=mobile_weights,
    )
    geometric *= _product(
        _power(rho, h - central_charge / 24.0)
        for rho, h in zip(rhos, internal_weights)
    )
    geometric *= character

    zamolodchikov = _lambda_n(
        kappa=central_charge - 1.0,
        theta3=theta3,
        z=z,
        mobile_positions=mobile_positions,
        corner_weights=corners,
        mobile_weights=mobile_weights,
    )
    zamolodchikov *= _product(
        _power(rho, h - (central_charge - 1.0) / 24.0)
        for rho, h in zip(rhos, internal_weights)
    )
    normalization_error = abs(geometric / zamolodchikov - 1.0)
    return identity_error, normalization_error


def _product(values) -> complex:
    result = 1.0 + 0.0j
    for value in values:
        result *= value
    return result


def main() -> None:
    cases = (
        (0.07 + 0.83j, ()),
        (0.13 + 0.91j, (0.31 + 0.07j,)),
        (-0.21 + 1.17j, (0.18 + 0.03j, 0.44 - 0.05j, 0.71 + 0.02j)),
    )
    tolerance = 2.0e-12
    print("sphere-n pillow conformal-factor checks")
    for index, (tau, mobile_fractions) in enumerate(cases, start=1):
        identity_error, normalization_error = check_case(tau, mobile_fractions)
        print(
            f"case {index} ({len(mobile_fractions)} mobile): "
            f"theta identity error={identity_error:.3e}, "
            f"normalization error={normalization_error:.3e}"
        )
        if max(identity_error, normalization_error) > tolerance:
            raise AssertionError("pillow conformal-factor identity failed")
    print("conformal-factor checks passed")


if __name__ == "__main__":
    main()
