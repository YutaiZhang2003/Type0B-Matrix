#!/usr/bin/env python3
"""Fast checks for the torus two-point cusp Laplace expansion."""

from __future__ import annotations

import math

try:
    from genus1_two_point_cusp_laplace import (
        necklace_laplace_estimate,
        necklace_reduced_dozz_polynomial,
    )
except ImportError:  # pragma: no cover
    from plumbing.genus1_two_point_cusp_laplace import (
        necklace_laplace_estimate,
        necklace_reduced_dozz_polynomial,
    )


def check_threshold_value() -> None:
    tau = 0.13 + 20.0j
    z = 2.0 * math.pi * (0.31 + 0.25 * tau)
    # Independent full-block threshold-Gaussian Q=10 value.
    reference = -1.3022432252830427e-3
    estimates = [
        necklace_laplace_estimate(0.4, z, tau, max_x_degree=degree)
        for degree in range(5, 9)
    ]
    best = min(estimates, key=lambda item: abs(item.value / reference - 1.0))
    assert abs(best.value / reference - 1.0) < 5.0e-5


def check_symmetry_and_validation() -> None:
    polynomial = necklace_reduced_dozz_polynomial(0.4, 4)
    for (first, second), coefficient in polynomial.items():
        reflected = polynomial[(second, first)]
        assert abs(coefficient - reflected) < 1.0e-10 * max(abs(coefficient), 1.0)
    try:
        necklace_reduced_dozz_polynomial(1.1, 3)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("out-of-range x was accepted")


def main() -> None:
    check_threshold_value()
    check_symmetry_and_validation()
    print("genus1 two-point cusp Laplace checks passed")


if __name__ == "__main__":
    main()
