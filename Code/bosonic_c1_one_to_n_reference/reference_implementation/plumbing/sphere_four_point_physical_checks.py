#!/usr/bin/env python3
"""Numerical gate for the four-point physical collar prescription."""

from __future__ import annotations

import math

try:
    from sphere_four_point_physical import (
        EqualOneToThreeBRYKernel,
        integrate_equal_one_to_three_bry,
    )
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.sphere_four_point_physical import (
        EqualOneToThreeBRYKernel,
        integrate_equal_one_to_three_bry,
    )


def main() -> None:
    outgoing_energy = 0.35
    epsilon = 0.08
    kernel = EqualOneToThreeBRYKernel(
        outgoing_energy,
        epsilon,
        block_order=8,
        momentum_order=10,
        momentum_maximum=8.0,
        special_dps=35,
    )
    results = tuple(
        integrate_equal_one_to_three_bry(
            kernel,
            sobol_power=9,
            replicates=2,
            seed=8841,
            collar_radius=radius,
            subtraction_scheme="local_finite_part",
        )
        for radius in (0.14, 0.08)
    )
    outgoing = complex(outgoing_energy, epsilon / 3.0)
    incoming = 3.0 * outgoing
    benchmark = 2.0 * math.pi * outgoing**3 * incoming * (1.0 + 1.0j * incoming)
    mean = sum(result.mean for result in results) / len(results)
    collar_spread = abs(results[0].mean - results[1].mean)
    benchmark_error = abs(mean - benchmark)
    print("sphere-four direct physical finite part")
    for result in results:
        print(
            f"  rho={result.collar_radius:.3f}: I4={result.mean!r}, "
            f"SE=({result.standard_error_real:.3e},{result.standard_error_imag:.3e})"
        )
    print(f"  collar spread={collar_spread:.3e}")
    print(f"  published-benchmark error={benchmark_error:.3e}")
    if collar_spread > 2.5e-2:
        raise AssertionError("the four-point finite part is not collar stable")
    if benchmark_error > 2.5e-2:
        raise AssertionError("the four-point baby step misses the physical benchmark")
    print("\nphysical sphere-four collar check passed")


if __name__ == "__main__":
    main()
