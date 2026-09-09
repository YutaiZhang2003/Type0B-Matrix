#!/usr/bin/env python3
"""Check the sphere-four-point cusp faces of the CCY torus two-point block.

For the necklace descendant tensor ``F[n1,n2]``, cutting either cylinder at
level zero leaves an ordinary sphere four-point block.  This script compares
the direct torus descendant contraction with the independently implemented
CCY sphere four-point c-recursion on both faces.

This is the first certificate for the Kummer/Fourier--Jacobi program recorded
in ``Machine Notes/h-Recursion/``.  It intentionally tests only an established
factorization identity; it does not assume that a generic sphere five-point
block equals an ordinary torus two-point block.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PLUMBING_ROOT = (
    REPOSITORY_ROOT
    / "Code"
    / "bosonic_c1_one_to_n_reference"
    / "reference_implementation"
    / "plumbing"
)
sys.path.insert(0, str(PLUMBING_ROOT))

from ccy_sphere_four_point import sphere_four_point_ccy_coefficients  # noqa: E402
from torus_two_point_blocks import necklace_descendant_coefficients  # noqa: E402


CASES = (
    {
        "central_charge": 26.215,
        "h1": 0.91,
        "h2": 1.07,
        "d1": 0.27,
        "d2": 0.41,
    },
    {
        "central_charge": 8.7,
        "h1": 1.13,
        "h2": 0.83,
        "d1": 0.22,
        "d2": 0.57,
    },
)


def _maximum_error(left: object, right: object) -> float:
    return max(abs(complex(a) - complex(b)) for a, b in zip(left, right))


def check_case(case: dict[str, float], order: int) -> tuple[float, float]:
    c = case["central_charge"]
    h1 = case["h1"]
    h2 = case["h2"]
    d1 = case["d1"]
    d2 = case["d2"]

    edge1_face = necklace_descendant_coefficients(
        c, h1, h2, d1, d2, order, 0
    )[:, 0]
    edge1_sphere = sphere_four_point_ccy_coefficients(
        central_charge=c,
        external_weights=(h2, d1, d2, h2),
        internal_weight=h1,
        order=order,
    )

    edge2_face = necklace_descendant_coefficients(
        c, h1, h2, d1, d2, 0, order
    )[0, :]
    edge2_sphere = sphere_four_point_ccy_coefficients(
        central_charge=c,
        external_weights=(h1, d2, d1, h1),
        internal_weight=h2,
        order=order,
    )

    return (
        _maximum_error(edge1_face, edge1_sphere),
        _maximum_error(edge2_face, edge2_sphere),
    )


def main() -> None:
    order = 6
    tolerance = 5.0e-11
    print("CCY torus-two-point cusp factorization")
    print(f"comparison order: {order}")
    for index, case in enumerate(CASES, start=1):
        edge1_error, edge2_error = check_case(case, order)
        print(
            f"case {index}: edge-1 face error={edge1_error:.3e}, "
            f"edge-2 face error={edge2_error:.3e}"
        )
        if max(edge1_error, edge2_error) > tolerance:
            raise AssertionError(
                "torus two-point cusp face does not match the sphere "
                "four-point block"
            )
    print("cusp-factorization checks passed")


if __name__ == "__main__":
    main()
