#!/usr/bin/env python3
"""Numerically distinguish the E.103 and E.106 four-point pillow factors.

The external primaries ``(h1,h2,h3,h4)`` sit at ``(0,z,1,infinity)``.
For each generic test point this script

1. computes the sphere four-point block directly from PBW Verma-module
   Gram matrices through level six;
2. changes variables from ``z`` to ``q`` using ``z=lambda(q)``;
3. extracts the reduced elliptic block with the adjacent E.106 pairing
   ``(1-z)^((c-1)/24-h2-h3)``;
4. compares all coefficients with an independent Zamolodchikov h-recursion;
5. repeats the extraction with the literal E.103 pairing
   ``(1-z)^(c/24-h3-h4)`` and with the explicit character product retained
   in E.106 as printed.

The two extracted reduced blocks obey

``H_E103(q) = (1-lambda(q))^(h4-h2) H_E106(q)``.

Thus the literal E.103 candidate has the generic level-one obstruction
``16*(h2-h4)``.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PLUMBING_ROOT = (
    REPOSITORY_ROOT
    / "Code"
    / "bosonic_c1_one_to_n_reference"
    / "reference_implementation"
    / "plumbing"
)
sys.path.insert(0, str(PLUMBING_ROOT))

from ccy_sphere_four_point import (  # noqa: E402
    _truncated_product,
    _unit_series_power,
    sphere_four_point_elliptic_h_coefficients,
)
from ccy_sphere_four_point_checks import (  # noqa: E402
    _direct_sphere_four_point_coefficients,
)
from torus_two_point_blocks import modular_lambda_series  # noqa: E402
from virasoro_blocks import (  # noqa: E402
    central_charge_to_b,
    degenerate_weight,
    fusion_polynomial_for_weights,
    zamolodchikov_a_rs,
)


ORDER = 6
CASES = (
    (26.215, (0.13, 0.27, 0.41, 0.56), 0.91),
    (31.7, (0.19, 0.34, 0.48, 0.67), 1.07),
    (42.3, (0.16, 0.31, 0.52, 0.73), 1.23),
)


def h_recursion_coefficients(
    *,
    central_charge: complex,
    external_weights: tuple[complex, complex, complex, complex],
    internal_weight: complex,
    order: int,
) -> np.ndarray:
    """Return H_0,...,H_order from the independent h-recursion."""

    h1, h2, h3, h4 = map(complex, external_weights)
    b = central_charge_to_b(central_charge)

    @functools.lru_cache(maxsize=None)
    def coefficient(level: int, current_h: complex) -> complex:
        total = 1.0 + 0.0j if level == 0 else 0.0 + 0.0j
        for r in range(1, level + 1):
            for s in range(1, level // r + 1):
                null_level = r * s
                pole = degenerate_weight(r, s, b)
                residue = (
                    16**null_level
                    * zamolodchikov_a_rs(r, s, b)
                    * fusion_polynomial_for_weights(r, s, b, h1, h2)
                    * fusion_polynomial_for_weights(r, s, b, h4, h3)
                    / (current_h - pole)
                )
                total += residue * coefficient(
                    level - null_level,
                    pole + null_level,
                )
        return complex(total)

    return np.asarray(
        [coefficient(level, complex(internal_weight)) for level in range(order + 1)],
        dtype=np.complex128,
    )


def candidate_coefficients(
    plane: tuple[complex, ...],
    *,
    central_charge: complex,
    external_weights: tuple[complex, complex, complex, complex],
    internal_weight: complex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standard E.106-paired, literal E.103, and literal E.106 H."""

    order = len(plane) - 1
    h_e106 = np.asarray(
        sphere_four_point_elliptic_h_coefficients(
            plane,
            central_charge=central_charge,
            external_weights=external_weights,
            internal_weight=internal_weight,
        ),
        dtype=np.complex128,
    )
    _h1, h2, _h3, h4 = map(complex, external_weights)
    one_minus_lambda = -np.asarray(modular_lambda_series(order), dtype=np.complex128)
    one_minus_lambda[0] += 1.0
    e103_over_e106 = _unit_series_power(
        one_minus_lambda,
        h4 - h2,
        order,
    )
    h_e103 = _truncated_product(h_e106, e103_over_e106, order)

    # E.106 as printed retains P(q)=prod_n(1-q^(2n))^(-1/2) after also
    # shifting every anomaly exponent from c to c-1.  Solving that literal
    # formula for its H therefore multiplies the standard H by P(q)^(-1).
    character_inverse = np.zeros(order + 1, dtype=np.complex128)
    character_inverse[0] = 1.0
    for integer in range(1, order // 2 + 1):
        factor = np.zeros(order + 1, dtype=np.complex128)
        factor[0] = 1.0
        factor[2 * integer] = -1.0
        character_inverse = _truncated_product(
            character_inverse,
            _unit_series_power(factor, 0.5, order),
            order,
        )
    h_e106_literal = _truncated_product(h_e106, character_inverse, order)
    return h_e106, h_e103, h_e106_literal


def relative_error(value: complex, target: complex) -> float:
    return abs(complex(value) - complex(target)) / max(1.0, abs(complex(target)))


def print_complex(value: complex) -> str:
    value = complex(value)
    if abs(value.imag) < 5.0e-12:
        return f"{value.real:+.12e}"
    return f"{value.real:+.8e}{value.imag:+.8e}j"


def compare_case(
    case_index: int,
    central_charge: complex,
    external_weights: tuple[complex, complex, complex, complex],
    internal_weight: complex,
) -> tuple[float, float, float]:
    plane = _direct_sphere_four_point_coefficients(
        central_charge=central_charge,
        external_weights=external_weights,
        internal_weight=internal_weight,
        order=ORDER,
    )
    e106, e103, e106_literal = candidate_coefficients(
        plane,
        central_charge=central_charge,
        external_weights=external_weights,
        internal_weight=internal_weight,
    )
    recursive = h_recursion_coefficients(
        central_charge=central_charge,
        external_weights=external_weights,
        internal_weight=internal_weight,
        order=ORDER,
    )

    errors_e106 = [relative_error(value, target) for value, target in zip(e106, recursive)]
    errors_e103 = [relative_error(value, target) for value, target in zip(e103, recursive)]
    errors_e106_literal = [
        relative_error(value, target) for value, target in zip(e106_literal, recursive)
    ]
    h2 = complex(external_weights[1])
    h4 = complex(external_weights[3])
    observed_level_one = e103[1] - recursive[1]
    predicted_level_one = 16.0 * (h2 - h4)

    print(
        f"case {case_index}: c={central_charge}, h_i={external_weights}, "
        f"h={internal_weight}"
    )
    print("  n        PBW / E.106             h-recursion              PBW / E.103")
    for level in range(ORDER + 1):
        print(
            f"  {level:d}  {print_complex(e106[level]):>23}  "
            f"{print_complex(recursive[level]):>23}  "
            f"{print_complex(e103[level]):>23}"
        )
    print(
        "  max relative error: "
        f"E.106 pairing={max(errors_e106):.3e}, "
        f"E.103={max(errors_e103):.3e}, "
        f"literal E.106 product={max(errors_e106_literal):.3e}"
    )
    print(
        "  level-one E.103 mismatch: "
        f"observed={print_complex(observed_level_one)}, "
        f"16(h2-h4)={print_complex(predicted_level_one)}"
    )
    print(
        "  literal E.106 product mismatch at q^2: "
        f"{print_complex(e106_literal[2] - recursive[2])}"
    )
    return max(errors_e106), max(errors_e103), max(errors_e106_literal)


def main() -> None:
    print("sphere four-point numerical PBW test of E.103 versus E.106")
    worst_e106 = 0.0
    best_e103 = float("inf")
    best_e106_literal = float("inf")
    for case_index, case in enumerate(CASES, start=1):
        error_e106, error_e103, error_e106_literal = compare_case(case_index, *case)
        worst_e106 = max(worst_e106, error_e106)
        best_e103 = min(best_e103, error_e103)
        best_e106_literal = min(best_e106_literal, error_e106_literal)
    if worst_e106 > 5.0e-9:
        raise AssertionError(f"E.106 failed: worst relative error {worst_e106:.3e}")
    if best_e103 < 1.0e-3:
        raise AssertionError("literal E.103 was not numerically distinguished from E.106")
    if best_e106_literal < 1.0e-3:
        raise AssertionError("the explicit E.106 product was not numerically detected")
    print(
        "result: the E.106 pairing passes through q^6; literal E.103 fails at q^1, "
        "and retaining the explicit E.106 product after the c-1 shift fails at q^2"
    )


if __name__ == "__main__":
    main()
