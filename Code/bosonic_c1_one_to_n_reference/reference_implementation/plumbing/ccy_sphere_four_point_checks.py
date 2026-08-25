#!/usr/bin/env python3
"""Independent checks of the punctured CCY recursion on the four-punctured sphere."""

from __future__ import annotations

import numpy as np

try:
    from ccy_sphere_four_point import (
        sphere_four_point_ccy_block,
        sphere_four_point_ccy_coefficients,
        sphere_four_point_coefficients_in_elliptic_nome,
        sphere_four_point_elliptic_descendant_block,
        sphere_four_point_elliptic_h_coefficients,
    )
    from torus_two_point_blocks import elliptic_nome
    from virasoro_plumbing_graph import (
        inverse_verma_gram_matrix,
        rho_primary_descendants,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_sphere_four_point import (
        sphere_four_point_ccy_block,
        sphere_four_point_ccy_coefficients,
        sphere_four_point_coefficients_in_elliptic_nome,
        sphere_four_point_elliptic_descendant_block,
        sphere_four_point_elliptic_h_coefficients,
    )
    from plumbing.torus_two_point_blocks import elliptic_nome
    from plumbing.virasoro_plumbing_graph import (
        inverse_verma_gram_matrix,
        rho_primary_descendants,
    )


def _relative_error(value: complex, target: complex) -> float:
    return abs(complex(value) - complex(target)) / max(abs(complex(target)), 1.0e-300)


def _direct_sphere_four_point_coefficients(
    *,
    central_charge: complex,
    external_weights: tuple[complex, complex, complex, complex],
    internal_weight: complex,
    order: int,
) -> tuple[complex, ...]:
    """Return the defining Verma-module descendant contraction."""

    h1, h2, h3, h4 = external_weights
    coefficients = []
    for level in range(order + 1):
        basis, inverse_gram, _condition = inverse_verma_gram_matrix(
            level,
            internal_weight,
            central_charge,
        )
        left = np.asarray(
            [
                rho_primary_descendants(
                    (),
                    (),
                    descendant,
                    h4,
                    h3,
                    internal_weight,
                    central_charge,
                )
                for descendant in basis
            ],
            dtype=np.complex128,
        )
        right = np.asarray(
            [
                rho_primary_descendants(
                    descendant,
                    (),
                    (),
                    internal_weight,
                    h2,
                    h1,
                    central_charge,
                )
                for descendant in basis
            ],
            dtype=np.complex128,
        )
        coefficients.append(complex(left @ inverse_gram @ right))
    return tuple(coefficients)


def check_ccy_against_standard_descendant_sum() -> None:
    cases = (
        (
            26.215,
            (0.13, 0.27, 0.41, 0.56),
            0.91,
            0.17 + 0.06j,
        ),
        (
            8.7,
            (0.22, 0.31, 0.47, 0.62),
            1.07,
            -0.12 + 0.08j,
        ),
    )
    order = 5
    print("sphere four-point CCY c-recursion vs direct descendant sum")
    for case_index, (central_charge, external_weights, internal_weight, z) in enumerate(
        cases,
        start=1,
    ):
        recursive = sphere_four_point_ccy_coefficients(
            central_charge=central_charge,
            external_weights=external_weights,
            internal_weight=internal_weight,
            order=order,
        )
        direct = _direct_sphere_four_point_coefficients(
            central_charge=central_charge,
            external_weights=external_weights,
            internal_weight=internal_weight,
            order=order,
        )
        coefficient_errors = tuple(
            _relative_error(recursive_value, direct_value)
            for recursive_value, direct_value in zip(recursive, direct)
        )
        recursive_value = sphere_four_point_ccy_block(
            z,
            central_charge=central_charge,
            external_weights=external_weights,
            internal_weight=internal_weight,
            order=order,
            include_primary_power=True,
        )
        h1, h2, _, _ = external_weights
        direct_descendant_value = sum(
            coefficient * z**level for level, coefficient in enumerate(direct)
        )
        direct_value = z ** (internal_weight - h1 - h2) * direct_descendant_value
        pointwise_error = _relative_error(recursive_value, direct_value)
        max_coefficient_error = max(coefficient_errors)
        print(
            f"  case {case_index}: max coefficient error={max_coefficient_error:.3e}, "
            f"pointwise error={pointwise_error:.3e}"
        )
        for level, (recursive_coefficient, direct_coefficient, error) in enumerate(
            zip(recursive, direct, coefficient_errors)
        ):
            print(
                f"    level {level}: CCY={recursive_coefficient!r}, "
                f"direct={direct_coefficient!r}, error={error:.3e}"
        )
        if max_coefficient_error > 2.0e-11:
            raise AssertionError(
                f"case {case_index} CCY coefficients disagree with the standard block"
            )
        if pointwise_error > 2.0e-12:
            raise AssertionError(f"case {case_index} full four-point block is inconsistent")


def check_elliptic_nome_reexpansion() -> None:
    external_weights = (1.07, 1.19, 1.11, 1.03)
    internal_weight = 1.37
    coefficients = sphere_four_point_ccy_coefficients(
        central_charge=25.0,
        external_weights=external_weights,
        internal_weight=internal_weight,
        order=12,
    )
    elliptic = sphere_four_point_coefficients_in_elliptic_nome(coefficients)
    z = 0.035 - 0.021j
    q = elliptic_nome(z)
    plane_value = sum(value * z**level for level, value in enumerate(coefficients))
    elliptic_value = sum(value * q**level for level, value in enumerate(elliptic))
    error = _relative_error(elliptic_value, plane_value)
    print("\nelliptic-nome re-expansion")
    print(f"  q={q!r}, relative error={error:.3e}")
    # scipy's complex hypergeometric inversion of lambda is the limiting
    # operation here; the algebraic re-expansion itself is exact by order.
    if error > 2.0e-9:
        raise AssertionError("elliptic re-expansion does not reproduce the plane block")

    h_coefficients = sphere_four_point_elliptic_h_coefficients(
        coefficients,
        central_charge=25.0,
        external_weights=external_weights,
        internal_weight=internal_weight,
    )
    stripped_value = sphere_four_point_elliptic_descendant_block(
        z,
        h_coefficients,
        central_charge=25.0,
        external_weights=external_weights,
        internal_weight=internal_weight,
        nome=q,
    )
    stripped_error = _relative_error(stripped_value, plane_value)
    print(f"  stripped-H relative error={stripped_error:.3e}")
    if stripped_error > 2.0e-9:
        raise AssertionError("stripped elliptic H block does not reproduce the plane block")


def main() -> None:
    check_ccy_against_standard_descendant_sum()
    check_elliptic_nome_reexpansion()
    print("\nsphere four-point CCY checks passed")


if __name__ == "__main__":
    main()
