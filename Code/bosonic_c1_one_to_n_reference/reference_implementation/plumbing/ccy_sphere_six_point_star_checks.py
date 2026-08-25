#!/usr/bin/env python3
"""Independent checks of the six-point star-channel block."""

from __future__ import annotations

try:
    from ccy_sphere_six_point_star import (
        sphere_six_point_star_c_coefficients,
        sphere_six_point_star_direct_coefficients,
        sphere_six_point_star_global_coefficient,
    )
except ImportError:  # pragma: no cover
    from plumbing.ccy_sphere_six_point_star import (
        sphere_six_point_star_c_coefficients,
        sphere_six_point_star_direct_coefficients,
        sphere_six_point_star_global_coefficient,
    )


def _relative_error(value: complex, target: complex) -> float:
    return abs(complex(value) - complex(target)) / max(abs(complex(target)), 1.0e-300)


def check_global_level_one() -> None:
    external = (0.17, 0.29, 0.43, 0.58, 0.71, 0.83)
    internal = (0.93, 1.08, 1.21)
    direct = sphere_six_point_star_direct_coefficients(
        central_charge=1.0e9,
        external_weights=external,
        internal_weights=internal,
        order1=1,
        order2=1,
        order3=1,
    )
    errors = {}
    for key, value in direct.items():
        global_value = sphere_six_point_star_global_coefficient(
            *key,
            external_weights=external,
            internal_weights=internal,
        )
        errors[key] = _relative_error(global_value, value)
    worst = max(errors, key=errors.get)
    print("global star block at level one")
    print(f"  max relative error={errors[worst]:.3e} at {worst}")
    if errors[worst] > 3.0e-15:
        raise AssertionError("global star coefficient has the wrong convention")


def check_c_recursion_against_direct() -> None:
    cases = (
        (26.215, (0.17, 0.29, 0.43, 0.58, 0.71, 0.83), (0.93, 1.08, 1.21)),
        (31.7, (0.21, 0.34, 0.49, 0.63, 0.79, 0.92), (1.03, 1.19, 1.37)),
    )
    order = 4
    print("\nstar c-recursion vs descendant definition")
    for index, (central_charge, external, internal) in enumerate(cases, start=1):
        direct = sphere_six_point_star_direct_coefficients(
            central_charge=central_charge,
            external_weights=external,
            internal_weights=internal,
            order1=order,
            order2=order,
            order3=order,
            max_total_order=order,
        )
        recursive = sphere_six_point_star_c_coefficients(
            central_charge=central_charge,
            external_weights=external,
            internal_weights=internal,
            order1=order,
            order2=order,
            order3=order,
            max_total_order=order,
        )
        errors = {key: _relative_error(recursive[key], value) for key, value in direct.items()}
        worst = max(errors, key=errors.get)
        print(f"  case {index}: max relative error={errors[worst]:.3e} at {worst}")
        if errors[worst] > 8.0e-10:
            raise AssertionError(f"case {index} star recursion disagrees")


def run() -> None:
    check_global_level_one()
    check_c_recursion_against_direct()
    print("\nall sphere six-point star checks passed")


if __name__ == "__main__":
    run()

