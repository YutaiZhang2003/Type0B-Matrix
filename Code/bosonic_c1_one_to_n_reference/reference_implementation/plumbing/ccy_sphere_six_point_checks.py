#!/usr/bin/env python3
"""Independent checks of the CCY sphere six-point comb block."""

from __future__ import annotations

try:
    from ccy_sphere_six_point import (
        evaluate_sphere_six_point_series,
        sphere_six_point_c_coefficients,
        sphere_six_point_direct_coefficients,
        sphere_six_point_global_coefficients,
        sphere_six_point_h_c25_limit,
        sphere_six_point_h_coefficients,
        sphere_six_point_primary_factor,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.ccy_sphere_six_point import (
        evaluate_sphere_six_point_series,
        sphere_six_point_c_coefficients,
        sphere_six_point_direct_coefficients,
        sphere_six_point_global_coefficients,
        sphere_six_point_h_c25_limit,
        sphere_six_point_h_coefficients,
        sphere_six_point_primary_factor,
    )


def _relative_error(value: complex, target: complex) -> float:
    return abs(complex(value) - complex(target)) / max(abs(complex(target)), 1.0e-300)


def _maximum_table_error(
    observed: dict[tuple[int, int, int], complex],
    target: dict[tuple[int, int, int], complex],
) -> tuple[float, tuple[int, int, int]]:
    errors = {key: _relative_error(observed[key], target[key]) for key in target}
    key = max(errors, key=errors.get)
    return errors[key], key


def check_global_level_one() -> None:
    external = (0.17, 0.29, 0.43, 0.58, 0.71, 0.83)
    internal = (0.93, 1.08, 1.21)
    direct = sphere_six_point_direct_coefficients(
        central_charge=1.0e9,
        external_weights=external,
        internal_weights=internal,
        order1=1,
        order2=1,
        order3=1,
    )
    global_table = sphere_six_point_global_coefficients(
        external_weights=external,
        internal_weights=internal,
        order1=1,
        order2=1,
        order3=1,
    )
    error, key = _maximum_table_error(global_table, direct)
    print("global six-point block at level one")
    print(f"  max relative error={error:.3e} at {key}")
    if error > 3.0e-15:
        raise AssertionError("global six-point coefficient has the wrong convention")


def check_h_and_c_recursions_against_direct() -> None:
    cases = (
        (
            26.215,
            (0.17, 0.29, 0.43, 0.58, 0.71, 0.83),
            (0.93, 1.08, 1.21),
        ),
        (
            31.7,
            (0.21, 0.34, 0.49, 0.63, 0.79, 0.92),
            (1.03, 1.19, 1.37),
        ),
    )
    order = 4
    print("\nsix-point h/c recursions vs descendant definition")
    for index, (central_charge, external, internal) in enumerate(cases, start=1):
        direct = sphere_six_point_direct_coefficients(
            central_charge=central_charge,
            external_weights=external,
            internal_weights=internal,
            order1=order,
            order2=order,
            order3=order,
            max_total_order=order,
        )
        h_recursive = sphere_six_point_h_coefficients(
            central_charge=central_charge,
            external_weights=external,
            internal_weights=internal,
            order1=order,
            order2=order,
            order3=order,
            max_total_order=order,
        )
        c_recursive = sphere_six_point_c_coefficients(
            central_charge=central_charge,
            external_weights=external,
            internal_weights=internal,
            order1=order,
            order2=order,
            order3=order,
            max_total_order=order,
        )
        h_error, h_key = _maximum_table_error(h_recursive, direct)
        c_error, c_key = _maximum_table_error(c_recursive, direct)
        hc_error, hc_key = _maximum_table_error(h_recursive, c_recursive)
        print(
            f"  case {index}: h/direct={h_error:.3e} at {h_key}, "
            f"c/direct={c_error:.3e} at {c_key}, h/c={hc_error:.3e} at {hc_key}"
        )
        if max(h_error, c_error, hc_error) > 8.0e-10:
            raise AssertionError(f"case {index} six-point recursions disagree")


def check_c25_h_regulator() -> None:
    external = (1.17, 1.09, 1.13, 1.21, 1.07, 1.15)
    internal = (1.31, 1.47, 1.66)
    order = 4
    exact = sphere_six_point_c_coefficients(
        central_charge=25.0,
        external_weights=external,
        internal_weights=internal,
        order1=order,
        order2=order,
        order3=order,
        max_total_order=order,
    )
    regulated, estimates = sphere_six_point_h_c25_limit(
        external_weights=external,
        internal_weights=internal,
        order1=order,
        order2=order,
        order3=order,
        max_total_order=order,
    )
    error, key = _maximum_table_error(regulated, exact)
    estimated = max(abs(value) for value in estimates.values())
    print("\nc=25 resonant six-point h-recursion regulator")
    print(f"  max h/c relative error={error:.3e} at {key}")
    print(f"  max absolute fit-shift estimate={estimated:.3e}")
    if error > 3.0e-5:
        raise AssertionError("regulated c=25 six-point h-recursion has not converged")


def check_series_and_primary_factor() -> None:
    coefficients = {
        (0, 0, 0): 1.0,
        (1, 0, 0): 2.0,
        (0, 1, 0): 3.0,
        (0, 0, 1): 5.0,
        (1, 1, 1): 7.0,
    }
    q1, q2, q3 = 0.1, -0.2, 0.3
    observed = evaluate_sphere_six_point_series(q1, q2, q3, coefficients)
    expected = 1.0 + 2.0 * q1 + 3.0 * q2 + 5.0 * q3 + 7.0 * q1 * q2 * q3
    if abs(observed - expected) > 1.0e-15:
        raise AssertionError("trivariate block-series evaluator is inconsistent")

    external = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
    internal = (0.8, 0.9, 1.1)
    z1, z2, z3 = 0.02, 0.2, 0.5
    primary = sphere_six_point_primary_factor(
        z1,
        z2,
        z3,
        external_weights=external,
        internal_weights=internal,
    )
    target = (
        z1 ** (internal[0] - external[0] - external[1])
        * z2 ** (internal[1] - external[2] - internal[0])
        * z3 ** (internal[2] - external[3] - internal[1])
    )
    if _relative_error(primary, target) > 2.0e-15:
        raise AssertionError("six-point primary-coordinate factor is inconsistent")
    print("\nseries and primary-factor conventions passed")


def run() -> None:
    check_global_level_one()
    check_h_and_c_recursions_against_direct()
    check_c25_h_regulator()
    check_series_and_primary_factor()
    print("\nall sphere six-point CCY checks passed")


if __name__ == "__main__":
    run()

