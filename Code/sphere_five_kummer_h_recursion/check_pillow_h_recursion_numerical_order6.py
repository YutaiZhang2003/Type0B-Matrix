#!/usr/bin/env python3
"""Numerical PBW check of the sphere five-point pillow h-recursion to order 6.

The direct five-point Virasoro block is evaluated from its two-edge PBW
descendant contraction in the plane comb frame.  Its bivariate series is
then changed from (z/t,t) to the aligned pillow variables (p1,p2), and all
conformal factors and the pillow character are stripped.  The resulting
H_5 coefficients are compared with the proposed two-channel h-recursion.

The comparison uses every coefficient with n1+n2 <= 6.  It also reports two
normalization failures inherited from the four-point audit: the order-(1,1)
failure produced by the literal (1-z) exponent printed in E.103, and the
universal order-(2,2) failure produced by retaining the pillow character
after shifting the universal prefactor from c to c-1.
"""

from __future__ import annotations

import functools
import sys
import time
from pathlib import Path
from typing import Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PLUMBING_ROOT = (
    REPOSITORY_ROOT
    / "Code"
    / "bosonic_c1_one_to_n_reference"
    / "reference_implementation"
    / "plumbing"
)
sys.path.insert(0, str(PLUMBING_ROOT))

from ccy_sphere_five_point import sphere_five_point_direct_coefficients  # noqa: E402
from virasoro_blocks import (  # noqa: E402
    central_charge_to_b,
    degenerate_weight,
    fusion_polynomial_for_weights,
    zamolodchikov_a_rs,
)


Series2 = dict[tuple[int, int], complex]


CASES = (
    {
        "central_charge": 26.215,
        "external_weights": (0.17, 0.29, 0.43, 0.58, 0.71),
        "internal_weights": (0.93, 1.08),
    },
    {
        "central_charge": 31.7,
        "external_weights": (0.21, 0.34, 0.49, 0.63, 0.79),
        "internal_weights": (1.03, 1.19),
    },
    {
        "central_charge": 42.3,
        "external_weights": (0.13, 0.31, 0.47, 0.69, 0.83),
        "internal_weights": (0.87, 1.14),
    },
)


def _clean(series: Mapping[tuple[int, int], complex], order: int) -> Series2:
    return {
        (i, j): complex(value)
        for (i, j), value in series.items()
        if i >= 0 and j >= 0 and i + j <= order and value != 0.0
    }


def add(left: Series2, right: Series2, order: int) -> Series2:
    out = dict(left)
    for key, value in right.items():
        out[key] = out.get(key, 0.0j) + value
    return _clean(out, order)


def scale(series: Series2, coefficient: complex, order: int) -> Series2:
    return _clean({key: coefficient * value for key, value in series.items()}, order)


def multiply(left: Series2, right: Series2, order: int) -> Series2:
    out: Series2 = {}
    for (i1, j1), value1 in left.items():
        for (i2, j2), value2 in right.items():
            key = (i1 + i2, j1 + j2)
            if sum(key) <= order:
                out[key] = out.get(key, 0.0j) + value1 * value2
    return _clean(out, order)


def monomial(i: int, j: int, coefficient: complex, order: int) -> Series2:
    return {} if i + j > order else {(i, j): complex(coefficient)}


def unit_power(series: Series2, exponent: complex, order: int) -> Series2:
    constant = series.get((0, 0), 0.0j)
    if abs(constant - 1.0) > 1.0e-14:
        raise ValueError(f"unit series has constant {constant!r}, not one")
    remainder = dict(series)
    remainder[(0, 0)] = remainder.get((0, 0), 0.0j) - 1.0
    remainder = _clean(remainder, order)
    out: Series2 = {(0, 0): 1.0 + 0.0j}
    power: Series2 = {(0, 0): 1.0 + 0.0j}
    binomial = 1.0 + 0.0j
    for k in range(1, order + 1):
        power = multiply(power, remainder, order)
        if not power:
            break
        binomial *= (complex(exponent) - (k - 1)) / k
        out = add(out, scale(power, binomial, order), order)
    return out


def one_plus_monomial(i: int, j: int, order: int) -> Series2:
    return add({(0, 0): 1.0 + 0.0j}, monomial(i, j, 1.0, order), order)


def pillow_map_units(order: int) -> tuple[Series2, Series2, Series2, Series2, Series2]:
    """Return z/(16q), t/(4p2), (z/t)/(4p1), theta3, and chi^-1."""

    one: Series2 = {(0, 0): 1.0 + 0.0j}

    # z/(16q)=prod_n (1+q^(2n))^8/(1+q^(2n-1))^8,
    # where q=p1*p2 has total bidegree two.
    z_unit = dict(one)
    n = 1
    while 4 * n - 2 <= order:
        if 4 * n <= order:
            z_unit = multiply(
                z_unit,
                unit_power(one_plus_monomial(2 * n, 2 * n, order), 8, order),
                order,
            )
        z_unit = multiply(
            z_unit,
            unit_power(
                one_plus_monomial(2 * n - 1, 2 * n - 1, order),
                -8,
                order,
            ),
            order,
        )
        n += 1

    # Product formula for t=z sn^2(Kw/pi|z), in the aligned convention
    # exp(iw)=-p1.  The leading coordinate is t=4p2+....
    y_unit = unit_power(one_plus_monomial(1, 0, order), 2, order)
    n = 1
    while 4 * n - 2 <= order:
        if 4 * n <= order:
            y_unit = multiply(
                y_unit,
                unit_power(one_plus_monomial(2 * n, 2 * n, order), 4, order),
                order,
            )
        y_unit = multiply(
            y_unit,
            unit_power(
                one_plus_monomial(2 * n - 1, 2 * n - 1, order),
                -4,
                order,
            ),
            order,
        )
        n += 1

    n = 1
    while 4 * n - 3 <= order:
        factors = (
            (2 * n + 1, 2 * n, 2),
            (2 * n - 1, 2 * n, 2),
            (2 * n, 2 * n - 1, -2),
            (2 * n - 2, 2 * n - 1, -2),
        )
        for i, j, exponent in factors:
            if i + j <= order:
                y_unit = multiply(
                    y_unit,
                    unit_power(one_plus_monomial(i, j, order), exponent, order),
                    order,
                )
        n += 1

    x_unit = multiply(z_unit, unit_power(y_unit, -1, order), order)

    theta3 = dict(one)
    n = 1
    while 2 * n * n <= order:
        theta3 = add(theta3, monomial(n * n, n * n, 2.0, order), order)
        n += 1

    chi_inverse = dict(one)
    n = 1
    while 4 * n <= order:
        chi_inverse = multiply(
            chi_inverse,
            unit_power(
                add(one, monomial(2 * n, 2 * n, -1.0, order), order),
                0.5,
                order,
            ),
            order,
        )
        n += 1
    return z_unit, y_unit, x_unit, theta3, chi_inverse


def direct_reduced_coefficients(
    order: int,
    *,
    plane: Series2,
    central_charge: complex,
    external_weights: tuple[complex, complex, complex, complex, complex],
    internal_weights: tuple[complex, complex],
    literal_e103_pairing: bool = False,
) -> Series2:
    """Strip the exact pillow prefactor from the direct plane PBW block."""

    d1, d2, d3, d4, d5 = map(complex, external_weights)
    h1, h2 = map(complex, internal_weights)
    c = complex(central_charge)
    a = h2 - h1
    z_unit, y_unit, x_unit, theta3, chi_inverse = pillow_map_units(order)
    one: Series2 = {(0, 0): 1.0 + 0.0j}

    t_series = scale(multiply({(0, 1): 1.0 + 0.0j}, y_unit, order), 4, order)
    x_series = scale(multiply({(1, 0): 1.0 + 0.0j}, x_unit, order), 4, order)
    mobile_unit = multiply(
        add(one, scale(t_series, -1, order), order),
        add(one, scale(x_series, -1, order), order),
        order,
    )
    z_series = scale(
        multiply({(1, 1): 1.0 + 0.0j}, z_unit, order),
        16,
        order,
    )
    one_minus_z = add(one, scale(z_series, -1, order), order)

    theta_exponent = c / 2 - 4 * (d1 + d2 + d4 + d5) - 2 * d3
    upper_exponent = (
        -c / 24 + d4 + d5
        if literal_e103_pairing
        else -c / 24 + d2 + d4
    )
    prefactor = dict(one)
    for unit, exponent in (
        (z_unit, h1 - c / 24),
        (y_unit, a),
        (mobile_unit, d3 / 2),
        (theta3, -theta_exponent),
        (one_minus_z, upper_exponent),
    ):
        prefactor = multiply(prefactor, unit_power(unit, exponent, order), order)
    prefactor = multiply(prefactor, chi_inverse, order)

    descendant: Series2 = {}
    for (n1, n2), coefficient in plane.items():
        term = multiply(
            monomial(n1, n2, coefficient * 4 ** (n1 + n2), order),
            unit_power(x_unit, n1, order),
            order,
        )
        term = multiply(term, unit_power(y_unit, n2, order), order)
        descendant = add(descendant, term, order)
    return multiply(prefactor, descendant, order)


def proposed_coefficients(
    order: int,
    *,
    central_charge: complex,
    external_weights: tuple[complex, complex, complex, complex, complex],
    internal_weights: tuple[complex, complex],
) -> Series2:
    d1, d2, d3, d4, d5 = map(complex, external_weights)
    h1, h2 = map(complex, internal_weights)
    b = central_charge_to_b(complex(central_charge))
    initial_a = h2 - h1

    @functools.lru_cache(maxsize=None)
    def coefficient(n1: int, n2: int, current_h: complex, current_a: complex) -> complex:
        total = 1.0 + 0.0j if (n1, n2) == (0, 0) else 0.0 + 0.0j
        for r in range(1, n1 + 1):
            for s in range(1, n1 // r + 1):
                level = r * s
                pole = degenerate_weight(r, s, b)
                residue = (
                    4**level
                    * zamolodchikov_a_rs(r, s, b)
                    * fusion_polynomial_for_weights(r, s, b, d1, d2)
                    * fusion_polynomial_for_weights(r, s, b, pole + current_a, d3)
                    / (current_h - pole)
                )
                total += residue * coefficient(
                    n1 - level,
                    n2,
                    pole + level,
                    current_a - level,
                )
        for r in range(1, n2 + 1):
            for s in range(1, n2 // r + 1):
                level = r * s
                pole = degenerate_weight(r, s, b)
                residue = (
                    4**level
                    * zamolodchikov_a_rs(r, s, b)
                    * fusion_polynomial_for_weights(r, s, b, d5, d4)
                    * fusion_polynomial_for_weights(r, s, b, pole - current_a, d3)
                    / (current_h + current_a - pole)
                )
                total += residue * coefficient(
                    n1,
                    n2 - level,
                    pole - current_a,
                    current_a + level,
                )
        return complex(total)

    return {
        (n1, n2): coefficient(n1, n2, h1, initial_a)
        for n1 in range(order + 1)
        for n2 in range(order + 1 - n1)
    }


def relative_error(value: complex, target: complex) -> float:
    return abs(complex(value) - complex(target)) / max(1.0, abs(complex(target)))


def compare_case(case: dict[str, object], order: int) -> tuple[float, tuple[int, int]]:
    central_charge = complex(case["central_charge"])
    external = tuple(complex(value) for value in case["external_weights"])
    internal = tuple(complex(value) for value in case["internal_weights"])

    start = time.perf_counter()
    plane = sphere_five_point_direct_coefficients(
        central_charge=central_charge,
        external_weights=external,
        internal_weights=internal,
        order1=order,
        order2=order,
        max_total_order=order,
    )
    pbw_seconds = time.perf_counter() - start
    direct = direct_reduced_coefficients(
        order,
        plane=plane,
        central_charge=central_charge,
        external_weights=external,
        internal_weights=internal,
    )
    proposed = proposed_coefficients(
        order,
        central_charge=central_charge,
        external_weights=external,
        internal_weights=internal,
    )
    errors = {key: relative_error(direct[key], proposed[key]) for key in proposed}
    worst_key = max(errors, key=errors.get)

    literal = direct_reduced_coefficients(
        2,
        plane={key: value for key, value in plane.items() if sum(key) <= 2},
        central_charge=central_charge,
        external_weights=external,
        internal_weights=internal,
        literal_e103_pairing=True,
    )
    literal_mismatch = literal[(1, 1)] - proposed[(1, 1)]
    predicted_mismatch = 16.0 * (external[1] - external[4])
    mismatch_error = abs(literal_mismatch - predicted_mismatch)

    # A hybrid formula that keeps the pillow character after the c -> c-1
    # shift multiplies the correctly extracted H by chi^{-1}.  Since
    # chi^{-1}=1-q^2/2+... and q=p1*p2, its first error is -1/2 at (2,2).
    _z_unit, _y_unit, _x_unit, _theta3, chi_inverse = pillow_map_units(4)
    direct_order4 = {key: value for key, value in direct.items() if sum(key) <= 4}
    hybrid = multiply(direct_order4, chi_inverse, 4)
    hybrid_mismatch = hybrid[(2, 2)] - proposed[(2, 2)]
    hybrid_error = abs(hybrid_mismatch + 0.5)
    print(
        f"  PBW={pbw_seconds:.2f}s, max relative error={errors[worst_key]:.3e} "
        f"at {worst_key}"
    )
    print(
        "  literal E.103 mixed mismatch="
        f"{literal_mismatch.real:+.12e}{literal_mismatch.imag:+.12e}j, "
        f"error vs 16(d2-d5)={mismatch_error:.3e}"
    )
    print(
        "  hybrid c-1-plus-character mismatch at (2,2)="
        f"{hybrid_mismatch.real:+.12e}{hybrid_mismatch.imag:+.12e}j, "
        f"error vs -1/2={hybrid_error:.3e}"
    )
    if hybrid_error > 2.0e-8:
        raise AssertionError("five-point character normalization check failed")
    return errors[worst_key], worst_key


def main() -> None:
    order = 6
    tolerance = 2.0e-8
    print("sphere five-point pillow h-recursion vs direct PBW")
    print(f"total bidegree: n1+n2 <= {order} ({(order + 1) * (order + 2) // 2} coefficients)")
    worst = 0.0
    for index, case in enumerate(CASES, start=1):
        print(f"case {index}: c={case['central_charge']}, " f"h={case['internal_weights']}")
        error, _ = compare_case(case, order)
        worst = max(worst, error)
    print(f"global maximum relative error={worst:.3e}")
    if worst > tolerance:
        raise AssertionError("pillow h-recursion disagrees with direct PBW data")
    print("all order-six numerical PBW checks passed")


if __name__ == "__main__":
    main()
