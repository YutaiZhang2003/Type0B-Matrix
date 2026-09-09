#!/usr/bin/env python3
"""High-precision PBW check of the sphere five-point pillow recursion.

The direct plane block is computed from exact rational PBW Gram matrices and
three-point tensors.  Its change to the aligned pillow variables is also
performed with exact rational bivariate series.  Only the final comparison
with the independent fixed-c h-recursion is numerical, at 80 decimal digits.

Every coefficient with n1+n2 <= 10 is tested (66 coefficients per case).
The samples are deliberately asymmetric and include both signs of
a=h2-h1.  In addition to agreement, the checker verifies that all seven
weights are distinct, no tested PBW coefficient vanishes, and every
recursion denominator visited through order ten stays away from a Kac pole.
The script also retains the two normalization regression tests: the literal
E.103 corner pairing and the forbidden c-1-plus-character hybrid.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Mapping

import mpmath as mp
import sympy as sp

from check_pillow_h_recursion_symbolic_order4 import (
    direct_pbw_coefficients,
    series2_add,
    series2_monomial,
    series2_mul,
    series2_scale,
    series2_unit_power,
)


Series2 = dict[tuple[int, int], sp.Expr]


ORDER = 10
DECIMAL_DIGITS = 80
CASES = (
    {
        "central_charge": "26.215",
        # Code ordering: (d1,d2,d_mobile,d_at_1,d_at_infinity).
        "external_weights": ("0.17", "0.29", "0.43", "0.58", "0.71"),
        "internal_weights": ("0.93", "1.08"),
    },
    {
        "central_charge": "31.7",
        "external_weights": ("0.21", "0.34", "0.49", "0.63", "0.79"),
        "internal_weights": ("1.03", "1.19"),
    },
    {
        "central_charge": "42.3",
        "external_weights": ("0.13", "0.31", "0.47", "0.69", "0.83"),
        "internal_weights": ("0.87", "1.14"),
    },
    {
        # Non-monotone external weights and negative a=h2-h1.
        "central_charge": "29.3761",
        "external_weights": ("0.113", "0.367", "0.811", "1.237", "0.524"),
        "internal_weights": ("1.417", "0.682"),
    },
    {
        # A second non-monotone sample with a comparatively large positive a.
        "central_charge": "37.219",
        "external_weights": ("0.907", "0.223", "1.041", "0.376", "0.658"),
        "internal_weights": ("0.749", "1.533"),
    },
)


def one_plus_monomial(i: int, j: int, order: int) -> Series2:
    return series2_add(
        {(0, 0): sp.S.One},
        series2_monomial(i, j, sp.S.One, order),
        order,
    )


def pillow_map_units(order: int) -> tuple[Series2, Series2, Series2, Series2, Series2]:
    """Return exact Z=z/(16q), Y=t/(4p2), X=(z/t)/(4p1), theta3, chi^-1."""

    one: Series2 = {(0, 0): sp.S.One}

    z_unit = dict(one)
    n = 1
    while 4 * n - 2 <= order:
        if 4 * n <= order:
            z_unit = series2_mul(
                z_unit,
                series2_unit_power(
                    one_plus_monomial(2 * n, 2 * n, order),
                    sp.Integer(8),
                    order,
                ),
                order,
            )
        z_unit = series2_mul(
            z_unit,
            series2_unit_power(
                one_plus_monomial(2 * n - 1, 2 * n - 1, order),
                sp.Integer(-8),
                order,
            ),
            order,
        )
        n += 1

    y_unit = series2_unit_power(
        one_plus_monomial(1, 0, order),
        sp.Integer(2),
        order,
    )
    n = 1
    while 4 * n - 2 <= order:
        if 4 * n <= order:
            y_unit = series2_mul(
                y_unit,
                series2_unit_power(
                    one_plus_monomial(2 * n, 2 * n, order),
                    sp.Integer(4),
                    order,
                ),
                order,
            )
        y_unit = series2_mul(
            y_unit,
            series2_unit_power(
                one_plus_monomial(2 * n - 1, 2 * n - 1, order),
                sp.Integer(-4),
                order,
            ),
            order,
        )
        n += 1

    n = 1
    while 4 * n - 3 <= order:
        for i, j, exponent in (
            (2 * n + 1, 2 * n, 2),
            (2 * n - 1, 2 * n, 2),
            (2 * n, 2 * n - 1, -2),
            (2 * n - 2, 2 * n - 1, -2),
        ):
            if i + j <= order:
                y_unit = series2_mul(
                    y_unit,
                    series2_unit_power(
                        one_plus_monomial(i, j, order),
                        sp.Integer(exponent),
                        order,
                    ),
                    order,
                )
        n += 1

    x_unit = series2_mul(
        z_unit,
        series2_unit_power(y_unit, sp.Integer(-1), order),
        order,
    )

    theta3 = dict(one)
    n = 1
    while 2 * n * n <= order:
        theta3 = series2_add(
            theta3,
            series2_monomial(n * n, n * n, sp.Integer(2), order),
            order,
        )
        n += 1

    chi_inverse = dict(one)
    n = 1
    while 4 * n <= order:
        factor = series2_add(
            one,
            series2_monomial(2 * n, 2 * n, -sp.S.One, order),
            order,
        )
        chi_inverse = series2_mul(
            chi_inverse,
            series2_unit_power(factor, sp.Rational(1, 2), order),
            order,
        )
        n += 1
    return z_unit, y_unit, x_unit, theta3, chi_inverse


def direct_reduced_coefficients(
    order: int,
    *,
    plane: Series2,
    central_charge: sp.Expr,
    external_weights: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr],
    internal_weights: tuple[sp.Expr, sp.Expr],
    literal_e103_pairing: bool = False,
) -> Series2:
    """Strip the complete pillow prefactor from exact plane PBW data."""

    d1, d2, d3, d4, d5 = external_weights
    h1, h2 = internal_weights
    c = central_charge
    a = h2 - h1
    z_unit, y_unit, x_unit, theta3, chi_inverse = pillow_map_units(order)
    one: Series2 = {(0, 0): sp.S.One}

    t_series = series2_scale(
        series2_mul({(0, 1): sp.S.One}, y_unit, order),
        sp.Integer(4),
        order,
    )
    x_series = series2_scale(
        series2_mul({(1, 0): sp.S.One}, x_unit, order),
        sp.Integer(4),
        order,
    )
    mobile_unit = series2_mul(
        series2_add(one, series2_scale(t_series, -sp.S.One, order), order),
        series2_add(one, series2_scale(x_series, -sp.S.One, order), order),
        order,
    )
    z_series = series2_scale(
        series2_mul({(1, 1): sp.S.One}, z_unit, order),
        sp.Integer(16),
        order,
    )
    one_minus_z = series2_add(
        one,
        series2_scale(z_series, -sp.S.One, order),
        order,
    )

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
        prefactor = series2_mul(
            prefactor,
            series2_unit_power(unit, exponent, order),
            order,
        )
    prefactor = series2_mul(prefactor, chi_inverse, order)

    descendant: Series2 = {}
    for (n1, n2), coefficient in plane.items():
        term = series2_mul(
            series2_monomial(
                n1,
                n2,
                coefficient * sp.Integer(4) ** (n1 + n2),
                order,
            ),
            series2_unit_power(x_unit, sp.Integer(n1), order),
            order,
        )
        term = series2_mul(
            term,
            series2_unit_power(y_unit, sp.Integer(n2), order),
            order,
        )
        descendant = series2_add(descendant, term, order)
    return series2_mul(prefactor, descendant, order)


def mp_number(value: str | sp.Expr) -> mp.mpf:
    if isinstance(value, str):
        return mp.mpf(value)
    return mp.mpf(str(sp.N(value, DECIMAL_DIGITS + 10)))


def background_data(central_charge: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    q_background = mp.sqrt((central_charge - 1) / 6)
    b = (q_background + mp.sqrt(q_background**2 - 4)) / 2
    return q_background, b


def degenerate_weight(r: int, s: int, q_background: mp.mpf, b: mp.mpf) -> mp.mpf:
    momentum = r * b + s / b
    return (q_background**2 - momentum**2) / 4


def a_rs(r: int, s: int, b: mp.mpf) -> mp.mpf:
    value = mp.mpf("0.5")
    for p in range(1 - r, r + 1):
        for ell in range(1 - s, s + 1):
            if (p, ell) in {(0, 0), (r, s)}:
                continue
            value /= p * b + ell / b
    return value


def fusion_polynomial(
    r: int,
    s: int,
    *,
    top: mp.mpf,
    bottom: mp.mpf,
    q_background: mp.mpf,
    b: mp.mpf,
) -> mp.mpf | mp.mpc:
    lambda_top = mp.sqrt(q_background**2 - 4 * top)
    lambda_bottom = mp.sqrt(q_background**2 - 4 * bottom)
    value: mp.mpf | mp.mpc = mp.mpf(1)
    for p in range(1 - r, r, 2):
        for ell in range(1 - s, s, 2):
            shift = p * b + ell / b
            value *= (lambda_top + lambda_bottom + shift) / 2
            value *= (lambda_top - lambda_bottom + shift) / 2
    return value


def proposed_coefficients(
    order: int,
    *,
    central_charge: mp.mpf,
    external_weights: tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf, mp.mpf],
    internal_weights: tuple[mp.mpf, mp.mpf],
    denominator_audit: list[mp.mpf] | None = None,
) -> dict[tuple[int, int], mp.mpf | mp.mpc]:
    d1, d2, d3, d4, d5 = external_weights
    h1, h2 = internal_weights
    initial_a = h2 - h1
    q_background, b = background_data(central_charge)

    @functools.lru_cache(maxsize=None)
    def coefficient(
        n1: int,
        n2: int,
        current_h: mp.mpf | mp.mpc,
        current_a: mp.mpf | mp.mpc,
    ) -> mp.mpf | mp.mpc:
        total: mp.mpf | mp.mpc = mp.mpf(1) if (n1, n2) == (0, 0) else mp.mpf(0)
        for r in range(1, n1 + 1):
            for s in range(1, n1 // r + 1):
                level = r * s
                pole = degenerate_weight(r, s, q_background, b)
                denominator = current_h - pole
                if denominator_audit is not None:
                    denominator_audit.append(abs(denominator))
                residue = (
                    mp.mpf(4) ** level
                    * a_rs(r, s, b)
                    * fusion_polynomial(
                        r,
                        s,
                        top=d1,
                        bottom=d2,
                        q_background=q_background,
                        b=b,
                    )
                    * fusion_polynomial(
                        r,
                        s,
                        top=pole + current_a,
                        bottom=d3,
                        q_background=q_background,
                        b=b,
                    )
                    / denominator
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
                pole = degenerate_weight(r, s, q_background, b)
                denominator = current_h + current_a - pole
                if denominator_audit is not None:
                    denominator_audit.append(abs(denominator))
                residue = (
                    mp.mpf(4) ** level
                    * a_rs(r, s, b)
                    * fusion_polynomial(
                        r,
                        s,
                        top=d5,
                        bottom=d4,
                        q_background=q_background,
                        b=b,
                    )
                    * fusion_polynomial(
                        r,
                        s,
                        top=pole - current_a,
                        bottom=d3,
                        q_background=q_background,
                        b=b,
                    )
                    / denominator
                )
                total += residue * coefficient(
                    n1,
                    n2 - level,
                    pole - current_a,
                    current_a + level,
                )
        return total

    return {
        (n1, n2): coefficient(n1, n2, h1, initial_a)
        for n1 in range(order + 1)
        for n2 in range(order + 1 - n1)
    }


def relative_error(value: mp.mpf | mp.mpc, target: mp.mpf | mp.mpc) -> mp.mpf:
    return abs(value - target) / max(mp.mpf(1), abs(target))


def compare_case(case: Mapping[str, object], order: int) -> tuple[mp.mpf, tuple[int, int]]:
    c_exact = sp.Rational(str(case["central_charge"]))
    external_exact = tuple(sp.Rational(value) for value in case["external_weights"])
    internal_exact = tuple(sp.Rational(value) for value in case["internal_weights"])
    if len(set(external_exact + internal_exact)) != 7:
        raise AssertionError("genericity audit failed: the seven weights are not distinct")
    if internal_exact[0] == internal_exact[1]:
        raise AssertionError("genericity audit failed: h1=h2")

    start = time.perf_counter()
    plane = direct_pbw_coefficients(
        order,
        c=c_exact,
        h1=internal_exact[0],
        h2=internal_exact[1],
        weights=external_exact,
    )
    pbw_seconds = time.perf_counter() - start
    start = time.perf_counter()
    direct_exact = direct_reduced_coefficients(
        order,
        plane=plane,
        central_charge=c_exact,
        external_weights=external_exact,
        internal_weights=internal_exact,
    )
    transform_seconds = time.perf_counter() - start

    c_mp = mp_number(c_exact)
    external_mp = tuple(mp_number(value) for value in external_exact)
    internal_mp = tuple(mp_number(value) for value in internal_exact)
    recursive = proposed_coefficients(
        order,
        central_charge=c_mp,
        external_weights=external_mp,
        internal_weights=internal_mp,
        denominator_audit=(denominator_audit := []),
    )
    direct = {key: mp_number(value) for key, value in direct_exact.items()}
    zero_coefficients = [
        key for key, value in direct_exact.items() if key != (0, 0) and value == 0
    ]
    if zero_coefficients:
        raise AssertionError(
            f"genericity audit failed: vanishing PBW coefficients {zero_coefficients}"
        )
    minimum_coefficient = min(
        abs(value) for key, value in direct.items() if key != (0, 0)
    )
    minimum_denominator = min(denominator_audit)
    if minimum_denominator < mp.mpf("1e-4"):
        raise AssertionError(
            "genericity audit failed: recursion sample is too close to a Kac pole"
        )
    errors = {key: relative_error(direct[key], recursive[key]) for key in recursive}
    worst_key = max(errors, key=errors.get)

    literal_exact = direct_reduced_coefficients(
        2,
        plane={key: value for key, value in plane.items() if sum(key) <= 2},
        central_charge=c_exact,
        external_weights=external_exact,
        internal_weights=internal_exact,
        literal_e103_pairing=True,
    )
    literal_mismatch = mp_number(literal_exact[(1, 1)]) - recursive[(1, 1)]
    literal_target = 16 * (external_mp[1] - external_mp[4])

    _, _, _, _, chi_inverse = pillow_map_units(4)
    direct_order4 = {key: value for key, value in direct_exact.items() if sum(key) <= 4}
    hybrid = series2_mul(direct_order4, chi_inverse, 4)
    hybrid_mismatch = mp_number(hybrid[(2, 2)]) - recursive[(2, 2)]

    print(
        f"  exact PBW={pbw_seconds:.2f}s, exact pillow transform={transform_seconds:.2f}s"
    )
    print(
        f"  genericity: a={mp.nstr(internal_mp[1]-internal_mp[0], 8)}, "
        f"min |PBW coefficient|={mp.nstr(minimum_coefficient, 8)}, "
        f"min |recursion denominator|={mp.nstr(minimum_denominator, 8)}"
    )
    print(
        f"  max relative error={mp.nstr(errors[worst_key], 8)} at {worst_key}; "
        f"direct={mp.nstr(direct[worst_key], 16)}, "
        f"recursion={mp.nstr(recursive[worst_key], 16)}"
    )
    print(
        "  E.103 mismatch error="
        f"{mp.nstr(abs(literal_mismatch-literal_target), 8)}; "
        "hybrid-character mismatch error="
        f"{mp.nstr(abs(hybrid_mismatch+mp.mpf('0.5')), 8)}"
    )
    return errors[worst_key], worst_key


def main() -> None:
    mp.mp.dps = DECIMAL_DIGITS
    coefficient_count = (ORDER + 1) * (ORDER + 2) // 2
    tolerance = mp.mpf("1e-60")
    print("sphere five-point pillow h-recursion: high-precision PBW check")
    print(
        f"total bidegree n1+n2 <= {ORDER}: {coefficient_count} coefficients/case, "
        f"{DECIMAL_DIGITS}-digit recursion"
    )
    worst = mp.mpf(0)
    worst_case = 0
    worst_key = (0, 0)
    for index, case in enumerate(CASES, start=1):
        print(
            f"case {index}: c={case['central_charge']}, "
            f"d={case['external_weights']}, h={case['internal_weights']}"
        )
        error, key = compare_case(case, ORDER)
        if error > worst:
            worst = error
            worst_case = index
            worst_key = key
    print(
        f"global maximum relative error={mp.nstr(worst, 8)} "
        f"at case {worst_case}, coefficient {worst_key}"
    )
    if worst > tolerance:
        raise AssertionError(
            f"order-{ORDER} recursion check failed tolerance {mp.nstr(tolerance, 3)}"
        )
    print(f"all order-{ORDER} high-precision PBW checks passed")


if __name__ == "__main__":
    main()
