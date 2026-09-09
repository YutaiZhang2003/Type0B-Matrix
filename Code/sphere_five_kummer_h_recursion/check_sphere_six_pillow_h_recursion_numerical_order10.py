#!/usr/bin/env python3
"""High-precision order-ten PBW check of the sphere six-point pillow recursion.

For each generic sample, the plane comb block is built from exact rational
Virasoro PBW Gram matrices and three-point tensors.  The complete
sphere-to-pillow conversion is also performed with exact rational
trivariate series.  Only the independent three-edge h-recursion is evaluated
numerically, at 80 decimal digits.  All 286 coefficients with
``n1+n2+n3 <= 10`` are compared.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Mapping

import mpmath as mp
import sympy as sp

from check_pillow_h_recursion_numerical_order10 import (
    a_rs,
    background_data,
    degenerate_weight,
    fusion_polynomial,
    mp_number,
    relative_error,
)
from check_sphere_six_pillow_h_recursion_symbolic_order3 import (
    direct_pbw_coefficients,
    direct_reduced_pillow_coefficients,
)


ORDER = 10
DECIMAL_DIGITS = 80
CASES = (
    {
        "label": "ascending moderate",
        "central_charge": "26.215",
        "external_weights": ("0.17", "0.29", "0.43", "0.58", "0.71", "0.86"),
        "internal_weights": ("0.9371", "1.0837", "1.3321"),
    },
    {
        "label": "descending then rising",
        # Non-monotone external weights and mixed-sign fixed differences.
        "central_charge": "29.3761",
        "external_weights": (
            "0.113",
            "0.367",
            "0.811",
            "1.237",
            "0.524",
            "0.946",
        ),
        "internal_weights": ("1.4193", "0.6871", "1.1098"),
    },
    {
        "label": "large positive then negative difference",
        "central_charge": "37.219",
        "external_weights": (
            "0.907",
            "0.223",
            "1.041",
            "0.376",
            "0.658",
            "1.284",
        ),
        "internal_weights": ("0.7517", "1.5372", "0.8894"),
    },
    {
        "label": "strictly descending",
        "central_charge": "26.215",
        "external_weights": ("0.17", "0.29", "0.43", "0.58", "0.71", "0.86"),
        "internal_weights": ("1.6423", "1.2711", "0.8237"),
    },
    {
        "label": "near-equal nonmonotone",
        "central_charge": "26.215",
        "external_weights": ("0.17", "0.29", "0.43", "0.58", "0.71", "0.86"),
        "internal_weights": ("0.9043", "0.9187", "0.8871"),
    },
    {
        "label": "strictly ascending wide",
        "central_charge": "29.3761",
        "external_weights": (
            "0.113",
            "0.367",
            "0.811",
            "1.237",
            "0.524",
            "0.946",
        ),
        "internal_weights": ("0.6189", "1.2473", "1.8891"),
    },
    {
        "label": "central peak",
        "central_charge": "29.3761",
        "external_weights": (
            "0.113",
            "0.367",
            "0.811",
            "1.237",
            "0.524",
            "0.946",
        ),
        "internal_weights": ("1.1267", "1.8429", "0.5679"),
    },
    {
        "label": "central valley",
        "central_charge": "37.219",
        "external_weights": (
            "0.907",
            "0.223",
            "1.041",
            "0.376",
            "0.658",
            "1.284",
        ),
        "internal_weights": ("1.7731", "0.5427", "1.2863"),
    },
    {
        "label": "near-equal descending tail",
        "central_charge": "37.219",
        "external_weights": (
            "0.907",
            "0.223",
            "1.041",
            "0.376",
            "0.658",
            "1.284",
        ),
        "internal_weights": ("1.1047", "1.1299", "1.0873"),
    },
    {
        "label": "wide peak then intermediate",
        "central_charge": "26.215",
        "external_weights": ("0.17", "0.29", "0.43", "0.58", "0.71", "0.86"),
        "internal_weights": ("0.4831", "1.9637", "1.2249"),
    },
)


def audit_fixed_difference_denominators(
    order: int,
    *,
    central_charge: mp.mpf,
    internal_weights: tuple[mp.mpf, mp.mpf, mp.mpf],
) -> tuple[mp.mpf, tuple[object, ...]]:
    """Exhaustively audit every denominator visited through ``order``.

    This follows the same three affine shift maps as the recursion but omits
    all fusion-polynomial arithmetic, so unsuitable samples are rejected
    before the expensive PBW contraction.
    """

    h1, h2, h3 = internal_weights
    initial_a2 = h2 - h1
    initial_a3 = h3 - h1
    q_background, b = background_data(central_charge)
    best_value = mp.inf
    best_context: tuple[object, ...] = ()

    @functools.lru_cache(maxsize=None)
    def visit(
        n1: int,
        n2: int,
        n3: int,
        current_h: mp.mpf | mp.mpc,
        current_a2: mp.mpf | mp.mpc,
        current_a3: mp.mpf | mp.mpc,
    ) -> None:
        nonlocal best_value, best_context
        levels = (n1, n2, n3)
        for edge, available in enumerate(levels):
            for r in range(1, available + 1):
                for s in range(1, available // r + 1):
                    level = r * s
                    pole = degenerate_weight(r, s, q_background, b)
                    if edge == 0:
                        denominator = current_h - pole
                        shifted = (
                            pole + level,
                            current_a2 - level,
                            current_a3 - level,
                        )
                    elif edge == 1:
                        denominator = current_h + current_a2 - pole
                        shifted = (
                            pole - current_a2,
                            current_a2 + level,
                            current_a3,
                        )
                    else:
                        denominator = current_h + current_a3 - pole
                        shifted = (
                            pole - current_a3,
                            current_a2,
                            current_a3 + level,
                        )
                    magnitude = abs(denominator)
                    if magnitude < best_value:
                        best_value = magnitude
                        best_context = (
                            edge + 1,
                            r,
                            s,
                            n1,
                            n2,
                            n3,
                            current_h,
                            current_a2,
                            current_a3,
                            pole,
                        )
                    remainder = list(levels)
                    remainder[edge] -= level
                    visit(*remainder, *shifted)

    for n1 in range(order + 1):
        for n2 in range(order + 1 - n1):
            for n3 in range(order + 1 - n1 - n2):
                visit(n1, n2, n3, h1, initial_a2, initial_a3)
    return best_value, best_context


def proposed_coefficients(
    order: int,
    *,
    central_charge: mp.mpf,
    external_weights: tuple[mp.mpf, ...],
    internal_weights: tuple[mp.mpf, mp.mpf, mp.mpf],
    denominator_audit: list[tuple[mp.mpf, tuple[object, ...]]] | None = None,
) -> dict[tuple[int, int, int], mp.mpf | mp.mpc]:
    """Return coefficients from the three-edge fixed-difference recursion."""

    d1, d2, d3, d4, d5, d6 = external_weights
    h1, h2, h3 = internal_weights
    initial_a2 = h2 - h1
    initial_a3 = h3 - h1
    q_background, b = background_data(central_charge)

    @functools.lru_cache(maxsize=None)
    def coefficient(
        n1: int,
        n2: int,
        n3: int,
        current_h: mp.mpf | mp.mpc,
        current_a2: mp.mpf | mp.mpc,
        current_a3: mp.mpf | mp.mpc,
    ) -> mp.mpf | mp.mpc:
        total: mp.mpf | mp.mpc = (
            mp.mpf(1) if (n1, n2, n3) == (0, 0, 0) else mp.mpf(0)
        )
        levels = (n1, n2, n3)
        for edge, available in enumerate(levels):
            for r in range(1, available + 1):
                for s in range(1, available // r + 1):
                    level = r * s
                    pole = degenerate_weight(r, s, q_background, b)
                    if edge == 0:
                        denominator = current_h - pole
                        left = fusion_polynomial(
                            r,
                            s,
                            top=d1,
                            bottom=d2,
                            q_background=q_background,
                            b=b,
                        )
                        right = fusion_polynomial(
                            r,
                            s,
                            top=pole + current_a2,
                            bottom=d3,
                            q_background=q_background,
                            b=b,
                        )
                        shifted = (
                            pole + level,
                            current_a2 - level,
                            current_a3 - level,
                        )
                        plumbing_factor = mp.mpf(4) ** level
                    elif edge == 1:
                        denominator = current_h + current_a2 - pole
                        left = fusion_polynomial(
                            r,
                            s,
                            top=pole - current_a2,
                            bottom=d3,
                            q_background=q_background,
                            b=b,
                        )
                        right = fusion_polynomial(
                            r,
                            s,
                            top=pole + current_a3 - current_a2,
                            bottom=d4,
                            q_background=q_background,
                            b=b,
                        )
                        shifted = (
                            pole - current_a2,
                            current_a2 + level,
                            current_a3,
                        )
                        plumbing_factor = mp.mpf(1)
                    else:
                        denominator = current_h + current_a3 - pole
                        left = fusion_polynomial(
                            r,
                            s,
                            top=pole + current_a2 - current_a3,
                            bottom=d4,
                            q_background=q_background,
                            b=b,
                        )
                        right = fusion_polynomial(
                            r,
                            s,
                            top=d6,
                            bottom=d5,
                            q_background=q_background,
                            b=b,
                        )
                        shifted = (
                            pole - current_a3,
                            current_a2,
                            current_a3 + level,
                        )
                        plumbing_factor = mp.mpf(4) ** level
                    if denominator_audit is not None:
                        denominator_audit.append(
                            (
                                abs(denominator),
                                (
                                    edge + 1,
                                    r,
                                    s,
                                    n1,
                                    n2,
                                    n3,
                                    current_h,
                                    current_a2,
                                    current_a3,
                                    pole,
                                ),
                            )
                        )
                    remainder = list(levels)
                    remainder[edge] -= level
                    total += (
                        plumbing_factor
                        * a_rs(r, s, b)
                        * left
                        * right
                        / denominator
                        * coefficient(*remainder, *shifted)
                    )
        return total

    return {
        (n1, n2, n3): coefficient(
            n1, n2, n3, h1, initial_a2, initial_a3
        )
        for n1 in range(order + 1)
        for n2 in range(order + 1 - n1)
        for n3 in range(order + 1 - n1 - n2)
    }


def compare_case(
    case: Mapping[str, object],
    order: int,
) -> tuple[mp.mpf, tuple[int, int, int]]:
    c_exact = sp.Rational(str(case["central_charge"]))
    external_exact = tuple(sp.Rational(value) for value in case["external_weights"])
    internal_exact = tuple(sp.Rational(value) for value in case["internal_weights"])
    all_weights = external_exact + internal_exact
    if len(set(all_weights)) != 9:
        raise AssertionError("genericity audit failed: the nine weights are not distinct")
    if len(set(internal_exact)) != 3:
        raise AssertionError("genericity audit failed: internal weights coincide")

    start = time.perf_counter()
    plane = direct_pbw_coefficients(
        order,
        c=c_exact,
        internal_weights=internal_exact,
        external_weights=external_exact,
    )
    pbw_seconds = time.perf_counter() - start
    start = time.perf_counter()
    direct_exact = direct_reduced_pillow_coefficients(
        order,
        plane_coefficients=plane,
        c=c_exact,
        internal_weights=internal_exact,
        external_weights=external_exact,
    )
    transform_seconds = time.perf_counter() - start

    c_mp = mp_number(c_exact)
    external_mp = tuple(mp_number(value) for value in external_exact)
    internal_mp = tuple(mp_number(value) for value in internal_exact)
    start = time.perf_counter()
    minimum_denominator, minimum_denominator_context = (
        audit_fixed_difference_denominators(
            order,
            central_charge=c_mp,
            internal_weights=internal_mp,
        )
    )
    audit_seconds = time.perf_counter() - start
    if minimum_denominator < mp.mpf("1e-4"):
        raise AssertionError(
            "genericity audit failed: sample is too close to a Kac pole; "
            f"context={minimum_denominator_context}"
        )
    start = time.perf_counter()
    recursive = proposed_coefficients(
        order,
        central_charge=c_mp,
        external_weights=external_mp,
        internal_weights=internal_mp,
    )
    recursion_seconds = time.perf_counter() - start
    direct = {key: mp_number(value) for key, value in direct_exact.items()}

    zero_coefficients = [
        key for key, value in direct_exact.items() if key != (0, 0, 0) and value == 0
    ]
    if zero_coefficients:
        raise AssertionError(
            f"genericity audit failed: vanishing PBW coefficients {zero_coefficients}"
        )
    minimum_coefficient = min(
        abs(value) for key, value in direct.items() if key != (0, 0, 0)
    )
    errors = {key: relative_error(direct[key], recursive[key]) for key in recursive}
    worst_key = max(errors, key=errors.get)
    a2 = internal_mp[1] - internal_mp[0]
    a3 = internal_mp[2] - internal_mp[0]
    print(
        f"  exact PBW={pbw_seconds:.2f}s, exact pillow={transform_seconds:.2f}s, "
        f"denominator audit={audit_seconds:.2f}s, "
        f"80-digit recursion={recursion_seconds:.2f}s"
    )
    print(
        f"  genericity: (a2,a3)=({mp.nstr(a2, 8)},{mp.nstr(a3, 8)}), "
        f"min |PBW coefficient|={mp.nstr(minimum_coefficient, 8)}, "
        f"min |denominator|={mp.nstr(minimum_denominator, 8)}"
    )
    print(
        f"  max relative error={mp.nstr(errors[worst_key], 8)} at {worst_key}; "
        f"direct={mp.nstr(direct[worst_key], 16)}, "
        f"recursion={mp.nstr(recursive[worst_key], 16)}"
    )
    return errors[worst_key], worst_key


def main() -> None:
    mp.mp.dps = DECIMAL_DIGITS
    coefficient_count = (ORDER + 1) * (ORDER + 2) * (ORDER + 3) // 6
    tolerance = mp.mpf("1e-58")
    print("sphere six-point pillow h-recursion: high-precision PBW check")
    print(
        f"total degree n1+n2+n3 <= {ORDER}: "
        f"{coefficient_count} coefficients/case, {DECIMAL_DIGITS}-digit recursion"
    )
    worst = mp.mpf(0)
    worst_case = 0
    worst_key = (0, 0, 0)
    for index, case in enumerate(CASES, start=1):
        print(
            f"case {index}: c={case['central_charge']}, "
            f"h={case['internal_weights']} [{case['label']}]",
            flush=True,
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
    print(f"all order-{ORDER} high-precision six-point PBW checks passed")


if __name__ == "__main__":
    main()
