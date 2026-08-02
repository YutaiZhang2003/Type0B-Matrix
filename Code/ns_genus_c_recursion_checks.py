#!/usr/bin/env python3
"""Low-level checks for the all-NS genus-g c-recursion note.

The script tests ingredients that are independent of the plumbing graph:

* the NS Kac-pole equation and its c-plane Jacobian;
* the osp(1|2) edge metric and the sphere global-block coefficients;
* suppression of the first non-global state at level 3/2 as c -> infinity;
* the ungraded/graded NS vacuum character in the plumbing convention.

Twice-levels are integers.  Thus ``twice_level=3`` means level 3/2.
The script intentionally does not assume a higher-genus super-Schottky
product; that product must be checked against direct vacuum sewing first.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Union

from superconformal_blocks import NSSphereFourPointBlock


Number = Union[complex, float]


def rising(value: Number, order: int) -> complex:
    """Return the rising Pochhammer symbol ``(value)_order``."""

    if order < 0:
        raise ValueError("order must be non-negative")
    result = 1.0 + 0.0j
    for offset in range(order):
        result *= complex(value) + offset
    return result


def osp_edge_norm(weight: Number, n: int, epsilon: int) -> complex:
    r"""Return the norm of L_-1^n G_-1/2^epsilon |h>.

    The exact global metric is

        n! (2 h)_(n+epsilon),   epsilon in {0,1}.
    """

    if n < 0:
        raise ValueError("n must be non-negative")
    if epsilon not in (0, 1):
        raise ValueError("epsilon must be 0 or 1")
    return math.factorial(n) * rising(2.0 * complex(weight), n + epsilon)


def sphere_bottom_global_coefficient(
    *,
    twice_level: int,
    internal_weight: Number,
    h1: Number,
    h2: Number,
    h3: Number,
    h4: Number,
) -> complex:
    """Bottom-component four-point osp(1|2) coefficient at one level."""

    if twice_level < 0:
        raise ValueError("twice_level must be non-negative")
    epsilon = twice_level % 2
    n = twice_level // 2
    h = complex(internal_weight)
    left = rising(h + complex(h3) - complex(h4) + epsilon / 2.0, n)
    right = rising(h + complex(h2) - complex(h1) + epsilon / 2.0, n)
    return left * right / osp_edge_norm(h, n, epsilon)


def ns_degenerate_weight(r: int, s: int, b: Number) -> complex:
    """Return h_(r,s)(c(b)) in the ordinary-c convention."""

    b = complex(b)
    q_background = b + 1.0 / b
    degenerate_momentum = r * b + s / b
    return (q_background * q_background - degenerate_momentum * degenerate_momentum) / 8.0


@dataclass(frozen=True)
class NSPoleData:
    r: int
    s: int
    h: complex
    b: complex
    b_squared: complex
    c: complex
    dc_dh: complex
    jacobian: complex


def ns_c_pole(r: int, s: int, weight: Number) -> NSPoleData:
    """Return the branch c_(r,s)(h) used by the project NS recursion."""

    if r < 2 or s < 1 or (r + s) % 2:
        raise ValueError("NS c-poles require r>=2, s>=1, and r+s even")
    h = complex(weight)
    discriminant = cmath.sqrt(
        16.0 * h * h + 8.0 * (r * s - 1.0) * h + (r - s) ** 2
    )
    x = -(4.0 * h + r * s - 1.0 + discriminant) / (r * r - 1.0)
    b = cmath.sqrt(x)
    c_value = 7.5 + 3.0 * x + 3.0 / x
    dx_dh = -(
        4.0 + (16.0 * h + 4.0 * (r * s - 1.0)) / discriminant
    ) / (r * r - 1.0)
    dc_dh = 3.0 * (1.0 - 1.0 / (x * x)) * dx_dh
    return NSPoleData(
        r=r,
        s=s,
        h=h,
        b=b,
        b_squared=x,
        c=c_value,
        dc_dh=dc_dh,
        jacobian=-dc_dh,
    )


def ns_inverse_null_slope(r: int, s: int, b: Number) -> complex:
    """Return A_(r,s) in the Belavin-Geiko/HJS normalization."""

    if r < 1 or s < 1 or (r + s) % 2:
        raise ValueError("NS labels require positive r,s with r+s even")
    b = complex(b)
    result = 0.5 + 0.0j
    for p in range(1 - r, r + 1):
        for q in range(1 - s, s + 1):
            if (p + q) % 2 or (p, q) in ((0, 0), (r, s)):
                continue
            result *= math.sqrt(2.0) / (p * b + q / b)
    return result


def momentum_from_weight(weight: Number, b: Number) -> complex:
    """Choose the principal lambda satisfying h=(Q^2-lambda^2)/8."""

    b = complex(b)
    return cmath.sqrt((b + 1.0 / b) ** 2 - 8.0 * complex(weight))


def ns_fusion_polynomial(
    *,
    r: int,
    s: int,
    alpha: int,
    first_weight: Number,
    second_weight: Number,
    b: Number,
) -> complex:
    """Return P_(r,s)^alpha for an ordered pair of NS weights."""

    if alpha not in (0, 1):
        raise ValueError("alpha must be 0 or 1")
    if r < 1 or s < 1 or (r + s) % 2:
        raise ValueError("NS labels require positive r,s with r+s even")
    b = complex(b)
    lambda_i = momentum_from_weight(first_weight, b)
    lambda_j = momentum_from_weight(second_weight, b)
    congruence = 2 if alpha == 0 else 0
    denominator = 2.0 * math.sqrt(2.0)
    result = 1.0 + 0.0j
    for p in range(1 - r, r, 2):
        for q in range(1 - s, s, 2):
            if (p + q - r - s) % 4 != congruence:
                continue
            shift = p * b + q / b
            result *= (lambda_i - lambda_j + shift) / denominator
            result *= (lambda_i + lambda_j + shift) / denominator
    return result


def ns_vacuum_character_coefficients(
    max_twice_level: int, *, lift_sign: int = 1
) -> tuple[int, ...]:
    r"""Return coefficients of prod_(n>=2) (1+s q^(n-1/2))/(1-q^n)."""

    if max_twice_level < 0:
        raise ValueError("max_twice_level must be non-negative")
    if lift_sign not in (-1, 1):
        raise ValueError("lift_sign must be +1 or -1")
    coefficients = [0] * (max_twice_level + 1)
    coefficients[0] = 1
    for bosonic_mode in range(4, max_twice_level + 1, 2):
        for level in range(bosonic_mode, max_twice_level + 1):
            coefficients[level] += coefficients[level - bosonic_mode]
    for fermionic_mode in range(3, max_twice_level + 1, 2):
        for level in range(max_twice_level, fermionic_mode - 1, -1):
            coefficients[level] += lift_sign * coefficients[level - fermionic_mode]
    return tuple(coefficients)


def enumerate_vacuum_fock_levels(
    max_twice_level: int, *, lift_sign: int = 1
) -> tuple[int, ...]:
    """Independently enumerate NS vacuum Fock occupations through a cutoff."""

    states: list[tuple[int, int]] = [(0, 1)]
    for bosonic_mode in range(4, max_twice_level + 1, 2):
        enlarged: list[tuple[int, int]] = []
        for level, multiplicity in states:
            occupation = 0
            while level + occupation * bosonic_mode <= max_twice_level:
                enlarged.append((level + occupation * bosonic_mode, multiplicity))
                occupation += 1
        states = enlarged
    for fermionic_mode in range(3, max_twice_level + 1, 2):
        states = states + [
            (level + fermionic_mode, lift_sign * multiplicity)
            for level, multiplicity in states
            if level + fermionic_mode <= max_twice_level
        ]
    coefficients = [0] * (max_twice_level + 1)
    for level, multiplicity in states:
        coefficients[level] += multiplicity
    return tuple(coefficients)


def level_three_half_gram(c: Number, weight: Number) -> tuple[tuple[complex, complex], ...]:
    r"""Gram matrix in {G_-3/2|h>, L_-1 G_-1/2|h>} at level 3/2."""

    c = complex(c)
    h = complex(weight)
    return (
        (2.0 * h + 2.0 * c / 3.0, 4.0 * h),
        (4.0 * h, 2.0 * h * (2.0 * h + 1.0)),
    )


def invert_two_by_two(matrix: tuple[tuple[complex, complex], ...]) -> tuple[tuple[complex, complex], ...]:
    """Invert a nonsingular 2 by 2 matrix."""

    (a, b), (c, d) = matrix
    determinant = a * d - b * c
    if determinant == 0:
        raise ZeroDivisionError("singular 2 by 2 matrix")
    return ((d / determinant, -b / determinant), (-c / determinant, a / determinant))


def bilinear(
    left: Iterable[Number],
    matrix: tuple[tuple[complex, complex], ...],
    right: Iterable[Number],
) -> complex:
    """Return left^T matrix right for two-component vectors."""

    left_values = tuple(complex(value) for value in left)
    right_values = tuple(complex(value) for value in right)
    if len(left_values) != 2 or len(right_values) != 2:
        raise ValueError("bilinear expects two-component vectors")
    return sum(
        left_values[row] * matrix[row][column] * right_values[column]
        for row in range(2)
        for column in range(2)
    )


@dataclass(frozen=True)
class CheckSummary:
    pole_equation_error: float
    jacobian_relative_error: float
    max_global_seed_error: float
    level_three_half_error_c1: float
    level_three_half_error_c2: float
    vacuum_character_match: bool
    first_ungraded_vacuum_coefficients: tuple[int, ...]
    first_graded_vacuum_coefficients: tuple[int, ...]


def run_checks() -> CheckSummary:
    """Run all analytic/numerical checks and return a compact summary."""

    pole = ns_c_pole(3, 1, 0.83)
    pole_error = abs(ns_degenerate_weight(3, 1, pole.b) - pole.h)
    step = 1.0e-6
    c_plus = ns_c_pole(3, 1, pole.h + step).c
    c_minus = ns_c_pole(3, 1, pole.h - step).c
    finite_difference = (c_plus - c_minus) / (2.0 * step)
    jacobian_error = abs((finite_difference - pole.dc_dh) / pole.dc_dh)

    weights = dict(h1=0.37, h2=0.61, h3=0.48, h4=0.29)
    block = NSSphereFourPointBlock(
        c=51.0,
        internal_weight=0.83,
        star2=False,
        star3=False,
        **weights,
    )
    global_errors = []
    for twice_level in range(9):
        expected = sphere_bottom_global_coefficient(
            twice_level=twice_level,
            internal_weight=0.83,
            **weights,
        )
        global_errors.append(abs(expected - block.seed_coefficient(twice_level)))

    h = 0.83
    left = (1.2, 0.7)
    right = (-0.3, 1.1)
    global_level_three_half = left[1] * right[1] / osp_edge_norm(h, 1, 1)
    finite_c_values = []
    for c_value in (1.0e3, 1.0e6):
        inverse = invert_two_by_two(level_three_half_gram(c_value, h))
        finite_c_values.append(bilinear(left, inverse, right))
    suppression_errors = [
        abs(value - global_level_three_half) for value in finite_c_values
    ]

    ungraded = ns_vacuum_character_coefficients(16, lift_sign=1)
    graded = ns_vacuum_character_coefficients(16, lift_sign=-1)
    vacuum_match = (
        ungraded == enumerate_vacuum_fock_levels(16, lift_sign=1)
        and graded == enumerate_vacuum_fock_levels(16, lift_sign=-1)
    )

    summary = CheckSummary(
        pole_equation_error=float(pole_error),
        jacobian_relative_error=float(jacobian_error),
        max_global_seed_error=float(max(global_errors)),
        level_three_half_error_c1=float(suppression_errors[0]),
        level_three_half_error_c2=float(suppression_errors[1]),
        vacuum_character_match=vacuum_match,
        first_ungraded_vacuum_coefficients=ungraded[:13],
        first_graded_vacuum_coefficients=graded[:13],
    )

    if summary.pole_equation_error > 1.0e-11:
        raise AssertionError("the c-pole branch does not solve h_(r,s)(c)=h")
    if summary.jacobian_relative_error > 1.0e-7:
        raise AssertionError("the analytic c-pole Jacobian failed its finite-difference check")
    if summary.max_global_seed_error > 1.0e-13:
        raise AssertionError("the osp sphere coefficients disagree with the production seed")
    if not summary.level_three_half_error_c2 < summary.level_three_half_error_c1 / 100.0:
        raise AssertionError("the non-global level-3/2 contribution is not suppressed at large c")
    if not summary.vacuum_character_match:
        raise AssertionError("the NS vacuum product disagrees with direct Fock enumeration")
    return summary


def _json_default(value: object) -> object:
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()
    summary = run_checks()
    if args.json:
        print(json.dumps(asdict(summary), indent=2, default=_json_default))
        return
    print("all-NS c-recursion checks: PASS")
    for key, value in asdict(summary).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
