#!/usr/bin/env python3
"""Exact three-way all-NS theta-block check through total level 2.

The three computations are

1. direct super-Virasoro PBW sewing;
2. the fixed-weight Zamolodchikov ``c``-recursion; and
3. the two-Virasoro branching formula, with both ordinary Virasoro blocks
   evaluated by their own ``c``-recursions, divided by the auxiliary NS
   Majorana block using the ordinary formal-series product of the human note.

All coefficients are exact SymPy expressions.  The comparison is made after
the common Liouville parametrization

    c = 3/2 + 3 Q^2,       h_i = Q^2/8 - P_i^2/2,
    Q = b + b^{-1}.

Twice-levels are used throughout.  Generic symbolic identities are checked
through physical total level ``3/2``.  A separate exact-rational extension
checks all 35 coefficients through total level ``2``; it sees the
super-Virasoro ``(2,2)`` residue and the ordinary Virasoro ``(2,1)`` residue
in each Virasoro factor, with branch labels through ``k_i=2 n_i=+/-2``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import json
from itertools import product
import math
from typing import Mapping, Sequence

import sympy as sp

from ns_genus2_symbolic_low_order import (
    C,
    ExactDirectThetaOracle,
    ExactThetaRecursion,
    H0,
    H1,
    HINF,
    falling,
    level_tuples,
    rising,
    theta_orientation_sign,
)
from python.free_majorana_pair_of_pants import (
    majorana_three_point,
    ns_fermion_states_at_twice_level,
)


B, P0, P1, PINF = sp.symbols("b P_0 P_1 P_infinity", nonzero=True)
Q = B + 1 / B
C_LIOUVILLE = sp.Rational(3, 2) + 3 * Q**2


def ns_weight(momentum: sp.Expr) -> sp.Expr:
    """NS weight in the ordinary-central-charge convention."""

    return sp.cancel(Q**2 / 8 - momentum**2 / 2)


WEIGHTS = (ns_weight(P0), ns_weight(P1), ns_weight(PINF))


def _integer_part_of_half_integer(numerator: int) -> int:
    """Return ``Int(numerator/2)`` with truncation toward zero."""

    magnitude = abs(int(numerator)) // 2
    return -magnitude if numerator < 0 else magnitude


def s_even(x: sp.Expr, r: int, b: sp.Expr = B) -> sp.Expr:
    """Exact symbolic even blow-up product."""

    r = int(r)
    if r < 0:
        sign = -1 if r % 2 else 1
        return sp.expand(sign * s_even(b + 1 / b - x, -r, b))
    value = sp.Pow(2, -sp.Rational(r * r, 2))
    for i in range(1, 2 * r):
        for j in range(1, 2 * r - i + 1):
            if (i + j) % 2 == 0:
                value *= x + (i - 1) * b + (j - 1) / b
    return sp.factor(value)


def s_odd(x: sp.Expr, r: int, b: sp.Expr = B) -> sp.Expr:
    """Exact symbolic odd blow-up product."""

    r = int(r)
    if r < 0:
        return s_odd(b + 1 / b - x, -r, b)
    value = sp.Pow(2, -sp.Rational(r * (r + 1), 2))
    for i in range(1, 2 * r + 1):
        for j in range(1, 2 * r - i + 2):
            if (i + j) % 2 == 1:
                value *= x + (i - 1) * b + (j - 1) / b
    return sp.factor(value)


def blow_up_factor(
    *,
    alpha: sp.Expr,
    middle_label: int,
    bra_momentum: sp.Expr,
    bra_label: int,
    ket_momentum: sp.Expr,
    ket_label: int,
    b: sp.Expr = B,
) -> sp.Expr:
    r"""Return ``l(alpha,middle_label|P_bra,k_bra,P_ket,k_ket)``."""

    even_channel = (middle_label + bra_label + ket_label) % 2 == 0
    value = sp.S.One
    for sigma, tau in product((-1, 1), repeat=2):
        x = alpha + sigma * bra_momentum + tau * ket_momentum
        twice_r = middle_label + sigma * bra_label + tau * ket_label
        if even_channel:
            if twice_r % 2:
                raise ArithmeticError("invalid even blow-up-product parity")
            value *= s_even(x, twice_r // 2, b)
        else:
            if twice_r % 2 == 0:
                raise ArithmeticError("invalid odd blow-up-product parity")
            value *= s_odd(x, _integer_part_of_half_integer(twice_r), b)
    return sp.factor(value)


def branch_norm(momentum: sp.Expr, label: int, b: sp.Expr = B) -> sp.Expr:
    """Exact norm ``N_k(P)`` from the identity specialization."""

    return sp.factor(
        s_even(2 * momentum, label, b)
        * s_even(-2 * momentum, -label, b)
    )


def branching_coefficient_squared(
    *,
    momenta: Sequence[sp.Expr],
    labels: Sequence[int],
    b: sp.Expr = B,
) -> sp.Expr:
    r"""Return the normalized coefficient ``B_a^2`` in trinion slot order.

    The tuple order is ``(bra, inserted, ket)``.  Thus the numerator is

    ``l(Q/2+P_inserted,k_inserted | P_bra,k_bra,P_ket,k_ket)``.
    """

    if len(momenta) != 3 or len(labels) != 3:
        raise ValueError("three momenta and three branch labels are required")
    p_bra, p_middle, p_ket = tuple(momenta)
    k_bra, k_middle, k_ket = (int(value) for value in labels)
    numerator = blow_up_factor(
        alpha=(b + 1 / b) / 2 + p_middle,
        middle_label=k_middle,
        bra_momentum=p_bra,
        bra_label=k_bra,
        ket_momentum=p_ket,
        ket_label=k_ket,
        b=b,
    )
    denominator = sp.prod(
        branch_norm(momentum, label, b)
        for momentum, label in zip(momenta, labels)
    )
    return sp.factor(sp.cancel(numerator**2 / denominator))


def two_virasoro_parameters(
    momentum: sp.Expr, label: int, b: sp.Expr = B
) -> tuple[tuple[sp.Expr, sp.Expr], tuple[sp.Expr, sp.Expr]]:
    r"""Return ``((c^(1),h_k^(1)),(c^(2),h_k^(2)))`` without radicals."""

    k = int(label)
    d1_squared = 2 - 2 * b**2
    b1_squared = 4 * b**2 / d1_squared
    q1_squared = b1_squared + 2 + 1 / b1_squared
    h1 = q1_squared / 4 - (momentum + k * b) ** 2 / d1_squared

    d2_squared = 2 - 2 / b**2
    inverse_b2_squared = 4 / (b**2 * d2_squared)
    q2_squared = inverse_b2_squared + 2 + 1 / inverse_b2_squared
    h2 = q2_squared / 4 - (momentum + k / b) ** 2 / d2_squared
    return (
        (sp.cancel(1 + 6 * q1_squared), sp.cancel(h1)),
        (sp.cancel(1 + 6 * q2_squared), sp.cancel(h2)),
    )


def virasoro_global_three_point(
    *, levels: Sequence[int], weights: Sequence[sp.Expr]
) -> sp.Expr:
    """Exact ``L_-1`` three-form in the CCY ``(infinity,one,zero)`` frame."""

    i_level, j_level, k_level = (int(value) for value in levels)
    h_bra, h_middle, h_ket = tuple(weights)
    two_edge = sp.S.Zero
    for p_level in range(min(i_level, k_level) + 1):
        two_edge += (
            sp.binomial(i_level, p_level)
            * falling(2 * h_ket + k_level - 1, p_level)
            * falling(sp.Integer(k_level), p_level)
            * rising(h_ket + h_middle - h_bra, k_level - p_level)
            * rising(
                h_bra + h_middle - h_ket + p_level - k_level,
                i_level - p_level,
            )
        )
    middle_factor = rising(
        h_bra
        + i_level
        - h_middle
        - j_level
        + 1
        - h_ket
        - k_level,
        j_level,
    )
    return sp.expand(middle_factor * two_edge)


def virasoro_global_theta_coefficient(
    *, levels: Sequence[int], weights: Sequence[sp.Expr]
) -> sp.Expr:
    """Exact global ``sl(2)`` theta-network coefficient."""

    rho = virasoro_global_three_point(levels=levels, weights=weights)
    denominator = sp.prod(
        sp.factorial(level) * rising(2 * weight, level)
        for level, weight in zip(levels, weights)
    )
    return sp.cancel(rho**2 / denominator)


def virasoro_21_pole_and_residue(
    *, edge: int, weights: Sequence[sp.Expr]
) -> tuple[sp.Expr, sp.Expr]:
    r"""Exact CCY ``(2,1)`` pole and ``-dc/dh A P^2`` residue.

    This is the first ordinary-Virasoro term in the fixed-weight
    Zamolodchikov recursion.  It is kept in the same selected ``b^2`` branch
    as :mod:`python.ccy_genus2_block`.
    """

    h_edge = weights[edge]
    x = -sp.Rational(2, 3) * (2 * h_edge + 1)
    pole_c = 13 + 6 * (x + 1 / x)
    q_squared = x + 2 + 1 / x
    fusion_pairs = (
        (weights[2], weights[1]),
        (weights[2], weights[0]),
        (weights[0], weights[1]),
    )
    top_weight, bottom_weight = fusion_pairs[edge]
    lambda_top_squared = q_squared - 4 * top_weight
    lambda_bottom_squared = q_squared - 4 * bottom_weight
    polynomial = (
        (lambda_top_squared + lambda_bottom_squared - x) ** 2
        - 4 * lambda_top_squared * lambda_bottom_squared
    ) / 16
    # For (r,s)=(2,1), the universally simplified CCY prefactor
    # -dc_(r,s)/dh A_(r,s) is exactly 2/x^2.
    residue = 2 * polynomial**2 / x**2
    return sp.cancel(pole_c), sp.factor(residue)


def ordinary_virasoro_c_recursion_series(
    *,
    c: sp.Expr,
    weights: Sequence[sp.Expr],
    max_total_level: int,
) -> dict[tuple[int, int, int], sp.Expr]:
    """Coefficientwise genus-two Virasoro ``c``-recursion through level 2."""

    cutoff = int(max_total_level)
    if cutoff < 0 or cutoff > 2:
        raise ValueError("the exact ordinary-Virasoro recursion supports levels 0..2")
    weight_tuple = tuple(weights)

    @lru_cache(maxsize=None)
    def coefficient(
        current_c: sp.Expr,
        current_weights: tuple[sp.Expr, sp.Expr, sp.Expr],
        levels: tuple[int, int, int],
    ) -> sp.Expr:
        total = virasoro_global_theta_coefficient(
            levels=levels, weights=current_weights
        )
        for edge, edge_level in enumerate(levels):
            if edge_level < 2:
                continue
            pole_c, residue = virasoro_21_pole_and_residue(
                edge=edge, weights=current_weights
            )
            shifted_levels = list(levels)
            shifted_levels[edge] -= 2
            shifted_weights = list(current_weights)
            shifted_weights[edge] += 2
            total += residue / (current_c - pole_c) * coefficient(
                pole_c,
                tuple(shifted_weights),
                tuple(shifted_levels),
            )
        return sp.factor(sp.together(total))

    return {
        levels: coefficient(c, weight_tuple, levels)
        for levels in level_tuples(cutoff)
    }


def multiply_three_variable_series(
    left: Mapping[tuple[int, int, int], sp.Expr],
    right: Mapping[tuple[int, int, int], sp.Expr],
    *,
    max_total_level: int,
) -> dict[tuple[int, int, int], sp.Expr]:
    """Multiply two total-degree-truncated three-variable series."""

    cutoff = int(max_total_level)
    result = {levels: sp.S.Zero for levels in level_tuples(cutoff)}
    for left_levels, left_value in left.items():
        for right_levels, right_value in right.items():
            levels = tuple(
                left_levels[edge] + right_levels[edge] for edge in range(3)
            )
            if sum(levels) <= cutoff:
                result[levels] += left_value * right_value
    return {
        levels: sp.factor(sp.together(value)) for levels, value in result.items()
    }


def hatted_two_virasoro_series(
    *,
    max_total_twice_level: int = 4,
    target_levels: Sequence[tuple[int, int, int]] | None = None,
) -> dict[tuple[int, int, int], sp.Expr]:
    r"""Return the branching expansion using Virasoro ``c``-recursion."""

    cutoff = int(max_total_twice_level)
    if cutoff > 4:
        raise ValueError("this exact low-order implementation supports cutoff <= 4")
    targets = tuple(target_levels) if target_levels is not None else tuple(level_tuples(cutoff))
    target_cutoff = max((sum(levels) for levels in targets), default=0)
    coefficients = {levels: sp.S.Zero for levels in targets}
    momenta = (P0, P1, PINF)
    maximum_label = math.isqrt(target_cutoff)
    label_range = tuple(range(-maximum_label, maximum_label + 1))
    for labels in product(label_range, repeat=3):
        base_levels = tuple(label * label for label in labels)
        base_total = sum(base_levels)
        if base_total > target_cutoff:
            continue
        if not any(
            all(base_levels[edge] <= target[edge] for edge in range(3))
            for target in targets
        ):
            continue
        branch = branching_coefficient_squared(momenta=momenta, labels=labels)
        # The exact PBW oracle includes the fixed theta BPZ linear bit in
        # addition to the quadratic Koszul sign.  Applying the same ledger to
        # the branching-vector parities puts all three methods in one frame.
        branch *= theta_orientation_sign(base_levels)
        copy_central_charges: list[sp.Expr] = []
        copy_weights: list[list[sp.Expr]] = [[], []]
        for momentum, label in zip(momenta, labels):
            parameters = two_virasoro_parameters(momentum, label)
            if not copy_central_charges:
                copy_central_charges.extend((parameters[0][0], parameters[1][0]))
            copy_weights[0].append(parameters[0][1])
            copy_weights[1].append(parameters[1][1])
        remaining_physical_level = (target_cutoff - base_total) // 2
        first_block = ordinary_virasoro_c_recursion_series(
            c=copy_central_charges[0],
            weights=copy_weights[0],
            max_total_level=remaining_physical_level,
        )
        second_block = ordinary_virasoro_c_recursion_series(
            c=copy_central_charges[1],
            weights=copy_weights[1],
            max_total_level=remaining_physical_level,
        )
        virasoro_product = multiply_three_variable_series(
            first_block,
            second_block,
            max_total_level=remaining_physical_level,
        )
        for descendant_levels, descendant_coefficient in virasoro_product.items():
            levels = tuple(
                base_levels[edge] + 2 * descendant_levels[edge]
                for edge in range(3)
            )
            if levels in coefficients:
                coefficients[levels] += branch * descendant_coefficient
    # The lower-order coefficients are small enough that a common denominator
    # greatly accelerates the two subsequent series divisions.  Keep the new
    # total-level-two shell factored term by term until it is actually used.
    return {
        levels: sp.together(value) if sum(levels) <= 3 else value
        for levels, value in coefficients.items()
    }


def auxiliary_majorana_series(
    *, max_total_twice_level: int = 3
) -> dict[tuple[int, int, int], sp.Expr]:
    """Directly sew the auxiliary free Majorana block coefficientwise."""

    cutoff = int(max_total_twice_level)
    states = tuple(
        ns_fermion_states_at_twice_level(level) for level in range(cutoff + 1)
    )
    coefficients: dict[tuple[int, int, int], sp.Expr] = {}
    for levels in level_tuples(cutoff):
        coefficient = 0
        for bra in states[levels[0]]:
            for middle in states[levels[1]]:
                for ket in states[levels[2]]:
                    rho = majorana_three_point(bra, middle, ket)
                    coefficient += rho * rho
        coefficients[levels] = sp.Integer(
            theta_orientation_sign(levels) * coefficient
        )
    if coefficients[(0, 0, 0)] != 1:
        raise AssertionError("auxiliary Majorana series must have unit constant term")
    return coefficients


def divide_multivariate_series(
    numerator: Mapping[tuple[int, int, int], sp.Expr],
    denominator: Mapping[tuple[int, int, int], sp.Expr],
    *,
    max_total_twice_level: int,
    target_levels: Sequence[tuple[int, int, int]] | None = None,
) -> dict[tuple[int, int, int], sp.Expr]:
    """Ordinary triangular formal-series division in three variables."""

    zero = (0, 0, 0)
    if denominator.get(zero) != 1:
        raise ValueError("series denominator must have unit constant term")
    quotient: dict[tuple[int, int, int], sp.Expr] = {}
    targets = (
        tuple(target_levels)
        if target_levels is not None
        else tuple(level_tuples(int(max_total_twice_level)))
    )
    targets = tuple(sorted(targets, key=lambda levels: (sum(levels), levels)))
    for levels in targets:
        value = numerator.get(levels, sp.S.Zero)
        for denominator_levels, denominator_coefficient in denominator.items():
            if denominator_levels == zero or denominator_coefficient == 0:
                continue
            remainder = tuple(
                levels[edge] - denominator_levels[edge] for edge in range(3)
            )
            if min(remainder) < 0 or remainder not in quotient:
                continue
            value -= denominator_coefficient * quotient[remainder]
        quotient[levels] = sp.cancel(value)
    return quotient


def audit_level_tuples(max_total_twice_level: int) -> tuple[tuple[int, int, int], ...]:
    """Return the exact coefficient set used by the three-way audit."""

    cutoff = int(max_total_twice_level)
    if cutoff <= 3:
        return tuple(level_tuples(cutoff))
    if cutoff == 4:
        return tuple(level_tuples(3))
    raise ValueError("the exact audit supports maximum twice-level 0 through 4")


@dataclass(frozen=True)
class ThreeWayCheckSummary:
    coefficient_count: int
    max_total_twice_level: int
    max_physical_total_level: str
    direct_vs_recursion_zero_count: int
    direct_vs_two_virasoro_zero_count: int
    double_virasoro_mismatch_count: int
    first_double_virasoro_mismatch: str
    checked_kac_channels: tuple[str, ...]
    auxiliary_majorana_coefficients: Mapping[str, str]
    representative_coefficients: Mapping[str, str]


@dataclass(frozen=True)
class HigherLevelCheckSummary:
    coefficient_count_per_sample: int
    top_shell_coefficient_count: int
    exact_rational_sample_count: int
    direct_vs_recursion_zero_count: int
    direct_vs_two_virasoro_zero_count: int
    double_virasoro_mismatch_count: int
    top_shell_double_virasoro_mismatch_count: int
    first_double_virasoro_mismatch: str
    maximum_physical_total_level: str
    checked_kac_channels: tuple[str, ...]
    samples: tuple[str, ...]


def run_checks(*, max_total_twice_level: int = 4) -> ThreeWayCheckSummary:
    """Run the exact coefficientwise comparison and return a compact audit."""

    cutoff = int(max_total_twice_level)
    if cutoff < 0 or cutoff > 4:
        raise ValueError("the three-way symbolic check supports cutoffs 0 through 4")
    symbolic_weights = (H0, H1, HINF)
    liouville_substitution = dict(
        zip((C, H0, H1, HINF), (C_LIOUVILLE, *WEIGHTS))
    )
    direct = ExactDirectThetaOracle(c=C, weights=symbolic_weights)
    recursive = ExactThetaRecursion()
    audited_levels = audit_level_tuples(cutoff)
    hatted = hatted_two_virasoro_series(
        max_total_twice_level=cutoff,
        target_levels=audited_levels,
    )
    majorana = auxiliary_majorana_series(max_total_twice_level=cutoff)
    ordinary_quotient = divide_multivariate_series(
        hatted,
        majorana,
        max_total_twice_level=cutoff,
        target_levels=audited_levels,
    )
    direct_vs_recursion: dict[tuple[int, int, int], sp.Expr] = {}
    direct_vs_two_virasoro: dict[tuple[int, int, int], sp.Expr] = {}
    direct_coefficients: dict[tuple[int, int, int], sp.Expr] = {}
    for levels in audited_levels:
        direct_generic = sp.cancel(direct.coefficient(levels))
        parity = sum(levels) % 2
        recursion_generic = recursive.coefficient(
            c=C,
            weights=symbolic_weights,
            levels=levels,
            sectors=(parity, parity),
        )
        recursion_difference = sp.cancel(
            sp.together(direct_generic - recursion_generic)
        )
        direct_value = sp.cancel(direct_generic.subs(liouville_substitution))
        direct_coefficients[levels] = direct_value
        direct_vs_recursion[levels] = recursion_difference
        ordinary_difference = direct_value - ordinary_quotient[levels]
        direct_vs_two_virasoro[levels] = sp.cancel(
            sp.together(ordinary_difference)
        )

    bad_recursion = {
        levels: value for levels, value in direct_vs_recursion.items() if value != 0
    }
    if bad_recursion:
        levels, value = next(iter(bad_recursion.items()))
        raise AssertionError(
            f"direct/c-recursion mismatch at twice-level {levels}: {sp.factor(value)}"
        )
    representative_levels = tuple(
        levels
        for levels in ((0, 0, 0), (1, 0, 0), (1, 1, 0), (3, 0, 0), (1, 1, 1))
        if sum(levels) <= cutoff
    )
    nonzero_majorana = {
        str(levels): str(value)
        for levels, value in majorana.items()
        if value != 0
    }
    two_virasoro_mismatches = {
        levels: sp.factor(value)
        for levels, value in direct_vs_two_virasoro.items()
        if value != 0
    }
    first_two_virasoro_mismatch = next(
        iter(two_virasoro_mismatches.items()), None
    )
    return ThreeWayCheckSummary(
        coefficient_count=len(direct_coefficients),
        max_total_twice_level=cutoff,
        max_physical_total_level=str(
            sp.Rational(max(sum(levels) for levels in audited_levels), 2)
        ),
        direct_vs_recursion_zero_count=sum(
            value == 0 for value in direct_vs_recursion.values()
        ),
        direct_vs_two_virasoro_zero_count=sum(
            value == 0 for value in direct_vs_two_virasoro.values()
        ),
        double_virasoro_mismatch_count=len(two_virasoro_mismatches),
        first_double_virasoro_mismatch=(
            "none"
            if first_two_virasoro_mismatch is None
            else (
                f"{first_two_virasoro_mismatch[0]}: "
                f"{first_two_virasoro_mismatch[1]}"
            )
        ),
        checked_kac_channels=(
            ("SCA (3,1)",)
            if max(sum(levels) for levels in audited_levels) >= 3
            else ()
        ),
        auxiliary_majorana_coefficients=nonzero_majorana,
        representative_coefficients={
            str(levels): str(sp.factor(direct_coefficients[levels]))
            for levels in representative_levels
        },
    )


DEFAULT_LEVEL_TWO_SAMPLES: tuple[
    tuple[sp.Rational, sp.Rational, sp.Rational, sp.Rational], ...
] = (
    (
        sp.Rational(3, 2),
        sp.Rational(2, 7),
        sp.Rational(3, 11),
        sp.Rational(5, 13),
    ),
    (
        sp.Rational(4, 3),
        sp.Rational(1, 5),
        sp.Rational(2, 9),
        sp.Rational(4, 11),
    ),
    (
        sp.Rational(5, 4),
        -sp.Rational(2, 7),
        sp.Rational(1, 6),
        sp.Rational(3, 10),
    ),
)


def run_level_two_exact_samples(
    *,
    samples: Sequence[
        tuple[sp.Rational, sp.Rational, sp.Rational, sp.Rational]
    ] = DEFAULT_LEVEL_TWO_SAMPLES,
) -> HigherLevelCheckSummary:
    """Check the complete level-two truncation at generic exact-rational data.

    The lower 20 identities are proved as generic rational functions by
    :func:`run_checks`.  This extension evaluates all 35 coefficients through
    total physical level two, including all 15 coefficients on the new shell,
    at several nonsingular rational points.  Every operation remains exact;
    this is deliberately reported as an exact sample check rather than a
    generic symbolic proof.
    """

    sample_values = tuple(tuple(sp.Rational(value) for value in sample) for sample in samples)
    if not sample_values:
        raise ValueError("at least one exact-rational sample is required")
    full_levels = tuple(level_tuples(4))
    hatted = hatted_two_virasoro_series(
        max_total_twice_level=4,
        target_levels=full_levels,
    )
    majorana = auxiliary_majorana_series(max_total_twice_level=4)
    symbolic_weights = (H0, H1, HINF)
    direct = ExactDirectThetaOracle(c=C, weights=symbolic_weights)
    recursive = ExactThetaRecursion()
    direct_generic = {
        levels: sp.cancel(direct.coefficient(levels)) for levels in full_levels
    }
    recursive_generic = {
        levels: recursive.coefficient(
            c=C,
            weights=symbolic_weights,
            levels=levels,
            sectors=(sum(levels) % 2, sum(levels) % 2),
        )
        for levels in full_levels
    }

    recursion_zero_count = 0
    two_virasoro_zero_count = 0
    two_virasoro_mismatch_count = 0
    top_shell_two_virasoro_mismatch_count = 0
    first_two_virasoro_mismatch = "none"
    sample_labels: list[str] = []
    for b_value, p0_value, p1_value, pinf_value in sample_values:
        q_value = b_value + 1 / b_value
        c_value = sp.Rational(3, 2) + 3 * q_value**2
        weight_values = (
            q_value**2 / 8 - p0_value**2 / 2,
            q_value**2 / 8 - p1_value**2 / 2,
            q_value**2 / 8 - pinf_value**2 / 2,
        )
        direct_substitution = dict(
            zip((C, H0, H1, HINF), (c_value, *weight_values))
        )
        branch_substitution = {
            B: b_value,
            P0: p0_value,
            P1: p1_value,
            PINF: pinf_value,
        }
        direct_values = {
            levels: sp.cancel(value.subs(direct_substitution))
            for levels, value in direct_generic.items()
        }
        hatted_values = {
            levels: sp.cancel(value.subs(branch_substitution))
            for levels, value in hatted.items()
        }
        ordinary_quotient = divide_multivariate_series(
            hatted_values,
            majorana,
            max_total_twice_level=4,
            target_levels=full_levels,
        )
        for levels in full_levels:
            recursion_value = sp.cancel(
                recursive_generic[levels].subs(direct_substitution)
            )
            recursion_difference = sp.cancel(
                direct_values[levels] - recursion_value
            )
            if recursion_difference != 0:
                raise AssertionError(
                    "exact sampled direct/c-recursion mismatch at "
                    f"sample {(b_value, p0_value, p1_value, pinf_value)}, "
                    f"twice-level {levels}: {recursion_difference}"
                )
            recursion_zero_count += 1

            two_virasoro_difference = sp.cancel(
                direct_values[levels] - ordinary_quotient[levels]
            )
            if two_virasoro_difference == 0:
                two_virasoro_zero_count += 1
            else:
                two_virasoro_mismatch_count += 1
                if sum(levels) == 4:
                    top_shell_two_virasoro_mismatch_count += 1
                if first_two_virasoro_mismatch == "none":
                    first_two_virasoro_mismatch = (
                        f"sample {(b_value, p0_value, p1_value, pinf_value)}, "
                        f"twice-level {levels}: {two_virasoro_difference}"
                    )
        sample_labels.append(
            f"b={b_value}, P=({p0_value},{p1_value},{pinf_value})"
        )

    return HigherLevelCheckSummary(
        coefficient_count_per_sample=len(full_levels),
        top_shell_coefficient_count=sum(sum(levels) == 4 for levels in full_levels),
        exact_rational_sample_count=len(sample_values),
        direct_vs_recursion_zero_count=recursion_zero_count,
        direct_vs_two_virasoro_zero_count=two_virasoro_zero_count,
        double_virasoro_mismatch_count=two_virasoro_mismatch_count,
        top_shell_double_virasoro_mismatch_count=(
            top_shell_two_virasoro_mismatch_count
        ),
        first_double_virasoro_mismatch=first_two_virasoro_mismatch,
        maximum_physical_total_level="2",
        checked_kac_channels=(
            "SCA (3,1)",
            "SCA (2,2)",
            "Vir (2,1) in both factors",
        ),
        samples=tuple(sample_labels),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-twice-level",
        type=int,
        default=4,
        help="maximum total twice-level (supported: 0 through 4)",
    )
    parser.add_argument(
        "--skip-level-two-samples",
        action="store_true",
        help="skip the complete 35-coefficient exact-rational level-two extension",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary = run_checks(max_total_twice_level=args.max_twice_level)
    higher = (
        run_level_two_exact_samples()
        if args.max_twice_level >= 4 and not args.skip_level_two_samples
        else None
    )
    if args.json:
        payload = {"symbolic": summary.__dict__}
        if higher is not None:
            payload["level_two_exact_samples"] = higher.__dict__
        print(json.dumps(payload, indent=2))
        return
    print("human-convention all-NS theta-block audit: COMPLETE")
    print(
        "  exact zero identities (direct vs c-recursion): "
        f"{summary.direct_vs_recursion_zero_count}/{summary.coefficient_count}"
    )
    print(
        "  exact zero identities (direct vs ordinary-quotient two Virasoro): "
        f"{summary.direct_vs_two_virasoro_zero_count}/{summary.coefficient_count}"
    )
    print(
        "  ordinary-product double-Virasoro mismatches: "
        f"{summary.double_virasoro_mismatch_count}/{summary.coefficient_count}"
    )
    if summary.double_virasoro_mismatch_count:
        print(
            "  first mismatch: "
            f"{summary.first_double_virasoro_mismatch}"
        )
    print(f"  maximum physical total level: {summary.max_physical_total_level}")
    print(f"  Kac channels reached: {', '.join(summary.checked_kac_channels)}")
    print(f"  auxiliary Majorana series: {dict(summary.auxiliary_majorana_coefficients)}")
    print("  representative common coefficients:")
    for levels, value in summary.representative_coefficients.items():
        print(f"    {levels}: {value}")
    if higher is not None:
        identity_count = (
            higher.coefficient_count_per_sample
            * higher.exact_rational_sample_count
        )
        print("complete level-two exact-rational audit: COMPLETE")
        print(
            "  exact zero identities (direct vs c-recursion): "
            f"{higher.direct_vs_recursion_zero_count}/{identity_count}"
        )
        print(
            "  exact zero identities (direct vs ordinary-quotient two Virasoro): "
            f"{higher.direct_vs_two_virasoro_zero_count}/{identity_count}"
        )
        print(
            "  ordinary-product double-Virasoro mismatches: "
            f"{higher.double_virasoro_mismatch_count}/{identity_count}"
        )
        print(
            "  new-shell double-Virasoro mismatches: "
            f"{higher.top_shell_double_virasoro_mismatch_count}/"
            f"{higher.top_shell_coefficient_count * higher.exact_rational_sample_count}"
        )
        if higher.double_virasoro_mismatch_count:
            print(f"  first mismatch: {higher.first_double_virasoro_mismatch}")
        print(
            "  coefficients per sample / new top shell: "
            f"{higher.coefficient_count_per_sample}/{higher.top_shell_coefficient_count}"
        )
        for sample in higher.samples:
            print(f"  sample: {sample}")


if __name__ == "__main__":
    main()
