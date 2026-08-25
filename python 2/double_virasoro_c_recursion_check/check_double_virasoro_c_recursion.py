#!/usr/bin/env python3
"""Compare the all-NS theta block from c-recursion and two Virasoro algebras.

The comparison retains every monomial of total physical level at most four.
Internally, series exponents are twice-levels, so no floating-point decision is
used to distinguish integer and half-integer powers.
"""

from __future__ import annotations

import argparse
import cmath
import itertools
import json
import math
import sys
from functools import lru_cache
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
PLUMBING = REPOSITORY / "plumbing"
if str(PLUMBING) not in sys.path:
    sys.path.insert(0, str(PLUMBING))

from virasoro_plumbing_graph import (  # noqa: E402
    direct_plumbing_graph_block,
    genus2_theta_graph,
)

from formal_series import (  # noqa: E402
    Exponent,
    Series,
    ZERO,
    add,
    clean,
    inverse,
    multiply,
    scale,
    shift,
    theta_inverse,
    theta_multiply,
    total_degree,
)
MAX_TOTAL_TWICE_LEVEL = 8
SQRT2 = math.sqrt(2.0)

# The generic rational test point used in the accompanying note.
B = 2.0
Q = B + 1.0 / B
C = 81.0 / 4.0
MOMENTA = (1.0 / 3.0, 2.0 / 5.0, 3.0 / 5.0)
WEIGHTS = tuple(0.5 * (Q * Q / 4.0 - momentum * momentum) for momentum in MOMENTA)

# Correlated branches.  They obey h_n^(1)+h_n^(2)=h+2n^2.
B1 = -1j * math.sqrt(8.0 / 3.0)
B2 = math.sqrt(3.0 / 2.0)
DENOMINATOR1 = 1j * math.sqrt(6.0)
DENOMINATOR2 = math.sqrt(3.0 / 2.0)
C1 = -21.0 / 4.0
C2 = 26.0

def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def rising(value: complex, order: int) -> complex:
    result = 1.0 + 0.0j
    for index in range(order):
        result *= value + index
    return result


def falling(value: complex, order: int) -> complex:
    result = 1.0 + 0.0j
    for index in range(order):
        result *= value - index
    return result


def ccy_s(first_level: int, third_level: int, h1: complex, h2: complex, h3: complex) -> complex:
    result = 0.0j
    for contracted in range(min(first_level, third_level) + 1):
        result += (
            math.comb(first_level, contracted)
            * falling(2.0 * h3 + third_level - 1.0, contracted)
            * falling(third_level, contracted)
            * rising(h3 + h2 - h1, third_level - contracted)
            * rising(
                h1 + h2 - h3 + contracted - third_level,
                first_level - contracted,
            )
        )
    return result


def global_ground_value(parity: int, occupations: tuple[int, int, int], weights: tuple[complex, complex, complex]) -> complex:
    h1, h2, h3 = weights
    if parity == 0:
        return {
            (0, 0, 0): 1.0,
            (1, 1, 0): h1 + h2 - h3,
            (1, 0, 1): h1 - h2 + h3,
            (0, 1, 1): h1 - h2 - h3,
        }.get(occupations, 0.0)
    return {
        (1, 0, 0): 1.0,
        (0, 1, 0): 1.0,
        (0, 0, 1): -1.0,
        (1, 1, 1): -(h1 + h2 + h3 - 0.5),
    }.get(occupations, 0.0)


def global_osp_series(
    parity: int,
    weights: tuple[complex, complex, complex],
    lifts: tuple[int, int, int],
    max_total_twice_level: int,
) -> Series:
    """Equation (3.10) of the main notes as a coefficient dictionary."""

    result: Series = {}
    for occupations in itertools.product((0, 1), repeat=3):
        if sum(occupations) % 2 != parity:
            continue
        ground = global_ground_value(parity, occupations, weights)
        if ground == 0:
            continue
        theta_sign = (-1) ** (
            occupations[0] * occupations[1]
            + occupations[0] * occupations[2]
            + occupations[1] * occupations[2]
        )
        ground_lift = math.prod(lifts[index] ** occupations[index] for index in range(3))
        for first in range(max_total_twice_level // 2 + 1):
            for second in range(max_total_twice_level // 2 + 1):
                for third in range(max_total_twice_level // 2 + 1):
                    levels = (first, second, third)
                    exponent = tuple(2 * levels[index] + occupations[index] for index in range(3))
                    if total_degree(exponent) > max_total_twice_level:
                        continue
                    shifted_weights = tuple(weights[index] + occupations[index] / 2.0 for index in range(3))
                    rho = (
                        ground
                        * ccy_s(first, third, *shifted_weights)
                        * rising(
                            shifted_weights[0]
                            + first
                            - shifted_weights[1]
                            - second
                            + 1.0
                            - shifted_weights[2]
                            - third,
                            second,
                        )
                    )
                    denominator = math.prod(
                        math.factorial(levels[index])
                        * rising(2.0 * weights[index], levels[index] + occupations[index])
                        for index in range(3)
                    )
                    coefficient = theta_sign * ground_lift * rho * rho / denominator
                    result[exponent] = result.get(exponent, 0.0j) + coefficient
    return clean(result)


def vacuum_series(lifts: tuple[int, int, int]) -> Series:
    """Large-c NS vacuum quotient through total level four.

    The table is in the block order ``(infinity, one, zero)``.  The extra
    CCY infinity-tube linear spin-frame sign has been removed, leaving the
    quadratic theta orientation used in Section 6.
    """

    monomials: dict[Exponent, float] = {
        ZERO: 1.0,
        (3, 3, 0): -1.0,
        (3, 0, 3): -1.0,
        (0, 3, 3): -1.0,
        (5, 3, 0): -3.0,
        (4, 4, 0): 1.0,
        (4, 0, 4): 1.0,
        (0, 4, 4): 1.0,
        (0, 5, 3): -3.0,
        (0, 3, 5): -3.0,
    }
    return {
        exponent: coefficient
        * math.prod(lifts[index] ** exponent[index] for index in range(3))
        for exponent, coefficient in monomials.items()
    }


def regular_seed_series(
    parity: int,
    weights: tuple[complex, complex, complex],
    lifts: tuple[int, int, int],
    max_total_twice_level: int,
) -> Series:
    """Large-c seed, including the graded vacuum/global convolution."""

    total: Series = {}
    for occupations in itertools.product((0, 1), repeat=3):
        if sum(occupations) % 2 != parity:
            continue
        global_piece = global_osp_series(
            parity,
            weights,
            lifts,
            max_total_twice_level,
        )
        # Select only this one component sector from the global sum.
        global_piece = {
            exponent: coefficient
            for exponent, coefficient in global_piece.items()
            if tuple(value % 2 for value in exponent) == occupations
        }
        shifted_lifts = tuple(((-1) ** occupations[index]) * lifts[index] for index in range(3))
        vacuum_piece = vacuum_series(shifted_lifts)
        total = add(total, multiply(vacuum_piece, global_piece, max_total_twice_level))
    return total


def ns_x_rs(r: int, s: int, weight: complex) -> complex:
    radical = (r - s) ** 2 + 8.0 * (r * s - 1.0) * weight + 16.0 * weight * weight
    return (r * s - 1.0 + 4.0 * weight + cmath.sqrt(radical)) / (1.0 - r * r)


def ns_c_rs(r: int, s: int, weight: complex) -> complex:
    x = ns_x_rs(r, s, weight)
    return 7.5 + 3.0 * (x + 1.0 / x)


def ns_dc_rs_dh(r: int, s: int, weight: complex) -> complex:
    x = ns_x_rs(r, s, weight)
    dh_dx = ((1.0 - r * r) - (1.0 - s * s) / (x * x)) / 8.0
    dc_dx = 3.0 * (1.0 - 1.0 / (x * x))
    return dc_dx / dh_dx


def ns_a_rs(r: int, s: int, b: complex) -> complex:
    result = 0.5 + 0.0j
    for p in range(1 - r, r + 1):
        for q in range(1 - s, s + 1):
            if (p + q) % 2 != 0 or (p, q) in {(0, 0), (r, s)}:
                continue
            result /= (p * b + q / b) / SQRT2
    return result


def ns_momentum_from_weight(weight: complex, b: complex) -> complex:
    background = b + 1.0 / b
    return cmath.sqrt(background * background - 8.0 * weight)


def ns_fusion_polynomial(
    r: int,
    s: int,
    parity: int,
    first_weight: complex,
    second_weight: complex,
    b: complex,
) -> complex:
    first = ns_momentum_from_weight(first_weight, b)
    second = ns_momentum_from_weight(second_weight, b)
    required = 2 if parity == 0 else 0
    result = 1.0 + 0.0j
    for p in range(1 - r, r, 2):
        for q in range(1 - s, s, 2):
            if (p + q - (r + s)) % 4 != required:
                continue
            lattice = p * b + q / b
            result *= (first - second + lattice) / (2.0 * SQRT2)
            result *= (first + second + lattice) / (2.0 * SQRT2)
    return result


def recursion_channels(max_total_twice_level: int) -> tuple[tuple[int, int], ...]:
    channels: list[tuple[int, int]] = []
    for r in range(2, max_total_twice_level + 1):
        for s in range(1, max_total_twice_level + 1):
            if (r + s) % 2 != 0 or r * s > max_total_twice_level:
                continue
            channels.append((r, s))
    return tuple(channels)


@lru_cache(maxsize=None)
def c_recursion_series_cached(
    central_charge: complex,
    weights: tuple[complex, complex, complex],
    parity: int,
    lifts: tuple[int, int, int],
    max_total_twice_level: int,
) -> tuple[tuple[Exponent, complex], ...]:
    total = regular_seed_series(parity, weights, lifts, max_total_twice_level)
    fusion_pairs = ((weights[2], weights[1]), (weights[2], weights[0]), (weights[0], weights[1]))
    for edge in range(3):
        for r, s in recursion_channels(max_total_twice_level):
            twice_level = r * s
            if twice_level > max_total_twice_level:
                continue
            x = ns_x_rs(r, s, weights[edge])
            b_pole = cmath.sqrt(x)
            pole = ns_c_rs(r, s, weights[edge])
            polynomial = ns_fusion_polynomial(
                r,
                s,
                parity,
                fusion_pairs[edge][0],
                fusion_pairs[edge][1],
                b_pole,
            )
            residue = (
                -ns_dc_rs_dh(r, s, weights[edge])
                * ns_a_rs(r, s, b_pole)
                * polynomial
                * polynomial
                / (central_charge - pole)
            )
            shifted_weights = list(weights)
            shifted_weights[edge] += twice_level / 2.0
            shifted_lifts = list(lifts)
            shifted_parity = parity
            if twice_level % 2:
                shifted_parity = 1 - parity
                for other in range(3):
                    if other != edge:
                        shifted_lifts[other] *= -1
            subblock = dict(
                c_recursion_series_cached(
                    pole,
                    tuple(shifted_weights),
                    shifted_parity,
                    tuple(shifted_lifts),
                    max_total_twice_level - twice_level,
                )
            )
            edge_shift = tuple(twice_level if index == edge else 0 for index in range(3))
            lift_factor = lifts[edge] ** twice_level
            total = add(
                total,
                scale(
                    shift(subblock, edge_shift, max_total_twice_level),
                    residue * lift_factor,
                ),
            )
    return tuple(sorted(total.items()))


def c_recursion_series(
    parity: int,
    lifts: tuple[int, int, int],
    max_total_twice_level: int = MAX_TOTAL_TWICE_LEVEL,
) -> Series:
    return dict(
        c_recursion_series_cached(
            complex(C),
            tuple(complex(weight) for weight in WEIGHTS),
            parity,
            lifts,
            max_total_twice_level,
        )
    )


def ell(x: complex, m: int) -> complex:
    if m == 0:
        return 1.0 + 0.0j
    if m < 0:
        reflected = ell(Q - x, -m)
        return ((-1) ** ((-m) // 2)) * reflected if m % 2 == 0 else reflected
    result = 2.0 ** (1.0 / 8.0) if m % 2 else 1.0
    required = m % 2
    for r in range(m):
        for s in range(m - r):
            if (r + s) % 2 != required:
                continue
            result *= x + r * B + s / B
    return result


def canonical_branching_coefficient_squared(labels: tuple[int, int, int]) -> complex:
    """Evaluate the compact four-product branching coefficient."""

    # Section 6 states the product for nonnegative labels.  A negative label
    # is first reflected by (P_i,n_i)->(-P_i,-n_i) on that leg.
    effective_momenta = tuple(
        -momentum if label < 0 else momentum
        for momentum, label in zip(MOMENTA, labels)
    )
    effective_labels = tuple(abs(label) for label in labels)
    n = tuple(label / 2.0 for label in effective_labels)
    parity = sum(labels) % 2
    combinations = (
        (1, 1, 1),
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
    )
    product_value = 1.0 + 0.0j
    for signs in combinations:
        argument = Q / 2.0 + sum(
            signs[index] * effective_momenta[index] for index in range(3)
        )
        product_index = int(round(2.0 * sum(signs[index] * n[index] for index in range(3))))
        product_value *= ell(argument, product_index)
    denominator = 1.0 + 0.0j
    for index in range(3):
        denominator *= ell(2.0 * effective_momenta[index], 2 * effective_labels[index])
        denominator *= ell(
            Q + 2.0 * effective_momenta[index], 2 * effective_labels[index]
        )
    return ((-1) ** parity) * product_value * product_value / denominator


def branch_weight(momentum: float, label: int, copy: int) -> complex:
    n = label / 2.0
    if copy == 1:
        branch_momentum = momentum / DENOMINATOR1 + n * B1
        background = B1 + 1.0 / B1
    elif copy == 2:
        branch_momentum = momentum / DENOMINATOR2 + n / B2
        background = B2 + 1.0 / B2
    else:
        raise ValueError("copy must be 1 or 2")
    return 0.25 * background * background - branch_momentum * branch_momentum


@lru_cache(maxsize=None)
def ordinary_virasoro_series(labels: tuple[int, int, int], copy: int, max_integer_level: int) -> tuple[tuple[Exponent, complex], ...]:
    weights = tuple(branch_weight(MOMENTA[index], labels[index], copy) for index in range(3))
    central_charge = C1 if copy == 1 else C2
    result = direct_plumbing_graph_block(
        genus2_theta_graph(),
        central_charge=central_charge,
        edge_weights=weights,
        q_values=(1.0, 1.0, 1.0),
        max_total_level=max_integer_level,
    )
    series = {
        tuple(2 * level for level in levels): coefficient
        for levels, coefficient in result.coefficient_by_levels.items()
    }
    return tuple(sorted(series.items()))


def _pfaffian(matrix: list[list[complex]]) -> complex:
    size = len(matrix)
    if size == 0:
        return 1.0 + 0.0j
    if size % 2:
        return 0.0j
    result = 0.0j
    for column in range(1, size):
        minor = [
            [matrix[row][other] for other in range(size) if other not in (0, column)]
            for row in range(size)
            if row not in (0, column)
        ]
        result += ((-1) ** (column + 1)) * matrix[0][column] * _pfaffian(minor)
    return result


def fermion_pair(first: tuple[int, int], second: tuple[int, int]) -> complex:
    """Pair two local fermion modes labelled by ``(slot, derivative order)``."""

    slot1, order1 = first
    slot2, order2 = second
    if slot1 == slot2:
        return 0.0j
    if slot1 > slot2:
        return -fermion_pair(second, first)
    if (slot1, slot2) == (0, 1):
        return -math.comb(order1, order2) if order2 <= order1 else 0.0
    if (slot1, slot2) == (0, 2):
        return -1.0 if order1 == order2 else 0.0
    if (slot1, slot2) == (1, 2):
        return ((-1) ** order1) * math.comb(order1 + order2, order1)
    raise AssertionError("unreachable fermion slot pair")


def fermion_three_point(states: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]) -> complex:
    insertions: list[tuple[int, int]] = []
    for slot, state in enumerate(states):
        # A state stores descending twice-modes; this is also the operator order.
        insertions.extend((slot, (mode - 1) // 2) for mode in state)
    matrix = [[0.0j for _ in insertions] for _ in insertions]
    for row in range(len(insertions)):
        for column in range(row + 1, len(insertions)):
            matrix[row][column] = fermion_pair(insertions[row], insertions[column])
            matrix[column][row] = -matrix[row][column]
    return _pfaffian(matrix)


def strict_fermion_states(max_twice_level: int) -> tuple[tuple[int, ...], ...]:
    odd_modes = tuple(range(1, max_twice_level + 1, 2))
    states: list[tuple[int, ...]] = []
    for occupied in itertools.product((0, 1), repeat=len(odd_modes)):
        state = tuple(reversed([mode for mode, use in zip(odd_modes, occupied) if use]))
        if sum(state) <= max_twice_level:
            states.append(state)
    return tuple(states)


def free_fermion_series(lifts: tuple[int, int, int], max_total_twice_level: int) -> Series:
    states = strict_fermion_states(max_total_twice_level)
    result: Series = {}
    for first in states:
        for second in states:
            for third in states:
                exponent = (sum(first), sum(second), sum(third))
                if total_degree(exponent) > max_total_twice_level:
                    continue
                parities = (len(first) % 2, len(second) % 2, len(third) % 2)
                if sum(parities) % 2:
                    continue
                rho = fermion_three_point((first, second, third))
                if rho == 0:
                    continue
                theta_sign = (-1) ** (
                    parities[0] * parities[1]
                    + parities[0] * parities[2]
                    + parities[1] * parities[2]
                )
                inverse_gram_sign = (-1) ** sum(parities)
                lift = math.prod(lifts[index] ** parities[index] for index in range(3))
                coefficient = theta_sign * inverse_gram_sign * lift * rho * rho
                result[exponent] = result.get(exponent, 0.0j) + coefficient
    return clean(result)


def double_virasoro_series(
    parity: int,
    lifts: tuple[int, int, int],
    max_total_twice_level: int = MAX_TOTAL_TWICE_LEVEL,
    *,
    ordinary_fermion_division: bool = False,
) -> Series:
    """Resolve the enlarged block and remove the auxiliary fermion factor.

    The default uses the theta-graded product.  Ordinary formal division is
    kept only to reproduce the first sign failure discussed in the note.
    """

    enlarged: Series = {}
    label_bound = math.isqrt(max_total_twice_level)
    for labels in itertools.product(range(-label_bound, label_bound + 1), repeat=3):
        base_exponent = tuple(label * label for label in labels)
        base_total = total_degree(base_exponent)
        if base_total > max_total_twice_level or sum(labels) % 2 != parity:
            continue
        remaining_integer_level = (max_total_twice_level - base_total) // 2
        first = dict(ordinary_virasoro_series(labels, 1, remaining_integer_level))
        second = dict(ordinary_virasoro_series(labels, 2, remaining_integer_level))
        product = multiply(first, second, max_total_twice_level - base_total)
        branch_sign = (-1) ** (
            labels[0] * labels[1]
            + labels[0] * labels[2]
            + labels[1] * labels[2]
        )
        lift = math.prod(lifts[index] ** labels[index] for index in range(3))
        branching = canonical_branching_coefficient_squared(labels)
        prefactor = branch_sign * lift * branching
        enlarged = add(
            enlarged,
            scale(shift(product, base_exponent, max_total_twice_level), prefactor),
        )
    auxiliary = free_fermion_series(lifts, max_total_twice_level)
    if ordinary_fermion_division:
        return multiply(
            enlarged,
            inverse(auxiliary, max_total_twice_level),
            max_total_twice_level,
        )
    return theta_multiply(
        enlarged,
        theta_inverse(auxiliary, max_total_twice_level),
        max_total_twice_level,
    )


def compare_series(left: Series, right: Series) -> tuple[float, float, Exponent]:
    exponents = set(left) | set(right)
    maximum_absolute = 0.0
    maximum_relative = 0.0
    worst = ZERO
    for exponent in exponents:
        difference = abs(left.get(exponent, 0.0j) - right.get(exponent, 0.0j))
        maximum_absolute = max(maximum_absolute, difference)
        scale_value = max(1.0, abs(left.get(exponent, 0.0j)), abs(right.get(exponent, 0.0j)))
        relative = difference / scale_value
        if relative > maximum_relative:
            maximum_relative = relative
            worst = exponent
    return maximum_absolute, maximum_relative, worst


def first_difference(left: Series, right: Series, tolerance: float) -> Exponent | None:
    for exponent in sorted(set(left) | set(right), key=lambda value: (sum(value), value)):
        scale_value = max(1.0, abs(left.get(exponent, 0.0j)), abs(right.get(exponent, 0.0j)))
        if abs(left.get(exponent, 0.0j) - right.get(exponent, 0.0j)) > tolerance * scale_value:
            return exponent
    return None


def encode_complex(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def encode_series(series: Series) -> list[dict[str, object]]:
    return [
        {
            "twice_levels": list(exponent),
            "levels": [value / 2.0 for value in exponent],
            "coefficient": encode_complex(coefficient),
        }
        for exponent, coefficient in sorted(series.items(), key=lambda item: (sum(item[0]), item[0]))
    ]


def tex_number(value: complex) -> str:
    require(abs(value.imag) < 5.0e-11, "the selected real test point produced a complex coefficient")
    real = 0.0 if abs(value.real) < 5.0e-13 else value.real
    return f"{real:.12g}"


def write_tex_tables(
    path: Path,
    master_series: dict[int, tuple[Series, Series]],
) -> None:
    lines = [
        "% Generated by check_double_virasoro_c_recursion.py; do not edit by hand.",
        "% The lift choice is (eta_1,eta_2,eta_3)=(1,1,1).",
    ]
    for parity in (0, 1):
        recursion, doubled = master_series[parity]
        lines.extend(
            [
                r"\begin{longtable}{@{}c r r@{}}",
                rf"\caption{{Coefficients of $\mathbf F_{parity}$ at $(\eta_1,\eta_2,\eta_3)=(1,1,1)$.}}",
                r"\label{tab:level-four-coefficients-" + str(parity) + r"}\\",
                r"\toprule",
                r"$(n_1,n_2,n_3)$ & $c$-recursion & double Virasoro\\",
                r"\midrule",
                r"\endfirsthead",
                r"\toprule",
                r"$(n_1,n_2,n_3)$ & $c$-recursion & double Virasoro\\",
                r"\midrule",
                r"\endhead",
                r"\midrule",
                r"\multicolumn{3}{r}{Continued on the next page}\\",
                r"\endfoot",
                r"\bottomrule",
                r"\endlastfoot",
            ]
        )
        for exponent in sorted(set(recursion) | set(doubled), key=lambda value: (sum(value), value)):
            levels = ",".join(
                str(value // 2) if value % 2 == 0 else rf"\frac{{{value}}}{{2}}"
                for value in exponent
            )
            lines.append(
                rf"$({levels})$ & ${tex_number(recursion.get(exponent, 0.0j))}$ & "
                rf"${tex_number(doubled.get(exponent, 0.0j))}$\\"
            )
        lines.extend([r"\end{longtable}", ""])
        if parity == 0:
            lines.extend([r"\clearpage", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    output_path: Path,
    tex_table_path: Path,
    tolerance: float,
    require_agreement: bool,
) -> None:
    require(abs(C1 - (1.0 + 6.0 * (B1 + 1.0 / B1) ** 2)) < 1.0e-13, "wrong c^(1)")
    require(abs(C2 - (1.0 + 6.0 * (B2 + 1.0 / B2) ** 2)) < 1.0e-13, "wrong c^(2)")
    for momentum, weight in zip(MOMENTA, WEIGHTS):
        for label in range(-2, 3):
            n = label / 2.0
            branch_sum = branch_weight(momentum, label, 1) + branch_weight(momentum, label, 2)
            require(abs(branch_sum - (weight + 2.0 * n * n)) < 2.0e-13, "branch-weight sum failed")

    ledger: list[dict[str, object]] = []
    master_series: dict[int, tuple[Series, Series]] = {}
    all_pass = True
    for lifts in itertools.product((1, -1), repeat=3):
        auxiliary = free_fermion_series(lifts, MAX_TOTAL_TWICE_LEVEL)
        auxiliary_inverse = theta_inverse(auxiliary, MAX_TOTAL_TWICE_LEVEL)
        inverse_check = theta_multiply(
            auxiliary,
            auxiliary_inverse,
            MAX_TOTAL_TWICE_LEVEL,
        )
        require(
            compare_series(inverse_check, {ZERO: 1.0})[1] < 5.0e-13,
            "theta-graded fermion inverse failed",
        )
        for parity in (0, 1):
            recursion = c_recursion_series(parity, lifts)
            doubled = double_virasoro_series(parity, lifts)
            if lifts == (1, 1, 1):
                master_series[parity] = (recursion, doubled)
            absolute, relative, worst = compare_series(recursion, doubled)
            first = first_difference(recursion, doubled, tolerance)
            passed = relative < tolerance
            all_pass = all_pass and passed
            print(
                f"eta={lifts} a={parity}: max_abs={absolute:.6e} "
                f"max_scaled={relative:.6e} worst={worst} "
                f"coefficients={len(set(recursion) | set(doubled))}"
            )
            ledger.append(
                {
                    "lifts": list(lifts),
                    "parity": parity,
                    "maximum_absolute_difference": absolute,
                    "maximum_scaled_difference": relative,
                    "worst_twice_levels": list(worst),
                    "first_different_twice_levels": None if first is None else list(first),
                    "coefficient_count": len(set(recursion) | set(doubled)),
                    "passed": passed,
                    "c_recursion": encode_series(recursion),
                    "double_virasoro": encode_series(doubled),
                }
            )

    ordinary_diagnostics: list[dict[str, object]] = []
    unit_lifts = (1, 1, 1)
    for parity in (0, 1):
        recursion = master_series[parity][0]
        ordinary = double_virasoro_series(
            parity,
            unit_lifts,
            ordinary_fermion_division=True,
        )
        absolute, relative, worst = compare_series(recursion, ordinary)
        first = first_difference(recursion, ordinary, tolerance)
        require(first is not None, "ordinary division unexpectedly agreed")
        ordinary_diagnostics.append(
            {
                "parity": parity,
                "first_different_twice_levels": list(first),
                "c_recursion_coefficient": encode_complex(recursion.get(first, 0.0j)),
                "ordinary_division_coefficient": encode_complex(ordinary.get(first, 0.0j)),
                "maximum_absolute_difference": absolute,
                "maximum_scaled_difference": relative,
                "worst_twice_levels": list(worst),
            }
        )
    output = {
        "description": "All-NS genus-two theta block through total level four",
        "max_total_level": 4,
        "parameters": {
            "b": B,
            "c": C,
            "momenta": list(MOMENTA),
            "weights": list(WEIGHTS),
            "c1": C1,
            "c2": C2,
        },
        "tolerance": tolerance,
        "all_passed": all_pass,
        "factorization_product": (
            "theta-graded convolution polarized from the theta sewing sign"
        ),
        "ordinary_division_diagnostic": ordinary_diagnostics,
        "comparisons": ledger,
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    write_tex_tables(tex_table_path, master_series)
    print(f"wrote {output_path}")
    print(f"wrote {tex_table_path}")
    if require_agreement:
        require(all_pass, "double-Virasoro and c-recursion series do not agree")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "level4_results.json",
        help="full coefficient ledger",
    )
    parser.add_argument("--tolerance", type=float, default=2.0e-9)
    parser.add_argument(
        "--tex-table",
        type=Path,
        default=HERE / "level4_tables.tex",
        help="complete coefficient tables for the accompanying note",
    )
    parser.add_argument(
        "--require-agreement",
        action="store_true",
        help="return a failing exit status if any coefficient differs",
    )
    arguments = parser.parse_args()
    run(
        arguments.output.resolve(),
        arguments.tex_table.resolve(),
        arguments.tolerance,
        arguments.require_agreement,
    )


if __name__ == "__main__":
    main()
