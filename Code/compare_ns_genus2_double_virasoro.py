#!/usr/bin/env python3
"""Compare the genus-two all-NS c-recursion with double Virasoro.

This is the coefficientwise implementation of the mature formalism in
``Human Notes/SCblock.tex``.  In particular it uses

* the parity-operator resolution in the enlarged ``SCA x F`` module;
* the all-label NS blow-up coefficient in the double-Virasoro sum;
* ordinary Virasoro fixed-weight c-recursion in each Virasoro factor; and
* the theta-polarized ``star`` inverse of the auxiliary Majorana block.

Levels in the public comparison are physical levels.  Internally, the NS and
Majorana series use twice-level triples.  The default cutoff is total physical
level four, so the double-Virasoro sum needs branch labels ``k=2n`` only in
``{-2,-1,0,1,2}`` and ordinary Virasoro descendants through total level four.

The NS c-recursion backend, direct PBW oracle, Majorana quotient, and
double-Virasoro answer all use the literal human-note theta sign

    (-1)^(p0 p1 + p0 pinfinity + p1 pinfinity).

No direct NS PBW coefficient is used to construct the double-Virasoro answer.
The direct oracle is evaluated only as a third, independent diagnostic.
"""

from __future__ import annotations

import argparse
import cmath
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import product
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import mpmath

CODE_DIRECTORY = Path(__file__).resolve().parent
PYTHON_DIRECTORY = CODE_DIRECTORY / "python"
if str(PYTHON_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIRECTORY))

from ns_genus12_finite_c_check import (
    DirectThetaOracle,
    recursion_theta_coefficient,
)
from ccy_genus2_block import (
    rho_lminus1_triple,
    sl2_descendant_norm,
)
from free_majorana_pair_of_pants import (
    majorana_three_point,
    ns_fermion_states_at_twice_level,
)
from two_virasoro_fusion import blow_up_factor, branch_norm
from virasoro_blocks import fusion_polynomial, momentum_from_weight


Level = tuple[int, int, int]
ScalarSeries = dict[Level, complex]


def level_tuples(max_total_level: int) -> Iterable[Level]:
    """Yield nonnegative triples of total degree at most the cutoff."""

    cutoff = int(max_total_level)
    for total in range(cutoff + 1):
        for level0 in range(total + 1):
            for level1 in range(total - level0 + 1):
                yield level0, level1, total - level0 - level1


def theta_quadratic_exponent(levels: Sequence[int]) -> int:
    """Return the human theta quadratic form on level parities."""

    p0, p1, pinfinity = (int(value) % 2 for value in levels)
    return (p0 * p1 + p0 * pinfinity + p1 * pinfinity) % 2


def theta_orientation_sign(levels: Sequence[int]) -> int:
    """Even-primary specialization of the current theta orientation sign."""

    return -1 if theta_quadratic_exponent(levels) else 1


def theta_cross_exponent(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the polarization of the theta quadratic form."""

    a = tuple(int(value) % 2 for value in left)
    b = tuple(int(value) % 2 for value in right)
    return sum(
        a[i] * b[j] + b[i] * a[j]
        for i in range(3)
        for j in range(i + 1, 3)
    ) % 2


def convolve_series(
    left: Mapping[Level, complex],
    right: Mapping[Level, complex],
    *,
    cutoff: int,
) -> ScalarSeries:
    """Ordinarily multiply two total-degree-truncated three-variable series."""

    result = {levels: 0.0j for levels in level_tuples(cutoff)}
    for left_levels, left_value in left.items():
        if left_value == 0:
            continue
        for right_levels, right_value in right.items():
            levels = tuple(
                left_levels[edge] + right_levels[edge] for edge in range(3)
            )
            if sum(levels) <= cutoff:
                result[levels] += left_value * right_value
    return result


def divide_theta_star_series(
    numerator: Mapping[Level, complex],
    denominator: Mapping[Level, complex],
    *,
    cutoff: int,
) -> ScalarSeries:
    """Triangular division in the human note's theta ``star`` algebra."""

    zero = (0, 0, 0)
    if abs(denominator.get(zero, 0.0) - 1.0) > 1.0e-14:
        raise ValueError("the auxiliary Majorana block must have constant term one")
    quotient: ScalarSeries = {}
    for levels in level_tuples(cutoff):
        value = complex(numerator.get(levels, 0.0))
        for denominator_levels, denominator_value in denominator.items():
            if denominator_levels == zero or denominator_value == 0:
                continue
            remainder = tuple(
                levels[edge] - denominator_levels[edge] for edge in range(3)
            )
            if min(remainder) < 0 or remainder not in quotient:
                continue
            sign = -1 if theta_cross_exponent(denominator_levels, remainder) else 1
            value -= sign * denominator_value * quotient[remainder]
        quotient[levels] = value
    return quotient


def divide_ordinary_series(
    numerator: Mapping[Level, complex],
    denominator: Mapping[Level, complex],
    *,
    cutoff: int,
) -> ScalarSeries:
    """Triangular ordinary division, retained as a diagnostic control."""

    zero = (0, 0, 0)
    if abs(denominator.get(zero, 0.0) - 1.0) > 1.0e-14:
        raise ValueError("the auxiliary Majorana block must have constant term one")
    quotient: ScalarSeries = {}
    for levels in level_tuples(cutoff):
        value = complex(numerator.get(levels, 0.0))
        for denominator_levels, denominator_value in denominator.items():
            if denominator_levels == zero or denominator_value == 0:
                continue
            remainder = tuple(
                levels[edge] - denominator_levels[edge] for edge in range(3)
            )
            if min(remainder) < 0 or remainder not in quotient:
                continue
            value -= denominator_value * quotient[remainder]
        quotient[levels] = value
    return quotient


def auxiliary_majorana_series(*, cutoff: int) -> ScalarSeries:
    """Sew the human-note NS Majorana block coefficient by coefficient."""

    states = tuple(
        ns_fermion_states_at_twice_level(level) for level in range(cutoff + 1)
    )
    coefficients: ScalarSeries = {}
    for levels in level_tuples(cutoff):
        coefficient = 0
        for state0 in states[levels[0]]:
            for state1 in states[levels[1]]:
                for state_infinity in states[levels[2]]:
                    rho = majorana_three_point(state0, state1, state_infinity)
                    coefficient += rho * rho
        coefficients[levels] = complex(
            theta_orientation_sign(levels) * coefficient
        )
    if coefficients[(0, 0, 0)] != 1:
        raise AssertionError("Majorana sewing did not have unit constant term")
    return coefficients


def virasoro_global_coefficient(
    *, weights: Sequence[complex], levels: Sequence[int]
) -> complex:
    """Global sl(2) theta-network coefficient in the CCY slot order."""

    i, j, k = (int(value) for value in levels)
    h0, h1, hinfinity = (complex(value) for value in weights)
    rho = rho_lminus1_triple(i, j, k, h0, h1, hinfinity)
    denominator = math.prod(
        sl2_descendant_norm(weight, level)
        for weight, level in zip((h0, h1, hinfinity), (i, j, k))
    )
    return rho * rho / denominator


# Through total Virasoro level four these are the only nonconstant terms of
# the large-c theta vacuum block.  They follow independently from direct
# vacuum-Verma sewing and from the primitive Schottky product.
VIRASORO_VACUUM_SEED_LEVEL4: Mapping[Level, complex] = {
    (0, 0, 0): 1.0,
    (0, 2, 2): 1.0,
    (2, 0, 2): 1.0,
    (2, 2, 0): 1.0,
}


def virasoro_regular_coefficient(
    *, weights: Sequence[complex], levels: Sequence[int]
) -> complex:
    """Large-c Virasoro coefficient in the Schottky frame through level four."""

    total = 0.0j
    for vacuum_levels, vacuum_value in VIRASORO_VACUUM_SEED_LEVEL4.items():
        remainder = tuple(
            int(levels[edge]) - vacuum_levels[edge] for edge in range(3)
        )
        if min(remainder) < 0:
            continue
        total += vacuum_value * virasoro_global_coefficient(
            weights=weights, levels=remainder
        )
    return total


def _virasoro_fusion_pair(
    weights: Sequence[complex], edge: int
) -> tuple[complex, complex]:
    if edge == 0:
        return complex(weights[2]), complex(weights[1])
    if edge == 1:
        return complex(weights[2]), complex(weights[0])
    if edge == 2:
        return complex(weights[0]), complex(weights[1])
    raise ValueError("a theta graph has three edges")


def virasoro_b_square_rs_from_h(r: int, s: int, h: complex) -> complex:
    """Stable CCY branch of the fixed-weight Virasoro pole parameter.

    The usual radical formula suffers exact cancellation for ``s=1`` when
    ``rs-1+2h`` is negative and all inputs are real.  The analytic CCY branch
    is linear in that case, so evaluate it before taking a principal square
    root.  For ``s>1`` a rationalized fallback removes the same numerical
    cancellation without changing the chosen root.
    """

    r = int(r)
    s = int(s)
    h = complex(h)
    a = r * s - 1.0 + 2.0 * h
    if s == 1:
        return 2.0 * a / (1.0 - r * r)
    discriminant = (r - s) ** 2 + 4.0 * (r * s - 1.0) * h + 4.0 * h * h
    root = cmath.sqrt(discriminant)
    numerator = a + root
    if abs(numerator) <= 1.0e-12 * max(1.0, abs(a), abs(root)):
        return (s * s - 1.0) / (root - a)
    return numerator / (1.0 - r * r)


def virasoro_c_rs_from_h(r: int, s: int, h: complex) -> complex:
    x = virasoro_b_square_rs_from_h(r, s, h)
    if x == 0:
        raise ZeroDivisionError(
            f"zero Virasoro b^2 at (r,s)=({r},{s}), h={h!r}"
        )
    return 13.0 + 6.0 * (x + 1.0 / x)


def virasoro_minus_dc_dh_times_a_rs(r: int, s: int, h: complex) -> complex:
    """Stable universal ``-dc/dh A_rs`` using the production cancellation."""

    x = virasoro_b_square_rs_from_h(r, s, h)
    numerator = -12.0 * x ** (2 * r * s - 1)
    denominator = (1.0 - r * r) * x * x - (1.0 - s * s)
    denominator_factors = [
        (p, ell)
        for p in range(1 - r, r + 1)
        for ell in range(1 - s, s + 1)
        if (p, ell) not in {(0, 0), (r, s)}
    ]
    remaining_numerator_factors = []
    for num_p, num_ell in ((1, -1), (1, 1)):
        matched = None
        matched_scale = 1
        for index, (den_p, den_ell) in enumerate(denominator_factors):
            if den_p and den_p * num_ell == den_ell * num_p:
                matched = index
                matched_scale = den_p // num_p
                break
        if matched is None:
            remaining_numerator_factors.append((num_p, num_ell))
        else:
            denominator *= matched_scale
            denominator_factors.pop(matched)
    for p, ell in remaining_numerator_factors:
        numerator *= p * x + ell
    for p, ell in denominator_factors:
        denominator *= p * x + ell
    if denominator == 0:
        raise ZeroDivisionError(
            f"singular Virasoro residue at (r,s)=({r},{s}), h={h!r}"
        )
    return numerator / denominator


def virasoro_residue_prefactor_for_weights(
    r: int,
    s: int,
    h_edge: complex,
    top_weight: complex,
    bottom_weight: complex,
) -> complex:
    """Return ``-dc/dh A_rs P_rs^2`` on the stable pole branch."""

    b_pole = cmath.sqrt(virasoro_b_square_rs_from_h(r, s, h_edge))
    lambda_top = momentum_from_weight(top_weight, b_pole)
    lambda_bottom = momentum_from_weight(bottom_weight, b_pole)
    polynomial = fusion_polynomial(
        r, s, b_pole, lambda_top, lambda_bottom
    )
    return (
        virasoro_minus_dc_dh_times_a_rs(r, s, h_edge)
        * polynomial
        * polynomial
    )


def ordinary_virasoro_c_recursion_series(
    *, c: complex, weights: Sequence[complex], cutoff: int
) -> ScalarSeries:
    """Ordinary genus-two Virasoro c-recursion through total level four."""

    if cutoff < 0 or cutoff > 4:
        raise ValueError("the explicit Virasoro vacuum seed supports levels 0..4")
    initial_weights = tuple(complex(value) for value in weights)

    @lru_cache(maxsize=None)
    def coefficient(
        current_c: complex,
        current_weights: tuple[complex, complex, complex],
        levels: Level,
    ) -> complex:
        total = virasoro_regular_coefficient(
            weights=current_weights, levels=levels
        )
        for edge, edge_level in enumerate(levels):
            top_weight, bottom_weight = _virasoro_fusion_pair(
                current_weights, edge
            )
            for r in range(2, edge_level + 1):
                for s in range(1, edge_level // r + 1):
                    rs = r * s
                    if rs > edge_level:
                        continue
                    pole = virasoro_c_rs_from_h(
                        r, s, current_weights[edge]
                    )
                    denominator = current_c - pole
                    if abs(denominator) < 1.0e-12:
                        raise ZeroDivisionError(
                            "coincident Virasoro c-recursion pole at "
                            f"edge={edge}, (r,s)=({r},{s}), levels={levels}"
                        )
                    residue = virasoro_residue_prefactor_for_weights(
                        r,
                        s,
                        current_weights[edge],
                        top_weight,
                        bottom_weight,
                    )
                    shifted_levels = list(levels)
                    shifted_levels[edge] -= rs
                    shifted_weights = list(current_weights)
                    shifted_weights[edge] += rs
                    total += residue / denominator * coefficient(
                        pole,
                        tuple(shifted_weights),
                        tuple(shifted_levels),
                    )
        return complex(total)

    return {
        levels: coefficient(complex(c), initial_weights, levels)
        for levels in level_tuples(cutoff)
    }


def two_virasoro_parameters(
    *, momentum: complex, label: int, b: complex
) -> tuple[tuple[complex, complex], tuple[complex, complex]]:
    """Return ``((c1,h1),(c2,h2))`` without choosing square roots."""

    momentum = complex(momentum)
    b = complex(b)
    k = int(label)
    d1_squared = 2.0 - 2.0 * b * b
    b1_squared = 4.0 * b * b / d1_squared
    q1_squared = b1_squared + 2.0 + 1.0 / b1_squared
    h1 = q1_squared / 4.0 - (momentum + k * b) ** 2 / d1_squared

    d2_squared = 2.0 - 2.0 / (b * b)
    inverse_b2_squared = 4.0 / (b * b * d2_squared)
    q2_squared = inverse_b2_squared + 2.0 + 1.0 / inverse_b2_squared
    h2 = q2_squared / 4.0 - (momentum + k / b) ** 2 / d2_squared
    return (
        (1.0 + 6.0 * q1_squared, h1),
        (1.0 + 6.0 * q2_squared, h2),
    )


def branching_coefficient_squared(
    *, b: complex, momenta: Sequence[complex], labels: Sequence[int], precision: int
) -> complex:
    """Evaluate the all-label human-note NS blow-up coefficient ``B_a^2``."""

    p0, p1, pinfinity = (complex(value) for value in momenta)
    k0, k1, kinfinity = (int(value) for value in labels)
    q = complex(b) + 1.0 / complex(b)
    numerator = blow_up_factor(
        q / 2.0 + p1,
        k1,
        p0,
        k0,
        pinfinity,
        kinfinity,
        b,
        precision=precision,
    )
    denominator = (
        branch_norm(p0, k0, b, precision=precision)
        * branch_norm(p1, k1, b, precision=precision)
        * branch_norm(pinfinity, kinfinity, b, precision=precision)
    )
    if denominator == 0:
        raise ZeroDivisionError(
            f"zero branching-vector norm for momenta={momenta}, labels={labels}"
        )
    return complex(numerator * numerator / denominator)


def double_virasoro_enlarged_series(
    *, b: complex, momenta: Sequence[complex], cutoff: int, precision: int = 70
) -> ScalarSeries:
    """Return the enlarged block ``D_a`` before Majorana star division."""

    momenta = tuple(complex(value) for value in momenta)
    coefficients = {levels: 0.0j for levels in level_tuples(cutoff)}
    maximum_label = math.isqrt(cutoff)
    label_range = range(-maximum_label, maximum_label + 1)
    block_cache: dict[
        tuple[complex, tuple[complex, complex, complex], int], ScalarSeries
    ] = {}

    def virasoro_block(
        c: complex, weights: Sequence[complex], remaining: int
    ) -> ScalarSeries:
        key = (complex(c), tuple(complex(value) for value in weights), int(remaining))
        if key not in block_cache:
            block_cache[key] = ordinary_virasoro_c_recursion_series(
                c=key[0], weights=key[1], cutoff=key[2]
            )
        return block_cache[key]

    with mpmath.workdps(precision):
        for labels in product(label_range, repeat=3):
            base_levels = tuple(label * label for label in labels)
            base_total = sum(base_levels)
            if base_total > cutoff:
                continue
            branch = branching_coefficient_squared(
                b=b,
                momenta=momenta,
                labels=labels,
                precision=precision,
            )
            branch *= theta_orientation_sign(base_levels)

            copy_central_charges: list[complex] = []
            copy_weights: list[list[complex]] = [[], []]
            for momentum, label in zip(momenta, labels):
                parameters = two_virasoro_parameters(
                    momentum=momentum, label=label, b=b
                )
                if not copy_central_charges:
                    copy_central_charges.extend(
                        (parameters[0][0], parameters[1][0])
                    )
                copy_weights[0].append(parameters[0][1])
                copy_weights[1].append(parameters[1][1])

            remaining = (cutoff - base_total) // 2
            first = virasoro_block(
                copy_central_charges[0], copy_weights[0], remaining
            )
            second = virasoro_block(
                copy_central_charges[1], copy_weights[1], remaining
            )
            product_series = convolve_series(first, second, cutoff=remaining)
            for descendant_levels, descendant_value in product_series.items():
                levels = tuple(
                    base_levels[edge] + 2 * descendant_levels[edge]
                    for edge in range(3)
                )
                if sum(levels) <= cutoff:
                    coefficients[levels] += branch * descendant_value
    return coefficients


def _validated_human_lifts(human_lifts: Sequence[int]) -> tuple[int, int, int]:
    """Validate lifts without applying an internal frame rephasing."""

    lifts = tuple(int(value) for value in human_lifts)
    if len(lifts) != 3 or any(value not in (-1, 1) for value in lifts):
        raise ValueError("all plumbing lifts must be +1 or -1")
    return lifts  # type: ignore[return-value]


def ns_c_recursion_series(
    *, c: complex, weights: Sequence[complex], cutoff: int
) -> ScalarSeries:
    """Return NS c-recursion coefficients in the literal human theta frame."""

    backend_lifts = _validated_human_lifts((1, 1, 1))
    coefficients: ScalarSeries = {}
    for levels in level_tuples(cutoff):
        parity = sum(levels) % 2
        coefficients[levels] = recursion_theta_coefficient(
            c=c,
            weights=weights,
            twice_levels=levels,
            sectors=(parity, parity),
            lifts=backend_lifts,
        )
    return coefficients


def direct_ns_series(
    *, c: complex, weights: Sequence[complex], cutoff: int
) -> ScalarSeries:
    """Return the independent direct-PBW diagnostic in the human theta frame."""

    oracle = DirectThetaOracle(c=c, weights=weights)
    backend_lifts = _validated_human_lifts((1, 1, 1))
    coefficients: ScalarSeries = {}
    for levels in level_tuples(cutoff):
        parity = sum(levels) % 2
        coefficients[levels] = oracle.coefficient(
            twice_levels=levels,
            sectors=(parity, parity),
            lifts=backend_lifts,
        )
    return coefficients


def evaluated_sector(
    series: Mapping[Level, complex],
    *,
    q_values: Sequence[complex],
    lifts: Sequence[int],
    sector: int,
) -> complex:
    """Evaluate one parity sector of a twice-level coefficient series."""

    if sector not in (0, 1):
        raise ValueError("the NS sector label must be zero or one")
    q0, q1, qinfinity = (complex(value) for value in q_values)
    eta0, eta1, etainfinity = (int(value) for value in lifts)
    total = 0.0j
    for levels, coefficient in series.items():
        if sum(levels) % 2 != sector:
            continue
        total += coefficient * math.prod(
            (q ** (level / 2.0)) * (eta ** (level % 2))
            for q, eta, level in zip(
                (q0, q1, qinfinity),
                (eta0, eta1, etainfinity),
                levels,
            )
        )
    return total


@dataclass(frozen=True)
class Sample:
    b: float
    momenta: tuple[float, float, float]


DEFAULT_SAMPLES = (
    Sample(b=1.5, momenta=(2.0 / 7.0, 3.0 / 11.0, 5.0 / 13.0)),
    Sample(b=1.6, momenta=(0.19, 0.29, 0.37)),
)


@dataclass(frozen=True)
class SampleResult:
    b: float
    momenta: tuple[float, float, float]
    central_charge: float
    weights: tuple[float, float, float]
    coefficient_count: int
    maximum_c_recursion_vs_double_virasoro_error: float
    maximum_c_recursion_vs_double_virasoro_relative_error: float
    worst_twice_levels: Level
    maximum_direct_vs_c_recursion_error: float
    ordinary_quotient_mismatch_count: int
    star_quotient_mismatch_count: int
    maximum_evaluated_lift_sector_relative_error: float
    worst_lifts: tuple[int, int, int]
    worst_sector: int


@dataclass(frozen=True)
class ComparisonResult:
    maximum_total_physical_level: int
    maximum_total_twice_level: int
    q_values: tuple[float, float, float]
    sample_results: tuple[SampleResult, ...]


def _relative_error(left: complex, right: complex) -> float:
    return float(abs(left - right) / max(1.0, abs(left), abs(right)))


def compare_sample(
    sample: Sample,
    *,
    maximum_total_physical_level: int,
    q_values: Sequence[complex],
    precision: int,
    coefficient_tolerance: float,
) -> SampleResult:
    """Run the full coefficient and lift/sector comparison for one sample."""

    cutoff = 2 * int(maximum_total_physical_level)
    b = float(sample.b)
    momenta = tuple(float(value) for value in sample.momenta)
    q_background = b + 1.0 / b
    c = 1.5 + 3.0 * q_background * q_background
    weights = tuple(
        q_background * q_background / 8.0 - momentum * momentum / 2.0
        for momentum in momenta
    )

    hatted = double_virasoro_enlarged_series(
        b=b,
        momenta=momenta,
        cutoff=cutoff,
        precision=precision,
    )
    majorana = auxiliary_majorana_series(cutoff=cutoff)
    double_virasoro = divide_theta_star_series(
        hatted, majorana, cutoff=cutoff
    )
    ordinary_control = divide_ordinary_series(
        hatted, majorana, cutoff=cutoff
    )
    ns_recursion = ns_c_recursion_series(c=c, weights=weights, cutoff=cutoff)
    direct = direct_ns_series(c=c, weights=weights, cutoff=cutoff)

    worst_levels = (0, 0, 0)
    maximum_error = 0.0
    maximum_relative = 0.0
    maximum_direct_recursion = 0.0
    ordinary_mismatch_count = 0
    star_mismatch_count = 0
    for levels in level_tuples(cutoff):
        recursive = ns_recursion[levels]
        double = double_virasoro[levels]
        error = float(abs(recursive - double))
        relative = _relative_error(recursive, double)
        if relative > maximum_relative:
            maximum_relative = relative
            maximum_error = error
            worst_levels = levels
        maximum_direct_recursion = max(
            maximum_direct_recursion, abs(direct[levels] - recursive)
        )
        scale = max(1.0, abs(recursive), abs(double))
        if error > coefficient_tolerance * scale:
            star_mismatch_count += 1
        ordinary_error = abs(recursive - ordinary_control[levels])
        ordinary_scale = max(1.0, abs(recursive), abs(ordinary_control[levels]))
        if ordinary_error > coefficient_tolerance * ordinary_scale:
            ordinary_mismatch_count += 1

    worst_lifts = (1, 1, 1)
    worst_sector = 0
    maximum_evaluated_relative = 0.0
    for lifts in product((-1, 1), repeat=3):
        for sector in (0, 1):
            recursive_value = evaluated_sector(
                ns_recursion,
                q_values=q_values,
                lifts=lifts,
                sector=sector,
            )
            double_value = evaluated_sector(
                double_virasoro,
                q_values=q_values,
                lifts=lifts,
                sector=sector,
            )
            relative = _relative_error(recursive_value, double_value)
            if relative > maximum_evaluated_relative:
                maximum_evaluated_relative = relative
                worst_lifts = tuple(int(value) for value in lifts)
                worst_sector = sector

    return SampleResult(
        b=b,
        momenta=momenta,
        central_charge=float(c),
        weights=tuple(float(value) for value in weights),
        coefficient_count=sum(1 for _ in level_tuples(cutoff)),
        maximum_c_recursion_vs_double_virasoro_error=maximum_error,
        maximum_c_recursion_vs_double_virasoro_relative_error=maximum_relative,
        worst_twice_levels=worst_levels,
        maximum_direct_vs_c_recursion_error=float(maximum_direct_recursion),
        ordinary_quotient_mismatch_count=ordinary_mismatch_count,
        star_quotient_mismatch_count=star_mismatch_count,
        maximum_evaluated_lift_sector_relative_error=maximum_evaluated_relative,
        worst_lifts=worst_lifts,
        worst_sector=worst_sector,
    )


def run_comparison(
    *,
    maximum_total_physical_level: int = 4,
    samples: Sequence[Sample] = DEFAULT_SAMPLES,
    q_values: Sequence[complex] = (0.013, 0.017, 0.011),
    precision: int = 70,
    coefficient_tolerance: float = 2.0e-8,
) -> ComparisonResult:
    """Compare both methods for every coefficient, lift, and NS parity sector."""

    maximum_total_physical_level = int(maximum_total_physical_level)
    if maximum_total_physical_level < 0 or maximum_total_physical_level > 4:
        raise ValueError("this comparison supports total physical levels 0 through 4")
    if not samples:
        raise ValueError("at least one generic sample is required")
    if len(q_values) != 3 or any(abs(complex(value)) >= 1 for value in q_values):
        raise ValueError("three plumbing coordinates with modulus below one are required")

    results = tuple(
        compare_sample(
            sample,
            maximum_total_physical_level=maximum_total_physical_level,
            q_values=q_values,
            precision=precision,
            coefficient_tolerance=coefficient_tolerance,
        )
        for sample in samples
    )
    return ComparisonResult(
        maximum_total_physical_level=maximum_total_physical_level,
        maximum_total_twice_level=2 * maximum_total_physical_level,
        q_values=tuple(float(complex(value).real) for value in q_values),
        sample_results=results,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-total-level", type=int, default=4)
    parser.add_argument("--precision", type=int, default=70)
    parser.add_argument("--coefficient-tolerance", type=float, default=2.0e-8)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_comparison(
        maximum_total_physical_level=args.max_total_level,
        precision=args.precision,
        coefficient_tolerance=args.coefficient_tolerance,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2))
        return

    print("all-NS genus-two c-recursion / double-Virasoro comparison")
    print(f"  maximum total physical level: {result.maximum_total_physical_level}")
    print(f"  maximum total twice-level: {result.maximum_total_twice_level}")
    for index, sample in enumerate(result.sample_results, start=1):
        print(f"  sample {index}: b={sample.b}, P={sample.momenta}")
        print(
            "    c-recursion vs double Virasoro: "
            f"max relative={sample.maximum_c_recursion_vs_double_virasoro_relative_error:.3e}, "
            f"worst={sample.worst_twice_levels}"
        )
        print(
            "    direct PBW vs c-recursion: "
            f"max absolute={sample.maximum_direct_vs_c_recursion_error:.3e}"
        )
        print(
            "    star / ordinary quotient mismatches: "
            f"{sample.star_quotient_mismatch_count}/"
            f"{sample.ordinary_quotient_mismatch_count}"
        )
        print(
            "    all lifts and both sectors: "
            f"max relative={sample.maximum_evaluated_lift_sector_relative_error:.3e}"
        )


if __name__ == "__main__":
    main()
