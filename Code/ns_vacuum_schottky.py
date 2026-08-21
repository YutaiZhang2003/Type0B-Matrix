#!/usr/bin/env python3
"""Large-c NS vacuum block from lifted Schottky primitive links.

This is the N=1 super-Virasoro extension of the CCY/Chen-Wu large-c
vacuum-link construction.  For one *unoriented* primitive Schottky class
``[gamma]`` it evaluates

    prod_{m>=2} (1 + u_gamma * k_gamma**(m-1))
                 / (1 - k_gamma**m),

where ``k_gamma`` is the attracting PSL(2) multiplier and ``u_gamma`` is
the half-multiplier of a chosen SL(2) lift, ``u_gamma**2 = k_gamma``.
Equivalently, if ``u_gamma = eta_gamma*sqrt(k_gamma)``, the numerator is
``1 + eta_gamma*k_gamma**(m-1/2)``.  Keeping ``u_gamma`` rather than a
separately chosen square root is essential: products of lifted words fix
the spin signs consistently.

The primitive-word enumerator is reused from ``Code/python``.  The theta
plumbing generators are the explicit CCY-frame maps of arXiv:2401.13900,
Eq. (2.20).  The checks at the bottom verify

* the SL(2) lift relation u_gamma^2 = k_gamma, including inverse and cyclic
  representatives;
* reduction to the two lifted genus-one NS vacuum characters;
* the exterior-Fock trace-log sign independently with finite matrices;
* the first three fermionic links of the genus-two theta plumbing chart;
* the corrected primitive product against every directly sewn theta-vacuum
  coefficient through total level eight;
* exact reduction of the bosonic part to the existing CCY evaluator.

Run from the repository root with

    python3 Code/ns_vacuum_schottky.py
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from genus2_vacuum_blocks import (  # noqa: E402
    Word,
    canonical_primitive_key,
    cyclic_rotations,
    inverse_word,
    primitive_conjugacy_words,
    schottky_vacuum_block,
    word_multiplier,
)
from plumbing_algorithms import (  # noqa: E402
    GeneratorData,
    Mobius,
    dilation,
    generator_data_from_mobius,
)


Matrix2 = tuple[complex, complex, complex, complex]


@dataclass(frozen=True)
class NSPrimitiveContribution:
    """Bosonic and fermionic logarithms for one primitive class."""

    word: Word
    multiplier: complex
    half_multiplier: complex
    bosonic_log: complex
    fermionic_log: complex


@dataclass(frozen=True)
class NSVacuumBlockResult:
    """A word- and oscillator-truncated lifted Schottky product."""

    max_word_length: int
    max_mode: int
    value: complex
    log_value: complex
    primitive_count: int
    contributions: tuple[NSPrimitiveContribution, ...]


@dataclass(frozen=True)
class CheckSummary:
    """Numerical errors reported by the self-checking command-line entry."""

    max_lift_square_relative_error: float
    max_representative_relative_error: float
    genus_one_character_error: float
    exterior_trace_log_error: float
    bosonic_ccy_error: float
    theta_leading_error_coarse: float
    theta_leading_error_fine: float
    theta_error_ratio: float
    theta_level_six_remainder_coarse: float
    theta_level_six_remainder_fine: float
    theta_level_six_remainder_ratio: float
    theta_level_eight_remainder_coarse: float
    theta_level_eight_remainder_fine: float
    theta_level_eight_remainder_ratio: float


def ccy_theta_generators(
    p_one: complex, p_two: complex, p_three: complex
) -> list[GeneratorData]:
    r"""Return the Schottky generators of the CCY theta plumbing frame.

    With the edge order ``(0,1,infinity)=(p_one,p_two,p_three)``, the maps are

        gamma_1(z) = p_one*p_three*z,
        gamma_2(z) = ((1-p_two)z-1/p_one)/(z-1/p_one).

    These are Eq. (2.20) of arXiv:2401.13900.  In particular the second map
    depends on ``p_one``, not ``p_three``.  Confusing those two plumbing
    parameters preserves the three leading pinching links but fails at the
    next order.
    """

    p_one = complex(p_one)
    p_two = complex(p_two)
    p_three = complex(p_three)
    first_multiplier = p_one * p_three
    first = GeneratorData(
        gamma=dilation(first_multiplier),
        attracting=0.0j,
        repelling=None,
        multiplier=first_multiplier,
    )
    second_map = Mobius(
        p_one * (1.0 - p_two),
        -1.0,
        p_one,
        -1.0,
    ).normalized()
    return [first, generator_data_from_mobius(second_map)]


def mobius_matrix(mobius: Mobius) -> Matrix2:
    """Return a projective Mobius matrix as four complex entries."""

    return complex(mobius.a), complex(mobius.b), complex(mobius.c), complex(mobius.d)


def matrix_multiply(left: Matrix2, right: Matrix2) -> Matrix2:
    """Multiply two 2-by-2 matrices stored in row-major order."""

    a, b, c, d = left
    e, f, g, h = right
    return (
        a * e + b * g,
        a * f + b * h,
        c * e + d * g,
        c * f + d * h,
    )


def matrix_inverse_sl2(matrix: Matrix2) -> Matrix2:
    """Invert a determinant-one matrix without changing its lift sign."""

    a, b, c, d = matrix
    return d, -b, -c, a


def sl2_generator_lift(generator: GeneratorData, sign: int = 1) -> Matrix2:
    """Choose one of the two SL(2) lifts of a PSL(2) generator.

    The projective matrix stored by the plumbing code is divided by a square
    root of its determinant.  ``sign`` selects the remaining central
    ambiguity.  The same signed matrix, inverted in SL(2), is used for the
    inverse letter.
    """

    if sign not in (-1, 1):
        raise ValueError("an SL(2) generator lift sign must be +1 or -1")
    matrix = mobius_matrix(generator.gamma)
    a, b, c, d = matrix
    determinant = a * d - b * c
    if determinant == 0:
        raise ValueError("cannot lift a singular projective matrix")
    scale = cmath.sqrt(determinant)
    lifted = tuple(sign * entry / scale for entry in matrix)
    lifted_det = lifted[0] * lifted[3] - lifted[1] * lifted[2]
    if abs(lifted_det - 1.0) > 2.0e-8:
        raise ArithmeticError(f"SL(2) normalization failed: det={lifted_det!r}")
    return lifted  # type: ignore[return-value]


def lifted_word_matrix(
    generators: Sequence[GeneratorData],
    word: Word,
    generator_lift_signs: Sequence[int],
) -> Matrix2:
    """Multiply the chosen SL(2) lifts along a reduced Schottky word."""

    if len(generator_lift_signs) != len(generators):
        raise ValueError("one lift sign is required for every Schottky generator")
    lifts = [
        sl2_generator_lift(generator, int(sign))
        for generator, sign in zip(generators, generator_lift_signs)
    ]
    matrix: Matrix2 = (1.0 + 0.0j, 0.0j, 0.0j, 1.0 + 0.0j)
    for letter in word:
        lifted = lifts[letter // 2]
        if letter % 2:
            lifted = matrix_inverse_sl2(lifted)
        matrix = matrix_multiply(matrix, lifted)
    return matrix


def spin_half_multiplier(
    generators: Sequence[GeneratorData],
    word: Word,
    generator_lift_signs: Sequence[int],
) -> complex:
    """Return the attracting half-multiplier of a lifted Schottky word.

    For a determinant-one lift with eigenvalues ``lambda`` and
    ``lambda**(-1)``, choose the eigenvalue outside the unit circle and set
    ``u=1/lambda``.  It then follows that ``u**2`` is the attracting PSL(2)
    multiplier.  The sign of ``u`` records the spin lift.
    """

    matrix = lifted_word_matrix(generators, word, generator_lift_signs)
    trace = matrix[0] + matrix[3]
    root = cmath.sqrt(trace * trace - 4.0)
    eigenvalue_a = 0.5 * (trace + root)
    eigenvalue_b = 0.5 * (trace - root)
    eigenvalue_large = (
        eigenvalue_a if abs(eigenvalue_a) >= abs(eigenvalue_b) else eigenvalue_b
    )
    if eigenvalue_large == 0:
        raise ArithmeticError("lifted loxodromic word has a zero eigenvalue")
    half_multiplier = 1.0 / eigenvalue_large
    if abs(half_multiplier) > 1.0:
        # This branch is only relevant extremely close to |u|=1.  Inverting
        # preserves the sign of the chosen lift and selects the attracting root.
        half_multiplier = 1.0 / half_multiplier
    return complex(half_multiplier)


def ns_oscillator_log_factor(
    multiplier: complex,
    half_multiplier: complex,
    *,
    max_mode: int,
) -> tuple[complex, complex]:
    r"""Return the bosonic and fermionic logs for one primitive class.

    With ``m=2,3,...`` the factors are

        (1-k**m)**(-1),       1 + u*k**(m-1).

    The second expression has weight ``m-1/2`` because ``u**2=k``.
    """

    if max_mode < 2:
        raise ValueError("max_mode must be at least 2")
    k = complex(multiplier)
    u = complex(half_multiplier)
    if not abs(k) < 1.0:
        raise ValueError(f"expected |k|<1, received {abs(k):.6g}")
    bosonic = 0.0j
    fermionic = 0.0j
    for mode in range(2, int(max_mode) + 1):
        bosonic -= cmath.log(1.0 - k**mode)
        fermionic += cmath.log(1.0 + u * k ** (mode - 1))
    return bosonic, fermionic


def ns_schottky_vacuum_block(
    generators: Sequence[GeneratorData],
    generator_lift_signs: Sequence[int],
    *,
    max_word_length: int = 8,
    max_mode: int = 50,
) -> NSVacuumBlockResult:
    """Evaluate the large-c NS vacuum product over primitive classes."""

    words = primitive_conjugacy_words(len(generators), int(max_word_length))
    contributions: list[NSPrimitiveContribution] = []
    log_value = 0.0j
    for word in words:
        multiplier = word_multiplier(generators, word)
        half_multiplier = spin_half_multiplier(generators, word, generator_lift_signs)
        bosonic, fermionic = ns_oscillator_log_factor(
            multiplier,
            half_multiplier,
            max_mode=int(max_mode),
        )
        log_value += bosonic + fermionic
        contributions.append(
            NSPrimitiveContribution(
                word=word,
                multiplier=multiplier,
                half_multiplier=half_multiplier,
                bosonic_log=bosonic,
                fermionic_log=fermionic,
            )
        )
    return NSVacuumBlockResult(
        max_word_length=int(max_word_length),
        max_mode=int(max_mode),
        value=cmath.exp(log_value),
        log_value=log_value,
        primitive_count=len(contributions),
        contributions=tuple(contributions),
    )


def genus_one_ns_vacuum_character(
    q: complex,
    lift_sign: int,
    *,
    max_mode: int = 50,
) -> complex:
    """The lifted genus-one character computed through the Schottky code."""

    generator = GeneratorData(
        gamma=dilation(complex(q)),
        attracting=0.0j,
        repelling=None,
        multiplier=complex(q),
    )
    return ns_schottky_vacuum_block(
        [generator],
        [int(lift_sign)],
        max_word_length=1,
        max_mode=int(max_mode),
    ).value


def direct_genus_one_product(
    q: complex,
    lift_sign: int,
    *,
    max_mode: int,
) -> complex:
    """Independent direct oscillator product for the NS vacuum character."""

    value = 1.0 + 0.0j
    for mode in range(2, int(max_mode) + 1):
        value *= (1.0 + lift_sign * q ** (mode - 0.5)) / (1.0 - q**mode)
    return value


def exterior_trace_log_error() -> float:
    r"""Check det(1+K)=exp(sum (-1)^(s+1) tr K^s/s)."""

    kernel = np.array(
        [
            [0.08 + 0.01j, -0.02, 0.01j],
            [0.03, -0.04 + 0.02j, 0.015],
            [-0.01j, 0.025, 0.05],
        ],
        dtype=complex,
    )
    direct = np.linalg.det(np.eye(3, dtype=complex) + kernel)
    power = np.eye(3, dtype=complex)
    logarithm = 0.0j
    for winding in range(1, 80):
        power = power @ kernel
        logarithm += ((-1) ** (winding + 1)) * np.trace(power) / winding
    reconstructed = cmath.exp(logarithm)
    return float(abs(direct - reconstructed))


def lift_consistency_errors() -> tuple[float, float]:
    """Check squares and representative-independence for primitive theta words."""

    generators = ccy_theta_generators(0.013, 0.009, 0.011)
    signs = (1, -1)
    square_error = 0.0
    representative_error = 0.0
    for word in primitive_conjugacy_words(2, 5):
        k = word_multiplier(generators, word)
        u = spin_half_multiplier(generators, word, signs)
        square_error = max(square_error, abs(u * u - k) / max(1.0e-300, abs(k)))

        representatives = cyclic_rotations(word) + cyclic_rotations(inverse_word(word))
        for representative in representatives:
            u_rep = spin_half_multiplier(generators, representative, signs)
            representative_error = max(
                representative_error,
                abs(u_rep - u) / max(1.0e-300, abs(u)),
            )
        if canonical_primitive_key(word) != word:
            raise AssertionError("primitive-word enumerator did not return a canonical word")
    return float(square_error), float(representative_error)


def theta_lift_signs(edge_lifts: Sequence[int]) -> tuple[int, int]:
    """Map human-note theta lifts to Schottky-generator lift signs.

    The determinant-one representatives used by
    :func:`ccy_theta_generators` have the raw shortest-link signs ``(+,+,-)``
    on ``(0,infinity)``, ``(1,infinity)``, and ``(0,1)``.  The human note
    absorbs the corresponding infinity-frame rephasing into its plumbing
    lift, so its vacuum coefficient has ``(-,-,-)``.  Equivalently the raw
    Schottky edge lifts are ``(eta_0,eta_1,-eta_infinity)``.  This conversion
    belongs here at the geometric backend boundary; callers always pass the
    public human-note lifts.
    """

    if len(edge_lifts) != 3 or any(sign not in (-1, 1) for sign in edge_lifts):
        raise ValueError("theta plumbing needs three edge lift signs")
    eta_zero, eta_one, eta_infty = (int(sign) for sign in edge_lifts)
    return -eta_zero * eta_infty, eta_zero * eta_one


def theta_expected_leading_coefficient(
    scales: Sequence[float],
    edge_lifts: Sequence[int],
) -> float:
    r"""Coefficient of t^3 from the three shortest fermionic theta links.

    In the literal human-note theta frame, all three shortest two-supercurrent
    vacuum links carry a minus:

        -eta_0 eta_infinity, -eta_0 eta_1, -eta_1 eta_infinity.
    """

    if len(scales) != 3 or len(edge_lifts) != 3:
        raise ValueError("three theta edge scales and lifts are required")
    x_zero, x_one, x_infty = (float(value) for value in scales)
    eta_zero, eta_one, eta_infty = (int(value) for value in edge_lifts)
    return float(
        -eta_zero * eta_infty * (x_zero * x_infty) ** 1.5
        -eta_one * eta_infty * (x_one * x_infty) ** 1.5
        -eta_zero * eta_one * (x_zero * x_one) ** 1.5
    )


def theta_scaled_leading_error(t: float, edge_lifts: Sequence[int]) -> float:
    """Compare the full theta product with its derived t^3 leading term."""

    scales = (1.2, 0.8, 1.1)
    q_values = tuple(t * scale for scale in scales)
    generators = ccy_theta_generators(*q_values)
    result = ns_schottky_vacuum_block(
        generators,
        theta_lift_signs(edge_lifts),
        max_word_length=4,
        max_mode=8,
    )
    numerical = (result.value - 1.0).real / (t**3)
    expected = theta_expected_leading_coefficient(scales, edge_lifts)
    return float(abs(numerical - expected))


def theta_level_six_remainder(t: float, edge_lifts: Sequence[int]) -> float:
    """Compare the primitive product with direct theta sewing through level six."""

    # Imported locally to keep the vacuum-product evaluator independent of the
    # finite-c recursion when it is used as a library.
    from ns_genus12_finite_c_check import THETA_VACUUM_SEED_LEVEL6

    scales = (0.8, 1.1, 1.3)
    q_values = tuple(float(t) * scale for scale in scales)
    product_value = ns_schottky_vacuum_block(
        ccy_theta_generators(*q_values),
        theta_lift_signs(edge_lifts),
        max_word_length=8,
        max_mode=16,
    ).value
    direct_truncation = 0.0 + 0.0j
    for levels, coefficient in THETA_VACUUM_SEED_LEVEL6.items():
        monomial = math.prod(
            q ** (level / 2.0) * int(lift) ** (level % 2)
            for q, level, lift in zip(q_values, levels, edge_lifts)
        )
        direct_truncation += coefficient * monomial
    return float(abs(product_value - direct_truncation))


def theta_level_eight_remainder(t: float, edge_lifts: Sequence[int]) -> float:
    """Compare the primitive product with direct theta sewing through level eight."""

    from ns_genus12_finite_c_check import THETA_VACUUM_SEED_LEVEL8

    scales = (0.8, 1.1, 1.3)
    q_values = tuple(float(t) * scale for scale in scales)
    product_value = ns_schottky_vacuum_block(
        ccy_theta_generators(*q_values),
        theta_lift_signs(edge_lifts),
        max_word_length=8,
        max_mode=16,
    ).value
    direct_truncation = 0.0 + 0.0j
    for levels, coefficient in THETA_VACUUM_SEED_LEVEL8.items():
        monomial = math.prod(
            q ** (level / 2.0) * int(lift) ** (level % 2)
            for q, level, lift in zip(q_values, levels, edge_lifts)
        )
        direct_truncation += coefficient * monomial
    return float(abs(product_value - direct_truncation))


def bosonic_ccy_error() -> float:
    """Compare the denominator of the NS product to the existing CCY code."""

    generators = ccy_theta_generators(0.013, 0.009, 0.011)
    ns_result = ns_schottky_vacuum_block(
        generators,
        (1, -1),
        max_word_length=5,
        max_mode=12,
    )
    bosonic_log = sum(item.bosonic_log for item in ns_result.contributions)
    ccy_result = schottky_vacuum_block(
        generators,
        max_word_length=5,
        max_mode=12,
        tolerance=0.0,
        channel="theta-check",
    )
    return float(abs(cmath.exp(bosonic_log) - ccy_result.value))


def run_checks() -> CheckSummary:
    """Run all numerical checks and raise if a structural check fails."""

    square_error, representative_error = lift_consistency_errors()

    genus_one_error = 0.0
    for lift_sign in (-1, 1):
        via_schottky = genus_one_ns_vacuum_character(0.037, lift_sign, max_mode=18)
        direct = direct_genus_one_product(0.037, lift_sign, max_mode=18)
        genus_one_error = max(genus_one_error, abs(via_schottky - direct))

    edge_lift_choices = tuple(product((-1, 1), repeat=3))
    coarse = max(
        theta_scaled_leading_error(0.012, edge_lifts)
        for edge_lifts in edge_lift_choices
    )
    fine = max(
        theta_scaled_leading_error(0.006, edge_lifts)
        for edge_lifts in edge_lift_choices
    )
    ratio = fine / coarse
    level_six_coarse_by_lift = tuple(
        theta_level_six_remainder(0.08, edge_lifts)
        for edge_lifts in edge_lift_choices
    )
    level_six_fine_by_lift = tuple(
        theta_level_six_remainder(0.04, edge_lifts)
        for edge_lifts in edge_lift_choices
    )
    level_six_ratio = max(
        fine_value / coarse_value
        for coarse_value, fine_value in zip(
            level_six_coarse_by_lift, level_six_fine_by_lift
        )
    )
    level_eight_coarse_by_lift = tuple(
        theta_level_eight_remainder(0.08, edge_lifts)
        for edge_lifts in edge_lift_choices
    )
    level_eight_fine_by_lift = tuple(
        theta_level_eight_remainder(0.04, edge_lifts)
        for edge_lifts in edge_lift_choices
    )
    # Individual lift sectors can have an accidentally small leading
    # remainder.  The ratio of the maxima is the stable all-sector estimate
    # of the first omitted total level.
    level_eight_ratio = max(level_eight_fine_by_lift) / max(
        level_eight_coarse_by_lift
    )

    summary = CheckSummary(
        max_lift_square_relative_error=square_error,
        max_representative_relative_error=representative_error,
        genus_one_character_error=float(genus_one_error),
        exterior_trace_log_error=exterior_trace_log_error(),
        bosonic_ccy_error=bosonic_ccy_error(),
        theta_leading_error_coarse=coarse,
        theta_leading_error_fine=fine,
        theta_error_ratio=float(ratio),
        theta_level_six_remainder_coarse=max(level_six_coarse_by_lift),
        theta_level_six_remainder_fine=max(level_six_fine_by_lift),
        theta_level_six_remainder_ratio=float(level_six_ratio),
        theta_level_eight_remainder_coarse=max(level_eight_coarse_by_lift),
        theta_level_eight_remainder_fine=max(level_eight_fine_by_lift),
        theta_level_eight_remainder_ratio=float(level_eight_ratio),
    )

    if summary.max_lift_square_relative_error > 2.0e-7:
        raise AssertionError(summary)
    if summary.max_representative_relative_error > 2.0e-7:
        raise AssertionError(summary)
    if summary.genus_one_character_error > 2.0e-13:
        raise AssertionError(summary)
    if summary.exterior_trace_log_error > 2.0e-13:
        raise AssertionError(summary)
    if summary.bosonic_ccy_error > 2.0e-13:
        raise AssertionError(summary)
    if not summary.theta_error_ratio < 0.65:
        raise AssertionError(summary)
    # Halving all plumbing parameters suppresses a level-seven remainder by
    # 2^(-7).  The loose upper bound leaves room for higher-level terms while
    # excluding any mismatch through total level six.
    if not summary.theta_level_six_remainder_ratio < 0.012:
        raise AssertionError(summary)
    # The first omitted term after physical total level eight is level nine,
    # so uniform halving predicts a ratio near 2^(-9).
    if not summary.theta_level_eight_remainder_ratio < 0.006:
        raise AssertionError(summary)
    return summary


def json_ready(value: object) -> object:
    """Convert dataclass output, including complex numbers, to JSON values."""

    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--theta-example",
        action="store_true",
        help="also print one lifted genus-two theta product",
    )
    args = parser.parse_args()

    summary = run_checks()
    output: dict[str, object] = {"checks": asdict(summary), "status": "PASS"}
    if args.theta_example:
        generators = ccy_theta_generators(0.013, 0.009, 0.011)
        example = ns_schottky_vacuum_block(
            generators,
            theta_lift_signs((1, -1, 1)),
            max_word_length=6,
            max_mode=20,
        )
        output["theta_example"] = {
            "value": example.value,
            "primitive_count": example.primitive_count,
            "generator_lift_signs": theta_lift_signs((1, -1, 1)),
        }
    print(json.dumps(json_ready(output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
