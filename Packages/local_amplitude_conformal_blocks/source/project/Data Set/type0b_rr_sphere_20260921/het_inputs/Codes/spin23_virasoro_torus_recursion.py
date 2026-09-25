#!/usr/bin/env python3
r"""Ordinary-Virasoro recursion for the cut theta graph.

The two-Virasoro Ramond construction needs an ordinary genus-two theta block
with the NS edge at infinity fixed at descendant level zero.  Setting that
plumbing parameter to zero leaves a two-punctured torus block.  In the PDF's
local pair-of-pants slot order

.. math::

   (\text{cut NS at infinity},\text{edge 2 at }1,
    \text{edge 3 at }0)

this module evaluates

.. math::

   \mathcal F_\theta
   (c;h_{\rm cut},h_{\rm previous},h_{\rm current};
    0,q_{\rm previous},q_{\rm current})

by the Cho--Collier--Yin central-charge recursion.  The regular part is the
global :math:`\mathfrak{sl}_2` block times the exact rank-one
:math:`c=\infty` vacuum block in this single-edge degeneration.  Since the
first (cut) edge is fixed at level zero, the global block is an adaptive
two-edge sum.  Keeping this slot order explicit is essential: a Mobius
permutation preserves the primary coefficient but changes descendant local
coordinates unless the states are transformed as well.

Only descendant powers are returned.  Primary factors
``q**(h-c/24)`` belong to the outer superconformal sewing and are not included.
The implementation is intentionally self-contained so the SO(23) release does
not depend on the much larger plumbing package.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Literal, Mapping


LevelPair = tuple[int, int]
CoefficientTable = dict[LevelPair, complex]


def _validate_order(order: int) -> int:
    if not isinstance(order, int) or order < 0:
        raise ValueError("order must be a nonnegative integer")
    return order


def _log_product(start: complex, count: int) -> complex:
    total = 0.0j
    for offset in range(count):
        factor = complex(start) + offset
        if factor == 0:
            raise ZeroDivisionError("a normalized global coefficient hit a zero")
        total += cmath.log(factor)
    return total


def _normalized_rho_two_edge(
    previous_level: int,
    current_level: int,
    h_previous: complex,
    h_cut: complex,
    h_current: complex,
) -> complex:
    r"""Return the normalized form with the infinity cut edge at level zero.

    More explicitly, this is

    .. math::

       \frac{\rho(h_{\rm cut},L_{-1}^{j}h_{\rm previous},
                    L_{-1}^{k}h_{\rm current})}
            {\sqrt{j!(2h_{\rm previous})_j\,
                         k!(2h_{\rm current})_k}}}.

    The unnormalized numerator is the exact global-Ward result

    .. math::

       (h_3+h_2-h_1)_k
       (h_1-h_2-h_3-k)^{\underline{j}}.

    The products are accumulated logarithmically to avoid overflow.
    """

    j = _validate_order(previous_level)
    k = _validate_order(current_level)
    h_1 = complex(h_cut)
    h_2 = complex(h_previous)
    h_3 = complex(h_current)
    rising_start = h_3 + h_2 - h_1
    falling_start = h_1 - h_2 - h_3 - k
    numerator_factors = tuple(
        [rising_start + offset for offset in range(k)]
        + [falling_start - offset for offset in range(j)]
    )
    if any(factor == 0 for factor in numerator_factors):
        return 0.0j
    numerator_log = sum(
        (cmath.log(factor) for factor in numerator_factors),
        0.0j,
    )
    denominator_log = 0.5 * (
        math.lgamma(j + 1.0)
        + _log_product(2.0 * h_2, j)
        + math.lgamma(k + 1.0)
        + _log_product(2.0 * h_3, k)
    )
    return complex(cmath.exp(numerator_log - denominator_log))


def _rising_pochhammer(value: complex, order: int) -> complex:
    """Return the rising factorial ``(value)_order``."""

    result = 1.0 + 0.0j
    for offset in range(order):
        result *= complex(value) + offset
    return result


def _falling_pochhammer(value: complex, order: int) -> complex:
    """Return the falling factorial of the requested nonnegative order."""

    result = 1.0 + 0.0j
    for offset in range(order):
        result *= complex(value) - offset
    return result


def _normalized_rho_necklace_two_edge(
    infinity_level: int,
    zero_level: int,
    h_infinity: complex,
    h_middle: complex,
    h_zero: complex,
) -> complex:
    r"""Return normalized ``rho(L_-1^i h1,h2,L_-1^k h3)``.

    This is the Cho--Collier--Yin global three-point coefficient with the
    middle state primary.  It differs from :func:`_normalized_rho_two_edge`,
    whose first argument is the level on the *middle* theta edge.  Keeping
    these two local-coordinate orderings separate is essential for comparing
    a cut theta graph with the cyclic necklace oracle.
    """

    i = _validate_order(infinity_level)
    k = _validate_order(zero_level)
    h_1 = complex(h_infinity)
    h_2 = complex(h_middle)
    h_3 = complex(h_zero)
    numerator = 0.0j
    for p in range(min(i, k) + 1):
        numerator += (
            math.comb(i, p)
            * _falling_pochhammer(2.0 * h_3 + k - 1.0, p)
            * _falling_pochhammer(k, p)
            * _rising_pochhammer(h_3 + h_2 - h_1, k - p)
            * _rising_pochhammer(h_1 + h_2 - h_3 + p - k, i - p)
        )
    norm_infinity = math.factorial(i) * _rising_pochhammer(2.0 * h_1, i)
    norm_zero = math.factorial(k) * _rising_pochhammer(2.0 * h_3, k)
    denominator = cmath.sqrt(norm_infinity * norm_zero)
    if denominator == 0:
        raise ZeroDivisionError("the normalized necklace coefficient is singular")
    return complex(numerator / denominator)


def global_torus_two_point_seed(
    *,
    h_cut: complex,
    h_previous: complex,
    h_current: complex,
    q_previous: complex,
    q_current: complex,
    tolerance: float = 1.0e-13,
    max_outer_order: int = 96,
) -> complex:
    r"""Return the correctly oriented cut-theta global block.

    Both oriented vertices retain the global edge arguments
    ``(cut NS, edge 2, edge 3)``.  Their bilinear global coefficients are
    therefore the same, and their product is the square used below.  The
    plumbing shell index is ``previous_level + current_level``.
    """

    h_cut = complex(h_cut)
    h_previous = complex(h_previous)
    h_current = complex(h_current)
    q_previous = complex(q_previous)
    q_current = complex(q_current)
    if abs(q_previous) >= 1 or abs(q_current) >= 1:
        raise ValueError("both plumbing parameters must lie in the unit disk")
    if not math.isfinite(tolerance) or not 0 < tolerance < 1:
        raise ValueError("tolerance must lie strictly between zero and one")
    if not isinstance(max_outer_order, int) or max_outer_order < 3:
        raise ValueError("max_outer_order must be an integer of at least three")

    total = 0.0j
    small_shells = 0
    last_shell = math.inf
    previous_powers = tuple(q_previous**level for level in range(max_outer_order + 1))
    current_powers = tuple(q_current**level for level in range(max_outer_order + 1))
    for shell_level in range(max_outer_order + 1):
        shell = 0.0j
        shell_norm = 0.0
        for previous_level in range(shell_level + 1):
            current_level = shell_level - previous_level
            edge_factor = (
                previous_powers[previous_level] * current_powers[current_level]
            )
            if edge_factor == 0:
                continue
            normalized_rho = _normalized_rho_two_edge(
                previous_level,
                current_level,
                h_previous,
                h_cut,
                h_current,
            )
            term = edge_factor * normalized_rho * normalized_rho
            shell += term
            shell_norm += abs(term)
        total += shell
        last_shell = shell_norm
        if shell_norm <= tolerance * max(1.0, abs(total)):
            small_shells += 1
            if small_shells >= 3:
                return complex(total)
        else:
            small_shells = 0
    raise ArithmeticError(
        "global cut-theta seed did not converge by outer order "
        f"{max_outer_order}; last shell={last_shell:.3e}"
    )


def _rank_one_multiplier(q_previous: complex, q_current: complex) -> complex:
    r"""Return the exact surviving Schottky multiplier when the third edge pinches."""

    q_previous = complex(q_previous)
    q_current = complex(q_current)
    trace = q_previous + q_current - 1.0
    determinant = q_previous * q_current
    root = cmath.sqrt(trace * trace - 4.0 * determinant)
    eigenvalue_a = 0.5 * (trace + root)
    eigenvalue_b = 0.5 * (trace - root)
    large = eigenvalue_a if abs(eigenvalue_a) >= abs(eigenvalue_b) else eigenvalue_b
    if large == 0:
        raise ZeroDivisionError("the surviving Schottky cycle is singular")
    multiplier = determinant / (large * large)
    if abs(multiplier) > 1:
        multiplier = 1.0 / multiplier
    if abs(multiplier) >= 1:
        raise ValueError("the surviving Schottky multiplier is outside the unit disk")
    return complex(multiplier)


@lru_cache(maxsize=32768)
def rank_one_vacuum_seed(
    q_previous: complex,
    q_current: complex,
    tolerance: float = 1.0e-15,
    max_mode: int = 100000,
) -> complex:
    r"""Return :math:`\prod_{n\ge2}(1-k^n)^{-1}` for the surviving cycle."""

    multiplier = _rank_one_multiplier(q_previous, q_current)
    product = 1.0 + 0.0j
    power = multiplier * multiplier
    for mode in range(2, max_mode + 1):
        product /= 1.0 - power
        if abs(power) <= tolerance:
            return complex(product)
        power *= multiplier
    raise ArithmeticError("rank-one vacuum product did not converge")


def _multiply_univariate(
    left: tuple[complex, ...],
    right: tuple[complex, ...],
    maximum_degree: int,
) -> tuple[complex, ...]:
    """Multiply two univariate coefficient tuples through one degree."""

    result = [0.0j] * (maximum_degree + 1)
    for left_degree, left_value in enumerate(left):
        if left_value == 0:
            continue
        for right_degree, right_value in enumerate(right):
            degree = left_degree + right_degree
            if degree > maximum_degree:
                break
            result[degree] += left_value * right_value
    return tuple(result)


@lru_cache(maxsize=128)
def _rank_one_vacuum_coefficients(
    maximum_previous_level: int,
    maximum_current_level: int,
) -> Mapping[LevelPair, complex]:
    r"""Expand the rank-one vacuum factor as a bivariate series.

    If ``k`` is the surviving Schottky multiplier and

    .. math::

       x=\frac{q_pq_c}{(1-q_p-q_c)^2},

    then ``x=k/(1+k)^2`` and hence
    :math:`k=\sum_{m\geq1}C_mx^m`, where ``C_m`` are Catalan numbers.
    This gives an exact finite expansion of
    :math:`\prod_{n\geq2}(1-k^n)^{-1}` at any requested bidegree, without
    fitting values of the algebraic multiplier.
    """

    max_x_degree = min(maximum_previous_level, maximum_current_level)
    if max_x_degree == 0:
        return {(0, 0): 1.0 + 0.0j}

    k_of_x = [0.0j] * (max_x_degree + 1)
    for degree in range(1, max_x_degree + 1):
        k_of_x[degree] = complex(
            math.comb(2 * degree, degree) // (degree + 1)
        )

    # Coefficients of prod_{n>=2} (1-k^n)^(-1): partitions with no part 1.
    vacuum_in_k = [0] * (max_x_degree + 1)
    vacuum_in_k[0] = 1
    for oscillator in range(2, max_x_degree + 1):
        for degree in range(oscillator, max_x_degree + 1):
            vacuum_in_k[degree] += vacuum_in_k[degree - oscillator]

    vacuum_in_x = [0.0j] * (max_x_degree + 1)
    k_power = tuple([1.0 + 0.0j] + [0.0j] * max_x_degree)
    for power, multiplicity in enumerate(vacuum_in_k):
        if multiplicity:
            for degree, coefficient in enumerate(k_power):
                vacuum_in_x[degree] += multiplicity * coefficient
        if power != max_x_degree:
            k_power = _multiply_univariate(
                k_power,
                tuple(k_of_x),
                max_x_degree,
            )

    result: CoefficientTable = {(0, 0): complex(vacuum_in_x[0])}
    for x_degree in range(1, max_x_degree + 1):
        x_coefficient = vacuum_in_x[x_degree]
        if x_coefficient == 0:
            continue
        for extra_previous in range(
            maximum_previous_level - x_degree + 1
        ):
            for extra_current in range(
                maximum_current_level - x_degree + 1
            ):
                total_extra = extra_previous + extra_current
                denominator_coefficient = (
                    math.comb(
                        2 * x_degree + total_extra - 1,
                        total_extra,
                    )
                    * math.comb(total_extra, extra_previous)
                )
                key = (
                    x_degree + extra_previous,
                    x_degree + extra_current,
                )
                result[key] = result.get(key, 0.0j) + (
                    x_coefficient * denominator_coefficient
                )
    return result


def _convolve_tables(
    left: Mapping[LevelPair, complex],
    right: Mapping[LevelPair, complex],
    maximum_previous_level: int,
    maximum_current_level: int,
) -> CoefficientTable:
    """Convolve two bivariate series inside a rectangular cutoff."""

    result: CoefficientTable = {}
    for (left_previous, left_current), left_value in left.items():
        if left_value == 0:
            continue
        for (right_previous, right_current), right_value in right.items():
            previous = left_previous + right_previous
            current = left_current + right_current
            if (
                previous > maximum_previous_level
                or current > maximum_current_level
            ):
                continue
            key = (previous, current)
            result[key] = result.get(key, 0.0j) + left_value * right_value
    return result


@lru_cache(maxsize=32768)
def _regular_seed_coefficients(
    h_cut: complex,
    h_previous: complex,
    h_current: complex,
    maximum_previous_level: int,
    maximum_current_level: int,
    include_vacuum_seed: bool,
) -> Mapping[LevelPair, complex]:
    """Return the exact finite bivariate regular seed."""

    global_coefficients = {
        (previous_level, current_level): _normalized_rho_two_edge(
            previous_level,
            current_level,
            h_previous,
            h_cut,
            h_current,
        )
        ** 2
        for previous_level in range(maximum_previous_level + 1)
        for current_level in range(maximum_current_level + 1)
    }
    vacuum_coefficients = (
        _rank_one_vacuum_coefficients(
            maximum_previous_level,
            maximum_current_level,
        )
        if include_vacuum_seed
        else {(0, 0): 1.0 + 0.0j}
    )
    return _convolve_tables(
        global_coefficients,
        vacuum_coefficients,
        maximum_previous_level,
        maximum_current_level,
    )


@lru_cache(maxsize=32768)
def _regular_necklace_seed_coefficients(
    h_cut: complex,
    h_previous: complex,
    h_current: complex,
    maximum_previous_level: int,
    maximum_current_level: int,
    include_vacuum_seed: bool,
) -> Mapping[LevelPair, complex]:
    r"""Return the finite regular seed in cyclic necklace coordinates."""

    global_coefficients: CoefficientTable = {}
    for previous_level in range(maximum_previous_level + 1):
        for current_level in range(maximum_current_level + 1):
            left = _normalized_rho_necklace_two_edge(
                previous_level,
                current_level,
                h_previous,
                h_cut,
                h_current,
            )
            right = _normalized_rho_necklace_two_edge(
                current_level,
                previous_level,
                h_current,
                h_cut,
                h_previous,
            )
            global_coefficients[(previous_level, current_level)] = left * right
    vacuum_coefficients = (
        _rank_one_vacuum_coefficients(
            maximum_previous_level,
            maximum_current_level,
        )
        if include_vacuum_seed
        else {(0, 0): 1.0 + 0.0j}
    )
    return _convolve_tables(
        global_coefficients,
        vacuum_coefficients,
        maximum_previous_level,
        maximum_current_level,
    )


def _b_square_rs_from_h(r: int, s: int, weight: complex) -> complex:
    if r < 2 or s < 1:
        raise ValueError("c-recursion uses r >= 2 and s >= 1")
    h = complex(weight)
    radical = (r - s) ** 2 + 4.0 * (r * s - 1.0) * h + 4.0 * h * h
    root = cmath.sqrt(radical)
    linear = r * s - 1.0 + 2.0 * h
    denominator = 1.0 - r * r
    candidate = (linear + root) / denominator
    alternative = (linear - root) / denominator
    if abs(candidate) > 1.0e-13 * max(1.0, abs(alternative)):
        return candidate
    # For s=1 the quadratic obtained after multiplying the Kac equation by
    # b**2 has a spurious b**2=0 root.  Depending on the principal square-root
    # branch, the displayed CCY expression can select that root.  The other
    # root is the finite c-pole and is the required analytic continuation.
    candidate = alternative
    if candidate == 0:
        raise ZeroDivisionError("the c-recursion Kac equation has no finite root")
    return candidate


def _c_rs_from_h(r: int, s: int, weight: complex) -> complex:
    b_square = _b_square_rs_from_h(r, s, weight)
    return 13.0 + 6.0 * (b_square + 1.0 / b_square)


def _momentum_from_weight(weight: complex, b: complex) -> complex:
    q_background = b + 1.0 / b
    return cmath.sqrt(q_background * q_background - 4.0 * complex(weight))


def _fusion_polynomial(
    r: int,
    s: int,
    b: complex,
    top_weight: complex,
    bottom_weight: complex,
) -> complex:
    lambda_top = _momentum_from_weight(top_weight, b)
    lambda_bottom = _momentum_from_weight(bottom_weight, b)
    product = 1.0 + 0.0j
    for p in range(1 - r, r, 2):
        for ell in range(1 - s, s, 2):
            shift = p * b + ell / b
            product *= 0.5 * (lambda_top + lambda_bottom + shift)
            product *= 0.5 * (lambda_top - lambda_bottom + shift)
    return product


def _minus_dc_dh_times_a_rs(r: int, s: int, weight: complex) -> complex:
    r"""Evaluate the finite product ``-dc_rs/dh * A_rs`` after cancellation."""

    x = _b_square_rs_from_h(r, s, weight)
    numerator = -12.0 * x ** (2 * r * s - 1)
    denominator = (1.0 - r * r) * x * x - (1.0 - s * s)
    denominator_factors = [
        (p, ell)
        for p in range(1 - r, r + 1)
        for ell in range(1 - s, s + 1)
        if (p, ell) not in ((0, 0), (r, s))
    ]
    remaining_numerator: list[tuple[int, int]] = []
    for numerator_factor in ((1, -1), (1, 1)):
        num_p, num_ell = numerator_factor
        match = next(
            (
                index
                for index, (den_p, den_ell) in enumerate(denominator_factors)
                if den_p != 0 and den_p * num_ell == den_ell * num_p
            ),
            None,
        )
        if match is None:
            remaining_numerator.append(numerator_factor)
        else:
            den_p, _ = denominator_factors.pop(match)
            denominator *= den_p // num_p
    for p, ell in remaining_numerator:
        numerator *= p * x + ell
    for p, ell in denominator_factors:
        denominator *= p * x + ell
    if denominator == 0:
        raise ZeroDivisionError("the simplified c-recursion residue is singular")
    return complex(numerator / denominator)


def _residue_prefactor(
    r: int,
    s: int,
    edge_weight: complex,
    top_weight: complex,
    bottom_weight: complex,
) -> complex:
    b = cmath.sqrt(_b_square_rs_from_h(r, s, edge_weight))
    polynomial = _fusion_polynomial(
        r, s, b, top_weight, bottom_weight
    )
    return _minus_dc_dh_times_a_rs(r, s, edge_weight) * polynomial * polynomial


@dataclass(frozen=True)
class VirasoroTorusRecursionResult:
    """Value and numerical controls for one cut-theta c-recursion."""

    value: complex
    c: complex
    h_cut: complex
    h_previous: complex
    h_current: complex
    q_previous: complex
    q_current: complex
    recursion_order: int
    include_vacuum_seed: bool
    regular_tolerance: float
    regular_max_outer_order: int


def virasoro_torus_two_point_block(
    *,
    c: complex,
    h_cut: complex,
    h_previous: complex,
    h_current: complex,
    q_previous: complex,
    q_current: complex,
    recursion_order: int,
    include_vacuum_seed: bool = True,
    regular_tolerance: float = 1.0e-13,
    regular_max_outer_order: int = 96,
    pole_tolerance: float = 1.0e-11,
) -> VirasoroTorusRecursionResult:
    r"""Evaluate the stripped two-punctured-torus block by c-recursion.

    ``recursion_order`` truncates the total level carried by non-global
    Virasoro singular vectors.  The global and rank-one vacuum regular terms
    are independently resummed to ``regular_tolerance``.
    """

    order = _validate_order(recursion_order)
    c = complex(c)
    h_cut = complex(h_cut)
    h_previous = complex(h_previous)
    h_current = complex(h_current)
    q_previous = complex(q_previous)
    q_current = complex(q_current)
    if abs(q_previous) >= 1 or abs(q_current) >= 1:
        raise ValueError("both plumbing parameters must lie in the unit disk")
    if not math.isfinite(pole_tolerance) or pole_tolerance <= 0:
        raise ValueError("pole_tolerance must be finite and positive")
    vacuum = (
        rank_one_vacuum_seed(
            q_previous, q_current, min(1.0e-15, 0.1 * regular_tolerance)
        )
        if include_vacuum_seed
        else 1.0 + 0.0j
    )

    @lru_cache(maxsize=None)
    def recurse(
        current_c: complex,
        current_previous: complex,
        current_current: complex,
        remaining: int,
    ) -> complex:
        seed = vacuum * global_torus_two_point_seed(
            h_cut=h_cut,
            h_previous=current_previous,
            h_current=current_current,
            q_previous=q_previous,
            q_current=q_current,
            tolerance=regular_tolerance,
            max_outer_order=regular_max_outer_order,
        )
        total = seed
        for edge in ("previous", "current"):
            edge_weight = (
                current_previous if edge == "previous" else current_current
            )
            q_edge = q_previous if edge == "previous" else q_current
            top_weight, bottom_weight = (
                (current_current, h_cut)
                if edge == "previous"
                else (h_cut, current_previous)
            )
            for r in range(2, remaining + 1):
                for s in range(1, remaining // r + 1):
                    level = r * s
                    if level > remaining:
                        continue
                    pole_c = _c_rs_from_h(r, s, edge_weight)
                    denominator = current_c - pole_c
                    scale = max(1.0, abs(current_c), abs(pole_c))
                    if abs(denominator) <= pole_tolerance * scale:
                        raise ZeroDivisionError(
                            "ordinary c-recursion reached a confluent pole; "
                            "move the complete two-Virasoro expression to "
                            "generic b before taking its finite part"
                        )
                    residue = (
                        q_edge**level
                        * _residue_prefactor(
                            r,
                            s,
                            edge_weight,
                            top_weight,
                            bottom_weight,
                        )
                        / denominator
                    )
                    if edge == "previous":
                        tail = recurse(
                            pole_c,
                            edge_weight + level,
                            current_current,
                            remaining - level,
                        )
                    else:
                        tail = recurse(
                            pole_c,
                            current_previous,
                            edge_weight + level,
                            remaining - level,
                        )
                    total += residue * tail
        return complex(total)

    value = recurse(c, h_previous, h_current, order)
    return VirasoroTorusRecursionResult(
        value=value,
        c=c,
        h_cut=h_cut,
        h_previous=h_previous,
        h_current=h_current,
        q_previous=q_previous,
        q_current=q_current,
        recursion_order=order,
        include_vacuum_seed=bool(include_vacuum_seed),
        regular_tolerance=float(regular_tolerance),
        regular_max_outer_order=int(regular_max_outer_order),
    )


@dataclass(frozen=True)
class VirasoroTorusCoefficientResult:
    """Finite bivariate coefficient table from ordinary c-recursion."""

    coefficients: Mapping[LevelPair, complex]
    c: complex
    h_cut: complex
    h_previous: complex
    h_current: complex
    maximum_previous_level: int
    maximum_current_level: int
    recursion_order: int
    include_vacuum_seed: bool
    local_ordering: str

    def descendant_value(
        self,
        q_previous: complex,
        q_current: complex,
    ) -> complex:
        """Evaluate the retained rectangular descendant series."""

        q_previous = complex(q_previous)
        q_current = complex(q_current)
        return complex(
            sum(
                coefficient
                * q_previous**previous_level
                * q_current**current_level
                for (
                    previous_level,
                    current_level,
                ), coefficient in self.coefficients.items()
            )
        )


def virasoro_torus_two_point_coefficients(
    *,
    c: complex,
    h_cut: complex,
    h_previous: complex,
    h_current: complex,
    maximum_previous_level: int,
    maximum_current_level: int,
    recursion_order: int,
    include_vacuum_seed: bool = True,
    local_ordering: Literal["theta", "necklace"] = "theta",
    pole_tolerance: float = 1.0e-11,
) -> VirasoroTorusCoefficientResult:
    r"""Return ordinary cut-theta coefficients by CCY c-recursion.

    This is the coefficient-level counterpart of
    :func:`virasoro_torus_two_point_block`.  It retains every coefficient
    :math:`q_p^{N_p}q_c^{N_c}` inside the requested rectangle and therefore
    supports exact formal multiplication and division in the two-Virasoro
    Ramond identity.  ``recursion_order`` bounds the total singular-vector
    level along each recursive path, exactly as in the value evaluator.
    """

    if local_ordering not in ("theta", "necklace"):
        raise ValueError("local_ordering must be 'theta' or 'necklace'")
    for value, name in (
        (maximum_previous_level, "maximum_previous_level"),
        (maximum_current_level, "maximum_current_level"),
    ):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    order = _validate_order(recursion_order)
    if not math.isfinite(pole_tolerance) or pole_tolerance <= 0:
        raise ValueError("pole_tolerance must be finite and positive")
    c = complex(c)
    h_cut = complex(h_cut)
    h_previous = complex(h_previous)
    h_current = complex(h_current)

    @lru_cache(maxsize=None)
    def recurse(
        current_c: complex,
        current_previous: complex,
        current_current: complex,
        remaining: int,
        maximum_previous: int,
        maximum_current: int,
    ) -> Mapping[LevelPair, complex]:
        regular_seed = (
            _regular_seed_coefficients
            if local_ordering == "theta"
            else _regular_necklace_seed_coefficients
        )
        total = dict(
            regular_seed(
                h_cut,
                current_previous,
                current_current,
                maximum_previous,
                maximum_current,
                bool(include_vacuum_seed),
            )
        )
        for edge in ("previous", "current"):
            edge_weight = (
                current_previous if edge == "previous" else current_current
            )
            edge_cutoff = (
                maximum_previous if edge == "previous" else maximum_current
            )
            top_weight, bottom_weight = (
                (current_current, h_cut)
                if edge == "previous"
                else (h_cut, current_previous)
            )
            for r in range(2, remaining + 1):
                for s in range(1, remaining // r + 1):
                    level = r * s
                    if level > remaining or level > edge_cutoff:
                        continue
                    pole_c = _c_rs_from_h(r, s, edge_weight)
                    denominator = current_c - pole_c
                    scale = max(1.0, abs(current_c), abs(pole_c))
                    if abs(denominator) <= pole_tolerance * scale:
                        raise ZeroDivisionError(
                            "ordinary c-recursion reached a confluent pole; "
                            "assemble at generic b before taking its finite part"
                        )
                    prefactor = _residue_prefactor(
                        r,
                        s,
                        edge_weight,
                        top_weight,
                        bottom_weight,
                    ) / denominator
                    if edge == "previous":
                        tail = recurse(
                            pole_c,
                            edge_weight + level,
                            current_current,
                            remaining - level,
                            maximum_previous - level,
                            maximum_current,
                        )
                        shift = (level, 0)
                    else:
                        tail = recurse(
                            pole_c,
                            current_previous,
                            edge_weight + level,
                            remaining - level,
                            maximum_previous,
                            maximum_current - level,
                        )
                        shift = (0, level)
                    for (tail_previous, tail_current), tail_value in tail.items():
                        key = (
                            tail_previous + shift[0],
                            tail_current + shift[1],
                        )
                        total[key] = total.get(key, 0.0j) + prefactor * tail_value
        return total

    coefficients = recurse(
        c,
        h_previous,
        h_current,
        order,
        maximum_previous_level,
        maximum_current_level,
    )
    return VirasoroTorusCoefficientResult(
        coefficients=dict(coefficients),
        c=c,
        h_cut=h_cut,
        h_previous=h_previous,
        h_current=h_current,
        maximum_previous_level=maximum_previous_level,
        maximum_current_level=maximum_current_level,
        recursion_order=order,
        include_vacuum_seed=bool(include_vacuum_seed),
        local_ordering=local_ordering,
    )


__all__ = [
    "VirasoroTorusCoefficientResult",
    "VirasoroTorusRecursionResult",
    "global_torus_two_point_seed",
    "rank_one_vacuum_seed",
    "_normalized_rho_necklace_two_edge",
    "virasoro_torus_two_point_block",
    "virasoro_torus_two_point_coefficients",
]
