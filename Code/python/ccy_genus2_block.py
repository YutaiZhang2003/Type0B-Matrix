#!/usr/bin/env python3
"""Cho-Collier-Yin genus-two Virasoro block in the plumbing frame.

The block implemented here is the genus-two, no-puncture block built in the
theta-graph frame by gluing two 2-holed discs.  In the notation of Cho,
Collier, and Yin,

    F = sum q1^|A| q2^|C| q3^|E|
        G_h1^{AB} G_h2^{CD} G_h3^{EF}
        rho(L_-A h1, L_-C h2, L_-E h3)
        rho(L_-B h1, L_-D h2, L_-F h3).

The public evaluator uses their central-charge recursion.  The large-c regular
part is written as a global SL(2) block times the c=infinity vacuum block.  The
global block is evaluated by the closed form for L_-1 descendants.  The vacuum
block is approximated, when requested, by the theta-graph Schottky
primitive-class product truncated by word length and oscillator level.

As in CCY's generic plumbing construction, this block contains the descendant
powers q^level only; the separated primary propagator prod_i q_i^h_i is applied
by the Liouville sewing wrapper.
"""

from __future__ import annotations

import argparse
import cmath
import math
from dataclasses import dataclass
from functools import lru_cache

import mpmath

try:
    from genus2_vacuum_blocks import oscillator_log_factor, schottky_vacuum_block
    from plumbing_algorithms import (
        generators_for_theta,
        inverse_letter,
        reduced_words,
        schottky_theta_period_matrix_cross_ratio,
        theta_cusp_surviving_multipliers,
        word_mobius,
    )
    from virasoro_blocks import (
        fusion_polynomial,
        momentum_from_weight,
        zamolodchikov_a_rs,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.genus2_vacuum_blocks import oscillator_log_factor, schottky_vacuum_block
    from plumbing.plumbing_algorithms import (
        generators_for_theta,
        inverse_letter,
        reduced_words,
        schottky_theta_period_matrix_cross_ratio,
        theta_cusp_surviving_multipliers,
        word_mobius,
    )
    from plumbing.virasoro_blocks import (
        fusion_polynomial,
        momentum_from_weight,
        zamolodchikov_a_rs,
    )


def parse_complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def format_complex(value: complex) -> str:
    return f"{value.real:+.12e}{value.imag:+.12e}j"


def _as_complex(value: complex | float | int) -> complex:
    return complex(value)


def _validate_order(order: int) -> int:
    order = int(order)
    if order < 0:
        raise ValueError("order must be non-negative")
    return order


def rising_pochhammer(value: complex, order: int) -> complex:
    """Return value (value+1) ... (value+order-1)."""
    order = _validate_order(order)
    out = 1.0 + 0.0j
    value = _as_complex(value)
    for n in range(order):
        out *= value + n
    return out


def falling_pochhammer(value: complex, order: int) -> complex:
    """Return value (value-1) ... (value-order+1)."""
    order = _validate_order(order)
    out = 1.0 + 0.0j
    value = _as_complex(value)
    for n in range(order):
        out *= value - n
    return out


def sl2_descendant_norm(weight: complex, level: int) -> complex:
    """Return <h| L_1^n L_-1^n |h> = n! (2h)_n."""
    level = _validate_order(level)
    return math.factorial(level) * rising_pochhammer(2.0 * _as_complex(weight), level)


def rho_lminus1_two_edge(i_level: int, k_level: int, h1: complex, h2: complex, h3: complex) -> complex:
    r"""Return s_{ik}=rho(L_-1^i h1, h2, L_-1^k h3).

    This is the closed form quoted by Cho-Collier-Yin for the global block.
    """
    i_level = _validate_order(i_level)
    k_level = _validate_order(k_level)
    h1 = _as_complex(h1)
    h2 = _as_complex(h2)
    h3 = _as_complex(h3)
    total = 0.0 + 0.0j
    for p_level in range(min(i_level, k_level) + 1):
        total += (
            math.factorial(i_level)
            / (math.factorial(p_level) * math.factorial(i_level - p_level))
            * falling_pochhammer(2.0 * h3 + k_level - 1.0, p_level)
            * falling_pochhammer(k_level, p_level)
            * rising_pochhammer(h3 + h2 - h1, k_level - p_level)
            * rising_pochhammer(h1 + h2 - h3 + p_level - k_level, i_level - p_level)
        )
    return total


def rho_lminus1_triple(
    i_level: int,
    j_level: int,
    k_level: int,
    h1: complex,
    h2: complex,
    h3: complex,
) -> complex:
    r"""Return rho(L_-1^i h1, L_-1^j h2, L_-1^k h3)."""
    i_level = _validate_order(i_level)
    j_level = _validate_order(j_level)
    k_level = _validate_order(k_level)
    prefactor = rising_pochhammer(
        _as_complex(h1) + i_level - _as_complex(h2) - j_level + 1.0 - _as_complex(h3) - k_level,
        j_level,
    )
    return prefactor * rho_lminus1_two_edge(i_level, k_level, h1, h2, h3)


def genus2_global_sl2_block(
    h1: complex,
    h2: complex,
    h3: complex,
    q1: complex,
    q2: complex,
    q3: complex,
    order: int,
) -> complex:
    """Return the total-degree truncated genus-two global SL(2) block."""
    order = _validate_order(order)
    h1 = _as_complex(h1)
    h2 = _as_complex(h2)
    h3 = _as_complex(h3)
    q1 = _as_complex(q1)
    q2 = _as_complex(q2)
    q3 = _as_complex(q3)
    total = 0.0 + 0.0j
    for i_level in range(order + 1):
        norm1 = sl2_descendant_norm(h1, i_level)
        for j_level in range(order + 1 - i_level):
            norm2 = sl2_descendant_norm(h2, j_level)
            for k_level in range(order + 1 - i_level - j_level):
                rho = rho_lminus1_triple(i_level, j_level, k_level, h1, h2, h3)
                total += (
                    (q1**i_level)
                    * (q2**j_level)
                    * (q3**k_level)
                    * rho
                    * rho
                    / (norm1 * norm2 * sl2_descendant_norm(h3, k_level))
                )
    return total


@dataclass(frozen=True)
class GlobalSL2Resummation:
    """Value and endpoint-shell certificate for the resummed global block."""

    value: complex
    last_shell: complex
    endpoint_total: int
    converged: bool


def genus2_global_sl2_block_resummed(
    h1: complex,
    h2: complex,
    h3: complex,
    q1: complex,
    q2: complex,
    q3: complex,
    *,
    tolerance: float = 1.0e-13,
    max_endpoint_total: int = 52,
    working_precision: int = 50,
) -> GlobalSL2Resummation:
    r"""Resum the middle edge and adaptively sum the two endpoint edges.

    For fixed endpoint occupations ``i,k``, translation covariance gives

    .. math::

       \sum_{j\geq0}\frac{q_2^j\rho_{ijk}^2}{j!(2h_2)_j}
       = s_{ik}^2\,{}_2F_1(a_{ik},a_{ik};2h_2;q_2),

    where ``s_ik=rho(i,0,k)`` and
    ``a_ik=h2+h3+k-h1-i``.  Thus ``max_endpoint_total`` is only a
    convergence guard for the remaining representation, not the
    c-recursion order.  Three consecutive small endpoint shells are required
    before the result is certified.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    max_endpoint_total = _validate_order(max_endpoint_total)
    working_precision = int(working_precision)
    if working_precision < 20:
        raise ValueError("working_precision must be at least 20 decimal digits")

    h1 = _as_complex(h1)
    h2 = _as_complex(h2)
    h3 = _as_complex(h3)
    q1 = _as_complex(q1)
    q2 = _as_complex(q2)
    q3 = _as_complex(q3)
    if any(abs(q) >= 1 for q in (q1, q2, q3)):
        raise ValueError("global-block plumbing coordinates must satisfy |q_i| < 1")

    total = 0.0 + 0.0j
    last_shell = 0.0 + 0.0j
    small_shells = 0
    converged = False
    used = 0
    with mpmath.workdps(working_precision):
        for endpoint_total in range(max_endpoint_total + 1):
            shell = 0.0 + 0.0j
            for i_level in range(endpoint_total + 1):
                k_level = endpoint_total - i_level
                s_ik = rho_lminus1_two_edge(i_level, k_level, h1, h2, h3)
                a_ik = h2 + h3 + k_level - h1 - i_level
                shell += complex(
                    (mpmath.mpc(q1) ** i_level)
                    * (mpmath.mpc(q3) ** k_level)
                    * (mpmath.mpc(s_ik) ** 2)
                    / (
                        mpmath.factorial(i_level)
                        * mpmath.rf(2 * mpmath.mpc(h1), i_level)
                        * mpmath.factorial(k_level)
                        * mpmath.rf(2 * mpmath.mpc(h3), k_level)
                    )
                    * mpmath.hyp2f1(
                        mpmath.mpc(a_ik),
                        mpmath.mpc(a_ik),
                        2 * mpmath.mpc(h2),
                        mpmath.mpc(q2),
                    )
                )
            total += shell
            last_shell = shell
            used = endpoint_total
            scale = max(1.0, abs(total))
            if endpoint_total >= 3 and abs(shell) <= tolerance * scale:
                small_shells += 1
            else:
                small_shells = 0
            if small_shells >= 3:
                converged = True
                break

    return GlobalSL2Resummation(
        value=complex(total),
        last_shell=complex(last_shell),
        endpoint_total=used,
        converged=converged,
    )


def _certified_global_sl2_value(
    h1: complex,
    h2: complex,
    h3: complex,
    q1: complex,
    q2: complex,
    q3: complex,
) -> complex:
    result = genus2_global_sl2_block_resummed(h1, h2, h3, q1, q2, q3)
    if not result.converged:
        raise RuntimeError(
            "theta global block failed its pointwise endpoint-shell test: "
            f"endpoint_total={result.endpoint_total}, "
            f"last_shell={result.last_shell!r}, value={result.value!r}"
        )
    return result.value


def _matrix_eigenvalue_ratio(a: complex, b: complex, c: complex, d: complex) -> complex:
    trace = a + d
    determinant = a * d - b * c
    discriminant = trace * trace - 4.0 * determinant
    root = cmath.sqrt(discriminant)
    lam_plus = 0.5 * (trace + root)
    lam_minus = 0.5 * (trace - root)
    large = lam_plus if abs(lam_plus) >= abs(lam_minus) else lam_minus
    small = lam_minus if abs(lam_plus) >= abs(lam_minus) else lam_plus
    if abs(large) == 0:
        raise ZeroDivisionError("Mobius word has two zero eigenvalues")
    if abs(small) == 0:
        return 0.0 + 0.0j
    return small / large


def _inverse_word(word: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(inverse_letter(letter) for letter in reversed(word))


def _cyclic_rotations(word: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    if not word:
        return (word,)
    return tuple(word[idx:] + word[:idx] for idx in range(len(word)))


def _is_power_word(word: tuple[int, ...]) -> bool:
    length = len(word)
    for period in range(1, length):
        if length % period == 0 and word == word[:period] * (length // period):
            return True
    return False


def primitive_conjugacy_words(max_word_len: int) -> tuple[tuple[int, ...], ...]:
    """Return canonical primitive conjugacy classes for a rank-two Schottky group."""
    max_word_len = _validate_order(max_word_len)
    seen: set[tuple[int, ...]] = set()
    classes: list[tuple[int, ...]] = []
    for word in reduced_words(2, max_word_len):
        if not word:
            continue
        if word[0] == inverse_letter(word[-1]):
            continue
        if _is_power_word(word):
            continue
        orbit = _cyclic_rotations(word) + _cyclic_rotations(_inverse_word(word))
        key = min(orbit)
        if key in seen:
            continue
        seen.add(key)
        classes.append(key)
    return tuple(classes)


@lru_cache(maxsize=None)
def _primitive_conjugacy_words_cached(max_word_len: int) -> tuple[tuple[int, ...], ...]:
    return primitive_conjugacy_words(max_word_len)


@lru_cache(maxsize=128)
def genus2_vacuum_seed_schottky(
    q1: complex,
    q2: complex,
    q3: complex,
    *,
    max_word_len: int = 3,
    oscillator_level_max: int = 12,
) -> complex:
    r"""Approximate the theta-frame product over unoriented primitive classes.

    CCY write exponent ``-1/2`` while summing oriented conjugacy classes.  The
    primitive-word helper identifies each word with its inverse, so every
    retained class carries exponent ``-1``.
    """
    max_word_len = _validate_order(max_word_len)
    oscillator_level_max = _validate_order(oscillator_level_max)
    if oscillator_level_max < 2 or max_word_len == 0:
        return 1.0 + 0.0j

    surviving = theta_cusp_surviving_multipliers(q1, q2, q3)
    if surviving is not None:
        log_value = 0.0 + 0.0j
        for multiplier in surviving:
            contribution, _ = oscillator_log_factor(
                multiplier,
                max_mode=oscillator_level_max,
                tolerance=1.0e-14,
            )
            log_value += contribution
        return cmath.exp(log_value)

    return schottky_vacuum_block(
        generators_for_theta(q1, q2, q3),
        max_word_length=max_word_len,
        max_mode=oscillator_level_max,
        channel="theta",
        q_values=(complex(q1), complex(q2), complex(q3)),
    ).value


def b_square_rs_from_h(r: int, s: int, h: complex) -> complex:
    """Return b_{r,s}(h)^2 in the CCY c-recursion convention."""
    if r < 2 or s < 1:
        raise ValueError("c-recursion uses r >= 2 and s >= 1")
    h = _as_complex(h)
    radical = (r - s) ** 2 + 4.0 * (r * s - 1.0) * h + 4.0 * h * h
    return (r * s - 1.0 + 2.0 * h + cmath.sqrt(radical)) / (1.0 - r * r)


def central_charge_from_b_square(b_square: complex) -> complex:
    b_square = _as_complex(b_square)
    return 13.0 + 6.0 * (b_square + 1.0 / b_square)


def c_rs_from_h(r: int, s: int, h: complex) -> complex:
    return central_charge_from_b_square(b_square_rs_from_h(r, s, h))


def dc_rs_dh(r: int, s: int, h: complex) -> complex:
    """Return derivative d c_{r,s}(h) / d h."""
    x = b_square_rs_from_h(r, s, h)
    dh_dx = 0.25 * ((1.0 - r * r) - (1.0 - s * s) / (x * x))
    dc_dx = 6.0 * (1.0 - 1.0 / (x * x))
    if abs(dh_dx) == 0:
        raise ZeroDivisionError("dc_rs_dh hit a branch point")
    return dc_dx / dh_dx


def b_from_c_rs_h(r: int, s: int, h: complex) -> complex:
    return cmath.sqrt(b_square_rs_from_h(r, s, h))


def fusion_polynomial_for_weights(
    r: int,
    s: int,
    b: complex,
    top_weight: complex,
    bottom_weight: complex,
) -> complex:
    lambda_top = momentum_from_weight(top_weight, b)
    lambda_bottom = momentum_from_weight(bottom_weight, b)
    return fusion_polynomial(r, s, b, lambda_top, lambda_bottom)


def minus_dc_dh_times_a_rs(r: int, s: int, h: complex) -> complex:
    r"""Return ``-dc_{r,s}/dh * A_{r,s}`` with universal cancellations.

    With ``x=b^2``,

        A_{r,s} = 1/2 x^(2rs-1) / prod'(p x + ell),

    where the primed product omits ``(p,ell)=(0,0),(r,s)``.  Also

        dc/dh = 24 (x^2 - 1) / ((1-r^2)x^2 - (1-s^2)).

    This representation lets us cancel the ``x-1`` and ``x+1`` zeros before
    evaluating the product.  That avoids the resonant ``0 * infinity`` form
    seen at ``c=25`` in higher-order recursion terms.
    """

    x = b_square_rs_from_h(r, s, h)
    numerator = -12.0 * (x ** (2 * r * s - 1))
    denominator = (1.0 - r * r) * x * x - (1.0 - s * s)

    denominator_factors: list[tuple[int, int]] = []
    for p in range(1 - r, r + 1):
        for ell in range(1 - s, s + 1):
            if (p, ell) in {(0, 0), (r, s)}:
                continue
            denominator_factors.append((p, ell))

    numerator_factors: list[tuple[int, int]] = [(1, -1), (1, 1)]
    remaining_numerator_factors: list[tuple[int, int]] = []
    for num_p, num_ell in numerator_factors:
        matched_index = None
        matched_scale = 1
        for idx, (den_p, den_ell) in enumerate(denominator_factors):
            if den_p != 0 and den_p * num_ell == den_ell * num_p:
                matched_index = idx
                matched_scale = den_p // num_p
                break
        if matched_index is None:
            remaining_numerator_factors.append((num_p, num_ell))
        else:
            denominator *= matched_scale
            denominator_factors.pop(matched_index)

    for p, ell in remaining_numerator_factors:
        numerator *= p * x + ell
    for p, ell in denominator_factors:
        denominator *= p * x + ell
    if abs(denominator) == 0:
        raise ZeroDivisionError(f"singular simplified A_rs prefactor for r={r}, s={s}, h={h!r}")
    return numerator / denominator


def _is_finite_complex(value: complex) -> bool:
    return math.isfinite(value.real) and math.isfinite(value.imag)


def ccy_residue_prefactor_for_weights(
    r: int,
    s: int,
    h_edge: complex,
    top_weight: complex,
    bottom_weight: complex,
) -> complex:
    r"""Return the finite CCY residue prefactor.

    The formal prefactor is

        -dc_{r,s}(h)/dh * A_{r,s}(b_{r,s}(h)) * P_{r,s}^2.

    At resonant points, for example at c=25 along recursive shifted weights,
    the separate factors can look like 0 * infinity because A_{r,s} has a
    vanishing denominator while dc/dh vanishes.  The product has a finite
    analytic limit.  We evaluate that limit directly instead of requiring the
    caller to regulate b away from the resonant value.
    """

    h_edge = _as_complex(h_edge)
    top_weight = _as_complex(top_weight)
    bottom_weight = _as_complex(bottom_weight)

    def direct(current_h: complex) -> complex:
        b_pole = b_from_c_rs_h(r, s, current_h)
        polynomial = fusion_polynomial_for_weights(r, s, b_pole, top_weight, bottom_weight)
        return minus_dc_dh_times_a_rs(r, s, current_h) * polynomial * polynomial

    try:
        value = direct(h_edge)
        if _is_finite_complex(value):
            return value
    except ZeroDivisionError:
        pass

    scale = max(1.0, abs(h_edge))
    samples: list[tuple[float, complex]] = []
    for relative_step in (1.0e-5, 3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7, 3.0e-8):
        step = relative_step * scale
        try:
            value = direct(h_edge + step)
        except ZeroDivisionError:
            continue
        if _is_finite_complex(value):
            samples.append((step, value))

    if not samples:
        raise ZeroDivisionError(f"could not resolve resonant CCY residue for r={r}, s={s}, h={h_edge!r}")
    if len(samples) == 1:
        return samples[-1][1]

    step_a, value_a = samples[-2]
    step_b, value_b = samples[-1]
    if step_a == step_b:
        return value_b
    return (step_a * value_b - step_b * value_a) / (step_a - step_b)


@dataclass(frozen=True)
class CCYGenus2BlockResult:
    value: complex
    c: complex
    h1: complex
    h2: complex
    h3: complex
    q1: complex
    q2: complex
    q3: complex
    order: int
    include_vacuum_seed: bool
    vacuum_word_len: int
    vacuum_oscillator_level_max: int
    partial_fraction_pole_count: int = 0
    partial_fraction_coefficient_count: int = 0
    partial_fraction_max_pole_order: int = 0


def _matching_pole(poles: dict[complex, list[complex]], pole: complex, tolerance: float) -> complex | None:
    for existing in poles:
        if abs(existing - pole) < tolerance:
            return existing
    return None


@dataclass
class PartialFractionInC:
    """Finite partial-fraction representation of a meromorphic c-recursion term.

    The represented function is

        constant + sum_p sum_{k>=1} coeff[p][k-1] / (c-p)^k.

    Pole keys are merged within a numerical tolerance because CCY recursion
    revisits the same algebraic pole through different recursive paths.
    """

    constant: complex = 0.0 + 0.0j
    poles: dict[complex, list[complex]] | None = None

    def __post_init__(self) -> None:
        if self.poles is None:
            self.poles = {}

    def add_constant(self, value: complex) -> None:
        self.constant += complex(value)

    def add_pole_coefficient(
        self,
        pole: complex,
        order: int,
        coefficient: complex,
        *,
        pole_tolerance: float,
    ) -> None:
        if order <= 0:
            raise ValueError("partial-fraction pole order must be positive")
        coefficient = complex(coefficient)
        if abs(coefficient) == 0:
            return
        pole = complex(pole)
        key = _matching_pole(self.poles or {}, pole, pole_tolerance)
        if key is None:
            key = pole
            self.poles[key] = []
        coeffs = self.poles[key]
        while len(coeffs) < order:
            coeffs.append(0.0 + 0.0j)
        coeffs[order - 1] += coefficient

    def add(self, other: "PartialFractionInC", *, pole_tolerance: float) -> None:
        self.add_constant(other.constant)
        for pole, coeffs in (other.poles or {}).items():
            for idx, coeff in enumerate(coeffs, start=1):
                self.add_pole_coefficient(pole, idx, coeff, pole_tolerance=pole_tolerance)

    def finite_part_at(self, pole: complex, *, pole_tolerance: float) -> complex:
        pole = complex(pole)
        value = self.constant
        for existing, coeffs in (self.poles or {}).items():
            if abs(existing - pole) < pole_tolerance:
                continue
            delta = pole - existing
            value += sum(coeff / (delta**order) for order, coeff in enumerate(coeffs, start=1))
        return value

    def pole_coefficients_at(self, pole: complex, *, pole_tolerance: float) -> tuple[complex, ...]:
        key = _matching_pole(self.poles or {}, complex(pole), pole_tolerance)
        if key is None:
            return ()
        return tuple(self.poles[key])

    def add_residue_times_laurent_at(
        self,
        *,
        pole: complex,
        residue: complex,
        subblock: "PartialFractionInC",
        pole_tolerance: float,
    ) -> None:
        """Add residue/(c-pole) times the Laurent expansion of subblock at pole."""
        residue = complex(residue)
        finite_part = subblock.finite_part_at(pole, pole_tolerance=pole_tolerance)
        self.add_pole_coefficient(
            pole,
            1,
            residue * finite_part,
            pole_tolerance=pole_tolerance,
        )
        for order, coeff in enumerate(
            subblock.pole_coefficients_at(pole, pole_tolerance=pole_tolerance),
            start=1,
        ):
            self.add_pole_coefficient(
                pole,
                order + 1,
                residue * coeff,
                pole_tolerance=pole_tolerance,
            )

    def value(self, c: complex, *, pole_tolerance: float) -> complex:
        c = complex(c)
        total = self.constant
        for pole, coeffs in (self.poles or {}).items():
            delta = c - pole
            if abs(delta) < pole_tolerance:
                raise ZeroDivisionError(f"requested c={c!r} lies on a c-recursion pole {pole!r}")
            total += sum(coeff / (delta**order) for order, coeff in enumerate(coeffs, start=1))
        return total

    @property
    def max_pole_order(self) -> int:
        return max((len(coeffs) for coeffs in (self.poles or {}).values()), default=0)

    @property
    def pole_count(self) -> int:
        return len(self.poles or {})

    @property
    def coefficient_count(self) -> int:
        return sum(len(coeffs) for coeffs in (self.poles or {}).values())


def ccy_genus2_block_partial_fraction(
    *,
    h1: complex,
    h2: complex,
    h3: complex,
    q1: complex,
    q2: complex,
    q3: complex,
    order: int,
    include_vacuum_seed: bool = True,
    vacuum_word_len: int = 3,
    vacuum_oscillator_level_max: int = 12,
    pole_tolerance: float = 1.0e-12,
) -> PartialFractionInC:
    """Return the CCY genus-two block as a partial fraction in c."""
    order = _validate_order(order)
    q1 = _as_complex(q1)
    q2 = _as_complex(q2)
    q3 = _as_complex(q3)

    vacuum_seed = (
        genus2_vacuum_seed_schottky(
            q1,
            q2,
            q3,
            max_word_len=vacuum_word_len,
            oscillator_level_max=vacuum_oscillator_level_max,
        )
        if include_vacuum_seed
        else 1.0 + 0.0j
    )

    @lru_cache(maxsize=None)
    def recurse(current_h1: complex, current_h2: complex, current_h3: complex, remaining: int) -> PartialFractionInC:
        seed = vacuum_seed * _certified_global_sl2_value(
            current_h1,
            current_h2,
            current_h3,
            q1,
            q2,
            q3,
        )
        total = PartialFractionInC(constant=seed)
        weights = (current_h1, current_h2, current_h3)
        q_values = (q1, q2, q3)
        fusion_pairs = (
            (current_h3, current_h2),
            (current_h3, current_h1),
            (current_h1, current_h2),
        )
        for edge in range(3):
            h_edge = weights[edge]
            for r in range(2, remaining + 1):
                for s in range(1, remaining // r + 1):
                    level = r * s
                    if level > remaining:
                        continue
                    pole_c = c_rs_from_h(r, s, h_edge)
                    top_weight, bottom_weight = fusion_pairs[edge]
                    residue = (q_values[edge] ** level) * ccy_residue_prefactor_for_weights(
                        r, s, h_edge, top_weight, bottom_weight
                    )
                    shifted = list(weights)
                    shifted[edge] = h_edge + level
                    subblock = recurse(shifted[0], shifted[1], shifted[2], remaining - level)
                    total.add_residue_times_laurent_at(
                        pole=pole_c,
                        residue=residue,
                        subblock=subblock,
                        pole_tolerance=pole_tolerance,
                    )
        return total

    return recurse(_as_complex(h1), _as_complex(h2), _as_complex(h3), order)


def ccy_genus2_block(
    *,
    c: complex,
    h1: complex,
    h2: complex,
    h3: complex,
    q1: complex,
    q2: complex,
    q3: complex,
    order: int,
    include_vacuum_seed: bool = True,
    vacuum_word_len: int = 3,
    vacuum_oscillator_level_max: int = 12,
    pole_tolerance: float = 1.0e-12,
    collision_aware: bool = True,
) -> CCYGenus2BlockResult:
    """Evaluate the CCY genus-two block by c-recursion.

    ``order`` truncates only the null-vector c-recursion depth.  The global
    seed is resummed independently and must pass its pointwise endpoint-shell
    convergence test.  The optional vacuum seed is a finite Schottky
    primitive-class product and is therefore an approximation to the exact
    c=infinity vacuum block.
    """
    order = _validate_order(order)
    c = _as_complex(c)
    q1 = _as_complex(q1)
    q2 = _as_complex(q2)
    q3 = _as_complex(q3)

    if collision_aware:
        partial_fraction = ccy_genus2_block_partial_fraction(
            h1=h1,
            h2=h2,
            h3=h3,
            q1=q1,
            q2=q2,
            q3=q3,
            order=order,
            include_vacuum_seed=include_vacuum_seed,
            vacuum_word_len=vacuum_word_len,
            vacuum_oscillator_level_max=vacuum_oscillator_level_max,
            pole_tolerance=pole_tolerance,
        )
        return CCYGenus2BlockResult(
            value=partial_fraction.value(c, pole_tolerance=pole_tolerance),
            c=c,
            h1=_as_complex(h1),
            h2=_as_complex(h2),
            h3=_as_complex(h3),
            q1=q1,
            q2=q2,
            q3=q3,
            order=order,
            include_vacuum_seed=include_vacuum_seed,
            vacuum_word_len=int(vacuum_word_len),
            vacuum_oscillator_level_max=int(vacuum_oscillator_level_max),
            partial_fraction_pole_count=partial_fraction.pole_count,
            partial_fraction_coefficient_count=partial_fraction.coefficient_count,
            partial_fraction_max_pole_order=partial_fraction.max_pole_order,
        )

    vacuum_seed = (
        genus2_vacuum_seed_schottky(
            q1,
            q2,
            q3,
            max_word_len=vacuum_word_len,
            oscillator_level_max=vacuum_oscillator_level_max,
        )
        if include_vacuum_seed
        else 1.0 + 0.0j
    )

    @lru_cache(maxsize=None)
    def recurse(current_c: complex, current_h1: complex, current_h2: complex, current_h3: complex, remaining: int) -> complex:
        seed = vacuum_seed * _certified_global_sl2_value(
            current_h1,
            current_h2,
            current_h3,
            q1,
            q2,
            q3,
        )
        total = seed
        weights = (current_h1, current_h2, current_h3)
        q_values = (q1, q2, q3)
        fusion_pairs = (
            (current_h3, current_h2),
            (current_h3, current_h1),
            (current_h1, current_h2),
        )
        for edge in range(3):
            h_edge = weights[edge]
            for r in range(2, remaining + 1):
                for s in range(1, remaining // r + 1):
                    level = r * s
                    if level > remaining:
                        continue
                    pole_c = c_rs_from_h(r, s, h_edge)
                    denominator = current_c - pole_c
                    if abs(denominator) < pole_tolerance:
                        raise ZeroDivisionError(
                            f"central charge is too close to c_({r},{s})({h_edge!r})={pole_c!r}"
                        )
                    b_pole = b_from_c_rs_h(r, s, h_edge)
                    top_weight, bottom_weight = fusion_pairs[edge]
                    polynomial = fusion_polynomial_for_weights(r, s, b_pole, top_weight, bottom_weight)
                    residue = (
                        -dc_rs_dh(r, s, h_edge)
                        * (q_values[edge] ** level)
                        * zamolodchikov_a_rs(r, s, b_pole)
                        * polynomial
                        * polynomial
                        / denominator
                    )
                    shifted = list(weights)
                    shifted[edge] = h_edge + level
                    total += residue * recurse(pole_c, shifted[0], shifted[1], shifted[2], remaining - level)
        return total

    return CCYGenus2BlockResult(
        value=recurse(c, _as_complex(h1), _as_complex(h2), _as_complex(h3), order),
        c=c,
        h1=_as_complex(h1),
        h2=_as_complex(h2),
        h3=_as_complex(h3),
        q1=q1,
        q2=q2,
        q3=q3,
        order=order,
        include_vacuum_seed=include_vacuum_seed,
        vacuum_word_len=int(vacuum_word_len),
        vacuum_oscillator_level_max=int(vacuum_oscillator_level_max),
    )


def liouville_c_and_weight(b: float, momentum: float) -> tuple[float, float]:
    q_background = b + 1.0 / b
    return 1.0 + 6.0 * q_background * q_background, 0.25 * q_background * q_background + momentum * momentum


def run() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the CCY genus-two Virasoro block.")
    parser.add_argument("--b", type=float, help="Liouville b; if set, c and h_i are read from P_i.")
    parser.add_argument("--p1", type=float, default=0.2)
    parser.add_argument("--p2", type=float, default=0.3)
    parser.add_argument("--p3", type=float, default=0.25)
    parser.add_argument("--c", type=parse_complex)
    parser.add_argument("--h1", type=parse_complex)
    parser.add_argument("--h2", type=parse_complex)
    parser.add_argument("--h3", type=parse_complex)
    parser.add_argument("--q1", type=parse_complex, required=True)
    parser.add_argument("--q2", type=parse_complex, required=True)
    parser.add_argument("--q3", type=parse_complex, required=True)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--no-vacuum-seed", action="store_true")
    parser.add_argument("--vacuum-word-len", type=int, default=3)
    parser.add_argument("--vacuum-oscillator-level-max", type=int, default=12)
    parser.add_argument("--period-matrix-word-len", type=int, default=5)
    args = parser.parse_args()

    if args.b is not None:
        c_value, h1 = liouville_c_and_weight(args.b, args.p1)
        _, h2 = liouville_c_and_weight(args.b, args.p2)
        _, h3 = liouville_c_and_weight(args.b, args.p3)
    else:
        if args.c is None or args.h1 is None or args.h2 is None or args.h3 is None:
            raise ValueError("pass either --b with --p1/--p2/--p3 or explicit --c/--h1/--h2/--h3")
        c_value, h1, h2, h3 = args.c, args.h1, args.h2, args.h3

    result = ccy_genus2_block(
        c=c_value,
        h1=h1,
        h2=h2,
        h3=h3,
        q1=args.q1,
        q2=args.q2,
        q3=args.q3,
        order=args.order,
        include_vacuum_seed=not args.no_vacuum_seed,
        vacuum_word_len=args.vacuum_word_len,
        vacuum_oscillator_level_max=args.vacuum_oscillator_level_max,
    )

    print("CCY genus-two Virasoro block")
    print(f"  c={format_complex(result.c)}")
    print(f"  h1={format_complex(result.h1)}")
    print(f"  h2={format_complex(result.h2)}")
    print(f"  h3={format_complex(result.h3)}")
    print(f"  q1={format_complex(result.q1)}")
    print(f"  q2={format_complex(result.q2)}")
    print(f"  q3={format_complex(result.q3)}")
    print(f"  order={result.order}")
    print(f"  vacuum seed={result.include_vacuum_seed}")
    print(f"  c-poles={result.partial_fraction_pole_count}")
    print(f"  c-pole coefficients={result.partial_fraction_coefficient_count}")
    print(f"  max c-pole order={result.partial_fraction_max_pole_order}")
    print(f"  value={format_complex(result.value)}")
    omega = schottky_theta_period_matrix_cross_ratio(
        result.q1,
        result.q2,
        result.q3,
        max_word_len=args.period_matrix_word_len,
    )
    print(f"  period matrix word length={args.period_matrix_word_len}")
    print("  Omega=")
    for row in omega:
        print("    " + "  ".join(format_complex(complex(entry)) for entry in row))


if __name__ == "__main__":
    run()
