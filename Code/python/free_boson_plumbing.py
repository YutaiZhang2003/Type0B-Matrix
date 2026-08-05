#!/usr/bin/env python3
"""Free-boson plumbing diagnostics in genus two.

This module intentionally separates three pieces of data:

* the free-Heisenberg primitive-word oscillator product, computed from the same
  plumbing-derived Schottky generators used elsewhere in this directory;
* the noncompact scalar handle-momentum Gaussian
  ``det(Im Omega)^(-1/2)``;
* the Bergman-metric determinant candidate from the genus-two Petersson norm
  ``F = det(Im Omega)^(5/2) prod_even |theta[delta](0|Omega)|``.

The independent Fock-space pair-of-pants sum lives in
``free_boson_pair_of_pants.py``.  Its agreement with this primitive-word
product is checked before it is used as an all-level oscillator evaluation.  This
``n >= 1`` product is not the CCY ``n >= 2`` large-c Virasoro vacuum seed or
the spectral determinant of the Bergman Laplacian.

For one real scalar with its zero mode omitted, the canonical determinant
power is ``(det' Delta_B)^(-1/2)``.  The moduli-independent determinant
constant cancels in a theta/glasses ratio evaluated at the same period matrix.
"""

from __future__ import annotations

import argparse
import cmath
import math
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np

try:
    from genus2_vacuum_blocks import (
        PrimitiveClassContribution,
        format_complex,
        primitive_conjugacy_words,
        word_multiplier,
    )
    from plumbing_algorithms import (
        GeneratorData,
        generators_for_glasses,
        generators_for_theta,
        schottky_glasses_period_matrix,
        schottky_theta_period_matrix_cross_ratio,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.genus2_vacuum_blocks import (
        PrimitiveClassContribution,
        format_complex,
        primitive_conjugacy_words,
        word_multiplier,
    )
    from plumbing.plumbing_algorithms import (
        GeneratorData,
        generators_for_glasses,
        generators_for_theta,
        schottky_glasses_period_matrix,
        schottky_theta_period_matrix_cross_ratio,
    )


def parse_complex(value: str) -> complex:
    """Parse command-line complex numbers without importing Liouville code."""

    return complex(value.replace("i", "j"))


@dataclass(frozen=True)
class FreeBosonProductResult:
    """Truncated nonchiral scalar oscillator product in a Schottky frame."""

    channel: str
    q_values: tuple[complex, complex, complex]
    max_word_length: int
    max_mode: int
    tolerance: float
    chiral_log_product: complex
    nonchiral_log_value: float
    nonchiral_value: float
    primitive_count: int
    omitted_chiral_tail_estimate: float
    contributions: tuple[PrimitiveClassContribution, ...]


@dataclass(frozen=True)
class BergmanDeterminantCandidate:
    """Genus-two Bergman determinant candidate, up to an overall constant."""

    det_im_omega: float
    chi10_product: complex
    petersson_norm_delta2: float
    determinant_factor: float
    determinant_exponent: float
    partition_candidate: float


def noncompact_scalar_loop_momentum_factor(omega: np.ndarray) -> float:
    r"""Return the dimensionless handle Gaussian ``det(Im Omega)^(-1/2)``.

    This is the repository's dimensionless sewing convention.  With
    ``p=sqrt(alpha') k``, momentum states obey
    ``<p|p'>=delta(p-p')`` and use completeness measure ``dp``.  At genus
    two the two loop-momentum integrals give

    ``integral d^2p exp(-pi p^T Im(Omega) p)=det(Im Omega)^(-1/2)``.

    This is not Xi's physical-momentum measure.  Xi uses ``dk_I/(2 pi)`` on
    each handle, so at genus two the corresponding per-target-volume factor
    is smaller by ``(2 pi sqrt(alpha'))^2 = 4 pi^2 alpha'``.  Use
    :func:`xi_noncompact_scalar_loop_momentum_factor` for that convention.
    """

    omega = np.asarray(omega, dtype=np.complex128)
    if omega.shape != (2, 2):
        raise ValueError("the genus-two loop-momentum factor requires a 2x2 period matrix")
    omega = 0.5 * (omega + omega.T)
    det_im = float(np.linalg.det(np.asarray(omega.imag, dtype=np.float64)))
    if det_im <= 0.0:
        raise ValueError("Need det(Im Omega)>0 for the scalar loop-momentum factor")
    return float(det_im**-0.5)


def xi_noncompact_scalar_loop_momentum_factor(
    omega: np.ndarray,
    *,
    alpha_prime: float = 1.0,
) -> float:
    r"""Return Xi's genus-two ``dk_I/(2 pi)`` loop-momentum Gaussian.

    For one noncompact scalar divided by ordinary target-space volume,

    ``integral prod_{I=1}^2 [dk_I/(2 pi)]
       exp(-pi alpha' k^T Im(Omega) k)
       = 1/(4 pi^2 alpha' sqrt(det Im(Omega)))``.

    The oscillator partition is unaffected by this constant conversion.
    """

    alpha_prime = float(alpha_prime)
    if not math.isfinite(alpha_prime) or alpha_prime <= 0.0:
        raise ValueError("alpha_prime must be positive and finite")
    return float(
        noncompact_scalar_loop_momentum_factor(omega)
        / (4.0 * math.pi * math.pi * alpha_prime)
    )


def noncompact_scalar_zero_mode_factor(omega: np.ndarray) -> float:
    r"""Backward-compatible alias for the handle-momentum Gaussian.

    Despite the historical function name, this is not the connected constant
    target-space zero mode.  It is the dimensionless two-handle momentum
    Gaussian.  Xi's per-ordinary-target-volume Gaussian is returned by
    :func:`xi_noncompact_scalar_loop_momentum_factor`.
    """

    return noncompact_scalar_loop_momentum_factor(omega)


def noncompact_scalar_zero_mode_ratio(
    omega_numerator: np.ndarray,
    omega_denominator: np.ndarray,
) -> float:
    """Return the handle-momentum factor in the numerator/denominator direction."""

    return (
        noncompact_scalar_loop_momentum_factor(omega_numerator)
        / noncompact_scalar_loop_momentum_factor(omega_denominator)
    )


def torus_raw_oscillator_abs(
    q_value: complex,
    *,
    max_mode: int = 200,
    tolerance: float = 1.0e-15,
) -> float:
    """Return ``prod_{n>=1} |1-q^n|^-2`` in the raw sewing convention."""

    q = complex(q_value)
    if not abs(q) < 1.0:
        raise ValueError("torus raw oscillator requires |q|<1")
    log_value = 0.0
    for mode in range(1, int(max_mode) + 1):
        power = q**mode
        log_value += -2.0 * math.log(abs(1.0 - power))
        if abs(power) < float(tolerance):
            break
    return float(math.exp(log_value))


def dedekind_eta_abs_from_q(
    q_value: complex,
    *,
    max_mode: int = 200,
    tolerance: float = 1.0e-15,
) -> float:
    """Return ``|eta(tau)|`` for ``q = exp(2 pi i tau)``."""

    q = complex(q_value)
    q_abs = abs(q)
    if not 0.0 < q_abs < 1.0:
        raise ValueError("Dedekind eta product requires 0<|q|<1")
    log_value = (1.0 / 24.0) * math.log(q_abs)
    for mode in range(1, int(max_mode) + 1):
        power = q**mode
        log_value += math.log(abs(1.0 - power))
        if abs(power) < float(tolerance):
            break
    return float(math.exp(log_value))


def tau_imag_from_q(q_value: complex) -> float:
    """Return ``Im(tau)`` for ``q = exp(2 pi i tau)``."""

    q_abs = abs(complex(q_value))
    if not 0.0 < q_abs < 1.0:
        raise ValueError("tau_imag_from_q requires 0<|q|<1")
    return float(-math.log(q_abs) / (2.0 * math.pi))


def glasses_separating_F_asymptotic(
    q1: complex,
    q2: complex,
    q_bridge: complex,
    *,
    eta_max_mode: int = 200,
    eta_tolerance: float = 1.0e-15,
) -> float:
    r"""Leading separating asymptotic of the Bergman ``F`` in glasses plumbing.

    With the current theta-constant normalization,

    ``prod_even theta_delta(0|Omega)^2
       ~ 4096 (2*pi)^2 Omega_12^2 eta(tau_1)^24 eta(tau_2)^24``

    and the glasses period map has ``Omega_12 ~ q_bridge/(2*pi*i)``.  Therefore

    ``F ~ 64 |q_bridge| (Im tau_1 Im tau_2)^(5/2) |eta_1|^12 |eta_2|^12``.
    """

    y1 = tau_imag_from_q(q1)
    y2 = tau_imag_from_q(q2)
    eta1 = dedekind_eta_abs_from_q(q1, max_mode=eta_max_mode, tolerance=eta_tolerance)
    eta2 = dedekind_eta_abs_from_q(q2, max_mode=eta_max_mode, tolerance=eta_tolerance)
    return float(64.0 * abs(complex(q_bridge)) * ((y1 * y2) ** 2.5) * (eta1**12) * (eta2**12))


def theta_tropical_imag_period_matrix(q_zero: complex, q_one: complex, q_infty: complex) -> np.ndarray:
    """Leading imaginary period matrix in the maximally degenerate theta chart."""

    y_zero = tau_imag_from_q(q_zero)
    y_one = tau_imag_from_q(q_one)
    y_infty = tau_imag_from_q(q_infty)
    return np.asarray(
        [
            [y_zero + y_infty, y_infty],
            [y_infty, y_one + y_infty],
        ],
        dtype=np.float64,
    )


def theta_maximal_F_asymptotic(q_zero: complex, q_one: complex, q_infty: complex) -> float:
    r"""Leading maximal-degeneration asymptotic of the Bergman ``F``.

    In the theta graph plumbing frame all three tubes are long.  With the
    current theta-constant normalization,

    ``prod_even theta_delta(0|Omega)^2 ~ 4096 q_zero q_one q_infty``.

    Hence

    ``F ~ 64 |q_zero q_one q_infty|^(1/2) det(Im Omega_trop)^(5/2)``.
    """

    y_tropical = theta_tropical_imag_period_matrix(q_zero, q_one, q_infty)
    det_y = float(np.linalg.det(y_tropical))
    if det_y <= 0.0:
        raise ValueError("theta tropical period matrix must have positive determinant")
    q_product_abs = abs(complex(q_zero) * complex(q_one) * complex(q_infty))
    return float(64.0 * math.sqrt(q_product_abs) * (det_y**2.5))


def theta_maximal_raw_oscillator_asymptotic(
    q_zero: complex,
    q_one: complex,
    q_infty: complex,
) -> float:
    """Raw scalar oscillator factor in the maximally degenerate theta limit."""

    for q_value in (q_zero, q_one, q_infty):
        if not abs(complex(q_value)) < 1.0:
            raise ValueError("theta maximal asymptotic requires |q_i|<1")
    return 1.0


def glasses_separating_raw_oscillator_asymptotic(
    q1: complex,
    q2: complex,
    *,
    max_mode: int = 200,
    tolerance: float = 1.0e-15,
) -> float:
    """Raw scalar oscillator factor in the separating glasses limit."""

    return torus_raw_oscillator_abs(q1, max_mode=max_mode, tolerance=tolerance) * torus_raw_oscillator_abs(
        q2,
        max_mode=max_mode,
        tolerance=tolerance,
    )


def plumbing_over_bergman_frame_factor(
    raw_oscillator: float,
    exact_F: float,
    *,
    determinant_exponent: float = -0.5,
) -> float:
    r"""Return ``W_pl/B = Z_pl / Z_Bergman`` for a c=1 scalar oscillator.

    The Bergman candidate is ``Z_Bergman = F^(determinant_exponent/3)`` up to
    the moduli-independent determinant constant.  Hence the ratio measured by
    the free scalar is

    ``W_pl/B = Z_pl * F^(-determinant_exponent/3)``.
    """

    if raw_oscillator <= 0.0 or exact_F <= 0.0:
        raise ValueError("frame-factor inputs must be positive")
    return float(raw_oscillator) * (float(exact_F) ** (-float(determinant_exponent) / 3.0))


def bergman_over_plumbing_frame_factor(
    raw_oscillator: float,
    exact_F: float,
    *,
    determinant_exponent: float = -0.5,
) -> float:
    """Return the inverse free-scalar frame factor ``Z_Bergman / Z_pl``."""

    return 1.0 / plumbing_over_bergman_frame_factor(
        raw_oscillator,
        exact_F,
        determinant_exponent=determinant_exponent,
    )


def long_tube_normalized_frame_factor(
    raw_oscillator: float,
    exact_F: float,
    asymptotic_raw_oscillator: float,
    asymptotic_F: float,
    *,
    determinant_exponent: float = -0.5,
) -> float:
    """Normalize ``W_pl/B = Z_pl / Z_Bergman`` by its long-tube asymptotic."""

    if min(raw_oscillator, exact_F, asymptotic_raw_oscillator, asymptotic_F) <= 0.0:
        raise ValueError("all frame-factor inputs must be positive")
    frame = plumbing_over_bergman_frame_factor(
        raw_oscillator,
        exact_F,
        determinant_exponent=determinant_exponent,
    )
    frame_asymptotic = plumbing_over_bergman_frame_factor(
        asymptotic_raw_oscillator,
        asymptotic_F,
        determinant_exponent=determinant_exponent,
    )
    return float(frame / frame_asymptotic)


def free_boson_chiral_log_factor(
    multiplier: complex,
    *,
    max_mode: int,
    tolerance: float,
) -> tuple[complex, float]:
    """Return ``-sum_{n>=1} log(1-k^n)`` for one primitive multiplier."""

    k = complex(multiplier)
    k_abs = abs(k)
    if k_abs >= 1.0:
        raise ValueError(f"Schottky multiplier must have |k|<1, got |k|={k_abs:.6g}")
    if max_mode < 1:
        raise ValueError("max_mode must be at least 1")

    total = 0.0j
    last_mode = int(max_mode)
    for mode in range(1, int(max_mode) + 1):
        power = k**mode
        total += -cmath.log(1.0 - power)
        last_mode = mode
        if abs(power) < float(tolerance):
            break

    if k_abs == 0.0:
        tail = 0.0
    else:
        first = k_abs ** (last_mode + 1)
        tail = first / max(1.0e-300, (1.0 - k_abs) * (1.0 - first))
    return total, float(tail)


def free_boson_schottky_product(
    generators: Sequence[GeneratorData],
    *,
    channel: str,
    q_values: tuple[complex, complex, complex],
    max_word_length: int = 8,
    max_mode: int = 80,
    tolerance: float = 1.0e-14,
) -> FreeBosonProductResult:
    """Evaluate the exact free-Heisenberg ``n >= 1`` oscillator product.

    This is not the ``n >= 2`` large-c Virasoro vacuum seed used as one input
    to the CCY recursion, nor is it the Bergman spectral determinant.
    """

    if len(generators) != 2:
        raise ValueError("the genus-two free-boson product expects two Schottky generators")
    words = primitive_conjugacy_words(len(generators), int(max_word_length))
    contributions: list[PrimitiveClassContribution] = []
    chiral_log = 0.0j
    omitted = 0.0
    for word in words:
        multiplier = word_multiplier(generators, word)
        log_factor, tail = free_boson_chiral_log_factor(
            multiplier,
            max_mode=max_mode,
            tolerance=tolerance,
        )
        chiral_log += log_factor
        omitted += tail
        contributions.append(
            PrimitiveClassContribution(
                word=word,
                multiplier=multiplier,
                log_factor=log_factor,
            )
        )
    nonchiral_log = float(2.0 * chiral_log.real)
    return FreeBosonProductResult(
        channel=channel,
        q_values=tuple(complex(value) for value in q_values),
        max_word_length=int(max_word_length),
        max_mode=int(max_mode),
        tolerance=float(tolerance),
        chiral_log_product=chiral_log,
        nonchiral_log_value=nonchiral_log,
        nonchiral_value=float(math.exp(nonchiral_log)),
        primitive_count=len(contributions),
        omitted_chiral_tail_estimate=float(omitted),
        contributions=tuple(contributions),
    )


def glasses_free_boson_product(
    q1: complex,
    q2: complex,
    q_bridge: complex,
    *,
    max_word_length: int = 8,
    max_mode: int = 80,
    tolerance: float = 1.0e-14,
) -> FreeBosonProductResult:
    q_values = (complex(q1), complex(q2), complex(q_bridge))
    return free_boson_schottky_product(
        generators_for_glasses(*q_values),
        channel="glasses",
        q_values=q_values,
        max_word_length=max_word_length,
        max_mode=max_mode,
        tolerance=tolerance,
    )


def theta_free_boson_product(
    q_zero: complex,
    q_one: complex,
    q_infty: complex,
    *,
    max_word_length: int = 8,
    max_mode: int = 80,
    tolerance: float = 1.0e-14,
) -> FreeBosonProductResult:
    q_values = (complex(q_zero), complex(q_one), complex(q_infty))
    return free_boson_schottky_product(
        generators_for_theta(*q_values),
        channel="theta",
        q_values=q_values,
        max_word_length=max_word_length,
        max_mode=max_mode,
        tolerance=tolerance,
    )


def bergman_petersson_norm_delta2(
    omega: np.ndarray,
    *,
    theta_nmax: int | None = None,
    theta_tol: float = 1.0e-12,
    chi10_normalization: str = "product",
) -> tuple[float, complex, float]:
    r"""Return ``det(Im Omega)^(5/2) prod_even |theta[delta](0|Omega)|``.

    The repository's ``igusa_chi10_genus2(..., normalization="product")``
    returns ``prod_even theta[delta]^2``.  Thus the absolute even-theta product
    in the Petersson norm is ``abs(chi10_product)^(1/2)``.
    """

    omega = np.asarray(omega, dtype=np.complex128)
    im_omega = np.asarray(np.imag(omega), dtype=np.float64)
    det_im = float(np.linalg.det(im_omega))
    if det_im <= 0.0:
        raise ValueError("Need det(Im Omega)>0 for the Bergman determinant candidate")
    chi10 = complex(
        igusa_chi10_genus2(
            omega,
            nmax=theta_nmax,
            tol=theta_tol,
            normalization=chi10_normalization,
        )
    )
    norm = (det_im ** 2.5) * (abs(chi10) ** 0.5)
    return det_im, chi10, float(norm)


def genus2_even_characteristics() -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Return the ten even genus-two theta characteristics."""

    chars = []
    for a_bits in product((0, 1), repeat=2):
        for b_bits in product((0, 1), repeat=2):
            parity = (a_bits[0] * b_bits[0] + a_bits[1] * b_bits[1]) % 2
            if parity == 0:
                chars.append((tuple(int(x) for x in a_bits), tuple(int(x) for x in b_bits)))
    return tuple(chars)


def _theta_truncation_genus2(omega: np.ndarray, tol: float) -> int:
    """Choose a square lattice cutoff for direct genus-two theta summation."""

    im_omega = np.asarray(np.imag(omega), dtype=np.float64)
    evals = np.linalg.eigvalsh(im_omega)
    lam_min = float(np.min(evals))
    if lam_min <= 0.0:
        raise ValueError("Need Im(Omega) positive definite to evaluate theta constants")
    return max(4, int(math.ceil(math.sqrt(-math.log(max(float(tol), 1.0e-16)) / (math.pi * lam_min)))) + 2)


def riemann_theta_constant_genus2(
    omega: np.ndarray,
    characteristic: tuple[tuple[int, int], tuple[int, int]],
    *,
    nmax: int | None = None,
    tol: float = 1.0e-12,
) -> complex:
    r"""Compute a theta constant in the repository characteristic convention.

    For binary vectors ``a,b``, the implemented definition is

    ``sum_n exp(i*pi*((n+a/2)^T Omega (n+a/2) + (n+a/2)^T b))``.

    In particular, no additional characteristic-dependent phase is applied.
    """

    omega = np.asarray(omega, dtype=np.complex128)
    if omega.shape != (2, 2):
        raise ValueError("riemann_theta_constant_genus2 requires a 2x2 period matrix")
    if nmax is None:
        nmax = _theta_truncation_genus2(omega, tol)

    a_bits, b_bits = characteristic
    eps = 0.5 * np.asarray(a_bits, dtype=np.float64)
    delta = 0.5 * np.asarray(b_bits, dtype=np.float64)
    rng = np.arange(-int(nmax), int(nmax) + 1, dtype=np.float64)
    n1, n2 = np.meshgrid(rng, rng, indexing="ij")
    v1 = n1 + eps[0]
    v2 = n2 + eps[1]
    quad_form = omega[0, 0] * v1 * v1 + 2.0 * omega[0, 1] * v1 * v2 + omega[1, 1] * v2 * v2
    linear_part = 2.0 * (v1 * delta[0] + v2 * delta[1])
    terms = np.exp(1j * math.pi * (quad_form + linear_part))
    return complex(np.sum(terms))


def riemann_theta_constant_log_abs_genus2(
    omega: np.ndarray,
    characteristic: tuple[tuple[int, int], tuple[int, int]],
    *,
    nmax: int | None = None,
    tol: float = 1.0e-12,
) -> float:
    """Return ``log|theta[characteristic](0|Omega)|`` without underflow."""

    omega = np.asarray(omega, dtype=np.complex128)
    if omega.shape != (2, 2):
        raise ValueError("riemann_theta_constant_log_abs_genus2 requires a 2x2 period matrix")
    if nmax is None:
        nmax = _theta_truncation_genus2(omega, tol)

    a_bits, b_bits = characteristic
    eps = 0.5 * np.asarray(a_bits, dtype=np.float64)
    delta = 0.5 * np.asarray(b_bits, dtype=np.float64)
    rng = np.arange(-int(nmax), int(nmax) + 1, dtype=np.float64)
    n1, n2 = np.meshgrid(rng, rng, indexing="ij")
    v1 = n1 + eps[0]
    v2 = n2 + eps[1]
    quad_form = omega[0, 0] * v1 * v1 + 2.0 * omega[0, 1] * v1 * v2 + omega[1, 1] * v2 * v2
    linear_part = 2.0 * (v1 * delta[0] + v2 * delta[1])
    exponents = 1j * math.pi * (quad_form + linear_part)
    shift = float(np.max(exponents.real))
    scaled_sum = complex(np.sum(np.exp(exponents - shift)))
    if scaled_sum == 0.0:
        raise ValueError("theta-constant lattice sum cancelled to zero")
    return float(shift + math.log(abs(scaled_sum)))


def igusa_chi10_genus2(
    omega: np.ndarray,
    *,
    nmax: int | None = None,
    tol: float = 1.0e-12,
    normalization: str = "product",
) -> complex:
    r"""Return the weight-ten even-theta product in a named convention.

    ``product`` is Moore's raw product

    ``Psi_10 = prod_even theta[delta](0|Omega)^2``.

    ``string_note_2^-12`` is the Fourier-normalized form used in the string
    notes, ``chi_10 = 2^-12 Psi_10``.  The older name ``igusa_2^-12`` is kept
    as a compatibility alias.  ``igusa_2^-14`` is the separate algebraic
    convention ``-2^-14 Psi_10``.
    """

    product_form = 1.0 + 0.0j
    for characteristic in genus2_even_characteristics():
        theta = riemann_theta_constant_genus2(omega, characteristic, nmax=nmax, tol=tol)
        product_form *= theta * theta

    if normalization == "product":
        return product_form
    if normalization in {"string_note_2^-12", "igusa_2^-12"}:
        return (2.0**-12) * product_form
    if normalization == "igusa_2^-14":
        return -(2.0**-14) * product_form
    raise ValueError(f"unsupported chi10 normalization {normalization!r}")


def igusa_chi10_log_abs_genus2(
    omega: np.ndarray,
    *,
    nmax: int | None = None,
    tol: float = 1.0e-12,
    normalization: str = "product",
) -> float:
    """Return ``log|chi10(Omega)|`` while retaining arbitrarily long cusps."""

    log_abs = 2.0 * sum(
        riemann_theta_constant_log_abs_genus2(
            omega,
            characteristic,
            nmax=nmax,
            tol=tol,
        )
        for characteristic in genus2_even_characteristics()
    )
    if normalization == "product":
        return float(log_abs)
    if normalization in {"string_note_2^-12", "igusa_2^-12"}:
        return float(log_abs - 12.0 * math.log(2.0))
    if normalization == "igusa_2^-14":
        return float(log_abs - 14.0 * math.log(2.0))
    raise ValueError(f"unsupported chi10 normalization {normalization!r}")


def bergman_scalar_partition_candidate(
    omega: np.ndarray,
    *,
    determinant_exponent: float = -0.5,
    theta_nmax: int | None = None,
    theta_tol: float = 1.0e-12,
    chi10_normalization: str = "product",
) -> BergmanDeterminantCandidate:
    r"""Return a Bergman free-boson candidate from the KKK determinant formula.

    Klein-Kokotov-Korotkin give ``det Delta_B = const * F^(1/3)`` for
    ``F = det(Im Omega)^(5/2) prod_even |theta_even|``.  The default exponent
    is ``-1/2`` for one real Gaussian scalar with its zero mode omitted.  The
    overall moduli-independent constant is not fixed by this function.
    """

    det_im, chi10, norm = bergman_petersson_norm_delta2(
        omega,
        theta_nmax=theta_nmax,
        theta_tol=theta_tol,
        chi10_normalization=chi10_normalization,
    )
    determinant_factor = norm ** (1.0 / 3.0)
    partition_candidate = determinant_factor ** float(determinant_exponent)
    return BergmanDeterminantCandidate(
        det_im_omega=det_im,
        chi10_product=chi10,
        petersson_norm_delta2=norm,
        determinant_factor=float(determinant_factor),
        determinant_exponent=float(determinant_exponent),
        partition_candidate=float(partition_candidate),
    )


def _print_product(result: FreeBosonProductResult) -> None:
    print(f"{result.channel} raw free-boson oscillator")
    print(f"  q = ({', '.join(format_complex(value) for value in result.q_values)})")
    print(f"  primitive classes = {result.primitive_count}")
    print(f"  chiral log product = {format_complex(result.chiral_log_product)}")
    print(f"  nonchiral value = {result.nonchiral_value:.16e}")
    print(f"  omitted chiral tail estimate <= {result.omitted_chiral_tail_estimate:.3e}")


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Genus-two free-boson plumbing/Bergman diagnostic.")
    parser.add_argument("--channel", choices=("glasses", "theta"), default="glasses")
    parser.add_argument("--q", type=parse_complex, nargs=3, default=(0.15 + 0.0j, 0.15 + 0.0j, 0.15 + 0.0j))
    parser.add_argument("--max-word-length", type=int, default=8)
    parser.add_argument("--max-mode", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=1.0e-14)
    parser.add_argument("--period-word-length", type=int, default=8)
    parser.add_argument("--period-b-order", type=int, default=600)
    parser.add_argument("--theta-nmax", type=int, default=8)
    parser.add_argument("--theta-tol", type=float, default=1.0e-12)
    parser.add_argument(
        "--determinant-exponent",
        type=float,
        nargs="+",
        default=(-1.0, -0.5),
        help="powers of det Delta_B to report; constants are not fixed",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    q_values = tuple(complex(value) for value in args.q)
    if args.channel == "glasses":
        product = glasses_free_boson_product(
            *q_values,
            max_word_length=args.max_word_length,
            max_mode=args.max_mode,
            tolerance=args.tolerance,
        )
        omega = schottky_glasses_period_matrix(
            *q_values,
            max_word_len=args.period_word_length,
            b_order=args.period_b_order,
        )
    else:
        product = theta_free_boson_product(
            *q_values,
            max_word_length=args.max_word_length,
            max_mode=args.max_mode,
            tolerance=args.tolerance,
        )
        omega = schottky_theta_period_matrix_cross_ratio(*q_values, max_word_len=args.period_word_length)

    _print_product(product)
    print("Bergman determinant candidate")
    for exponent in args.determinant_exponent:
        candidate = bergman_scalar_partition_candidate(
            omega,
            determinant_exponent=float(exponent),
            theta_nmax=args.theta_nmax,
            theta_tol=args.theta_tol,
        )
        frame_factor = plumbing_over_bergman_frame_factor(
            product.nonchiral_value,
            candidate.petersson_norm_delta2,
            determinant_exponent=float(exponent),
        )
        print(f"  det exponent {float(exponent):+.6g}:")
        print(f"    det(Im Omega) = {candidate.det_im_omega:.16e}")
        print(f"    |chi10_product| = {abs(candidate.chi10_product):.16e}")
        print(f"    F = {candidate.petersson_norm_delta2:.16e}")
        print(f"    detDelta candidate = {candidate.determinant_factor:.16e}")
        print(f"    Bergman partition candidate = {candidate.partition_candidate:.16e}")
        print(f"    W_pl/B = Z_plumbing / Z_Bergman = {frame_factor:.16e}")


if __name__ == "__main__":
    run()
