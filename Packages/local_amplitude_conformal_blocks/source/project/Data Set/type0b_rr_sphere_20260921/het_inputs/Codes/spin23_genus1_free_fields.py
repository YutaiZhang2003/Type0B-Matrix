#!/usr/bin/env python3
r"""Genus-one theta, prime-form, and free-field primitives.

The torus coordinate in this module has periods ``1`` and ``tau``.  Theta
functions use

.. math::

   \vartheta\!\left[\begin{smallmatrix}a\\b\end{smallmatrix}\right]
   (z|\tau)
   =\sum_{n\in\mathbb Z}
    e^{\pi i(n+a)^2\tau+2\pi i(n+a)(z+b)}.

This differs by a factor of :math:`2\pi` in the coordinate from conventions
where the torus periods are :math:`2\pi` and :math:`2\pi\tau`.
"""

from __future__ import annotations

import cmath
import math
from typing import Sequence

import mpmath as mp
import numpy as np

from spin23_genus1_spin import TorusSpinStructure


def _validate_tau(tau: complex) -> complex:
    tau = complex(tau)
    if not math.isfinite(tau.real) or not math.isfinite(tau.imag):
        raise ValueError("tau must be finite")
    if tau.imag <= 0:
        raise ValueError("tau must lie in the upper half-plane")
    return tau


def _theta_cutoff(
    z: complex,
    tau: complex,
    a: float,
    precision: int,
) -> int:
    target = max(10, int(precision)) * math.log(10.0)
    shift = abs(z.imag) / tau.imag + abs(a)
    growth = math.pi * z.imag * z.imag / tau.imag
    return int(math.ceil(shift + math.sqrt((target + growth) / (math.pi * tau.imag)))) + 3


def theta_characteristic(
    a: float,
    b: float,
    z: complex,
    tau: complex,
    *,
    derivative_order: int = 0,
    precision: int = 40,
) -> complex:
    r"""Evaluate a genus-one theta function or a ``z`` derivative."""

    tau = _validate_tau(tau)
    z = complex(z)
    a = float(a)
    b = float(b)
    if a not in (0.0, 0.5) or b not in (0.0, 0.5):
        raise ValueError("a and b must each be 0 or 1/2")
    if not isinstance(derivative_order, int) or derivative_order < 0:
        raise ValueError("derivative_order must be a nonnegative integer")
    cutoff = _theta_cutoff(z, tau, a, precision)
    with mp.workdps(precision):
        z_mp = mp.mpc(z)
        tau_mp = mp.mpc(tau)
        total = mp.mpc(0)
        for n in range(-cutoff, cutoff + 1):
            shifted = mp.mpf(n) + a
            term = mp.exp(
                mp.pi * 1j * shifted * shifted * tau_mp
                + 2 * mp.pi * 1j * shifted * (z_mp + b)
            )
            if derivative_order:
                term *= (2 * mp.pi * 1j * shifted) ** derivative_order
            total += term
        return complex(total)


def dedekind_eta(tau: complex, *, precision: int = 40) -> complex:
    r"""Evaluate :math:`\eta(\tau)=q^{1/24}\prod_{n\geq1}(1-q^n)`."""

    tau = _validate_tau(tau)
    q = cmath.exp(2j * math.pi * tau)
    if abs(q) == 0:
        return 0.0j
    cutoff = int(math.ceil(precision * math.log(10.0) / -math.log(abs(q)))) + 3
    if cutoff > 1_000_000:
        raise ArithmeticError("eta product requires modular reduction at this tau")
    with mp.workdps(precision):
        q_mp = mp.mpc(q)
        product_value = mp.mpc(1)
        q_power = q_mp
        for _ in range(1, cutoff + 1):
            product_value *= 1 - q_power
            q_power *= q_mp
        return complex(mp.exp(mp.pi * 1j * mp.mpc(tau) / 12) * product_value)


def odd_theta(z: complex, tau: complex, *, precision: int = 40) -> complex:
    """Return the odd theta characteristic; its overall sign is immaterial."""

    return theta_characteristic(0.5, 0.5, z, tau, precision=precision)


def odd_theta_derivative(
    z: complex,
    tau: complex,
    *,
    precision: int = 40,
) -> complex:
    """Return the first coordinate derivative of the odd theta function."""

    return theta_characteristic(
        0.5,
        0.5,
        z,
        tau,
        derivative_order=1,
        precision=precision,
    )


def prime_form(
    z: complex,
    w: complex,
    tau: complex,
    *,
    precision: int = 40,
) -> complex:
    r"""Return :math:`E(z,w)=\vartheta_1(z-w)/\vartheta_1'(0)`.

    With the present period-one coordinate, :math:`E(z,w)\sim z-w` near
    coincidence.
    """

    denominator = odd_theta_derivative(0, tau, precision=precision)
    if denominator == 0:
        raise ArithmeticError("the odd theta derivative vanished numerically")
    return odd_theta(complex(z) - complex(w), tau, precision=precision) / denominator


def spin_characteristic(spin_structure: TorusSpinStructure) -> tuple[float, float]:
    """Convert spatial/time periodicities to a theta characteristic."""

    if not isinstance(spin_structure, TorusSpinStructure):
        raise TypeError("spin_structure must be a TorusSpinStructure")
    return (
        0.5 if spin_structure.spatial_periodic else 0.0,
        0.5 if spin_structure.temporal_periodic else 0.0,
    )


def szego_kernel(
    z: complex,
    tau: complex,
    spin_structure: TorusSpinStructure,
    *,
    precision: int = 40,
) -> complex:
    r"""Return the meromorphic Szegő kernel with residue one at ``z=0``.

    For an even spin structure this is

    .. math::

       S_\delta(z)=\frac{\vartheta_\delta(z)\vartheta_1'(0)}
       {\vartheta_\delta(0)\vartheta_1(z)}.

    For the odd structure the returned object is
    :math:`\partial_z\log\vartheta_1(z)`.  It is the standard meromorphic
    kernel before a separate prescription removes the fermion zero mode.
    """

    z = complex(z)
    a, b = spin_characteristic(spin_structure)
    theta_odd = odd_theta(z, tau, precision=precision)
    if theta_odd == 0:
        raise ZeroDivisionError("the Szego kernel is singular at a theta zero")
    derivative = odd_theta_derivative(0, tau, precision=precision)
    if spin_structure.arf_invariant:
        return odd_theta_derivative(z, tau, precision=precision) / theta_odd
    theta_zero = theta_characteristic(a, b, 0, tau, precision=precision)
    if theta_zero == 0:
        raise ArithmeticError("an even theta constant vanished numerically")
    return (
        theta_characteristic(a, b, z, tau, precision=precision)
        * derivative
        / (theta_zero * theta_odd)
    )


def odd_szego_nonzero_mode(
    z: complex,
    tau: complex,
    *,
    precision: int = 40,
) -> complex:
    r"""Return a doubly periodic inverse with the odd fermion zero mode removed.

    The meromorphic odd kernel is not single valued.  In the period-one
    coordinate, the periodic zero-mode-subtracted inverse is

    .. math::

       S_{\rm odd}^{\prime}(z|\tau)
       =\partial_z\log\vartheta_1(z|\tau)
        +2\pi i\frac{\operatorname{Im}z}{\tau_2}.

    The second term is nonholomorphic, as required when inverting the Dirac
    operator on the orthogonal complement of its constant zero mode.  A
    discontinuous chiral representative can be used instead, but must not be
    silently identified with the meromorphic kernel.
    """

    tau = _validate_tau(tau)
    z = complex(z)
    meromorphic = szego_kernel(
        z,
        tau,
        TorusSpinStructure(True, True),
        precision=precision,
    )
    return meromorphic + 2j * math.pi * z.imag / tau.imag


def torus_scalar_green(
    z: complex,
    w: complex,
    tau: complex,
    *,
    precision: int = 40,
) -> float:
    r"""Return the zero-average scalar Green-function kernel up to a constant.

    The convention is

    .. math::

       G(z,w)=-\log|E(z,w)|^2
       +\frac{2\pi[\operatorname{Im}(z-w)]^2}{\tau_2}.

    Its additive constant cancels from neutral vertex-operator correlators.
    """

    tau = _validate_tau(tau)
    difference = complex(z) - complex(w)
    form = prime_form(z, w, tau, precision=precision)
    if form == 0:
        return math.inf
    return -math.log(abs(form) ** 2) + 2 * math.pi * difference.imag**2 / tau.imag


def gaussian_vertex_factor(
    points: Sequence[complex],
    charges: Sequence[complex],
    tau: complex,
    *,
    covariance_scale: complex,
    require_neutrality: bool = True,
    neutrality_tolerance: float = 1.0e-12,
    precision: int = 40,
) -> complex:
    r"""Return a neutral Gaussian vertex-operator oscillator factor.

    This evaluates

    .. math::

       \exp\!\left[-s\sum_{i<j}k_i k_j G(z_i,z_j)\right],

    where ``s=covariance_scale`` is explicit so that spacelike, timelike,
    and analytic-continuation conventions cannot be confused.  Momentum
    zero-mode and compact winding sums are not included.
    """

    positions = tuple(complex(point) for point in points)
    momenta = tuple(complex(charge) for charge in charges)
    if not positions or len(positions) != len(momenta):
        raise ValueError("points and charges must have the same positive length")
    if require_neutrality and abs(sum(momenta)) > neutrality_tolerance:
        raise ValueError("the Gaussian vertex charges are not neutral")
    exponent = 0.0j
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            exponent -= (
                complex(covariance_scale)
                * momenta[i]
                * momenta[j]
                * torus_scalar_green(
                    positions[i],
                    positions[j],
                    tau,
                    precision=precision,
                )
            )
    return cmath.exp(exponent)


def noncompact_boson_partition_per_unit_volume(
    tau: complex,
    *,
    alpha_prime: float = 2.0,
    precision: int = 40,
) -> float:
    r"""Return the noncompact scalar partition function per zero-mode volume.

    The normalization is

    .. math::

       \frac{Z_X}{V_X}
       =\frac{1}{\sqrt{4\pi^2\alpha'\tau_2}\,|\eta(\tau)|^2}.

    At ``alpha_prime=2`` this is the normalization used in the original
    two-dimensional heterotic torus calculation.  The target-space zero-mode
    volume, or equivalently the momentum-conserving delta function in a
    correlator, is not included.
    """

    tau = _validate_tau(tau)
    alpha_prime = float(alpha_prime)
    if not math.isfinite(alpha_prime) or alpha_prime <= 0:
        raise ValueError("alpha_prime must be finite and positive")
    eta = dedekind_eta(tau, precision=precision)
    return 1.0 / (
        math.sqrt(4.0 * math.pi**2 * alpha_prime * tau.imag)
        * abs(eta) ** 2
    )


def noncompact_boson_vertex_correlator(
    points: Sequence[complex],
    momenta: Sequence[complex],
    tau: complex,
    *,
    alpha_prime: float = 2.0,
    target_signature: int = -1,
    include_partition_function: bool = True,
    neutrality_tolerance: float = 1.0e-12,
    precision: int = 40,
) -> complex:
    r"""Return a neutral noncompact-boson torus correlator without zero mode.

    ``target_signature=+1`` is a spacelike scalar and ``-1`` is the formal
    timelike continuation.  With the Green kernel used in this module the
    covariance scale is ``target_signature * alpha_prime / 2``.  The
    target-space zero-mode integral is deliberately excluded.
    """

    if target_signature not in (-1, 1):
        raise ValueError("target_signature must be +1 or -1")
    oscillator = gaussian_vertex_factor(
        points,
        momenta,
        tau,
        covariance_scale=target_signature * float(alpha_prime) / 2.0,
        require_neutrality=True,
        neutrality_tolerance=neutrality_tolerance,
        precision=precision,
    )
    if not include_partition_function:
        return oscillator
    return (
        noncompact_boson_partition_per_unit_volume(
            tau,
            alpha_prime=alpha_prime,
            precision=precision,
        )
        * oscillator
    )


def even_superghost_chiral_partition(
    tau: complex,
    spin_structure: TorusSpinStructure,
    *,
    precision: int = 40,
) -> complex:
    r"""Return the even-spin ``beta-gamma`` determinant after picture raising.

    In the trace convention of Xi Yin's string notes this is

    .. math::

       Z_{\beta\gamma,\delta}
       =\epsilon_1\epsilon_2\,
        \frac{\eta(\tau)}{\theta_\delta(0|\tau)},

    where ``epsilon_i`` is ``+1`` for periodic and ``-1`` for
    anti-periodic spinors.  Multiplication by two free Majorana determinants
    therefore reproduces the signed ``(psi, beta, gamma)`` trace.  The odd
    spin structure has zero modes and must not be evaluated with this
    determinant formula.
    """

    if spin_structure.arf_invariant:
        raise ValueError(
            "the odd beta-gamma system requires its zero-mode/PCO prescription"
        )
    a, b = spin_characteristic(spin_structure)
    theta_constant = theta_characteristic(
        a,
        b,
        0.0,
        tau,
        precision=precision,
    )
    if theta_constant == 0:
        raise ArithmeticError("an even theta constant vanished numerically")
    spatial_sign = 1 if spin_structure.spatial_periodic else -1
    temporal_sign = 1 if spin_structure.temporal_periodic else -1
    return (
        spatial_sign
        * temporal_sign
        * dedekind_eta(tau, precision=precision)
        / theta_constant
    )


def bc_torus_partition(
    tau: complex,
    *,
    precision: int = 40,
) -> float:
    r"""Return the nonchiral torus ``bc`` factor :math:`|\eta(\tau)|^4`.

    No modulus-measure factor, translation-volume quotient, GSO coefficient,
    puncture measure, or string coupling is included.  Keeping these factors
    separate prevents the genus-one projector coefficient from being counted
    twice.
    """

    tau = _validate_tau(tau)
    return abs(dedekind_eta(tau, precision=precision)) ** 4


def heterotic_fixed_puncture_measure_density(
    tau: complex,
    *,
    precision: int = 40,
) -> float:
    r"""Return the fixed-puncture heterotic factor :math:`|\eta|^4/2`.

    Xi Yin's period-:math:`2\pi` torus convention contains

    .. math::

       \frac{d^2\tau}{2}(2\pi)^2|\eta(\tau)|^4
       \prod_{j=2}^{n}d^2z_j.

    After fixing the first puncture and converting the coordinates and all
    weight-``(1,1)`` picture-zero matter insertions to period-one coordinates,
    the powers of ``2*pi`` cancel.  The remaining density multiplying
    ``d^2 tau prod_{j=2}^n d^2 z_j`` is ``|eta|^4/2``.  The diagonal GSO
    coefficient, the factor ``1/2`` from each picture-raised vertex, and the
    phase ``i**n`` are not included here.
    """

    return 0.5 * bc_torus_partition(tau, precision=precision)


def pfaffian(matrix: Sequence[Sequence[complex]], *, tolerance: float = 1.0e-14) -> complex:
    """Return the Pfaffian of an even-dimensional antisymmetric matrix."""

    array = np.asarray(matrix, dtype=np.complex128).copy()
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("the Pfaffian input must be square")
    size = array.shape[0]
    if size % 2:
        return 0.0j
    if size == 0:
        return 1.0 + 0.0j
    scale = max(1.0, float(np.max(np.abs(array))))
    if not np.allclose(array + array.T, 0.0, atol=tolerance * scale, rtol=0.0):
        raise ValueError("the Pfaffian input must be antisymmetric")

    result = 1.0 + 0.0j
    for pivot_index in range(0, size - 1, 2):
        candidates = np.abs(array[pivot_index, pivot_index + 1 :])
        offset = int(np.argmax(candidates))
        partner = pivot_index + 1 + offset
        if abs(array[pivot_index, partner]) <= tolerance * scale:
            return 0.0j
        if partner != pivot_index + 1:
            array[[pivot_index + 1, partner], :] = array[[partner, pivot_index + 1], :]
            array[:, [pivot_index + 1, partner]] = array[:, [partner, pivot_index + 1]]
            result *= -1
        pivot = array[pivot_index, pivot_index + 1]
        result *= pivot
        remaining = slice(pivot_index + 2, size)
        if pivot_index + 2 < size:
            left = array[pivot_index, remaining].copy()
            right = array[pivot_index + 1, remaining].copy()
            array[remaining, remaining] += (
                np.outer(right, left) - np.outer(left, right)
            ) / pivot
    return complex(result)


def majorana_wick_factor(
    points: Sequence[complex],
    tau: complex,
    spin_structure: TorusSpinStructure,
    *,
    precision: int = 40,
) -> complex:
    r"""Return the normalized Wick factor for one chiral Majorana fermion.

    For an even spin structure this is a Pfaffian and therefore vanishes for
    an odd number of insertions.  For the odd spin structure exactly one
    insertion saturates the constant fermion zero mode; the result is the
    alternating sum of Pfaffians of the remaining nonzero-mode fields.
    Overall chiral determinant factors are deliberately not included.
    """

    positions = tuple(complex(point) for point in points)

    def nonzero_mode_pfaffian(selected: tuple[complex, ...]) -> complex:
        matrix = np.zeros((len(selected), len(selected)), dtype=np.complex128)
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                difference = selected[i] - selected[j]
                value = (
                    odd_szego_nonzero_mode(
                        difference,
                        tau,
                        precision=precision,
                    )
                    if spin_structure.arf_invariant
                    else szego_kernel(
                        difference,
                        tau,
                        spin_structure,
                        precision=precision,
                    )
                )
                matrix[i, j] = value
                matrix[j, i] = -value
        return pfaffian(matrix)

    if not positions:
        return 0.0j if spin_structure.arf_invariant else 1.0 + 0.0j
    if spin_structure.arf_invariant:
        if len(positions) % 2 == 0:
            return 0.0j
        total = 0.0j
        for omitted in range(len(positions)):
            remaining = positions[:omitted] + positions[omitted + 1 :]
            total += (-1) ** omitted * nonzero_mode_pfaffian(remaining)
        return total
    if len(positions) % 2:
        return 0.0j
    return nonzero_mode_pfaffian(positions)


def majorana_chiral_partition(
    tau: complex,
    spin_structure: TorusSpinStructure,
    *,
    n_fermions: int = 1,
    precision: int = 40,
) -> complex:
    r"""Return the oscillator determinant of ``n_fermions`` real fermions.

    For an even spin structure this is

    .. math::

       \left(\frac{\vartheta_\delta(0|\tau)}{\eta(\tau)}
       \right)^{n_f/2}

    on the principal branch.  It vanishes in the odd spin structure because
    of the unsaturated constant zero modes.  GSO phases are not included.
    """

    if not isinstance(n_fermions, int):
        raise TypeError("n_fermions must be an integer")
    if n_fermions < 0:
        raise ValueError("n_fermions must be nonnegative")
    if n_fermions == 0:
        return 1.0 + 0.0j
    if spin_structure.arf_invariant:
        return 0.0j
    a, b = spin_characteristic(spin_structure)
    theta_constant = theta_characteristic(a, b, 0, tau, precision=precision)
    return cmath.exp(
        (n_fermions / 2.0)
        * cmath.log(theta_constant / dedekind_eta(tau, precision=precision))
    )


def flavored_majorana_wick_factor(
    points: Sequence[complex],
    flavors: Sequence[int],
    tau: complex,
    spin_structure: TorusSpinStructure,
    *,
    n_fermions: int,
    include_even_partition: bool = True,
    precision: int = 40,
) -> complex:
    r"""Return a free-fermion correlator with explicit flavor labels.

    Fields of different flavors do not contract.  In an odd spin structure,
    every one of the ``n_fermions`` zero modes must occur an odd number of
    times; otherwise the correlator is zero.  When all odd zero modes are
    saturated, this routine returns only the normalized Wick/zero-mode
    factor because the odd-sector nonzero-mode determinant depends on the
    chosen chiral path-integral normalization.  The even-sector determinant
    can be included with ``include_even_partition``.
    """

    positions = tuple(complex(point) for point in points)
    labels = tuple(flavors)
    if len(positions) != len(labels):
        raise ValueError("points and flavors must have equal length")
    if not isinstance(n_fermions, int) or n_fermions <= 0:
        raise ValueError("n_fermions must be a positive integer")
    if any(not isinstance(label, int) or not 0 <= label < n_fermions for label in labels):
        raise ValueError("every flavor must be an integer in [0,n_fermions)")

    grouped = tuple(
        tuple(point for point, label in zip(positions, labels) if label == flavor)
        for flavor in range(n_fermions)
    )
    if spin_structure.arf_invariant and any(len(group) % 2 == 0 for group in grouped):
        return 0.0j
    value = 1.0 + 0.0j
    for group in grouped:
        value *= majorana_wick_factor(
            group,
            tau,
            spin_structure,
            precision=precision,
        )
    if include_even_partition and not spin_structure.arf_invariant:
        value *= majorana_chiral_partition(
            tau,
            spin_structure,
            n_fermions=n_fermions,
            precision=precision,
        )
    return value


__all__ = [
    "bc_torus_partition",
    "dedekind_eta",
    "even_superghost_chiral_partition",
    "flavored_majorana_wick_factor",
    "gaussian_vertex_factor",
    "heterotic_fixed_puncture_measure_density",
    "majorana_chiral_partition",
    "majorana_wick_factor",
    "noncompact_boson_partition_per_unit_volume",
    "noncompact_boson_vertex_correlator",
    "odd_theta",
    "odd_theta_derivative",
    "odd_szego_nonzero_mode",
    "pfaffian",
    "prime_form",
    "spin_characteristic",
    "szego_kernel",
    "theta_characteristic",
    "torus_scalar_green",
]
