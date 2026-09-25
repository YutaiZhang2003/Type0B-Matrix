#!/usr/bin/env python3
r"""Candidate closed forms for the genuine Spin(23) singlet amplitudes.

The formulas use the raw-descendant normalization of
``spin23_singlet_amplitudes.py``.  They omit the common sphere normalization,
heterotic coupling, energy-conservation delta function, and asymptotic
reflection phases in exactly the same way as the numerical evaluator.

Set ``x_i = i*omega_i`` for the three outgoing energies and
``x_0 = x_1+x_2+x_3``.  Define

.. math::

   \Pi=x_0x_1x_2x_3=\omega_0\omega_1\omega_2\omega_3,
   \qquad d_{ij}=1+x_i+x_j,

and ``s_2=x_1*x_2+x_1*x_3+x_2*x_3``.  The candidates are

.. math::

   \mathcal M_{S\to SVV}^{ab}
   &=\delta^{ab}\,\pi\Pi\left(
     1+2x_0+\frac{x_0x_1}{d_{23}}\right),\\
   \mathcal M_{V\to VSS}^{ab}
   &=\delta^{ab}\,\pi\Pi\left(
     1+2x_0-\frac{x_2x_3}{d_{23}}\right),\\
   \mathcal M_{S\to SSS}
   &=\pi\Pi\left[
       \Pi\left(\frac1{d_{12}}+\frac1{d_{13}}+\frac1{d_{23}}\right)
       +(1+2x_0)(1+s_2-x_0^2)
     \right].

The pole residues are fixed by products of the exact three-point amplitudes.
The regular terms were selected by numerical coefficient fits and locked to
small integers.  In particular, both the fixed-incoming ``V -> VSS`` formula
and the compact polynomial in the all-singlet formula must remain labeled as
conjectural until independent analytic derivations are supplied.

Equivalently, with ``omega0=omega1+omega2+omega3`` and
``W=omega0*omega1*omega2*omega3``, the formulas are

.. math::

   \mathcal M_{S\to SVV}^{ab}
   &=\delta^{ab}\,\pi W\left[
       1+2i\omega_0-
       \frac{\omega_0\omega_1}{1+i(\omega_2+\omega_3)}
     \right],\\
   \mathcal M_{V\to VSS}^{ab}
   &=\delta^{ab}\,\pi W\left[
       1+2i\omega_0+
       \frac{\omega_2\omega_3}{1+i(\omega_2+\omega_3)}
     \right],\\
   \mathcal M_{S\to SSS}
   &=\pi W\left[
       W\sum_{1\leq i<j\leq3}\frac1{1+i(\omega_i+\omega_j)}
       +(1+2i\omega_0)
        \left(1+\frac12\sum_{r=0}^3\omega_r^2\right)
     \right].

The first expression also has the fully factorized form

.. math::

   \mathcal M_{S\to SVV}^{ab}
   =\delta^{ab}\,\pi W\,
     \frac{(1+i\omega_0)\,[1+i(2\omega_0-\omega_1)]}
          {1+i(\omega_0-\omega_1)}.
"""

from __future__ import annotations

import cmath
import math
from numbers import Number
from typing import Sequence


def _outgoing_x(
    omega1: complex,
    omega2: complex,
    omega3: complex,
) -> tuple[complex, complex, complex, complex]:
    """Return ``(x1, x2, x3, x0)`` with ``x_i=i*omega_i``."""

    x1, x2, x3 = (1j * complex(value) for value in (omega1, omega2, omega3))
    return x1, x2, x3, x1 + x2 + x3


def _pair_denominator(first: complex, second: complex) -> complex:
    """Return the resonance denominator ``1+x_i+x_j``."""

    denominator = 1 + first + second
    if denominator == 0:
        raise ZeroDivisionError("the candidate lies exactly on a pair-channel pole")
    return denominator


def s_to_svv_raw_candidate(
    omega_singlet: complex,
    omega_vector1: complex,
    omega_vector2: complex,
    *,
    delta_ab: Number = 1,
) -> complex:
    r"""Return the raw ``S -> S V^a V^b`` closed-form candidate.

    The first argument is the outgoing singlet energy.  The last two are the
    outgoing vector energies, so exchanging them leaves the answer invariant.
    """

    x1, x2, x3, x0 = _outgoing_x(
        omega_singlet,
        omega_vector1,
        omega_vector2,
    )
    product = x0 * x1 * x2 * x3
    d23 = _pair_denominator(x2, x3)
    pole = x0 * x1 / d23
    contact = 1 + 2 * x0
    return complex(delta_ab) * math.pi * product * (pole + contact)


def s_to_sss_raw_candidate(
    omega1: complex,
    omega2: complex,
    omega3: complex,
) -> complex:
    r"""Return the permutation-symmetric raw ``S -> S S S`` candidate."""

    x1, x2, x3, x0 = _outgoing_x(omega1, omega2, omega3)
    product = x0 * x1 * x2 * x3
    s2 = x1 * x2 + x1 * x3 + x2 * x3
    pole_sum = (
        1 / _pair_denominator(x1, x2)
        + 1 / _pair_denominator(x1, x3)
        + 1 / _pair_denominator(x2, x3)
    )
    contact = (1 + 2 * x0) * (1 + s2 - x0**2)
    return math.pi * product * (product * pole_sum + contact)


def v_to_vss_raw_candidate(
    omega_vector: complex,
    omega_singlet1: complex,
    omega_singlet2: complex,
    *,
    delta_ab: Number = 1,
) -> complex:
    r"""Return the raw ``V^a -> V^b S S`` closed-form candidate.

    The first argument is the outgoing vector energy and the last two are
    the outgoing singlet energies.  This numerically locked conjecture uses
    the fixed-incoming-vector block convention; it is not obtained by
    reassigning the incoming leg of :func:`s_to_svv_raw_candidate`.
    """

    x1, x2, x3, x0 = _outgoing_x(
        omega_vector,
        omega_singlet1,
        omega_singlet2,
    )
    product = x0 * x1 * x2 * x3
    d23 = _pair_denominator(x2, x3)
    pole = -(x2 * x3) / d23
    contact = 1 + 2 * x0
    return complex(delta_ab) * math.pi * product * (pole + contact)


def singlet_descendant_norm(omega: complex) -> complex:
    """Return the principal norm ``sqrt(1+omega**2)`` of one singlet leg."""

    return cmath.sqrt(1 + complex(omega) ** 2)


def s_to_svv_unit_descendants_candidate(
    omega_singlet: complex,
    omega_vector1: complex,
    omega_vector2: complex,
    *,
    delta_ab: Number = 1,
) -> complex:
    """Return the ``S -> SVV`` candidate with both singlet legs normalized."""

    omega0 = sum(
        map(complex, (omega_singlet, omega_vector1, omega_vector2))
    )
    return s_to_svv_raw_candidate(
        omega_singlet,
        omega_vector1,
        omega_vector2,
        delta_ab=delta_ab,
    ) / (
        singlet_descendant_norm(omega0)
        * singlet_descendant_norm(omega_singlet)
    )


def s_to_sss_unit_descendants_candidate(
    omega1: complex,
    omega2: complex,
    omega3: complex,
) -> complex:
    """Return the ``S -> SSS`` candidate with all four singlets normalized."""

    outgoing: Sequence[complex] = tuple(map(complex, (omega1, omega2, omega3)))
    omega0 = sum(outgoing)
    norm = singlet_descendant_norm(omega0)
    for omega in outgoing:
        norm *= singlet_descendant_norm(omega)
    return s_to_sss_raw_candidate(*outgoing) / norm


def v_to_vss_unit_descendants_candidate(
    omega_vector: complex,
    omega_singlet1: complex,
    omega_singlet2: complex,
    *,
    delta_ab: Number = 1,
) -> complex:
    """Return ``V -> VSS`` with the two singlet descendants normalized."""

    return v_to_vss_raw_candidate(
        omega_vector,
        omega_singlet1,
        omega_singlet2,
        delta_ab=delta_ab,
    ) / (
        singlet_descendant_norm(omega_singlet1)
        * singlet_descendant_norm(omega_singlet2)
    )


__all__ = [
    "s_to_sss_raw_candidate",
    "s_to_sss_unit_descendants_candidate",
    "s_to_svv_raw_candidate",
    "s_to_svv_unit_descendants_candidate",
    "singlet_descendant_norm",
    "v_to_vss_raw_candidate",
    "v_to_vss_unit_descendants_candidate",
]
