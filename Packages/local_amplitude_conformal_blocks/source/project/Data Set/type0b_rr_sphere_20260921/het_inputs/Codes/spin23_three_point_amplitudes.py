#!/usr/bin/env python3
r"""Tree-level Spin(23) heterotic three-point amplitudes.

The continuum spectrum contains a Spin(23) vector ``V^a`` and a singlet
``S``.  In the conventions used by the Spin(23) four-point calculation,

.. math::

   \mathcal V_{-1}^{a,\pm}(\omega)
   &=g_H c\bar c\,e^{-\varphi}\bar\lambda^a
     e^{\pm i\omega X^0}V_\omega,\\
   \mathcal S_{-1}^{\pm}(\omega)
   &=g_H c\bar c\,e^{-\varphi}e^{\pm i\omega X^0}
     \widetilde G_{-1/2}V_\omega.

On the sphere, two vertices are taken in picture ``-1`` and one in picture
zero.  The picture-zero vertex contributes ``G_-1/2 V``; its term
proportional to the time fermion has zero expectation value at three points.
Both nonzero amplitudes therefore use the odd NS super-Liouville structure
constant.  On the energy-conserving locus

.. math::

   \omega_0=\omega_1+\omega_2,

the exact ``b=1`` structure constant obeys

.. math::

   \widetilde C(\omega_0,\omega_1,\omega_2)
   =\omega_0\omega_1\omega_2.

The functions below omit the common sphere normalization, the string
coupling, the energy-conservation delta function, and tachyon-wall
in/out reflection phases.  This separation is intentional: the CFT
three-point function is fixed, whereas the asymptotic singlet phase convention
has not yet been independently derived in this repository.
"""

from __future__ import annotations

import cmath
from numbers import Number


def ns_weight(omega: complex) -> complex:
    r"""Return the on-shell NS Liouville weight ``(1 + omega**2) / 2``."""

    return (1 + complex(omega) ** 2) / 2


def odd_structure_constant_on_shell(
    omega1: complex,
    omega2: complex,
) -> complex:
    r"""Return the exact odd NS structure constant on the 1-to-2 locus.

    Parameters
    ----------
    omega1, omega2 : complex
        Outgoing energies.  The incoming energy is defined to be their sum.

    Returns
    -------
    complex
        ``omega0 * omega1 * omega2``, where
        ``omega0 = omega1 + omega2``.
    """

    omega1 = complex(omega1)
    omega2 = complex(omega2)
    omega0 = omega1 + omega2
    return omega0 * omega1 * omega2


def s_to_vv_raw(
    omega1: complex,
    omega2: complex,
    *,
    delta_ab: Number = 1,
) -> complex:
    r"""Return the reduced raw ``S -> V^a V^b`` three-point amplitude.

    The result is

    .. math::

       \frac{\mathcal A^{ab}_{S\to VV}}{\mathcal N_{H,3}}
       =\delta^{ab}\omega_0\omega_1\omega_2,
       \qquad \omega_0=\omega_1+\omega_2.

    ``N_H,3`` denotes the omitted common sphere normalization.

    Parameters
    ----------
    omega1, omega2 : complex
        Outgoing vector energies.
    delta_ab : number, optional
        Value of the Kronecker contraction for the selected vector indices.

    Returns
    -------
    complex
        Reduced raw amplitude.
    """

    return complex(delta_ab) * odd_structure_constant_on_shell(omega1, omega2)


def s_to_ss_raw(omega1: complex, omega2: complex) -> complex:
    r"""Return the reduced raw ``S -> S S`` three-point amplitude.

    The three left-moving level-one-half descendants give the exact Ward
    factor

    .. math::

       \rho_{111}=h_0+h_1+h_2-\frac12.

    Consequently,

    .. math::

       \frac{\mathcal A_{S\to SS}}{\mathcal N_{H,3}}
       =\omega_0\omega_1\omega_2
        \left(h_0+h_1+h_2-\frac12\right).

    Parameters
    ----------
    omega1, omega2 : complex
        Outgoing singlet energies.

    Returns
    -------
    complex
        Reduced raw amplitude.
    """

    omega1 = complex(omega1)
    omega2 = complex(omega2)
    omega0 = omega1 + omega2
    ward_factor = (
        ns_weight(omega0)
        + ns_weight(omega1)
        + ns_weight(omega2)
        - 0.5
    )
    return odd_structure_constant_on_shell(omega1, omega2) * ward_factor


def singlet_descendant_norm(omega: complex) -> complex:
    r"""Return the norm factor of ``G_-1/2 |V_omega>``.

    The NS algebra gives

    .. math::

       \langle V_\omega|G_{1/2}G_{-1/2}|V_\omega\rangle
       =2h(\omega)=1+\omega^2.

    The principal square root is used after analytic continuation.
    """

    return cmath.sqrt(1 + complex(omega) ** 2)


def s_to_vv_unit_descendants(
    omega1: complex,
    omega2: complex,
    *,
    delta_ab: Number = 1,
) -> complex:
    r"""Return ``S -> VV`` with the singlet descendant normalized to one.

    No tachyon-wall reflection phase is included.
    """

    omega0 = complex(omega1) + complex(omega2)
    return s_to_vv_raw(omega1, omega2, delta_ab=delta_ab) / (
        singlet_descendant_norm(omega0)
    )


def s_to_ss_unit_descendants(omega1: complex, omega2: complex) -> complex:
    r"""Return ``S -> SS`` with all three singlet descendants normalized.

    No tachyon-wall reflection phases are included.
    """

    omega0 = complex(omega1) + complex(omega2)
    denominator = (
        singlet_descendant_norm(omega0)
        * singlet_descendant_norm(omega1)
        * singlet_descendant_norm(omega2)
    )
    return s_to_ss_raw(omega1, omega2) / denominator


__all__ = [
    "ns_weight",
    "odd_structure_constant_on_shell",
    "s_to_ss_raw",
    "s_to_ss_unit_descendants",
    "s_to_vv_raw",
    "s_to_vv_unit_descendants",
    "singlet_descendant_norm",
]
