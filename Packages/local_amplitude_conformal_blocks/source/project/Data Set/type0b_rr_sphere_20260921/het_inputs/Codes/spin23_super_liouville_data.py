#!/usr/bin/env python3
r"""Delta-normalized :math:`b=1` N=1 super-Liouville three-point data.

The formulas in this module use the real-momentum normalization employed in
the Spin(23) sphere calculation.  NS primaries have

.. math::

   h_{\rm NS}(P)=\frac{1+P^2}{2},

while Ramond primaries have :math:`h_{\rm R}(P)=h_{\rm NS}(P)+1/16`.
The two R--R--NS constants are returned in the BRY even/odd convention.  The
dictionary to the chiral Hadasz--Jaskolski--Suchanek forms is

.. math::

   \rho^+\leftrightarrow C_{\rm even},\qquad
   \rho^-\leftrightarrow C_{\rm odd}.

At :math:`b=1` these expressions are the specialization of the exact
R--R--NS constants of Poghossian, arXiv:hep-th/9607120, eq. (A.69), after
the external fields are delta normalized.  They agree with the convention
used in arXiv:2201.05621.  No complex conjugation is applied under analytic
continuation of the momenta.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypeAlias

import mpmath as mp


Number: TypeAlias = complex | float | int


def _as_mp(value: Number) -> mp.mpc:
    """Convert one public numeric input to an mpmath complex number."""

    return mp.mpc(value)


def _upsilon_1_mp(x: mp.mpc) -> mp.mpc:
    return mp.barnesg(x) * mp.barnesg(2 - x)


def _upsilon_ns_mp(x: mp.mpc) -> mp.mpc:
    if x == 0:
        return mp.mpc(0)
    half = x / 2
    # This expression is algebraically equal to
    # gamma(half)/gamma(1-half) * Upsilon_1(half)^2, but it avoids a
    # cancellation between a pole and a zero near x=0.
    return (
        mp.gamma(half)
        * mp.gamma(1 - half)
        * mp.barnesg(half) ** 2
        * mp.barnesg(1 - half) ** 2
    )


def _upsilon_r_mp(x: mp.mpc) -> mp.mpc:
    return _upsilon_1_mp((x + 1) / 2) ** 2


def _n_ns_mp(momentum: mp.mpc) -> mp.mpc:
    if momentum == 0:
        return mp.mpc(0)
    return (
        mp.gamma(1 + 1j * momentum)
        / mp.gamma(1 - 1j * momentum)
        * _upsilon_ns_mp(2j * momentum)
    )


def _n_r_mp(momentum: mp.mpc) -> mp.mpc:
    return (
        mp.gamma(mp.mpf("0.5") + 1j * momentum)
        / mp.gamma(mp.mpf("0.5") - 1j * momentum)
        * _upsilon_r_mp(2j * momentum)
    )


@lru_cache(maxsize=4096)
def _cached_n_ns(momentum: complex, precision: int) -> mp.mpc:
    """Cache one-dimensional NS leg data at its requested precision."""

    with mp.workdps(precision):
        return +_n_ns_mp(mp.mpc(momentum))


@lru_cache(maxsize=4096)
def _cached_n_r(momentum: complex, precision: int) -> mp.mpc:
    """Cache one-dimensional Ramond leg data at its requested precision."""

    with mp.workdps(precision):
        return +_n_r_mp(mp.mpc(momentum))


def _momentum_combinations(
    p1: mp.mpc,
    p2: mp.mpc,
    p3: mp.mpc,
) -> tuple[mp.mpc, tuple[mp.mpc, mp.mpc, mp.mpc]]:
    total = p1 + p2 + p3
    differences = (p2 + p3 - p1, p1 + p3 - p2, p1 + p2 - p3)
    return total, differences


def ns_weight(momentum: Number) -> complex:
    r"""Return :math:`h_{\rm NS}(P)=(1+P^2)/2` at :math:`b=1`."""

    p = complex(momentum)
    return (1.0 + p * p) / 2.0


def ramond_liouville_weight(momentum: Number) -> complex:
    r"""Return :math:`h_{\rm R}(P)=(1+P^2)/2+1/16` at :math:`b=1`."""

    return ns_weight(momentum) + 1.0 / 16.0


def upsilon_1(x: Number, *, precision: int = 40) -> complex:
    r"""Return :math:`\Upsilon_1(x)=G(x)G(2-x)`, with value one at ``x=1``."""

    with mp.workdps(precision):
        return complex(_upsilon_1_mp(_as_mp(x)))


def upsilon_ns(x: Number, *, precision: int = 40) -> complex:
    r"""Return the :math:`b=1` function :math:`\Upsilon_{\rm NS}(x)`."""

    with mp.workdps(precision):
        return complex(_upsilon_ns_mp(_as_mp(x)))


def upsilon_r(x: Number, *, precision: int = 40) -> complex:
    r"""Return the :math:`b=1` function :math:`\Upsilon_{\rm R}(x)`."""

    with mp.workdps(precision):
        return complex(_upsilon_r_mp(_as_mp(x)))


def ns_leg_factor(momentum: Number, *, precision: int = 40) -> complex:
    r"""Return the delta-normalized NS leg factor :math:`N_{\rm NS}(P)`."""

    with mp.workdps(precision):
        return complex(_n_ns_mp(_as_mp(momentum)))


def ramond_leg_factor(momentum: Number, *, precision: int = 40) -> complex:
    r"""Return the delta-normalized Ramond leg factor :math:`N_{\rm R}(P)`."""

    with mp.workdps(precision):
        return complex(_n_r_mp(_as_mp(momentum)))


@lru_cache(maxsize=65536)
def ns_structure_constants(
    p1: Number,
    p2: Number,
    p3: Number,
    *,
    precision: int = 40,
) -> tuple[complex, complex]:
    r"""Return :math:`(C_{\rm NS},\widetilde C_{\rm NS})` at :math:`b=1`.

    The first coefficient multiplies the even NS three-point form and the
    second multiplies the odd form.
    """

    with mp.workdps(precision):
        momenta = tuple(_as_mp(p) for p in (p1, p2, p3))
        total, differences = _momentum_combinations(*momenta)
        numerator = mp.fprod(
            _cached_n_ns(complex(momentum), precision)
            for momentum in momenta
        )
        even_denominator = _upsilon_ns_mp(1 + 1j * total) * mp.fprod(
            _upsilon_ns_mp(1 + 1j * difference) for difference in differences
        )
        odd_denominator = _upsilon_r_mp(1 + 1j * total) * mp.fprod(
            _upsilon_r_mp(1 + 1j * difference) for difference in differences
        )
        return (
            complex(0.5j * numerator / even_denominator),
            complex(1j * numerator / odd_denominator),
        )


@lru_cache(maxsize=65536)
def rr_ns_structure_constants(
    p1: Number,
    p2: Number,
    p3: Number,
    *,
    precision: int = 40,
) -> tuple[complex, complex]:
    r"""Return BRY's :math:`(C_{\rm even},C_{\rm odd})` for R-R-NS.

    ``p1`` and ``p2`` are Ramond momenta and ``p3`` is the NS momentum.
    Signed or complex momenta are accepted; reflection is implemented by
    analytic continuation of this formula, not by post-processing signs.
    """

    with mp.workdps(precision):
        p1_mp, p2_mp, p3_mp = (_as_mp(p) for p in (p1, p2, p3))
        total, (delta1, delta2, delta3) = _momentum_combinations(
            p1_mp,
            p2_mp,
            p3_mp,
        )
        numerator = (
            _cached_n_r(complex(p1_mp), precision)
            * _cached_n_r(complex(p2_mp), precision)
            * _cached_n_ns(complex(p3_mp), precision)
        )
        even_denominator = (
            _upsilon_r_mp(1 + 1j * total)
            * _upsilon_r_mp(1 + 1j * delta3)
            * _upsilon_ns_mp(1 + 1j * delta1)
            * _upsilon_ns_mp(1 + 1j * delta2)
        )
        odd_denominator = (
            _upsilon_ns_mp(1 + 1j * total)
            * _upsilon_ns_mp(1 + 1j * delta3)
            * _upsilon_r_mp(1 + 1j * delta1)
            * _upsilon_r_mp(1 + 1j * delta2)
        )
        common = -0.5j * numerator
        return (
            complex(common / even_denominator),
            complex(common / odd_denominator),
        )


def rr_ns_chiral_structure_constant(
    p1: Number,
    p2: Number,
    p3: Number,
    *,
    structure_sign: int,
    precision: int = 40,
) -> complex:
    r"""Return one HJS chiral R--R--NS coefficient.

    ``structure_sign=+1`` selects :math:`C_{\rm even}` and ``-1`` selects
    :math:`C_{\rm odd}`.  These signs should not be confused with the two
    nonchiral Ramond-family coefficients
    :math:`(C_{\rm even}\mathbin\pm C_{\rm odd})/2`.
    """

    if structure_sign not in (-1, 1):
        raise ValueError("structure_sign must be +1 or -1")
    even, odd = rr_ns_structure_constants(
        p1,
        p2,
        p3,
        precision=precision,
    )
    return even if structure_sign == 1 else odd


__all__ = [
    "ns_leg_factor",
    "ns_structure_constants",
    "ns_weight",
    "ramond_leg_factor",
    "ramond_liouville_weight",
    "rr_ns_chiral_structure_constant",
    "rr_ns_structure_constants",
    "upsilon_1",
    "upsilon_ns",
    "upsilon_r",
]
