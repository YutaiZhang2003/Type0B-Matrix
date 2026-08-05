"""BRY b=1 N=1 super-Liouville NS structure constants.

The formulas and normalization follow arXiv:2201.05621, section 3.1.  The
Barnes-double-gamma combination is evaluated at b=1 through mpmath's Barnes
G-function, using Upsilon_1(x) = G(x) G(2-x).
"""

from __future__ import annotations

from typing import Union

import mpmath


Number = Union[complex, float]


def _mp(value: Number) -> mpmath.mpc:
    return mpmath.mpc(value)


def _upsilon_1_mp(x: mpmath.mpc) -> mpmath.mpc:
    return mpmath.barnesg(x) * mpmath.barnesg(2 - x)


def _upsilon_ns_mp(x: mpmath.mpc) -> mpmath.mpc:
    if x == 0:
        return mpmath.mpc(0)
    half = x / 2
    # This form is algebraically identical to
    # Gamma(half)/Gamma(1-half) * Upsilon_1(half)^2, but behaves better
    # close to the zero at x=0.
    return (
        mpmath.gamma(half)
        * mpmath.gamma(1 - half)
        * mpmath.barnesg(half) ** 2
        * mpmath.barnesg(1 - half) ** 2
    )


def _upsilon_r_mp(x: mpmath.mpc) -> mpmath.mpc:
    return _upsilon_1_mp((x + 1) / 2) ** 2


def _n_ns_mp(momentum: mpmath.mpc) -> mpmath.mpc:
    if momentum == 0:
        return mpmath.mpc(0)
    return (
        mpmath.gamma(1 + 1j * momentum)
        / mpmath.gamma(1 - 1j * momentum)
        * _upsilon_ns_mp(2j * momentum)
    )


def _n_r_mp(momentum: mpmath.mpc) -> mpmath.mpc:
    return (
        mpmath.gamma(mpmath.mpf("0.5") + 1j * momentum)
        / mpmath.gamma(mpmath.mpf("0.5") - 1j * momentum)
        * _upsilon_r_mp(2j * momentum)
    )


def upsilon_1(x: Number, precision: int = 30) -> complex:
    """Return the normalized b=1 Upsilon function, Upsilon_1(1)=1."""

    with mpmath.workdps(precision):
        return complex(_upsilon_1_mp(_mp(x)))


def upsilon_ns(x: Number, precision: int = 30) -> complex:
    """Return BRY's Upsilon_NS(x) at b=1."""

    with mpmath.workdps(precision):
        return complex(_upsilon_ns_mp(_mp(x)))


def upsilon_r(x: Number, precision: int = 30) -> complex:
    """Return BRY's Upsilon_R(x) at b=1."""

    with mpmath.workdps(precision):
        return complex(_upsilon_r_mp(_mp(x)))


def n_ns(momentum: Number, precision: int = 30) -> complex:
    """Return the delta-normalized NS external-leg factor N_NS(P)."""

    with mpmath.workdps(precision):
        return complex(_n_ns_mp(_mp(momentum)))


def n_r(momentum: Number, precision: int = 30) -> complex:
    """Return the delta-normalized Ramond external-leg factor N_R(P)."""

    with mpmath.workdps(precision):
        return complex(_n_r_mp(_mp(momentum)))


def _momentum_combinations(
    p1: mpmath.mpc, p2: mpmath.mpc, p3: mpmath.mpc
) -> tuple[mpmath.mpc, tuple[mpmath.mpc, mpmath.mpc, mpmath.mpc]]:
    total = p1 + p2 + p3
    differences = (p2 + p3 - p1, p1 + p3 - p2, p1 + p2 - p3)
    return total, differences


def ns_structure_constant(
    p1: Number, p2: Number, p3: Number, precision: int = 30
) -> complex:
    """Return C(P1,P2,P3) in the real BRY phase convention."""

    with mpmath.workdps(precision):
        momenta = (_mp(p1), _mp(p2), _mp(p3))
        total, differences = _momentum_combinations(*momenta)
        numerator = mpmath.fprod(_n_ns_mp(momentum) for momentum in momenta)
        denominator = _upsilon_ns_mp(1 + 1j * total) * mpmath.fprod(
            _upsilon_ns_mp(1 + 1j * difference) for difference in differences
        )
        return complex(0.5j * numerator / denominator)


def ns_tilde_structure_constant(
    p1: Number, p2: Number, p3: Number, precision: int = 30
) -> complex:
    """Return tilde C(P1,P2,P3) in the real BRY phase convention."""

    with mpmath.workdps(precision):
        momenta = (_mp(p1), _mp(p2), _mp(p3))
        total, differences = _momentum_combinations(*momenta)
        numerator = mpmath.fprod(_n_ns_mp(momentum) for momentum in momenta)
        denominator = _upsilon_r_mp(1 + 1j * total) * mpmath.fprod(
            _upsilon_r_mp(1 + 1j * difference) for difference in differences
        )
        return complex(1j * numerator / denominator)


def rr_ns_structure_constants(
    p1: Number, p2: Number, p3: Number, precision: int = 30
) -> tuple[complex, complex]:
    r"""Return BRY's (C_even,C_odd) for R(P1) R(P2) NS(P3).

    Signed or complex momenta are accepted.  Analytic continuation of the
    formula then implements the BRY reflection rules rather than applying
    them as separate floating-point sign corrections.
    """

    with mpmath.workdps(precision):
        p1_mp, p2_mp, p3_mp = _mp(p1), _mp(p2), _mp(p3)
        total, differences = _momentum_combinations(p1_mp, p2_mp, p3_mp)
        delta1, delta2, delta3 = differences
        numerator = _n_r_mp(p1_mp) * _n_r_mp(p2_mp) * _n_ns_mp(p3_mp)

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
    sign: int,
    precision: int = 30,
) -> complex:
    r"""Return the coefficient of the HJS chiral sign branch.

    Crossing fixes the dictionary

        HJS chiral +  <->  BRY C_even,
        HJS chiral -  <->  BRY C_odd.

    These signs are not the BRY nonchiral Ramond-family labels, whose
    physical coefficients are C_pm=(C_even +/- C_odd)/2.
    """

    sign = int(sign)
    if sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
    c_even, c_odd = rr_ns_structure_constants(p1, p2, p3, precision)
    return c_even if sign == 1 else c_odd


__all__ = [
    "n_ns",
    "n_r",
    "ns_structure_constant",
    "ns_tilde_structure_constant",
    "rr_ns_chiral_structure_constant",
    "rr_ns_structure_constants",
    "upsilon_1",
    "upsilon_ns",
    "upsilon_r",
]
