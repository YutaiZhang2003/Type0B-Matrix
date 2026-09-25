#!/usr/bin/env python3
"""Numerical and analytic tools for the sphere 1->3 amplitude in the
2d Spin(23) heterotic string.

The code implements the vector four-point channel discussed in the accompanying
LaTeX note.  It uses the central-charge (c-) recursion for N=1 NS
super-Virasoro blocks, converts the resulting sewing coefficients to H(qhat),
truncates in the elliptic nome and restores the plane prefactor. The public
amplitude uses endpoint/bulk/infinite-tail momentum quadrature and a
crossing-patched plane-moduli integral. Explicit 'sewing' is a validation mode.

Conventions
-----------
External legs are ordered as

    leg 4: incoming, z_4 = infinity, energy omega_0
    leg 3: outgoing, z_3 = 1,        energy omega_3
    leg 2: outgoing, z_2 = z,        energy omega_2
    leg 1: outgoing, z_1 = 0,        energy omega_1

The ``energies`` argument is [omega_1, omega_2, omega_3, omega_0].
The returned vector coefficients (A,B,C) multiply

    A delta^{a0 a3} delta^{a1 a2}
  + B delta^{a0 a2} delta^{a1 a3}
  + C delta^{a0 a1} delta^{a2 a3}.

All results omit the common heterotic sphere normalization, the energy delta
function, and a convention-dependent common phase/sign from the two odd
supermoduli.  The numerical routine is intended first for a convergent
Euclidean/sub-threshold domain.  Physical real energies require the analytic
counterterm continuation described in the note.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import mpmath as mp
import numpy as np
import scipy.special as sps
from numpy.polynomial.legendre import leggauss
if __package__:
    from .liouville_momentum_quadrature import threshold_weighted_rule
    from . import ns_elliptic_conversion as elliptic_conversion
else:
    # Preserve the standalone scripts shipped in this bundle as well as
    # package imports used by the mixed/Ramond sphere evaluators.
    from liouville_momentum_quadrature import threshold_weighted_rule
    import ns_elliptic_conversion as elliptic_conversion

mp.mp.dps = 40


# ---------------------------------------------------------------------------
# b=1 N=1 super-Liouville data
# ---------------------------------------------------------------------------

def h_of_p(momentum: complex) -> mp.mpc:
    p = mp.mpc(momentum)
    return (1 + p * p) / 2


def gamma_ratio(x: complex) -> mp.mpc:
    x = mp.mpc(x)
    return mp.gamma(x) / mp.gamma(1 - x)


def log_gamma_ratio(x: complex) -> mp.mpc:
    """Principal-branch logarithm of gamma(x)/gamma(1-x)."""
    x = mp.mpc(x)
    return mp.loggamma(x) - mp.loggamma(1 - x)


def _upsilon_1_barnes_direct(x: complex) -> mp.mpc:
    """Direct Barnes-G definition, retained as an audit reference."""
    x = mp.mpc(x)
    return mp.barnesg(x) * mp.barnesg(2 - x)


def log_upsilon_1(x: complex) -> mp.mpc:
    r"""Evaluate log Upsilon_1(x) after shift reduction.

    At b=1,

        Upsilon_1(x+1) = gamma(x) Upsilon_1(x),

    where gamma(x)=Gamma(x)/Gamma(1-x).  We first move the real part of
    ``x`` to the fixed strip 1/2 <= Re(x) < 3/2, evaluate the Barnes-G
    representation only there, and accumulate every shift in logarithmic
    form.  Exponentiating at the end prevents overflow and loss of scale in
    the DOZZ-like products.
    """
    x = mp.mpc(x)
    shift = int(mp.floor(mp.re(x) - mp.mpf("0.5")))
    base = x - shift
    result = mp.log(mp.barnesg(base)) + mp.log(mp.barnesg(2 - base))
    if shift > 0:
        for offset in range(shift):
            result += log_gamma_ratio(base + offset)
    elif shift < 0:
        for offset in range(-shift):
            result -= log_gamma_ratio(x + offset)
    return result


def upsilon_1(x: complex) -> mp.mpc:
    """Upsilon_1 normalized by Upsilon_1(1)=1."""
    return mp.exp(log_upsilon_1(x))


def log_upsilon_ns(x: complex) -> mp.mpc:
    x = mp.mpc(x)
    return log_gamma_ratio(x / 2) + 2 * log_upsilon_1(x / 2)


def log_upsilon_r(x: complex) -> mp.mpc:
    x = mp.mpc(x)
    return 2 * log_upsilon_1((x + 1) / 2)


def upsilon_ns(x: complex) -> mp.mpc:
    return mp.exp(log_upsilon_ns(x))


def upsilon_r(x: complex) -> mp.mpc:
    return mp.exp(log_upsilon_r(x))


def _log_ns_leg(momentum: complex) -> mp.mpc:
    momentum = mp.mpc(momentum)
    return (
        mp.loggamma(1 + 1j * momentum)
        - mp.loggamma(1 - 1j * momentum)
        + log_upsilon_ns(2j * momentum)
    )


def log_c_even(p1: complex, p2: complex, p3: complex) -> mp.mpc:
    """Logarithm of the even NS three-point structure constant at b=1."""
    ps = [mp.mpc(p1), mp.mpc(p2), mp.mpc(p3)]
    total = sum(ps)
    result = mp.log(1j / 2) - log_upsilon_ns(1 + 1j * total)
    for pi in ps:
        result += _log_ns_leg(pi)
        result -= log_upsilon_ns(1 + 1j * (total - 2 * pi))
    return result


def log_c_odd(p1: complex, p2: complex, p3: complex) -> mp.mpc:
    """Logarithm of the odd NS three-point structure constant at b=1."""
    ps = [mp.mpc(p1), mp.mpc(p2), mp.mpc(p3)]
    total = sum(ps)
    result = mp.log(1j) - log_upsilon_r(1 + 1j * total)
    for pi in ps:
        result += _log_ns_leg(pi)
        result -= log_upsilon_r(1 + 1j * (total - 2 * pi))
    return result


def c_even(p1: complex, p2: complex, p3: complex) -> mp.mpc:
    """Even NS structure constant, accumulated and exponentiated in log space."""
    return mp.exp(log_c_even(p1, p2, p3))


def c_odd(p1: complex, p2: complex, p3: complex) -> mp.mpc:
    """Odd NS structure constant, accumulated and exponentiated in log space."""
    return mp.exp(log_c_odd(p1, p2, p3))


# ---------------------------------------------------------------------------
# c-recursion for sphere N=1 NS blocks
# ---------------------------------------------------------------------------

def _seed_coefficient(
    level2: int,
    h4: complex,
    h3: complex,
    h2: complex,
    h1: complex,
    h: complex,
    star3: bool = False,
    star2: bool = False,
) -> mp.mpc:
    """Large-c seed f_m.  level2 is twice the descendant level."""
    if level2 % 2 == 0:
        n = level2 // 2
        a3 = h + h3 - h4 + (mp.mpf("0.5") if star3 else 0)
        a2 = h + h2 - h1 + (mp.mpf("0.5") if star2 else 0)
        return mp.rf(a3, n) * mp.rf(a2, n) / (
            mp.factorial(n) * mp.rf(2 * h, n)
        )

    n = (level2 - 1) // 2
    half = mp.mpf("0.5")
    if not star3 and not star2:
        return mp.rf(h + h3 - h4 + half, n) * mp.rf(
            h + h2 - h1 + half, n
        ) / (mp.factorial(n) * mp.rf(2 * h, n + 1))
    if not star3 and star2:
        return mp.rf(h + h3 - h4 + half, n) * mp.rf(
            h + h2 - h1, n + 1
        ) / (mp.factorial(n) * mp.rf(2 * h, n + 1))
    if star3 and not star2:
        return -mp.rf(h + h3 - h4, n + 1) * mp.rf(
            h + h2 - h1 + half, n
        ) / (mp.factorial(n) * mp.rf(2 * h, n + 1))
    return -mp.rf(h + h3 - h4, n + 1) * mp.rf(
        h + h2 - h1, n + 1
    ) / (mp.factorial(n) * mp.rf(2 * h, n + 1))


def _c_rs_and_derivative(h: complex, r: int, s: int) -> tuple[mp.mpc, mp.mpc, mp.mpc]:
    discriminant = 16 * h * h + 8 * (r * s - 1) * h + (r - s) ** 2
    root = mp.sqrt(discriminant)
    y = -(4 * h + r * s - 1 + root) / (r * r - 1)
    dy = -(4 + (16 * h + 4 * (r * s - 1)) / root) / (r * r - 1)
    c_rs = mp.mpf("7.5") + 3 * y + 3 / y
    dc_dh = 3 * dy * (1 - 1 / (y * y))
    return c_rs, dc_dh, mp.sqrt(y)


def _a_rs(b: complex, r: int, s: int) -> mp.mpc:
    product = mp.mpc(1)
    for p in range(1 - r, r + 1):
        for q in range(1 - s, s + 1):
            if (p + q) % 2 != 0:
                continue
            if (p == 0 and q == 0) or (p == r and q == s):
                continue
            product *= mp.sqrt(2) / (p * b + q / b)
    return product / 2


def _a_from_h(h: complex, q_background: complex) -> mp.mpc:
    return mp.sqrt(q_background * q_background / 4 - 2 * h)


def _fusion_polynomial(
    h_a: complex,
    h_b: complex,
    star_b: bool,
    b: complex,
    r: int,
    s: int,
) -> mp.mpc:
    q_background = b + 1 / b
    a1 = _a_from_h(h_a, q_background)
    a2 = _a_from_h(h_b, q_background)
    target = 0 if star_b else 2
    product = mp.mpc(1)
    for p in range(1 - r, r, 2):
        for q in range(1 - s, s, 2):
            if ((p + q) - (r + s)) % 4 != target:
                continue
            x = p * b + q / b
            product *= (2 * a1 - 2 * a2 - x) / (2 * mp.sqrt(2))
            product *= (2 * a1 + 2 * a2 + x) / (2 * mp.sqrt(2))
    return product


@dataclass
class NSBlockComputer:
    h4: complex
    h3: complex
    h2: complex
    h1: complex
    star3: bool = False
    star2: bool = False
    c: complex = 13.5
    max_level2: int = 16

    def __post_init__(self) -> None:
        self.h4 = mp.mpc(self.h4)
        self.h3 = mp.mpc(self.h3)
        self.h2 = mp.mpc(self.h2)
        self.h1 = mp.mpc(self.h1)
        self.c = mp.mpc(self.c)
        self._memo: dict[tuple[int, str, str], mp.mpc] = {}

    def coefficient(self, level2: int, h: complex, c: complex | None = None) -> mp.mpc:
        if level2 == 0:
            return mp.mpc(1)
        if c is None:
            c = self.c
        h = mp.mpc(h)
        c = mp.mpc(c)
        key = (int(level2), mp.nstr(h, 30), mp.nstr(c, 30))
        if key in self._memo:
            return self._memo[key]

        value = _seed_coefficient(
            level2,
            self.h4,
            self.h3,
            self.h2,
            self.h1,
            h,
            self.star3,
            self.star2,
        )
        for r in range(2, level2 + 1):
            for s in range(1, level2 + 1):
                rs = r * s
                if rs <= 1 or rs > level2 or (r + s) % 2 != 0:
                    continue
                c_rs, dc_dh, b = _c_rs_and_derivative(h, r, s)
                sign = (-1) ** rs if self.star3 else 1
                if level2 % 2 == 0:
                    star12 = self.star2
                    star43 = self.star3
                else:
                    star12 = not self.star2
                    star43 = not self.star3
                p12 = _fusion_polynomial(self.h1, self.h2, star12, b, r, s)
                p43 = _fusion_polynomial(self.h4, self.h3, star43, b, r, s)
                residue = sign * (-dc_dh) * _a_rs(b, r, s) * p12 * p43
                value += residue / (c - c_rs) * self.coefficient(
                    level2 - rs, h + mp.mpf(rs) / 2, c_rs
                )

        self._memo[key] = value
        return value


# ---------------------------------------------------------------------------
# Formal power-series conversion from z to elliptic nome q
# ---------------------------------------------------------------------------

def _series_multiply(a: np.ndarray, b: np.ndarray, length: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    out = np.zeros(length, dtype=np.complex128)
    for i in range(min(len(a), length)):
        if a[i] == 0:
            continue
        jmax = min(len(b), length - i)
        out[i : i + jmax] += a[i] * b[:jmax]
    return out


def _series_inverse(a: np.ndarray, length: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.complex128)
    out = np.zeros(length, dtype=np.complex128)
    out[0] = 1 / a[0]
    for n in range(1, length):
        out[n] = -sum(
            a[k] * out[n - k] for k in range(1, min(n + 1, len(a)))
        ) / a[0]
    return out


def _series_derivative(a: np.ndarray) -> np.ndarray:
    return np.array([n * a[n] for n in range(1, len(a))], dtype=np.complex128)


def _series_integral(a: np.ndarray, length: int, constant: complex = 0) -> np.ndarray:
    out = np.zeros(length, dtype=np.complex128)
    out[0] = constant
    for n in range(1, length):
        if n - 1 < len(a):
            out[n] = a[n - 1] / n
    return out


def _series_log(a: np.ndarray, length: int) -> np.ndarray:
    derivative = _series_derivative(a)
    inverse = _series_inverse(a, length)
    quotient = _series_multiply(derivative, inverse, length - 1)
    return _series_integral(quotient, length, constant=np.log(a[0]))


def _series_exp(a: np.ndarray, length: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.complex128)
    out = np.zeros(length, dtype=np.complex128)
    out[0] = np.exp(a[0])
    for n in range(1, length):
        out[n] = sum(
            k * a[k] * out[n - k] for k in range(1, min(n + 1, len(a)))
        ) / n
    return out


def _series_power(a: np.ndarray, power: complex, length: int) -> np.ndarray:
    return _series_exp(_series_log(a, length) * complex(power), length)


def _series_compose(polynomial: Sequence[complex], z_series: np.ndarray, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.complex128)
    current = np.zeros(length, dtype=np.complex128)
    current[0] = 1
    for n, coefficient in enumerate(polynomial):
        if n > 0:
            current = _series_multiply(current, z_series, length)
        out += complex(coefficient) * current
    return out


def _modular_series(length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return t-series z(t), u=z/(16 t^2), and theta_3(q=t^2)."""
    theta3 = np.zeros(length, dtype=np.complex128)
    theta3[0] = 1
    n = 1
    while 2 * n * n < length:
        theta3[2 * n * n] += 2
        n += 1

    theta2_reduced = np.zeros(length, dtype=np.complex128)
    n = 0
    while 2 * n * (n + 1) < length:
        theta2_reduced[2 * n * (n + 1)] += 1
        n += 1

    a2 = _series_multiply(theta2_reduced, theta2_reduced, length)
    a4 = _series_multiply(a2, a2, length)
    t2 = _series_multiply(theta3, theta3, length)
    t4 = _series_multiply(t2, t2, length)
    u = _series_multiply(a4, _series_inverse(t4, length), length)
    z = np.zeros(length, dtype=np.complex128)
    if length > 2:
        z[2:] = 16 * u[: length - 2]
    return z, u, theta3


class SewingNSBlock:
    """Ordinary plumbing block at (0,q,1,infinity), without a pillow prefactor.

    q_order is the integer-level cutoff; the odd component also includes
    level q_order+1/2.  The original primary weights and starred c-recursion
    seeds are retained.  No conversion through modular lambda is made.
    """

    series_parameter = "sewing"

    def __init__(self, block, h_internal, q_order=7):
        self.block = block
        self.h = mp.mpc(h_internal)
        self.q_order = int(q_order)
        self.exponent = complex(self.h - block.h1 - block.h2) - (0.5 if block.star2 else 0)
        self.even = np.asarray([complex(block.coefficient(2*n, self.h))
                                for n in range(self.q_order + 1)], dtype=complex)
        sign = -1 if block.star2 and block.star3 else 1
        self.odd = sign * np.asarray([complex(block.coefficient(2*n+1, self.h))
                                      for n in range(self.q_order + 1)], dtype=complex)

    def value(self, q_sew, parity, order=None, *, geometry=None, derivative=False):
        if parity not in ("e", "o"):
            raise ValueError("parity must be e or o")
        q_sew = np.asarray(q_sew, dtype=complex)
        coefficients = self.even if parity == "e" else self.odd
        if order is not None:
            if not 0 <= order <= self.q_order:
                raise ValueError("requested sewing order exceeds computed coefficients")
            coefficients = coefficients[:order+1]
        exponent = self.exponent + (0.5 if parity == "o" else 0)
        if derivative:
            coefficients = coefficients*(exponent+np.arange(len(coefficients)))
            exponent -= 1
        return np.exp(exponent * np.log(q_sew)) * np.polynomial.polynomial.polyval(q_sew, coefficients)

    def local_data(self, parity, order, total_order):
        if parity not in ("e", "o") or not 0 <= order <= self.q_order:
            raise ValueError("invalid parity or sewing order")
        return self.exponent+(parity == "o")/2, (self.even if parity == "e" else self.odd)[:min(order,total_order)+1]


class ResummedNSBlock(SewingNSBlock):
    """Production c-recursion -> truncated H(qhat) -> plane block."""

    series_parameter = "elliptic_nome"

    def __init__(self, block, h_internal, q_order=7):
        super().__init__(block, h_internal, q_order)
        weights = (block.h1, block.h2+block.star2/2, block.h3+block.star3/2, block.h4)
        self.elliptic = elliptic_conversion.EllipticComponentSeries(
            h_internal, weights, self.even, self.odd, q_order, c=block.c)

    def value(self, z, parity, order=None, *, geometry=None, derivative=False):
        if parity not in ("e", "o"):
            raise ValueError("parity must be e or o")
        return self.elliptic.value(z, int(parity == "o"), order, geometry=geometry, derivative=derivative)

    def local_data(self, parity, order, total_order):
        if parity not in ("e", "o"):
            raise ValueError("parity must be e or o")
        return self.elliptic.local_data(int(parity == "o"), order, total_order)

    def adjacent_component(self, starstar):
        """Apply the adjacent-star Ward identity BEFORE conversion/truncation.

        L_even = S_odd - dP_even, L_odd = S_even - dP_odd.
        This gives the actual (G V1,G V2,V3,V4) component, not a derivative
        of an independently truncated nome approximation.
        """
        n = np.arange(self.q_order+1)
        even = -(self.exponent+n)*self.even
        even[1:] += starstar.odd[:-1]
        odd = starstar.even-(self.exponent+.5+n)*self.odd
        b = self.block
        return elliptic_conversion.EllipticComponentSeries(
            self.h, (b.h1+.5,b.h2+.5,b.h3,b.h4), even, odd, self.q_order, c=b.c)


def plane_block(block, h_internal, order, series_parameter):
    elliptic_conversion.validate_representation(series_parameter)
    return (ResummedNSBlock if series_parameter == "elliptic_nome" else SewingNSBlock)(block,h_internal,order)


class EllipticNSBlock:
    """Legacy/independent elliptic-nome utility; not the HetSO(23) production path.

    We do not use the h-recursion formula.  This is useful in light of the 2026
    correction to one NS elliptic h-recursion block.
    """

    def __init__(self, block: NSBlockComputer, h_internal: complex, q_order: int = 7):
        self.block = block
        self.h = mp.mpc(h_internal)
        self.q_order = q_order
        self.length = 2 * q_order + 3
        self.z_series, self.u_series, self.theta3_series = _modular_series(self.length)
        self.even_h, self.odd_h = self._build_h_series()

    def _build_h_series(self) -> tuple[np.ndarray, np.ndarray]:
        length = self.length
        h = complex(self.h)
        h1 = complex(self.block.h1)
        h2 = complex(self.block.h2)
        h3 = complex(self.block.h3)
        h4 = complex(self.block.h4)
        h2a = h2 + (0.5 if self.block.star2 else 0)
        h3a = h3 + (0.5 if self.block.star3 else 0)
        theta_exponent = 6 - 4 * (h1 + h2a + h3a + h4)

        one_minus_z = -self.z_series.copy()
        one_minus_z[0] += 1
        prefactor = _series_multiply(
            _series_power(self.u_series, h - 0.5, length),
            _series_power(one_minus_z, h2a + h3a - 0.5, length),
            length,
        )
        prefactor = _series_multiply(
            prefactor,
            _series_power(self.theta3_series, -theta_exponent, length),
            length,
        )

        even_polynomial = [1] + [
            complex(self.block.coefficient(2 * n, self.h))
            for n in range(1, self.q_order + 1)
        ]
        even_z = _series_compose(even_polynomial, self.z_series, length)
        even_h = _series_multiply(prefactor, even_z, length)
        even_h[2 * self.q_order + 1 :] = 0

        odd_polynomial = [
            complex(self.block.coefficient(2 * n + 1, self.h))
            for n in range(self.q_order + 1)
        ]
        odd_z_polynomial = _series_compose(odd_polynomial, self.z_series, length)
        sqrt_u = _series_power(self.u_series, 0.5, length)
        sqrt_z = np.zeros(length, dtype=np.complex128)
        sqrt_z[1:] = 4 * sqrt_u[: length - 1]
        # BRY's star-star odd block contains an explicit minus sign.
        sign = -1 if self.block.star2 and self.block.star3 else 1
        odd_z = sign * _series_multiply(sqrt_z, odd_z_polynomial, length)
        odd_h = _series_multiply(prefactor, odd_z, length)
        odd_h[2 * self.q_order + 2 :] = 0
        return even_h, odd_h


def _theta3_grid(q: np.ndarray) -> np.ndarray:
    theta = np.ones_like(q)
    for n in range(1, 60):
        term = 2 * q ** (n * n)
        theta += term
        if np.max(np.abs(term)) < 1e-15:
            break
    return theta


def _q_grid(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # K(z) = RF(0,1-z,1). scipy's symmetric Carlson form supports complex arrays.
    k_z = sps.elliprf(np.zeros_like(z), 1 - z, np.ones_like(z))
    k_1mz = sps.elliprf(np.zeros_like(z), z, np.ones_like(z))
    q = np.exp(-np.pi * k_1mz / k_z)
    return q, _theta3_grid(q)


def _evaluate_elliptic_block(
    elliptic: EllipticNSBlock,
    z: np.ndarray,
    q: np.ndarray,
    theta3: np.ndarray,
    parity: str,
) -> np.ndarray:
    coefficients = elliptic.even_h if parity == "e" else elliptic.odd_h
    t = np.sqrt(q)
    h_series = np.zeros_like(q)
    for coefficient in coefficients[::-1]:
        h_series = h_series * t + coefficient

    h = complex(elliptic.h)
    h1 = complex(elliptic.block.h1)
    h2 = complex(elliptic.block.h2)
    h3 = complex(elliptic.block.h3)
    h4 = complex(elliptic.block.h4)
    h2a = h2 + (0.5 if elliptic.block.star2 else 0)
    h3a = h3 + (0.5 if elliptic.block.star3 else 0)
    theta_exponent = 6 - 4 * (h1 + h2a + h3a + h4)

    return np.exp(
        (h - 0.5) * np.log(16 * q)
        + (0.5 - h1 - h2a) * np.log(z)
        + (0.5 - h2a - h3a) * np.log(1 - z)
        + theta_exponent * np.log(theta3)
    ) * h_series


# ---------------------------------------------------------------------------
# Vector four-point correlator and moduli quadrature
# ---------------------------------------------------------------------------

@dataclass
class PKernel:
    p: float
    weight: float
    primary: SewingNSBlock
    starstar: SewingNSBlock
    even_structure: complex
    odd_structure: complex


def _build_s_channel_data(
    energies: Sequence[complex],
    q_order: int,
    p_nodes: int,
    p_max: float,
    *, momentum_scheme: str = "cutoff", momentum_threshold_options=None,
    series_parameter: str = "elliptic_nome",
) -> list[PKernel]:
    w1, w2, w3, w0 = [complex(x) for x in energies]
    h1, h2, h3, h4 = [h_of_p(x) for x in (w1, w2, w3, w0)]
    momenta, quadrature_weights = _reference_momentum_rule(
        p_nodes, p_max, momentum_scheme, momentum_threshold_options)
    data: list[PKernel] = []
    for p, weight in zip(momenta, quadrature_weights):
        hp = h_of_p(float(p))
        primary_computer = NSBlockComputer(
            h4, h3, h2, h1, False, False, max_level2=2 * q_order+1
        )
        star_computer = NSBlockComputer(
            h4, h3, h2, h1, True, True, max_level2=2 * q_order+1
        )
        data.append(
            PKernel(
                p=float(p),
                weight=float(weight) / np.pi,
                primary=plane_block(primary_computer, hp, q_order, series_parameter),
                starstar=plane_block(star_computer, hp, q_order, series_parameter),
                even_structure=complex(c_even(w1, w2, p) * c_even(w3, w0, p)),
                odd_structure=complex(c_odd(w1, w2, p) * c_odd(w3, w0, p)),
            )
        )
    return data


def _integrate_one_p_kernel(
    kernel: PKernel,
    energies: Sequence[complex],
    z_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    geometry=None,
) -> np.ndarray:
    w1, w2, w3, _ = [complex(x) for x in energies]
    z, weights = z_data[:2]
    zbar = np.conj(z)
    if geometry is None and kernel.primary.series_parameter == "elliptic_nome":
        geometry = elliptic_conversion.nome_geometry(z)
    geometry_bar = None if geometry is None else tuple(np.conj(a) for a in geometry)
    fe = kernel.primary.value(z, "e", geometry=geometry)
    fo = kernel.primary.value(z, "o", geometry=geometry)
    fe_star = kernel.starstar.value(z, "e", geometry=geometry)
    fo_star = kernel.starstar.value(z, "o", geometry=geometry)
    febar = kernel.primary.value(zbar, "e", geometry=geometry_bar)
    fobar = kernel.primary.value(zbar, "o", geometry=geometry_bar)

    primary_correlator = (
        kernel.even_structure * fe * febar
        + kernel.odd_structure * fo * fobar
    )
    two_descendant_correlator = (
        kernel.even_structure * fo_star * febar
        + kernel.odd_structure * fe_star * fobar
    )
    pco_bracket = two_descendant_correlator + (w2 * w3) / (1 - z) * primary_correlator
    time_factor = np.exp(
        -2 * w1 * w2 * np.log(np.abs(z))
        -2 * w2 * w3 * np.log(np.abs(1 - z))
    )
    base = weights * time_factor * pco_bracket
    return np.array(
        [np.sum(base / zbar), np.sum(-base), np.sum(base / (1 - zbar))]
    )


def _integrate_s_channel(
    data: Sequence[PKernel],
    energies: Sequence[complex],
    z_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    total = np.zeros(3, dtype=complex)
    geometry = (elliptic_conversion.nome_geometry(z_data[0])
                if data and data[0].primary.series_parameter == "elliptic_nome" else None)
    for kernel in data:
        total += kernel.weight * _integrate_one_p_kernel(kernel, energies, z_data, geometry)
    return total


def _unit_disk_excluding_lens(
    epsilon: float,
    theta_orders: tuple[int, int, int],
    radial_order: int,
    *,
    sewing: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Quadrature on |z|<=1 with the lens |1-z|<epsilon removed."""
    theta_inner = 2 * np.arcsin(epsilon / 2)
    theta_outer = np.arcsin(epsilon)
    intervals = [
        (-np.pi, -theta_outer),
        (-theta_outer, -theta_inner),
        (-theta_inner, 0),
        (0, theta_inner),
        (theta_inner, theta_outer),
        (theta_outer, np.pi),
    ]
    orders = [
        theta_orders[2],
        theta_orders[1],
        theta_orders[0],
        theta_orders[0],
        theta_orders[1],
        theta_orders[2],
    ]
    radial_nodes, radial_weights = leggauss(radial_order)
    z_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []

    for (start, end), order in zip(intervals, orders):
        theta_nodes, theta_weights = leggauss(order)
        theta = (start + end) / 2 + (end - start) * theta_nodes / 2
        theta_weight = (end - start) * theta_weights / 2
        for angle, angular_weight in zip(theta, theta_weight):
            cosine = np.cos(angle)
            sine = np.sin(angle)
            allowed: list[tuple[float, float]] = []
            if abs(sine) >= epsilon or cosine <= 0:
                allowed = [(0.0, 1.0)]
            else:
                delta = np.sqrt(max(0.0, epsilon * epsilon - sine * sine))
                r_minus = max(0.0, cosine - delta)
                r_plus = min(1.0, cosine + delta)
                if r_minus > 0:
                    allowed.append((0.0, r_minus))
                if r_plus < 1:
                    allowed.append((r_plus, 1.0))
            for r_start, r_end in allowed:
                if r_end <= r_start:
                    continue
                switch = 0.5/cosine if cosine > 0 else float("inf")
                cuts = [r_start,switch,r_end] if sewing and r_start < switch < r_end else [r_start,r_end]
                for lo,hi in zip(cuts[:-1],cuts[1:]):
                    radius = (lo + hi) / 2 + (hi - lo) * radial_nodes / 2
                    radial_weight = (hi - lo) * radial_weights / 2
                    z_parts.append(radius * np.exp(1j * angle))
                    weight_parts.append(angular_weight * radial_weight * radius)

    z = np.concatenate(z_parts)
    weights = np.concatenate(weight_parts)
    if sewing:
        return z, weights
    q, theta3 = _q_grid(z)
    return z, weights, q, theta3


def _full_plane_rest(
    original_data: Sequence[PKernel],
    swapped_data: Sequence[PKernel],
    energies: Sequence[complex],
    epsilon: float,
    theta_orders: tuple[int, int, int],
    radial_order: int,
    original_t=None,
    swapped_t=None,
    order=None,
) -> np.ndarray:
    z_data = _unit_disk_excluding_lens(
        epsilon, theta_orders=theta_orders, radial_order=radial_order, sewing=True
    )
    def bulk(s_data,t_data,ws):
        if t_data is None:
            return _integrate_s_channel(s_data,ws,z_data)
        z,weights=z_data
        if s_data and s_data[0].primary.series_parameter == "elliptic_nome":
            use_t=abs(elliptic_conversion.nome_geometry(1-z)[0])<abs(elliptic_conversion.nome_geometry(z)[0])
        else:
            use_t=np.abs(1-z)<np.abs(z)
        total=np.zeros(3,dtype=complex)
        if np.any(~use_t):
            total += _integrate_s_channel(s_data,ws,(z[~use_t],weights[~use_t]))
        if np.any(use_t):
            for kernel in t_data:
                total += kernel.weight*_t_kernel_integral(kernel,ws,1-z[use_t],weights[use_t],order)
        return total

    original = bulk(original_data,original_t,energies)
    swapped_energies = [energies[0], energies[2], energies[1], energies[3]]
    swapped = bulk(swapped_data,swapped_t,swapped_energies)
    # The exterior |z|>1 is mapped to the unit disk by z -> 1/z and 2<->3.
    return np.array(
        [
            original[0] + swapped[1],
            original[1] + swapped[0],
            original[2] + swapped[2],
        ]
    )


# Direct z-series is efficient in the small t-channel lens w=1-z.
@dataclass
class TKernel:
    p: float
    weight: float
    primary: NSBlockComputer
    starstar: NSBlockComputer
    even_structure: complex
    odd_structure: complex
    h_internal: complex
    resummed_primary: ResummedNSBlock | None = None
    resummed_adjacent: elliptic_conversion.EllipticComponentSeries | None = None


def _reference_momentum_rule(p_nodes, p_max, scheme, threshold_options):
    if scheme == "threshold_weighted":
        rule = threshold_weighted_rule(p_nodes, threshold_options)
        return rule.momenta, rule.weights
    if scheme != "cutoff" or threshold_options is not None:
        raise ValueError("reference momentum rule needs threshold_weighted or explicit cutoff without threshold options")
    nodes, weights = leggauss(p_nodes)
    return (nodes + 1) * p_max / 2, weights * p_max / 2


def _build_t_channel_data(
    energies: Sequence[complex],
    series_order: int,
    p_nodes: int,
    p_max: float,
    *, momentum_scheme: str = "cutoff", momentum_threshold_options=None,
    series_parameter: str = "elliptic_nome",
) -> list[TKernel]:
    w1, w2, w3, w0 = [complex(x) for x in energies]
    h1, h2, h3, h4 = [h_of_p(x) for x in (w1, w2, w3, w0)]
    momenta, quadrature_weights = _reference_momentum_rule(
        p_nodes, p_max, momentum_scheme, momentum_threshold_options)
    data: list[TKernel] = []
    # In the t channel the order is (4,1,2,3).
    for p, weight in zip(momenta, quadrature_weights):
        h_internal = h_of_p(float(p))
        data.append(
            TKernel(
                p=float(p),
                weight=float(weight) / np.pi,
                primary=NSBlockComputer(
                    h4, h1, h2, h3, False, False, max_level2=2 * series_order + 1
                ),
                starstar=NSBlockComputer(
                    h4, h1, h2, h3, True, True, max_level2=2 * series_order + 1
                ),
                even_structure=complex(c_even(w2, w3, p) * c_even(w1, w0, p)),
                odd_structure=complex(c_odd(w2, w3, p) * c_odd(w1, w0, p)),
                h_internal=complex(h_internal),
            )
        )
    elliptic_conversion.validate_representation(series_parameter)
    if series_parameter == "elliptic_nome":
        for kernel in data:
            kernel.resummed_primary = ResummedNSBlock(kernel.primary,kernel.h_internal,series_order)
            star = ResummedNSBlock(kernel.starstar,kernel.h_internal,series_order)
            kernel.resummed_adjacent = kernel.resummed_primary.adjacent_component(star)
    return data


def _direct_block(
    block: NSBlockComputer,
    h_internal: complex,
    z: np.ndarray,
    parity: str,
    order: int,
    derivative: bool = False,
) -> np.ndarray:
    z = np.asarray(z, dtype=complex)
    h = complex(h_internal)
    h1 = complex(block.h1)
    h2 = complex(block.h2)
    h2a = h2 + (0.5 if block.star2 else 0)
    exponent = h - h1 - h2a

    if parity == "e":
        coefficients = [1] + [
            complex(block.coefficient(2 * n, h)) for n in range(1, order + 1)
        ]
        if derivative:
            differentiated = [
                (exponent + n) * coefficients[n] for n in range(len(coefficients))
            ]
            return np.exp((exponent - 1) * np.log(z)) * np.polynomial.polynomial.polyval(
                z, differentiated
            )
        return np.exp(exponent * np.log(z)) * np.polynomial.polynomial.polyval(
            z, coefficients
        )

    sign = -1 if block.star2 and block.star3 else 1
    coefficients = [
        complex(block.coefficient(2 * n + 1, h)) for n in range(order + 1)
    ]
    if derivative:
        differentiated = [
            (exponent + 0.5 + n) * coefficients[n]
            for n in range(len(coefficients))
        ]
        return sign * np.exp((exponent - 0.5) * np.log(z)) * np.polynomial.polynomial.polyval(
            z, differentiated
        )
    return sign * np.exp((exponent + 0.5) * np.log(z)) * np.polynomial.polynomial.polyval(
        z, coefficients
    )


def _t_kernel_integral(
    kernel: TKernel,
    energies: Sequence[complex],
    w: np.ndarray,
    weights: np.ndarray,
    order: int,
) -> np.ndarray:
    w1, w2, w3, _ = [complex(x) for x in energies]
    wbar = np.conj(w)

    if kernel.resummed_primary is not None:
        return _resummed_t_kernel_integral(kernel, energies, w, weights, order)

    fe = _direct_block(kernel.primary, kernel.h_internal, w, "e", order)
    fo = _direct_block(kernel.primary, kernel.h_internal, w, "o", order)
    febar = _direct_block(kernel.primary, kernel.h_internal, wbar, "e", order)
    fobar = _direct_block(kernel.primary, kernel.h_internal, wbar, "o", order)
    fe_star = _direct_block(kernel.starstar, kernel.h_internal, w, "e", order)
    fo_star = _direct_block(kernel.starstar, kernel.h_internal, w, "o", order)

    # Ward identity for stars on the adjacent pair (2,1):
    # B_21 = B_23 - d_w A.  Internal parity is exchanged by the stars.
    pair_even = fo_star - _direct_block(
        kernel.primary, kernel.h_internal, w, "e", order, derivative=True
    )
    pair_odd = fe_star - _direct_block(
        kernel.primary, kernel.h_internal, w, "o", order, derivative=True
    )

    primary_correlator = (
        kernel.even_structure * fe * febar
        + kernel.odd_structure * fo * fobar
    )
    descendant_correlator = (
        kernel.even_structure * pair_even * febar
        + kernel.odd_structure * pair_odd * fobar
    )
    bracket = descendant_correlator + (w2 * w3) / w * primary_correlator
    z = 1 - w
    time_factor = np.exp(
        -2 * w1 * w2 * np.log(np.abs(z))
        -2 * w2 * w3 * np.log(np.abs(w))
    )
    base = weights * time_factor * bracket
    return np.array(
        [
            np.sum(base / (1 - wbar)),
            np.sum(-base),
            np.sum(base / wbar),
        ]
    )


def _resummed_t_kernel_integral(kernel, energies, w, weights, order, geometry=None):
    """Plane correlator with each crossed component truncated in qhat."""
    w1,w2,w3,_ = map(complex,energies)
    wb = np.conj(w)
    geometry = elliptic_conversion.nome_geometry(w) if geometry is None else geometry
    geometry_bar = tuple(np.conj(a) for a in geometry)
    p,l = kernel.resummed_primary,kernel.resummed_adjacent
    fe = p.value(w,'e',order,geometry=geometry)
    fo = p.value(w,'o',order,geometry=geometry)
    feb = p.value(wb,'e',order,geometry=geometry_bar)
    fob = p.value(wb,'o',order,geometry=geometry_bar)
    le = l.value(w,0,order,geometry=geometry)
    lo = l.value(w,1,order,geometry=geometry)
    g = kernel.even_structure*fe*feb+kernel.odd_structure*fo*fob
    descendant = kernel.even_structure*le*feb+kernel.odd_structure*lo*fob
    z = 1-w
    time_factor = np.exp(-2*w1*w2*np.log(abs(z))-2*w2*w3*np.log(abs(w)))
    base = weights*time_factor*(descendant+w2*w3/w*g)
    return np.array([np.sum(base/(1-wb)),np.sum(-base),np.sum(base/wb)])


def _lens_grid(
    epsilon: float,
    radial_order: int,
    angular_order: int,
    radial_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = leggauss(radial_order)
    x = (nodes + 1) / 2
    x_weight = weights / 2
    rho = epsilon * x**radial_power
    rho_weight = epsilon * radial_power * x ** (radial_power - 1) * x_weight
    phi_nodes, phi_weights = leggauss(angular_order)
    w_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for radius, radial_weight in zip(rho, rho_weight):
        phi_max = np.arccos(radius / 2)
        phi = phi_max * phi_nodes
        angular_weight = phi_max * phi_weights
        w_parts.append(radius * np.exp(1j * phi))
        weight_parts.append(radial_weight * angular_weight * radius)
    return np.concatenate(w_parts), np.concatenate(weight_parts)


def _lens_integral(
    data: Sequence[TKernel],
    energies: Sequence[complex],
    epsilon: float,
    order: int,
    radial_order: int,
    angular_order: int,
    radial_power: float,
) -> np.ndarray:
    w, weights = _lens_grid(epsilon, radial_order, angular_order, radial_power)
    total = np.zeros(3, dtype=complex)
    for kernel in data:
        total += kernel.weight * _t_kernel_integral(
            kernel, energies, w, weights, order
        )
    return total


def _full_lens_integral(
    original_data: Sequence[TKernel],
    swapped_data: Sequence[TKernel],
    energies: Sequence[complex],
    epsilon: float,
    order: int,
    radial_order: int,
    angular_order: int,
    radial_power: float,
) -> np.ndarray:
    original = _lens_integral(
        original_data,
        energies,
        epsilon,
        order,
        radial_order,
        angular_order,
        radial_power,
    )
    swapped_energies = [energies[0], energies[2], energies[1], energies[3]]
    swapped = _lens_integral(
        swapped_data,
        swapped_energies,
        epsilon,
        order,
        radial_order,
        angular_order,
        radial_power,
    )
    return np.array(
        [
            original[0] + swapped[1],
            original[1] + swapped[0],
            original[2] + swapped[2],
        ]
    )


def vector_amplitude_coefficients(
    energies: Sequence[complex],
    *,
    q_order: int = 6,
    p_nodes: int = 28,
    p_max: float = 5.0,
    momentum_scheme: str = "threshold_weighted",
    momentum_threshold_options=None,
    series_parameter: str = "elliptic_nome",
    epsilon: float = 0.05,
    theta_orders: tuple[int, int, int] = (80, 80, 200),
    radial_order: int = 72,
    lens_radial_order: int = 64,
    lens_angular_order: int = 160,
    lens_power: float = 3.0,
) -> np.ndarray:
    """Compute (A,B,C) in a convergent Euclidean/sub-threshold domain.

    C is evaluated by a cyclic relabeling, which avoids a large cancellation in
    the direct s-channel representation.
    """
    if len(energies) != 4:
        raise ValueError("energies must be [omega1, omega2, omega3, omega0]")
    if abs(complex(energies[3] - sum(energies[:3]))) > 1e-12:
        raise ValueError("energy conservation omega0=omega1+omega2+omega3 is required")

    def one_ordering(ws: Sequence[complex]) -> np.ndarray:
        swapped = [ws[0], ws[2], ws[1], ws[3]]
        options = dict(momentum_scheme=momentum_scheme, momentum_threshold_options=momentum_threshold_options,
                       series_parameter=series_parameter)
        s_data = _build_s_channel_data(ws, q_order, p_nodes, p_max, **options)
        s_swapped = _build_s_channel_data(swapped, q_order, p_nodes, p_max, **options)
        t_data = _build_t_channel_data(ws, q_order, p_nodes, p_max, **options)
        t_swapped = _build_t_channel_data(swapped, q_order, p_nodes, p_max, **options)
        rest = _full_plane_rest(
            s_data,
            s_swapped,
            ws,
            epsilon,
            theta_orders,
            radial_order,
            t_data,
            t_swapped,
            q_order,
        )
        lens = _full_lens_integral(
            t_data,
            t_swapped,
            ws,
            epsilon,
            q_order,
            lens_radial_order,
            lens_angular_order,
            lens_power,
        )
        return rest + lens

    original = one_ordering(energies)
    # C for (w1,w2,w3) is A after cyclic permutation (w1,w2,w3)->(w2,w3,w1).
    cyclic_energies = [energies[1], energies[2], energies[0], energies[3]]
    cyclic = one_ordering(cyclic_energies)
    return np.array([original[0], original[1], cyclic[0]])


# ---------------------------------------------------------------------------
# Resonance identities
# ---------------------------------------------------------------------------

def complex_beta_integral(a: complex, abar: complex, b: complex, bbar: complex) -> mp.mpc:
    """Single-valued complex beta integral in its analytic-continuation form."""
    return (
        mp.pi
        * mp.gamma(a)
        * mp.gamma(b)
        * mp.gamma(1 - abar - bbar)
        / (mp.gamma(1 - abar) * mp.gamma(1 - bbar) * mp.gamma(a + b))
    )


def vector_resonance_coefficients(
    omega1: complex, omega2: complex, omega3: complex
) -> tuple[mp.mpc, mp.mpc, mp.mpc]:
    """Exact reduced VVVV coefficients on omega0=i and sum omega_j=i.

    The result is in the common phase convention used in the note.
    """
    if abs(complex(omega1 + omega2 + omega3 - 1j)) > 1e-10:
        raise ValueError("resonance requires omega1+omega2+omega3=i")
    return (
        mp.pi * omega1 * omega2,
        mp.pi * omega1 * omega3,
        mp.pi * omega2 * omega3,
    )


def ssvv_resonance_coefficient(
    omega1: complex, omega2: complex, omega3: complex
) -> mp.mpc:
    """Canonical normalized V->VSS resonance coefficient, without delta^{ab}.

    Here omega1 is the outgoing vector energy and omega2,omega3 are singlet
    energies.  The overall common phase/sign is omitted.
    """
    if abs(complex(omega1 + omega2 + omega3 - 1j)) > 1e-10:
        raise ValueError("resonance requires omega1+omega2+omega3=i")
    numerator = -mp.pi * omega2 * omega3 * (1 + 1j * omega2) * (1 + 1j * omega3)
    denominator = mp.sqrt((1 + omega2**2) * (1 + omega3**2))
    return numerator / denominator


REFERENCE_SAMPLE = {
    "energies": ["0.10j", "0.12j", "0.14j", "0.36j"],
    "settings": {
        "q_order": 6,
        "p_nodes": 28,
        "p_max": 5.0,
        "epsilon": 0.04,
        "theta_orders": [100, 100, 240],
        "radial_order": 80,
        "lens_radial_order": 72,
        "lens_angular_order": 180,
        "lens_power": 3.0,
    },
    "coefficients": [
        -0.0024982,
        -0.0025521,
        -0.0026108,
    ],
    "estimated_absolute_moduli_quadrature_error": 2.0e-6,
}


def _parse_complex_list(text: str) -> list[complex]:
    values = [complex(item.strip()) for item in text.split(",") if item.strip()]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("expected four comma-separated complex numbers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resonance",
        default="0.2j,0.3j,0.5j",
        help="three outgoing resonance energies, comma-separated (sum must be 1j)",
    )
    parser.add_argument(
        "--numerical",
        type=_parse_complex_list,
        help="compute numerical VVVV coefficients for omega1,omega2,omega3,omega0",
    )
    parser.add_argument("--quick", action="store_true", help="use a faster, lower-accuracy grid")
    parser.add_argument("--show-reference", action="store_true", help="print the stored reference sample")
    args = parser.parse_args()

    resonance_values = [complex(x.strip()) for x in args.resonance.split(",")]
    if len(resonance_values) != 3:
        parser.error("--resonance requires three values")
    a, b, c = vector_resonance_coefficients(*resonance_values)
    print("Exact VVVV resonance coefficients (common normalization omitted):")
    print("  A =", mp.nstr(a, 18))
    print("  B =", mp.nstr(b, 18))
    print("  C =", mp.nstr(c, 18))
    print("Exact canonical normalized V->VSS resonance coefficient:")
    print("  ", mp.nstr(ssvv_resonance_coefficient(*resonance_values), 18))

    if args.show_reference:
        print("\nStored reference sample:")
        print(json.dumps(REFERENCE_SAMPLE, indent=2))

    if args.numerical is not None:
        start = time.time()
        if args.quick:
            settings = dict(
                q_order=5,
                p_nodes=14,
                p_max=4.5,
                epsilon=0.06,
                theta_orders=(36, 36, 96),
                radial_order=36,
                lens_radial_order=32,
                lens_angular_order=80,
                lens_power=3.0,
            )
        else:
            settings = dict(
                q_order=6,
                p_nodes=28,
                p_max=5.0,
                epsilon=0.04,
                theta_orders=(100, 100, 240),
                radial_order=80,
                lens_radial_order=72,
                lens_angular_order=180,
                lens_power=3.0,
            )
        coefficients = vector_amplitude_coefficients(args.numerical, **settings)
        print("\nNumerical VVVV coefficients (common normalization omitted):")
        for name, value in zip("ABC", coefficients):
            print(f"  {name} = {value.real:.10g} {value.imag:+.3g}j")
        print(f"elapsed: {time.time()-start:.1f} s")


if __name__ == "__main__":
    main()
