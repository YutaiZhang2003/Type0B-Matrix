#!/usr/bin/env python3
r"""Generic-real-``b`` delta-normalized N=1 super-Liouville constants.

The public constants use the same BRY continuum normalization as
``super_liouville_structure_constants.py``: every primary has two-point
function ``pi delta(P-P')`` and every internal edge is integrated with
``dP/pi``.  The formulas are written in terms of

    Upsilon_NS(x) = Upsilon_b(x/2) Upsilon_b((x+Q)/2),
    Upsilon_R (x) = Upsilon_b((x+b)/2) Upsilon_b((x+1/b)/2).

For positive continuum momentum, the external-leg square roots are chosen
positive.  At generic ``b`` the Upsilon square roots are accompanied by the
leg metrics of Poghossian's reflection-symmetric normalization,

    g_NS(P) = |gamma(1-iP/b) gamma(-iPb)|^(1/2),
    g_R (P) = |gamma(1/2+iPb) gamma(1/2+iP/b)|^(-1/2).

The RRNS coefficient also carries the relative factor ``b**(-2)``.  These
three factors all become one at ``b=1``; omitting them therefore passes a
self-dual regression while giving the wrong relative NS/R normalization at
generic ``b``.  At ``b=1`` the resulting convention is exactly

    N_NS(P) = i sqrt(Upsilon_NS(2iP) Upsilon_NS(-2iP)),
    N_R (P) =   sqrt(Upsilon_R (2iP) Upsilon_R (-2iP))

used by the existing BRY implementation.  Consequently the functions below
reduce to its real ``C``, ``tilde C``, ``C_even`` and ``C_odd`` without an
additional phase conversion.  The Human-Note odd NS coefficient remains
``i*tilde C`` and must be applied only by the sewing assembler.

The optional cosmological factor is the common, momentum-independent factor
left after delta normalization.  With

    K_b = (pi*mu/2) gamma((1+b^2)/2) b^(2-2b^2),

one three-point constant is multiplied by ``K_b^(-Q/(2b))``.  Hence a
genus-two theta graph, which has two pants, carries ``K_b^(-Q/b)``.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Sequence

import mpmath as mp


HERE = Path(__file__).resolve().parent
PLUMBING = HERE.parent / "genus_2_cross_channel"
if str(PLUMBING) not in sys.path:
    sys.path.insert(0, str(PLUMBING))

from liouville_torus import UpsilonB  # noqa: E402


def _python_complex(value: mp.mpc | complex | float) -> complex:
    z = mp.mpc(value)
    return complex(float(mp.re(z)), float(mp.im(z)))


@dataclass
class GenericSuperLiouvilleConstants:
    """Numerical generic-``b`` structure constants in BRY normalization."""

    b: float
    dps: int = 35
    mu: complex = 1.0
    include_cosmological_prefactor: bool = False

    def __post_init__(self) -> None:
        self.b = float(self.b)
        self.dps = int(self.dps)
        if not math.isfinite(self.b) or self.b <= 0.0:
            raise ValueError("b must be finite and positive")
        if self.dps < 20:
            raise ValueError("dps must be at least 20")
        if complex(self.mu) == 0.0:
            raise ValueError("mu must be nonzero")
        self.special = UpsilonB(self.b, dps=self.dps)

    @property
    def q_background(self) -> mp.mpf:
        self.special._set_precision()
        b = mp.mpf(self.b)
        return b + 1 / b

    def log_upsilon_ns(self, x: complex | mp.mpc) -> mp.mpc:
        self.special._set_precision()
        x = mp.mpc(x)
        q = self.q_background
        return self.special.log_upsilon(x / 2) + self.special.log_upsilon(
            (x + q) / 2
        )

    def log_upsilon_r(self, x: complex | mp.mpc) -> mp.mpc:
        self.special._set_precision()
        x = mp.mpc(x)
        b = mp.mpf(self.b)
        return self.special.log_upsilon((x + b) / 2) + self.special.log_upsilon(
            (x + 1 / b) / 2
        )

    def upsilon_ns(self, x: complex | mp.mpc) -> mp.mpc:
        return mp.exp(self.log_upsilon_ns(x))

    def upsilon_r(self, x: complex | mp.mpc) -> mp.mpc:
        return mp.exp(self.log_upsilon_r(x))

    def upsilon_ns_prime_zero(self) -> mp.mpc:
        # Upsilon_b(Q/2)=1 in UpsilonB's normalization and
        # Upsilon_b'(0)=Upsilon_b(b).
        self.special._set_precision()
        return mp.mpf("0.5") * self.special.upsilon(mp.mpf(self.b))

    def _positive_leg(self, momentum: float, sector: str) -> mp.mpf:
        """Positive square root of Upsilon_s(2iP) Upsilon_s(-2iP)."""

        self.special._set_precision()
        p = mp.mpf(momentum)
        if p < 0:
            raise ValueError("the positive-continuum leg convention requires P>=0")
        if p == 0:
            return mp.mpf(0)
        logarithm = (
            self.log_upsilon_ns(2j * p) + self.log_upsilon_ns(-2j * p)
            if sector == "NS"
            else self.log_upsilon_r(2j * p) + self.log_upsilon_r(-2j * p)
        )
        return mp.exp(mp.re(logarithm) / 2)

    def _symmetric_leg_metric(self, momentum: float, sector: str) -> mp.mpf:
        """Poghossian reflection-symmetric metric multiplying one leg.

        Only the absolute value is retained: the discarded unit phase is a
        continuum-primary rephasing and cancels from every diagonal sewing.
        """

        self.special._set_precision()
        p = mp.mpf(momentum)
        b = mp.mpf(self.b)
        if p < 0:
            raise ValueError("the positive-continuum leg convention requires P>=0")
        if p == 0:
            return mp.mpf(1)

        def log_little_gamma(argument: mp.mpc) -> mp.mpc:
            return mp.loggamma(argument) - mp.loggamma(1 - argument)

        if sector == "NS":
            logarithm = log_little_gamma(1 - 1j * p / b) + log_little_gamma(
                -1j * p * b
            )
            return mp.exp(mp.re(logarithm) / 2)
        if sector == "R":
            logarithm = log_little_gamma(mp.mpf("0.5") + 1j * p * b) + log_little_gamma(
                mp.mpf("0.5") + 1j * p / b
            )
            return mp.exp(-mp.re(logarithm) / 2)
        raise ValueError("sector must be NS or R")

    def _normalized_leg(self, momentum: float, sector: str) -> mp.mpf:
        return self._positive_leg(momentum, sector) * self._symmetric_leg_metric(
            momentum, sector
        )

    def cosmological_three_point_factor(self) -> complex:
        """Return the common delta-normalized factor for one pair of pants."""

        if not self.include_cosmological_prefactor:
            return 1.0 + 0.0j
        self.special._set_precision()
        b = mp.mpf(self.b)
        q = b + 1 / b
        x = (1 + b * b) / 2
        log_gamma_ratio = mp.loggamma(x) - mp.loggamma(1 - x)
        log_base = (
            mp.log(mp.pi * mp.mpc(self.mu) / 2)
            + log_gamma_ratio
            + (2 - 2 * b * b) * mp.log(b)
        )
        return _python_complex(mp.exp(-q * log_base / (2 * b)))

    @staticmethod
    def _combinations(
        p1: float, p2: float, p3: float
    ) -> tuple[mp.mpf, tuple[mp.mpf, mp.mpf, mp.mpf]]:
        values = tuple(mp.mpf(value) for value in (p1, p2, p3))
        total = sum(values)
        differences = (
            values[1] + values[2] - values[0],
            values[0] + values[2] - values[1],
            values[0] + values[1] - values[2],
        )
        return total, differences

    def _denominator(self, sectors: Sequence[str], arguments: Sequence[mp.mpc]) -> mp.mpc:
        factors = []
        for sector, argument in zip(sectors, arguments):
            factors.append(
                self.upsilon_ns(argument)
                if sector == "NS"
                else self.upsilon_r(argument)
            )
        return mp.fprod(factors)

    def ns_constants(self, p1: float, p2: float, p3: float) -> tuple[complex, complex]:
        """Return ``(C, tilde_C)`` for three NS primaries."""

        self.special._set_precision()
        momenta = (float(p1), float(p2), float(p3))
        total, differences = self._combinations(*momenta)
        q2 = self.q_background / 2
        arguments = (q2 + 1j * total,) + tuple(
            q2 + 1j * value for value in differences
        )
        numerator = self.upsilon_ns_prime_zero() * mp.fprod(
            self._normalized_leg(value, "NS") for value in momenta
        )
        common = mp.mpc(self.cosmological_three_point_factor())
        c_bottom = common * numerator / self._denominator(("NS",) * 4, arguments)
        c_top = common * 2 * numerator / self._denominator(("R",) * 4, arguments)
        return _python_complex(c_bottom), _python_complex(c_top)

    def rr_ns_constants(
        self, p_r1: float, p_r2: float, p_ns: float
    ) -> tuple[complex, complex]:
        """Return BRY ``(C_even,C_odd)`` for ``R(P1) R(P2) NS(P3)``."""

        self.special._set_precision()
        momenta = (float(p_r1), float(p_r2), float(p_ns))
        total, differences = self._combinations(*momenta)
        delta1, delta2, delta3 = differences
        q2 = self.q_background / 2
        numerator = self.upsilon_ns_prime_zero() * mp.mpf(self.b) ** (-2) * (
            self._normalized_leg(momenta[0], "R")
            * self._normalized_leg(momenta[1], "R")
            * self._normalized_leg(momenta[2], "NS")
        )
        common = mp.mpc(self.cosmological_three_point_factor())
        even = common * numerator / self._denominator(
            ("R", "R", "NS", "NS"),
            (
                q2 + 1j * total,
                q2 + 1j * delta3,
                q2 + 1j * delta1,
                q2 + 1j * delta2,
            ),
        )
        odd = common * numerator / self._denominator(
            ("NS", "NS", "R", "R"),
            (
                q2 + 1j * total,
                q2 + 1j * delta3,
                q2 + 1j * delta1,
                q2 + 1j * delta2,
            ),
        )
        return _python_complex(even), _python_complex(odd)


def hjs_rr_ns_constant(constants: Sequence[complex], sign: int) -> complex:
    """Map HJS sign ``+/-`` to BRY ``C_even/C_odd``."""

    sign = int(sign)
    if sign not in (-1, 1) or len(constants) != 2:
        raise ValueError("constants must be (C_even,C_odd) and sign must be +/-1")
    return complex(constants[0 if sign == 1 else 1])
