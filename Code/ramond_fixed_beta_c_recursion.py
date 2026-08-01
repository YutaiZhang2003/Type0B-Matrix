"""Fixed-beta c-recursion for the mixed sphere Ramond channel.

The block is

    <NS_4 R_3 R_2 NS_1>

with a generic long Ramond representation on the internal edge.  Its
analytic variables are

    (c; h_1, beta_2, beta_3, h_4; beta, sign_3, sign_2).

In particular, ``beta`` rather than ``h_R=c/24-beta**2`` is held fixed when
``c`` moves.  This converts the two HJS beta poles into two c poles for every
unordered Ramond Kac pair.  We use the canonical positive-beta sheet

    beta = beta_rs(b) = (r b + s/b)/(2 sqrt(2)).

The negative-beta description gives the same c poles after ``b -> -b`` and
must not be added a second time.

The raw HJS elliptic block has a polynomial regular part in c.  We therefore
recurse the oscillator-normalized series

    Hhat(c,q) = H(c,q) /
      [prod_{n>=1} (1+q**n)/(1-q**n)]**(c/6).

The pole kernel is unchanged by this normalization.  Direct Ward/Gram
calculation is promoted by an all-level global-family decomposition:
every positive-level quasi-primary endpoint coupling is O(c**-1/2),
so its sewn family is O(c**-1).  The surviving ground family gives

    S_beta(q) = (16 q / z(q))**(beta**2)
        * (1-z(q))**(h1+h4+1/16)
        * theta3(q)**(
            4*(h1+h4-beta2**2-beta3**2) + 1/4
        ).

Equivalently, after extracting the primary power in the local z-series,
the fixed-beta large-c block is

    (1-z)**(-c/12 + h1+h4+beta2**2+beta3**2).

The closed seed reproduces the direct q and q^2 terms and finite-c recursion
checks through q^4.  It is the production default at every order.  The
generic recursion still requires a long internal module, beta != 0; the
short quotient at beta = 0 has the same large-c regular term but different
finite-c Gram and pole data.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Dict, Sequence, Union

from mixed_ramond_sphere_blocks import (
    MixedRExchangeSphereFourPointBlock,
    _r_a_beta,
    _r_beta_prime,
    _r_ns_fusion_polynomial,
)
from superconformal_blocks import (
    NSSphereFourPointBlock,
    central_charge,
    _series_mul,
    _series_pow,
)


Number = Union[complex, float]
RegularSeed = Callable[[int, complex, int, int], complex]


@dataclass(frozen=True)
class RamondCPole:
    """One fixed-beta Ramond Kac pole on the canonical beta=+beta_rs sheet."""

    r: int
    s: int
    branch: int
    b: complex
    c: complex
    beta_prime: complex
    derivative_beta_c: complex


@dataclass(frozen=True)
class RamondCResidueCheck:
    """Numerical comparison of a predicted and measured c-plane residue."""

    pole: RamondCPole
    power: int
    predicted: complex
    measured: complex

    @property
    def relative_error(self) -> float:
        return abs(self.predicted - self.measured) / max(
            abs(self.measured), 1.0e-300
        )


def _sign(value: int, name: str) -> int:
    value = int(value)
    if value not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1")
    return value


def ramond_c_poles(
    internal_beta: Number,
    r: int,
    s: int,
    *,
    pole_tolerance: float = 1.0e-12,
) -> tuple[RamondCPole, RamondCPole]:
    """Return the two c poles associated with one unordered R Kac pair.

    Labels must obey ``r>s>=1`` and ``r+s`` odd.  The two roots solve

        r b^2 - 2 sqrt(2) beta b + s = 0.

    Using both roots and only one ordering of ``(r,s)`` avoids double
    counting the equivalent descriptions ``(s,r,1/b)``.
    """

    if r <= s or s < 1 or (r + s) % 2 != 1:
        raise ValueError("canonical R labels require r>s>=1 and r+s odd")
    beta = complex(internal_beta)
    discriminant = cmath.sqrt(2.0 * beta * beta - r * s)
    roots = (
        (math.sqrt(2.0) * beta + discriminant) / r,
        (math.sqrt(2.0) * beta - discriminant) / r,
    )
    result = []
    for branch, b_pole in zip((1, -1), roots):
        q_background = b_pole + 1.0 / b_pole
        c_pole = central_charge(b_pole)
        derivative_beta_b = (
            r - s / (b_pole * b_pole)
        ) / (2.0 * math.sqrt(2.0))
        derivative_c_b = (
            6.0
            * q_background
            * (1.0 - 1.0 / (b_pole * b_pole))
        )
        scale = max(1.0, abs(derivative_beta_b), abs(derivative_c_b))
        if abs(derivative_c_b) <= pole_tolerance * scale:
            raise ZeroDivisionError(
                "the b-to-c map is ramified at this R Kac pole"
            )
        derivative_beta_c = derivative_beta_b / derivative_c_b
        if abs(derivative_beta_c) <= pole_tolerance / scale:
            raise ZeroDivisionError(
                "colliding fixed-beta R Kac poles require a higher-order "
                "Laurent prescription"
            )
        result.append(
            RamondCPole(
                r=r,
                s=s,
                branch=branch,
                b=b_pole,
                c=c_pole,
                beta_prime=_r_beta_prime(b_pole, r, s),
                derivative_beta_c=derivative_beta_c,
            )
        )
    return result[0], result[1]


def ramond_oscillator_series(c: Number, max_power: int) -> tuple[complex, ...]:
    r"""Return coefficients of

    ``[prod_n (1+q^n)/(1-q^n)]^(c/6)`` through ``q^max_power``.
    """

    if not isinstance(max_power, int) or max_power < 0:
        raise ValueError("max_power must be a nonnegative integer")
    product = [0.0j] * (max_power + 1)
    product[0] = 1.0 + 0.0j
    for n in range(1, max_power + 1):
        factor = [0.0j] * (max_power + 1)
        factor[0] = 1.0 + 0.0j
        for multiple in range(1, max_power // n + 1):
            factor[multiple * n] = 2.0 + 0.0j
        product = _series_mul(product, factor, max_power)
    return tuple(_series_pow(product, complex(c) / 6.0, max_power))


def normalize_hjs_series(
    coefficients: Sequence[Number], c: Number
) -> tuple[complex, ...]:
    """Divide a raw HJS q-series by the universal Ramond oscillator factor."""

    if not coefficients:
        return ()
    max_power = len(coefficients) - 1
    oscillator = ramond_oscillator_series(c, max_power)
    inverse = _series_pow(oscillator, -1.0, max_power)
    return tuple(
        _series_mul(
            [complex(value) for value in coefficients],
            inverse,
            max_power,
        )
    )


@lru_cache(maxsize=None)
def _ramond_large_c_seed_series(
    max_power: int,
    internal_beta: complex,
    beta2_r: complex,
    beta3_r: complex,
    h1_ns: complex,
    h4_ns: complex,
) -> tuple[complex, ...]:
    """Return the closed all-order large-c seed.

    The elliptic nome is the HJS nome

        q = exp[-pi K(1-z)/K(z)],

    so ``z(q)`` is the modular lambda function.  All powers are expanded as
    formal series about ``q=0``.
    """

    theta3_full, _, z_full = NSSphereFourPointBlock._elliptic_series_data(
        max_power + 1
    )
    theta3 = theta3_full[: max_power + 1]
    z_series = z_full[: max_power + 1]
    z_over_16q = [
        z_full[power + 1] / 16.0 for power in range(max_power + 1)
    ]
    one_minus_z = [-value for value in z_series]
    one_minus_z[0] += 1.0

    one_minus_exponent = h1_ns + h4_ns + 1.0 / 16.0
    theta_exponent = (
        4.0
        * (
            h1_ns
            + h4_ns
            - beta2_r * beta2_r
            - beta3_r * beta3_r
        )
        + 0.25
    )
    seed = _series_mul(
        _series_pow(
            z_over_16q,
            -internal_beta * internal_beta,
            max_power,
        ),
        _series_pow(one_minus_z, one_minus_exponent, max_power),
        max_power,
    )
    return tuple(
        _series_mul(
            seed,
            _series_pow(theta3, theta_exponent, max_power),
            max_power,
        )
    )


def ramond_large_c_seed_series(
    *,
    max_power: int,
    internal_beta: Number,
    beta2_r: Number,
    beta3_r: Number,
    h1_ns: Number,
    h4_ns: Number,
) -> tuple[complex, ...]:
    """Return the all-level ``S_beta(q)`` through ``q**max_power``.

    This public wrapper validates the truncation order and canonicalizes all
    parameters before dispatching to the cached formal-series evaluator.
    Direct Ward/Gram sewing and the HJS recursion provide independent
    low-order checks of the all-level global-family derivation.
    """

    if not isinstance(max_power, int) or max_power < 0:
        raise ValueError("max_power must be a nonnegative integer")
    return _ramond_large_c_seed_series(
        max_power,
        complex(internal_beta),
        complex(beta2_r),
        complex(beta3_r),
        complex(h1_ns),
        complex(h4_ns),
    )


def ramond_large_c_seed_candidate_series(
    **kwargs: object,
) -> tuple[complex, ...]:
    """Backward-compatible alias for ramond_large_c_seed_series."""

    return ramond_large_c_seed_series(**kwargs)


class FixedBetaRExchangeSphereFourPointBlock:
    r"""Oscillator-normalized fixed-beta c-recursive R-channel block."""

    def __init__(
        self,
        *,
        c: Number,
        h1_ns: Number,
        beta2_r: Number,
        beta3_r: Number,
        h4_ns: Number,
        internal_beta: Number,
        sign3: int = 1,
        sign2: int = 1,
        regular_seed: RegularSeed | None = None,
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.c = complex(c)
        self.h1 = complex(h1_ns)
        self.beta2 = complex(beta2_r)
        self.beta3 = complex(beta3_r)
        self.h4 = complex(h4_ns)
        self.internal_beta = complex(internal_beta)
        self.sign3 = _sign(sign3, "sign3")
        self.sign2 = _sign(sign2, "sign2")
        self.regular_seed = regular_seed
        self.pole_tolerance = float(pole_tolerance)
        if abs(self.internal_beta) <= self.pole_tolerance:
            raise ValueError("the generic long-R recursion requires beta != 0")

    def _seed(
        self,
        power: int,
        internal_beta: complex,
        sign3: int,
        sign2: int,
    ) -> complex:
        if power == 0:
            return 1.0 + 0.0j
        if power == 1:
            return (
                8.0 * internal_beta * internal_beta
                - 8.0 * self.beta2 * self.beta2
                - 8.0 * self.beta3 * self.beta3
                - 8.0 * self.h1
                - 8.0 * self.h4
                - 0.5
            )
        if power == 2:
            beta_squared = internal_beta * internal_beta
            beta2_squared = self.beta2 * self.beta2
            beta3_squared = self.beta3 * self.beta3
            difference = (
                beta_squared
                - beta2_squared
                - beta3_squared
                - self.h1
                - self.h4
            )
            return (
                32.0 * difference * difference
                - 16.0 * beta_squared
                + 12.0 * beta2_squared
                + 12.0 * beta3_squared
                - 4.0 * self.h1
                - 4.0 * self.h4
                - 3.0 / 8.0
            )
        if self.regular_seed is None:
            return ramond_large_c_seed_series(
                max_power=power,
                internal_beta=internal_beta,
                beta2_r=self.beta2,
                beta3_r=self.beta3,
                h1_ns=self.h1,
                h4_ns=self.h4,
            )[power]
        return complex(
            self.regular_seed(power, internal_beta, sign3, sign2)
        )

    def _residue_multiplier(
        self,
        pole: RamondCPole,
        sign3: int,
        sign2: int,
    ) -> complex:
        a_beta = _r_a_beta(
            pole.b, pole.r, pole.s, self.pole_tolerance
        )
        left = _r_ns_fusion_polynomial(
            b=pole.b,
            r=pole.r,
            s=pole.s,
            ramond_beta_value=self.beta3,
            ns_weight=self.h4,
            sign=sign3,
        )
        right = _r_ns_fusion_polynomial(
            b=pole.b,
            r=pole.r,
            s=pole.s,
            ramond_beta_value=self.beta2,
            ns_weight=self.h1,
            sign=sign2,
        )
        shift = pole.r * pole.s // 2
        return (
            -(16.0**shift)
            * a_beta
            * left
            * right
            / pole.derivative_beta_c
        )

    @lru_cache(maxsize=None)
    def _coefficient(
        self,
        power: int,
        internal_beta: complex,
        c: complex,
        sign3: int,
        sign2: int,
    ) -> complex:
        result = self._seed(power, internal_beta, sign3, sign2)
        for r in range(2, 2 * power + 1):
            for s in range(1, r):
                product = r * s
                if (
                    (r + s) % 2 != 1
                    or product % 2
                    or product // 2 > power
                ):
                    continue
                shift = product // 2
                for pole in ramond_c_poles(
                    internal_beta,
                    r,
                    s,
                    pole_tolerance=self.pole_tolerance,
                ):
                    denominator = c - pole.c
                    scale = max(1.0, abs(c), abs(pole.c))
                    if abs(denominator) <= self.pole_tolerance * scale:
                        raise ZeroDivisionError(
                            f"c-recursion encountered the ({r},{s}) "
                            f"branch {pole.branch:+d} pole"
                        )
                    result += (
                        self._residue_multiplier(pole, sign3, sign2)
                        / denominator
                        * self._coefficient(
                            power - shift,
                            pole.beta_prime,
                            pole.c,
                            sign3,
                            sign2,
                        )
                    )
        return result

    def coefficient(self, power: int) -> complex:
        """Return the coefficient of ``q^power`` in the normalized block."""

        if not isinstance(power, int) or power < 0:
            raise ValueError("power must be a nonnegative integer")
        return self._coefficient(
            power,
            self.internal_beta,
            self.c,
            self.sign3,
            self.sign2,
        )

    def normalized_coefficients(self, order: int) -> Dict[int, complex]:
        """Return normalized coefficients for powers ``0,...,order-1``."""

        if not isinstance(order, int) or order < 1:
            raise ValueError("order must be a positive integer")
        return {power: self.coefficient(power) for power in range(order)}

    def raw_coefficients(self, order: int) -> Dict[int, complex]:
        """Restore the universal oscillator factor."""

        normalized = [
            self.coefficient(power) for power in range(order)
        ]
        oscillator = ramond_oscillator_series(self.c, order - 1)
        raw = _series_mul(normalized, oscillator, order - 1)
        return {power: value for power, value in enumerate(raw)}

    def hjs_reference_coefficients(
        self, *, b: Number, order: int, normalized: bool = True
    ) -> Dict[int, complex]:
        """Evaluate the established HJS beta-recursion at the same data."""

        reference = MixedRExchangeSphereFourPointBlock.from_fixed_data(
            b=b,
            h1_ns=self.h1,
            beta2_r=self.beta2,
            beta3_r=self.beta3,
            h4_ns=self.h4,
            internal_beta=self.internal_beta,
            sign3=self.sign3,
            sign2=self.sign2,
            pole_tolerance=self.pole_tolerance,
        )
        if abs(reference.c - self.c) > self.pole_tolerance * max(
            1.0, abs(reference.c), abs(self.c)
        ):
            raise ValueError("b does not represent the block's central charge")
        raw = [
            reference.elliptic_coefficients(order)[power]
            for power in range(order)
        ]
        values = (
            normalize_hjs_series(raw, self.c) if normalized else tuple(raw)
        )
        return {power: value for power, value in enumerate(values)}


__all__ = [
    "FixedBetaRExchangeSphereFourPointBlock",
    "RamondCPole",
    "RamondCResidueCheck",
    "normalize_hjs_series",
    "ramond_c_poles",
    "ramond_large_c_seed_candidate_series",
    "ramond_large_c_seed_series",
    "ramond_oscillator_series",
]
