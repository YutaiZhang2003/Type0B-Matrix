"""Mixed NS/R sphere blocks in the ordinary-c BRY/HJS convention.

This module implements the two channels needed for the decisive RRNSNS
crossing test of Suchanek, arXiv:1012.2974:

* MixedNSExchangeSphereFourPointBlock for <NS_4 NS_3 R_2 R_1>, with an
  NS internal module;
* MixedRExchangeSphereFourPointBlock for <NS_4 R_3 R_2 NS_1>, with a
  generic long-R internal module.

The supercurrent is G_n, the central charge is the ordinary c, and
beta=iP/sqrt(2). Chiral-form signs are kept separate from the nonchiral
Type-0B Ramond-family label.
"""

from __future__ import annotations

import cmath
import math
from functools import lru_cache
from typing import Dict, Literal, Union

import mpmath

from ramond_sphere_blocks import ramond_beta, ramond_liouville_weight
from superconformal_blocks import central_charge, elliptic_nome, ns_liouville_weight


Number = Union[complex, float]
Parity = Literal["even", "odd"]


def _sign(value: int, name: str = "sign") -> int:
    value = int(value)
    if value not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1")
    return value


def _ns_degenerate_weight(b: complex, r: int, s: int) -> complex:
    if r < 1 or s < 1 or (r + s) % 2:
        raise ValueError("NS Kac labels require r,s >= 1 and r+s even")
    return (
        -(r * s - 1.0) / 4.0
        + (1.0 - r * r) * b * b / 8.0
        + (1.0 - s * s) / (8.0 * b * b)
    )


def _ns_a_factor(
    b: complex, r: int, s: int, pole_tolerance: float
) -> complex:
    result = 0.5 + 0.0j
    for p in range(1 - r, r + 1):
        for q in range(1 - s, s + 1):
            if (p + q) % 2 or (p, q) in ((0, 0), (r, s)):
                continue
            denominator = (p * b + q / b) / math.sqrt(2.0)
            if abs(denominator) < pole_tolerance:
                raise ZeroDivisionError(
                    "resonant NS A_rs factor; displace b from the rational point"
                )
            result /= denominator
    return result


def _rr_ns_fusion_polynomial(
    *,
    b: complex,
    r: int,
    s: int,
    lower_beta: complex,
    upper_beta: complex,
    upper_sign: int,
) -> complex:
    """Return P_NS|RR^{rs}[^{sign beta_upper}_{beta_lower}]."""

    upper_sign = _sign(upper_sign, "upper_sign")
    result = 1.0 + 0.0j
    for p in range(1 - r, r, 2):
        for q in range(1 - s, s, 2):
            offset = (p * b + q / b) / (2.0 * math.sqrt(2.0))
            selector = (p + q - r - s) % 4
            if selector == 2:
                result *= lower_beta - upper_sign * upper_beta + offset
            elif selector == 0:
                result *= lower_beta + upper_sign * upper_beta + offset
            else:
                raise AssertionError("invalid NS|RR fusion parity")
    return result


def _ns_ns_fusion_polynomial(
    *,
    b: complex,
    r: int,
    s: int,
    lower_weight: complex,
    upper_weight: complex,
    starred: bool,
) -> complex:
    """Return P_NS|NN^{rs}[^{(* )Delta_upper}_{Delta_lower}]."""

    q_background = b + 1.0 / b
    lower_lambda = cmath.sqrt(q_background * q_background - 8.0 * lower_weight)
    upper_lambda = cmath.sqrt(q_background * q_background - 8.0 * upper_weight)
    wanted_parity = 1 if starred else 0
    result = 1.0 + 0.0j
    denominator = 2.0 * math.sqrt(2.0)
    for k in range(r):
        for ell in range(s):
            if (k + ell) % 2 != wanted_parity:
                continue
            p = 1 - r + 2 * k
            q = 1 - s + 2 * ell
            linear = p * b + q / b
            result *= (lower_lambda + upper_lambda - linear) / denominator
            result *= (lower_lambda - upper_lambda - linear) / denominator
    return result


def _r_beta_rs(b: complex, r: int, s: int) -> complex:
    return (r * b + s / b) / (2.0 * math.sqrt(2.0))


def _r_beta_prime(b: complex, r: int, s: int) -> complex:
    return ((-1) ** s) * (r * b - s / b) / (2.0 * math.sqrt(2.0))


def _r_a_beta(
    b: complex, r: int, s: int, pole_tolerance: float
) -> complex:
    """Return A^R_rs for a pole in beta."""

    result = -(2.0 ** (r * s - 1.5)) + 0.0j
    for m in range(1 - r, r + 1):
        for n in range(1 - s, s + 1):
            # The Ramond Kac labels have r+s odd, but the inverse-norm
            # product itself runs over the even sublattice.  The ``odd''
            # condition printed below Eq. (4.23) of arXiv:1012.2974 is a
            # typo: at level one it contradicts the exact 2x2 Gram matrix.
            # With the even condition the product reproduces the residues
            # at both beta_{21} and beta_{12}.
            if (m + n) % 2 or (m, n) in ((0, 0), (r, s)):
                continue
            denominator = m * b + n / b
            if abs(denominator) < pole_tolerance:
                raise ZeroDivisionError(
                    "resonant Ramond A_rs factor; displace b from the rational point"
                )
            result /= denominator
    return result


def _r_ns_fusion_polynomial(
    *,
    b: complex,
    r: int,
    s: int,
    ramond_beta_value: complex,
    ns_weight: complex,
    sign: int,
) -> complex:
    """Return P_R|RNS^{rs}[^{sign beta_R}_{Delta_NS}]."""

    sign = _sign(sign)
    q_background = b + 1.0 / b
    # Suchanek parameterizes a physical NS momentum by
    # lambda=Q-2a=-2iP.  This is the opposite square-root branch from the
    # principal +2iP branch used by the BRY NS-only c-recursion.
    ns_lambda = -cmath.sqrt(q_background * q_background - 8.0 * ns_weight)
    result = 1.0 + 0.0j
    denominator = 2.0 * math.sqrt(2.0)
    for k in range(r):
        for ell in range(s):
            p = 1 - r + 2 * k
            q = 1 - s + 2 * ell
            offset = (p * b + q / b) / denominator
            if (k + ell) % 2 == 0:
                result *= ns_lambda / denominator - sign * ramond_beta_value - offset
            else:
                result *= ns_lambda / denominator + sign * ramond_beta_value - offset
    return result


class MixedNSExchangeSphereFourPointBlock:
    r"""Block for <NS_4 NS_3 R_2 R_1> with internal NS weight Delta."""

    def __init__(
        self,
        *,
        b: Number,
        p1_r: Number,
        p2_r: Number,
        p3_ns: Number,
        p4_ns: Number,
        internal_momentum: Number,
        sign2: int = 1,
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.b = complex(b)
        self.c = central_charge(self.b)
        self.beta1 = ramond_beta(p1_r)
        self.beta2 = ramond_beta(p2_r)
        self.h1 = ramond_liouville_weight(p1_r, self.b)
        self.h2 = ramond_liouville_weight(p2_r, self.b)
        self.h3 = ns_liouville_weight(p3_ns, self.b)
        self.h4 = ns_liouville_weight(p4_ns, self.b)
        self.internal_weight = ns_liouville_weight(internal_momentum, self.b)
        self.sign2 = _sign(sign2, "sign2")
        self.pole_tolerance = float(pole_tolerance)

    @classmethod
    def from_weights(
        cls,
        *,
        b: Number,
        h1_r: Number,
        h2_r: Number,
        h3_ns: Number,
        h4_ns: Number,
        internal_weight: Number,
        sign2: int = 1,
        pole_tolerance: float = 1.0e-12,
    ) -> "MixedNSExchangeSphereFourPointBlock":
        """Construct a fixed-weight block for central-charge pole tests.

        Unlike the momentum constructor, this keeps all five conformal
        weights fixed while ``c`` (and hence ``b``) changes.  The Ramond
        square roots use the same principal branch as the c-recursion
        validation layer.
        """

        instance = cls.__new__(cls)
        instance.b = complex(b)
        instance.c = central_charge(instance.b)
        instance.h1 = complex(h1_r)
        instance.h2 = complex(h2_r)
        instance.h3 = complex(h3_ns)
        instance.h4 = complex(h4_ns)
        instance.internal_weight = complex(internal_weight)
        instance.beta1 = cmath.sqrt(instance.c / 24.0 - instance.h1)
        instance.beta2 = cmath.sqrt(instance.c / 24.0 - instance.h2)
        instance.sign2 = _sign(sign2, "sign2")
        instance.pole_tolerance = float(pole_tolerance)
        return instance

    @lru_cache(maxsize=None)
    def _series(
        self,
        max_twice_power: int,
        parity: Parity,
        internal_weight: complex,
        sign2: int,
    ) -> tuple[complex, ...]:
        result = [0.0j] * (max_twice_power + 1)
        if parity == "even":
            result[0] = 1.0 + 0.0j

        for r in range(1, max_twice_power + 1):
            for s in range(1, max_twice_power // r + 1):
                shift = r * s
                if shift > max_twice_power or (r + s) % 2:
                    continue
                delta_rs = _ns_degenerate_weight(self.b, r, s)
                denominator = internal_weight - delta_rs
                if abs(denominator) < self.pole_tolerance:
                    raise ZeroDivisionError(
                        f"internal weight is too close to the ({r},{s}) NS pole"
                    )

                odd_kac = bool(r % 2)
                next_parity: Parity = parity
                next_sign2 = sign2
                phase = 1.0 + 0.0j
                if odd_kac:
                    next_parity = "odd" if parity == "even" else "even"
                    next_sign2 = -sign2
                    phase = cmath.exp(
                        (1j if parity == "even" else -1j) * math.pi / 4.0
                    )

                tail = self._series(
                    max_twice_power - shift,
                    next_parity,
                    complex(delta_rs + shift / 2.0),
                    next_sign2,
                )
                left = _ns_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    lower_weight=self.h4,
                    upper_weight=self.h3,
                    starred=(parity == "odd"),
                )
                right = _rr_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    lower_beta=self.beta1,
                    upper_beta=self.beta2,
                    upper_sign=sign2,
                )
                coefficient = (
                    16.0 ** (shift / 2.0)
                    * phase
                    * _ns_a_factor(self.b, r, s, self.pole_tolerance)
                    * left
                    * right
                    / denominator
                )
                for power, value in enumerate(tail):
                    result[power + shift] += coefficient * value
        return tuple(result)

    def elliptic_coefficients(self, order: int, parity: Parity) -> Dict[int, complex]:
        if order < 1:
            raise ValueError("order must be positive")
        if parity not in ("even", "odd"):
            raise ValueError("parity must be 'even' or 'odd'")
        max_twice = 2 * (order - 1) + (1 if parity == "odd" else 0)
        values = self._series(
            max_twice, parity, self.internal_weight, self.sign2
        )
        remainder = 0 if parity == "even" else 1
        return {
            power: value
            for power, value in enumerate(values)
            if power % 2 == remainder
        }

    def elliptic_block(self, z: Number, order: int, parity: Parity) -> complex:
        z = complex(z)
        q = elliptic_nome(z)
        series = sum(
            value * q ** (power / 2.0)
            for power, value in self.elliptic_coefficients(order, parity).items()
        )
        return self._elliptic_prefactor(z) * series

    def _elliptic_prefactor(self, z: Number) -> complex:
        """Return the mixed NS-exchange prefactor at fixed ``b``."""

        z = complex(z)
        q = elliptic_nome(z)
        theta3 = complex(mpmath.jtheta(3, 0, mpmath.mpc(q)))
        vacuum_shift = (self.c - 1.5) / 24.0
        return (
            (16.0 * q) ** (self.internal_weight - vacuum_shift)
            * z ** (vacuum_shift - self.h1 - self.h2)
            * (1.0 - z) ** (vacuum_shift - self.h2 - self.h3 + 1.0 / 16.0)
            * theta3
            ** (
                (self.c - 1.5) / 2.0
                - 4.0 * (self.h1 + self.h2 + self.h3 + self.h4)
                + 0.5
            )
        )

    def direct_leading_coefficients(self) -> Dict[str, complex]:
        """Return independent level-1/2 and level-1 Ward/Gram anchors."""

        odd_half = (
            cmath.exp(-1j * math.pi / 4.0)
            * (self.beta1 - self.sign2 * self.beta2)
            / (2.0 * self.internal_weight)
        )
        even_one = (
            (self.internal_weight + self.h3 - self.h4)
            * (self.internal_weight + self.h2 - self.h1)
            / (2.0 * self.internal_weight)
        )
        return {
            "even_level_0": 1.0 + 0.0j,
            "odd_level_half": odd_half,
            "even_level_1": even_one,
        }


class MixedRExchangeSphereFourPointBlock:
    r"""Block for <NS_4 R_3 R_2 NS_1> with internal long-R beta."""

    def __init__(
        self,
        *,
        b: Number,
        p1_ns: Number,
        p2_r: Number,
        p3_r: Number,
        p4_ns: Number,
        internal_momentum: Number,
        sign3: int = 1,
        sign2: int = 1,
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.b = complex(b)
        self.c = central_charge(self.b)
        self.h1 = ns_liouville_weight(p1_ns, self.b)
        self.h2 = ramond_liouville_weight(p2_r, self.b)
        self.h3 = ramond_liouville_weight(p3_r, self.b)
        self.h4 = ns_liouville_weight(p4_ns, self.b)
        self.beta2 = ramond_beta(p2_r)
        self.beta3 = ramond_beta(p3_r)
        self.internal_beta = ramond_beta(internal_momentum)
        self.sign3 = _sign(sign3, "sign3")
        self.sign2 = _sign(sign2, "sign2")
        self.pole_tolerance = float(pole_tolerance)
        if abs(self.internal_beta) <= self.pole_tolerance:
            raise ValueError(
                "MixedRExchangeSphereFourPointBlock represents a generic "
                "long-R module and requires nonzero internal momentum; "
                "construct the beta=0 short quotient separately"
            )

    @classmethod
    def from_fixed_data(
        cls,
        *,
        b: Number,
        h1_ns: Number,
        beta2_r: Number,
        beta3_r: Number,
        h4_ns: Number,
        internal_beta: Number,
        sign3: int = 1,
        sign2: int = 1,
        pole_tolerance: float = 1.0e-12,
    ) -> "MixedRExchangeSphereFourPointBlock":
        """Construct the HJS block in fixed-beta c-recursion variables.

        The internal and external Ramond ``beta`` parameters and the two NS
        weights are held fixed as ``b`` (hence ``c``) moves.  This is the
        analytic family needed to test a Ramond-channel c-recursion.
        """

        result = cls.__new__(cls)
        result.b = complex(b)
        result.c = central_charge(result.b)
        result.h1 = complex(h1_ns)
        result.h4 = complex(h4_ns)
        result.beta2 = complex(beta2_r)
        result.beta3 = complex(beta3_r)
        result.h2 = result.c / 24.0 - result.beta2 * result.beta2
        result.h3 = result.c / 24.0 - result.beta3 * result.beta3
        result.internal_beta = complex(internal_beta)
        result.sign3 = _sign(sign3, "sign3")
        result.sign2 = _sign(sign2, "sign2")
        result.pole_tolerance = float(pole_tolerance)
        if abs(result.internal_beta) <= result.pole_tolerance:
            raise ValueError(
                "MixedRExchangeSphereFourPointBlock represents a generic "
                "long-R module and requires internal_beta != 0"
            )
        return result

    @lru_cache(maxsize=None)
    def _series(
        self,
        max_power: int,
        internal_beta: complex,
        sign3: int,
        sign2: int,
    ) -> tuple[complex, ...]:
        result = [0.0j] * (max_power + 1)
        result[0] = 1.0 + 0.0j

        for r in range(1, 2 * max_power + 1):
            for s in range(1, (2 * max_power) // r + 1):
                product = r * s
                if (r + s) % 2 != 1 or product % 2:
                    continue
                shift = product // 2
                if shift > max_power:
                    continue

                beta_rs = _r_beta_rs(self.b, r, s)
                beta_prime = _r_beta_prime(self.b, r, s)
                a_factor = _r_a_beta(
                    self.b, r, s, self.pole_tolerance
                )
                tail_same = self._series(
                    max_power - shift, beta_prime, sign3, sign2
                )
                tail_flipped = self._series(
                    max_power - shift, beta_prime, -sign3, -sign2
                )

                left_same = _r_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    ramond_beta_value=self.beta3,
                    ns_weight=self.h4,
                    sign=sign3,
                )
                right_same = _r_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    ramond_beta_value=self.beta2,
                    ns_weight=self.h1,
                    sign=sign2,
                )
                left_flipped = _r_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    ramond_beta_value=self.beta3,
                    ns_weight=self.h4,
                    sign=-sign3,
                )
                right_flipped = _r_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    ramond_beta_value=self.beta2,
                    ns_weight=self.h1,
                    sign=-sign2,
                )
                denominator_plus = internal_beta - beta_rs
                denominator_minus = internal_beta + beta_rs
                scale = max(1.0, abs(internal_beta), abs(beta_rs))
                if (
                    abs(denominator_plus) <= self.pole_tolerance * scale
                    or abs(denominator_minus) <= self.pole_tolerance * scale
                ):
                    raise ZeroDivisionError(
                        f"internal beta is too close to the ({r},{s}) Ramond pole"
                    )
                common = 16.0**shift * a_factor
                coefficient_plus = (
                    common * left_same * right_same / denominator_plus
                )
                coefficient_minus = -(
                    common * left_flipped * right_flipped / denominator_minus
                )
                for power, value in enumerate(tail_same):
                    result[power + shift] += coefficient_plus * value
                for power, value in enumerate(tail_flipped):
                    result[power + shift] += coefficient_minus * value
        return tuple(result)

    def elliptic_coefficients(self, order: int) -> Dict[int, complex]:
        if order < 1:
            raise ValueError("order must be positive")
        values = self._series(
            order - 1, self.internal_beta, self.sign3, self.sign2
        )
        return {power: value for power, value in enumerate(values)}

    @property
    def internal_weight(self) -> complex:
        return self.c / 24.0 - self.internal_beta * self.internal_beta

    def level_one_gram_matrix(self) -> tuple[tuple[complex, complex], ...]:
        r"""Return the direct long-R Gram matrix at level one.

        The ordered positive-parity basis is
        ``(L_-1 w^+, G_-1 G_0 w^+)``.  This is an algebraic check that is
        independent of the elliptic recursion.
        """

        h = self.internal_weight
        kappa_squared = h - self.c / 24.0
        off_diagonal = 1.5 * kappa_squared
        return (
            (2.0 * h, off_diagonal),
            (
                off_diagonal,
                kappa_squared * (2.0 * h + self.c / 4.0),
            ),
        )

    def direct_level_one_coefficient(self) -> complex:
        r"""Return the local ``z^1`` coefficient by direct state sewing.

        Ward identities give the two level-one three-form vectors

        ``(h+h_3-h_4, -beta^2/2-sign3*beta*beta3)`` and
        ``(h+h_2-h_1, -beta^2/2-sign2*beta*beta2)``.

        Contracting them with the inverse of :meth:`level_one_gram_matrix`
        provides a low-order brute-force benchmark with no Kac recursion.
        """

        beta = self.internal_beta
        left = (
            self.internal_weight + self.h3 - self.h4,
            -0.5 * beta * beta - self.sign3 * beta * self.beta3,
        )
        right = (
            self.internal_weight + self.h2 - self.h1,
            -0.5 * beta * beta - self.sign2 * beta * self.beta2,
        )
        gram = self.level_one_gram_matrix()
        determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
        if abs(determinant) < self.pole_tolerance:
            raise ZeroDivisionError("the level-one Ramond Gram matrix is singular")
        return (
            left[0] * (gram[1][1] * right[0] - gram[0][1] * right[1])
            + left[1] * (-gram[1][0] * right[0] + gram[0][0] * right[1])
        ) / determinant

    def recursion_level_one_coefficient(self) -> complex:
        """Return the same local coefficient extracted algebraically from H(q)."""

        h = self.internal_weight
        vacuum_shift = (self.c - 1.5) / 24.0
        theta_exponent = (
            (self.c - 1.5) / 2.0
            - 4.0 * (self.h1 + self.h2 + self.h3 + self.h4)
            + 0.5
        )
        prefactor_coefficient = (
            0.5 * (h - vacuum_shift - 1.0 / 16.0)
            - (vacuum_shift - self.h2 - self.h3)
            + theta_exponent / 8.0
        )
        return prefactor_coefficient + self.elliptic_coefficients(2)[1] / 16.0

    def elliptic_block(self, z: Number, order: int) -> complex:
        z = complex(z)
        q = elliptic_nome(z)
        series = sum(
            value * q**power
            for power, value in self.elliptic_coefficients(order).items()
        )
        return self._elliptic_prefactor(z) * series

    def _elliptic_prefactor(self, z: Number) -> complex:
        """Return the mixed long-R-exchange prefactor at fixed ``b``."""

        z = complex(z)
        q = elliptic_nome(z)
        theta3 = complex(mpmath.jtheta(3, 0, mpmath.mpc(q)))
        vacuum_shift = (self.c - 1.5) / 24.0
        return (
            (16.0 * q)
            ** (self.internal_weight - vacuum_shift - 1.0 / 16.0)
            * z ** (vacuum_shift - self.h1 - self.h2 + 1.0 / 16.0)
            * (1.0 - z) ** (vacuum_shift - self.h2 - self.h3)
            * theta3
            ** (
                (self.c - 1.5) / 2.0
                - 4.0 * (self.h1 + self.h2 + self.h3 + self.h4)
                + 0.5
            )
        )


__all__ = [
    "MixedNSExchangeSphereFourPointBlock",
    "MixedRExchangeSphereFourPointBlock",
]
