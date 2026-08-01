"""Sphere four-point blocks with four Ramond external operators.

The implementation follows Hadasz--Jaskolski--Suchanek, arXiv:0810.1203,
but translates their notation to the ordinary-central-charge convention used
by BRY:

    S_n(HJS) = G_n(BRY),  Delta(HJS) = h(BRY),
    c(HJS) = c(BRY),      beta = i P / sqrt(2).

The external operators are Ramond primaries and the exchanged module is
Neveu--Schwarz.  The two signs attached to beta_3 and beta_2 label the two
chiral N-R-R three-point forms; they are not the nonchiral Type-0B
V^{R,+} and V^{R,-} family labels.

Levels and q powers are represented by twice their value.  This keeps the
odd block, whose elliptic series starts at q**(1/2), free of floating-point
dictionary keys.
"""

from __future__ import annotations

import cmath
import math
from functools import lru_cache
from typing import Dict, Literal, Union

import mpmath

from superconformal_blocks import central_charge, elliptic_nome, ns_liouville_weight


Number = Union[complex, float]
Parity = Literal["even", "odd"]


def b_from_c(c: Number) -> complex:
    """Return the principal b branch with c = 3/2 + 3(b + b^-1)^2."""

    c = complex(c)
    q_background = cmath.sqrt((c - 1.5) / 3.0)
    return 0.5 * (q_background + cmath.sqrt(q_background * q_background - 4.0))


def ramond_beta(momentum: Number) -> complex:
    """Translate a BRY Liouville momentum P to HJS beta = i P / sqrt(2)."""

    return 1j * complex(momentum) / math.sqrt(2.0)


def ramond_liouville_weight(momentum: Number, b: Number = 1.0) -> complex:
    """Return h_R(P) = c/24 + P^2/2 in the BRY ordinary-c convention."""

    return central_charge(b) / 24.0 + 0.5 * complex(momentum) ** 2


def ramond_g0_matrix(beta: Number) -> tuple[tuple[complex, complex], ...]:
    r"""Return the BRY/HJS G_0 action in the ordered basis (w^+, w^-).

    The columns are the images of the basis states:

        G_0 w^+ = i exp(-i pi/4) beta w^-,
        G_0 w^- = i exp(+i pi/4) beta w^+.

    Consequently G_0^2 = -beta^2 = h_R-c/24.  Keeping this phaseful
    matrix explicit is useful when the same convention is transferred to
    a Ramond internal edge.
    """

    beta = complex(beta)
    return (
        (0.0j, 1j * cmath.exp(1j * math.pi / 4.0) * beta),
        (1j * cmath.exp(-1j * math.pi / 4.0) * beta, 0.0j),
    )


class RamondExternalSphereFourPointBlock:
    r"""Four-R sphere block with an internal NS representation.

    The puncture order is (R_4, R_3, R_2, R_1) = (infinity, 1, z, 0).
    sign3 and sign2 are independently +1 or -1 and encode the HJS
    superscripts on beta_3 and beta_2:

        F[^{sign3 beta3, sign2 beta2}_{beta4, beta1}].

    The recursion is in the internal NS weight at fixed ordinary central
    charge.  Near b=1, use a small central-charge displacement so individual
    resonant Kac terms are separated before taking the continuous limit.
    """

    def __init__(
        self,
        *,
        b: Number,
        beta1: Number,
        beta2: Number,
        beta3: Number,
        beta4: Number,
        internal_weight: Number,
        sign3: int = 1,
        sign2: int = 1,
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.b = complex(b)
        self.c = central_charge(self.b)
        self.beta1 = complex(beta1)
        self.beta2 = complex(beta2)
        self.beta3 = complex(beta3)
        self.beta4 = complex(beta4)
        self.internal_weight = complex(internal_weight)
        self.sign3 = self._validate_sign(sign3, "sign3")
        self.sign2 = self._validate_sign(sign2, "sign2")
        self.pole_tolerance = float(pole_tolerance)

        self.h1 = self.c / 24.0 - self.beta1 * self.beta1
        self.h2 = self.c / 24.0 - self.beta2 * self.beta2
        self.h3 = self.c / 24.0 - self.beta3 * self.beta3
        self.h4 = self.c / 24.0 - self.beta4 * self.beta4

    @staticmethod
    def _validate_sign(value: int, name: str) -> int:
        value = int(value)
        if value not in (-1, 1):
            raise ValueError(f"{name} must be +1 or -1")
        return value

    @classmethod
    def from_liouville_momenta(
        cls,
        *,
        p1: Number,
        p2: Number,
        p3: Number,
        p4: Number,
        internal_momentum: Number,
        b: Number = 1.0,
        sign3: int = 1,
        sign2: int = 1,
        pole_tolerance: float = 1.0e-12,
    ) -> "RamondExternalSphereFourPointBlock":
        """Construct the block from BRY Liouville momenta."""

        return cls(
            b=b,
            beta1=ramond_beta(p1),
            beta2=ramond_beta(p2),
            beta3=ramond_beta(p3),
            beta4=ramond_beta(p4),
            internal_weight=ns_liouville_weight(internal_momentum, b),
            sign3=sign3,
            sign2=sign2,
            pole_tolerance=pole_tolerance,
        )

    def degenerate_weight(self, r: int, s: int) -> complex:
        """Return the NS Kac weight Delta_rs(c) in the HJS/BRY convention."""

        if r < 1 or s < 1 or (r + s) % 2:
            raise ValueError("NS Kac labels require r,s >= 1 and r+s even")
        b = self.b
        return (
            -(r * s - 1.0) / 4.0
            + (1.0 - r * r) * b * b / 8.0
            + (1.0 - s * s) / (8.0 * b * b)
        )

    def a_factor(self, r: int, s: int) -> complex:
        """Return the inverse NS null-norm slope A_rs(c)."""

        result = 0.5 + 0.0j
        for p in range(1 - r, r + 1):
            for q in range(1 - s, s + 1):
                if (p + q) % 2 or (p, q) in ((0, 0), (r, s)):
                    continue
                denominator = (p * self.b + q / self.b) / math.sqrt(2.0)
                if abs(denominator) < self.pole_tolerance:
                    raise ZeroDivisionError(
                        "resonant A_rs factor; displace b from the rational point "
                        "and take the continuous limit"
                    )
                result /= denominator
        return result

    def fusion_polynomial(
        self,
        r: int,
        s: int,
        *,
        lower_beta: Number,
        upper_beta: Number,
        upper_sign: int,
    ) -> complex:
        r"""Return HJS P_c^{rs}[^{+/- beta_upper}_{beta_lower}]."""

        upper_sign = self._validate_sign(upper_sign, "upper_sign")
        lower_beta = complex(lower_beta)
        upper_beta = complex(upper_beta)
        result = 1.0 + 0.0j

        for p in range(1 - r, r, 2):
            for q in range(1 - s, s, 2):
                offset = (p * self.b + q / self.b) / (2.0 * math.sqrt(2.0))
                selector = (p + q - (r + s)) % 4
                if selector == 2:
                    result *= lower_beta - upper_sign * upper_beta + offset
                elif selector == 0:
                    result *= lower_beta + upper_sign * upper_beta + offset
                else:
                    raise AssertionError("unexpected Ramond fusion-polynomial parity")
        return result

    def _residue_factor(self, r: int, s: int, sign3: int, sign2: int) -> complex:
        left = self.fusion_polynomial(
            r,
            s,
            lower_beta=self.beta4,
            upper_beta=self.beta3,
            upper_sign=sign3,
        )
        right = self.fusion_polynomial(
            r,
            s,
            lower_beta=self.beta1,
            upper_beta=self.beta2,
            upper_sign=sign2,
        )
        # For r,s odd, the two odd three-point forms contribute
        # (-e^{i pi/4})(e^{-i pi/4}) = -1.
        orientation_sign = 1.0 if r % 2 == 0 else -1.0
        return orientation_sign * self.a_factor(r, s) * left * right

    @lru_cache(maxsize=None)
    def _elliptic_series(
        self,
        max_twice_power: int,
        parity: Parity,
        internal_weight: complex,
        sign3: int,
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

                delta_rs = self.degenerate_weight(r, s)
                denominator = internal_weight - delta_rs
                if abs(denominator) < self.pole_tolerance:
                    raise ZeroDivisionError(
                        f"internal weight is too close to the ({r},{s}) NS Kac pole"
                    )

                odd_kac_level = bool(r % 2)
                next_parity: Parity
                if odd_kac_level:
                    next_parity = "odd" if parity == "even" else "even"
                    next_sign3, next_sign2 = -sign3, -sign2
                else:
                    next_parity = parity
                    next_sign3, next_sign2 = sign3, sign2

                tail = self._elliptic_series(
                    max_twice_power - shift,
                    next_parity,
                    complex(delta_rs + shift / 2.0),
                    next_sign3,
                    next_sign2,
                )
                coefficient = (
                    16.0 ** (shift / 2.0)
                    * self._residue_factor(r, s, sign3, sign2)
                    / denominator
                )
                for power, value in enumerate(tail):
                    result[power + shift] += coefficient * value

        return tuple(result)

    def elliptic_coefficients(self, order: int, parity: Parity) -> Dict[int, complex]:
        """Return H(q) coefficients keyed by twice the q power.

        The order matches the existing NS block API: the even series retains
        q^0 through q^(order-1), while the odd series retains q^(1/2) through
        q^(order-1/2).
        """

        if order < 1:
            raise ValueError("order must be positive")
        if parity not in ("even", "odd"):
            raise ValueError("parity must be 'even' or 'odd'")
        max_twice_power = 2 * (order - 1) + (1 if parity == "odd" else 0)
        coefficients = self._elliptic_series(
            max_twice_power,
            parity,
            self.internal_weight,
            self.sign3,
            self.sign2,
        )
        expected_remainder = 0 if parity == "even" else 1
        return {
            power: value
            for power, value in enumerate(coefficients)
            if power % 2 == expected_remainder
        }

    def elliptic_block(self, z: Number, order: int, parity: Parity) -> complex:
        """Evaluate the holomorphic four-R block on principal branches."""

        z = complex(z)
        q = elliptic_nome(z)
        coefficients = self.elliptic_coefficients(order, parity)
        elliptic_series = sum(
            coefficient * q ** (twice_power / 2.0)
            for twice_power, coefficient in coefficients.items()
        )
        return self._elliptic_prefactor(z) * elliptic_series

    def _elliptic_prefactor(self, z: Number) -> complex:
        """Return the universal HJS prefactor, separated for finite-part use."""

        z = complex(z)
        q = elliptic_nome(z)
        theta3 = complex(mpmath.jtheta(3, 0, mpmath.mpc(q)))
        vacuum_shift = (self.c - 1.5) / 24.0
        return (
            (16.0 * q) ** (self.internal_weight - vacuum_shift)
            * z ** (vacuum_shift - self.h1 - self.h2)
            * (1.0 - z) ** (vacuum_shift - self.h2 - self.h3)
            * theta3
            ** ((self.c - 1.5) / 2.0 - 4.0 * (self.h1 + self.h2 + self.h3 + self.h4))
        )

    def direct_leading_coefficients(self) -> Dict[str, complex]:
        """Return the level-0, level-1/2, and level-1 direct sewing anchors."""

        left_odd = self.beta4 - self.sign3 * self.beta3
        right_odd = self.beta1 - self.sign2 * self.beta2
        odd_half = -(left_odd * right_odd) / (2.0 * self.internal_weight)
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

    def diagonal_block_product(self, z: Number, order: int, parity: Parity) -> complex:
        """Return F(z)F(zbar) for one chiral sign branch.

        This is a pointwise nonchiral block product, not yet the full Type-0B
        four-R correlator.  The latter also needs the BRY structure constants
        and the chiral-to-nonchiral Ramond sewing matrix.
        """

        z = complex(z)
        return self.elliptic_block(z, order, parity) * self.elliptic_block(
            z.conjugate(), order, parity
        )


__all__ = [
    "RamondExternalSphereFourPointBlock",
    "b_from_c",
    "ramond_beta",
    "ramond_g0_matrix",
    "ramond_liouville_weight",
]
