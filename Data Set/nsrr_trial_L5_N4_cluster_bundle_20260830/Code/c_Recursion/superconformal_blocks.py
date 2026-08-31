"""NS super-Virasoro blocks for a four-punctured sphere.

This module implements the BRY c-recursion of arXiv:2201.05621, section 3.2,
converted to the fixed-parity three-point convention in
``Human Notes/SCblock.tex``.  Levels are represented internally by twice
their value, so half-integer levels never become floating-point dictionary
keys.

Only the Neveu--Schwarz sphere block is in scope here.  Structure constants,
the modulus integral, analytic continuation of the amplitude, Ramond blocks,
and higher-genus sewing are deliberately separate layers.
"""

from __future__ import annotations

import cmath
import math
from typing import Dict, List, Literal, Sequence, Union

import mpmath

from ns_recursion_recipe import (
    ns_c_pole_mp,
    ns_fusion_polynomial_mp,
    ns_inverse_null_slope_mp,
    ns_ordinary_edge_scalar_kernel_mp,
)


Number = Union[complex, float]
Parity = Literal["even", "odd"]


def central_charge(b: Number) -> complex:
    """Return c = 3/2 (1 + 2 Q^2), with Q = b + 1/b."""

    b = complex(b)
    q_background = b + 1.0 / b
    return 1.5 * (1.0 + 2.0 * q_background * q_background)


def ns_liouville_weight(momentum: Number, b: Number = 1.0) -> complex:
    """Return h(P) = 1/2 (Q^2/4 + P^2) for an NS Liouville primary."""

    b = complex(b)
    momentum = complex(momentum)
    q_background = b + 1.0 / b
    return 0.5 * (0.25 * q_background * q_background + momentum * momentum)


def elliptic_nome(z: Number) -> complex:
    """Return q(z) = exp[-pi K(1-z)/K(z)] on principal branches."""

    z_mp = mpmath.mpc(z)
    return complex(mpmath.exp(-mpmath.pi * mpmath.ellipk(1 - z_mp) / mpmath.ellipk(z_mp)))


def _rising(value: Number, order: int):
    """Return a rising factorial without coercing ``value``'s numeric type."""

    result = 1
    for offset in range(order):
        result *= value + offset
    return result


def _series_mul(left: Sequence[complex], right: Sequence[complex], order: int) -> List[complex]:
    result = [0.0j] * (order + 1)
    for i, left_value in enumerate(left[: order + 1]):
        if left_value == 0:
            continue
        for j, right_value in enumerate(right[: order + 1 - i]):
            result[i + j] += left_value * right_value
    return result


def _series_pow(series: Sequence[complex], exponent: complex, order: int) -> List[complex]:
    """Raise a power series with nonzero constant term to any complex power."""

    if not series or series[0] == 0:
        raise ValueError("a nonzero constant term is required for a series power")
    source = list(series[: order + 1]) + [0.0j] * max(0, order + 1 - len(series))
    result = [0.0j] * (order + 1)
    result[0] = source[0] ** exponent
    for n in range(1, order + 1):
        total = 0.0j
        for k in range(1, n + 1):
            total += ((exponent + 1.0) * k - n) * source[k] * result[n - k]
        result[n] = total / (n * source[0])
    return result


def _series_compose(coefficients: Sequence[complex], argument: Sequence[complex], order: int) -> List[complex]:
    """Compose sum coefficients[n] x^n with a series argument satisfying x(0)=0."""

    result = [0.0j] * (order + 1)
    power = [0.0j] * (order + 1)
    power[0] = 1.0 + 0.0j
    for coefficient in coefficients:
        for index in range(order + 1):
            result[index] += coefficient * power[index]
        power = _series_mul(power, argument, order)
    return result


class NSSphereFourPointBlock:
    """BRY c-recursion for an NS four-point sphere block.

    The argument order is ``(h4, marked h3, marked h2, h1; h | z)``.
    ``star2`` and ``star3`` mark the external level-1/2 descendants.  The
    stored ``h2`` and ``h3`` remain the weights of their underlying primaries;
    their numerical exponent weights are increased by 1/2 only where BRY's
    underlined-weight notation requires it.
    """

    def __init__(
        self,
        *,
        c: Number,
        h1: Number,
        h2: Number,
        h3: Number,
        h4: Number,
        internal_weight: Number,
        star2: bool = False,
        star3: bool = False,
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.c = complex(c)
        self.h1 = complex(h1)
        self.h2 = complex(h2)
        self.h3 = complex(h3)
        self.h4 = complex(h4)
        self.internal_weight = complex(internal_weight)
        self.star2 = bool(star2)
        self.star3 = bool(star3)
        self.pole_tolerance = pole_tolerance
        self._coefficient_cache: Dict[tuple[int, complex, complex], complex] = {}
        self._elliptic_coefficient_cache: Dict[
            tuple[int, Parity], Dict[int, complex]
        ] = {}

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
        star2: bool = False,
        star3: bool = False,
        pole_tolerance: float = 1.0e-12,
    ) -> "NSSphereFourPointBlock":
        """Construct a block directly from NS super-Liouville momenta."""

        return cls(
            c=central_charge(b),
            h1=ns_liouville_weight(p1, b),
            h2=ns_liouville_weight(p2, b),
            h3=ns_liouville_weight(p3, b),
            h4=ns_liouville_weight(p4, b),
            internal_weight=ns_liouville_weight(internal_momentum, b),
            star2=star2,
            star3=star3,
            pole_tolerance=pole_tolerance,
        )

    @property
    def background_charge(self) -> complex:
        """Return the principal solution Q of c = 3/2 (1 + 2 Q^2)."""

        return cmath.sqrt(self.c / 3.0 - 0.5)

    @property
    def exponent_h2(self) -> complex:
        return self.h2 + (0.5 if self.star2 else 0.0)

    @property
    def exponent_h3(self) -> complex:
        return self.h3 + (0.5 if self.star3 else 0.0)

    def seed_coefficient(self, twice_level: int, internal_weight: Number | None = None) -> complex:
        """Return the large-c seed f_m at level ``twice_level / 2``."""

        if twice_level < 0:
            raise ValueError("twice_level must be nonnegative")
        if twice_level == 0:
            return 1.0 + 0.0j
        # Do not normalize through Python's binary64 ``complex`` type here.
        # The multiprecision subclass deliberately passes ``mpmath`` numbers
        # into this seed, and resonant finite-part calculations amplify even a
        # tiny loss of precision before the Kac terms cancel.
        h = self.internal_weight if internal_weight is None else internal_weight

        if twice_level % 2 == 0:
            m = twice_level // 2
            left = h + self.h3 - self.h4 + (0.5 if self.star3 else 0.0)
            right = h + self.h2 - self.h1 + (0.5 if self.star2 else 0.0)
            return _rising(left, m) * _rising(right, m) / (
                math.factorial(m) * _rising(2.0 * h, m)
            )

        integer_part = (twice_level - 1) // 2
        if self.star3:
            left = _rising(h + self.h3 - self.h4, integer_part + 1)
        else:
            left = _rising(h + self.h3 - self.h4 + 0.5, integer_part)
        if self.star2:
            right = _rising(h + self.h2 - self.h1, integer_part + 1)
        else:
            right = _rising(h + self.h2 - self.h1 + 0.5, integer_part)
        # In the human-note convention the internal odd state occurs in the
        # third slot of the left trinion, so every odd seed carries a minus.
        sign = -1.0
        return sign * left * right / (
            math.factorial(integer_part) * _rising(2.0 * h, integer_part + 1)
        )

    @staticmethod
    def _pole_data(r: int, s: int, h: complex) -> tuple[complex, complex, complex]:
        discriminant = cmath.sqrt(
            16.0 * h * h + 8.0 * (r * s - 1.0) * h + (r - s) ** 2
        )
        b_squared = -(4.0 * h + r * s - 1.0 + discriminant) / (r * r - 1.0)
        b_pole = cmath.sqrt(b_squared)
        c_pole = 7.5 + 3.0 * b_squared + 3.0 / b_squared

        derivative_b_squared = -(
            4.0 + (16.0 * h + 4.0 * (r * s - 1.0)) / discriminant
        ) / (r * r - 1.0)
        derivative_c = 3.0 * (1.0 - 1.0 / (b_squared * b_squared)) * derivative_b_squared
        return b_pole, c_pole, derivative_c

    @staticmethod
    def _a_factor(r: int, s: int, b_pole: complex) -> complex:
        result = 0.5 + 0.0j
        for p in range(1 - r, r + 1):
            for q in range(1 - s, s + 1):
                if (p, q) in ((0, 0), (r, s)) or (p + q) % 2:
                    continue
                result *= math.sqrt(2.0) / (p * b_pole + q / b_pole)
        return result

    @staticmethod
    def _fusion_polynomial(
        r: int,
        s: int,
        b_pole: complex,
        first_weight: complex,
        second_weight: complex,
        second_is_starred: bool,
    ) -> complex:
        q_pole = b_pole + 1.0 / b_pole
        a_first = cmath.sqrt(q_pole * q_pole / 4.0 - 2.0 * first_weight)
        a_second = cmath.sqrt(q_pole * q_pole / 4.0 - 2.0 * second_weight)
        congruence = 0 if second_is_starred else 2
        result = 1.0 + 0.0j
        denominator = 2.0 * math.sqrt(2.0)
        for p in range(1 - r, r, 2):
            for q in range(1 - s, s, 2):
                if ((p + q) - (r + s)) % 4 != congruence:
                    continue
                linear = p * b_pole + q / b_pole
                result *= (2.0 * a_first - 2.0 * a_second - linear) / denominator
                result *= (2.0 * a_first + 2.0 * a_second + linear) / denominator
        return result

    def _residue(self, twice_level: int, r: int, s: int, h: complex) -> tuple[complex, complex]:
        b_pole, c_pole, derivative_c = self._pole_data(r, s, h)
        a_factor = self._a_factor(r, s, b_pole)

        use_star2 = self.star2
        use_star3 = self.star3
        if twice_level % 2:
            use_star2 = not use_star2
            use_star3 = not use_star3

        p12 = self._fusion_polynomial(
            r, s, b_pole, self.h1, self.h2, use_star2
        )
        p43 = self._fusion_polynomial(
            r, s, b_pole, self.h4, self.h3, use_star3
        )
        # Converting the BRY/component residue to the human-note three-form
        # leaves the level-transport phase (-1)^(rs), independently of which
        # external field is starred.
        sigma = (-1.0) ** (r * s)
        return sigma * (-derivative_c) * a_factor * p12 * p43, c_pole

    def _coefficient(self, twice_level: int, h: complex, c: complex) -> complex:
        key = (twice_level, h, c)
        if key in self._coefficient_cache:
            return self._coefficient_cache[key]
        if twice_level == 0:
            return 1.0 + 0.0j

        result = self.seed_coefficient(twice_level, h)
        # BRY's ancillary notebook includes rs = 2m.  This endpoint is needed
        # for the first (3,1) and (2,2) poles, despite a strict inequality in
        # the displayed equation of the paper.
        for r in range(2, twice_level + 1):
            for s in range(1, twice_level // r + 1):
                rs = r * s
                if (r + s) % 2 or rs > twice_level:
                    continue
                residue, c_pole = self._residue(twice_level, r, s, h)
                denominator = c - c_pole
                scale = max(1.0, abs(c), abs(c_pole))
                if abs(denominator) <= self.pole_tolerance * scale:
                    raise ZeroDivisionError(
                        f"c-recursion encountered the ({r},{s}) pole: "
                        f"c={c!r}, c_rs={c_pole!r}"
                    )
                result += residue / denominator * self._coefficient(
                    twice_level - rs, h + 0.5 * rs, c_pole
                )

        self._coefficient_cache[key] = result
        return result

    def coefficient(self, twice_level: int) -> complex:
        """Return F_m with ``twice_level = 2m``.

        Even values select the even block coefficients and odd values select
        the odd block coefficients.  ``coefficient(0)`` is the common F_0=1.
        """

        if not isinstance(twice_level, int) or twice_level < 0:
            raise ValueError("twice_level must be a nonnegative integer")
        return self._coefficient(twice_level, self.internal_weight, self.c)

    def z_block(self, z: Number, order: int, parity: Parity = "even") -> complex:
        """Evaluate the truncated local z-series.

        ``order`` is the number of retained coefficients: even blocks keep
        levels 0 through order-1, while odd blocks keep levels 1/2 through
        order-1/2.  This representation is intended for |z| < 1.
        """

        if order < 1:
            raise ValueError("order must be positive")
        z = complex(z)
        leading_power = self.internal_weight - self.h1 - self.exponent_h2
        if parity == "even":
            series = sum(self.coefficient(2 * level) * z**level for level in range(order))
        elif parity == "odd":
            # The signed large-c seeds fix the chiral phase exactly as in the
            # BRY ancillary notebook; no second overall sign is inserted.
            series = sum(
                self.coefficient(2 * level + 1) * z ** (level + 0.5)
                for level in range(order)
            )
        else:
            raise ValueError("parity must be 'even' or 'odd'")
        return z**leading_power * series

    def _global_z_block(self, z, h, parity: Parity):
        """Exact reduced global osp(1|2) block in one parity sector."""

        left = h + self.h3 - self.h4
        right = h + self.h2 - self.h1
        if parity == "even":
            return mpmath.hyp2f1(
                left + (mpmath.mpf("0.5") if self.star3 else 0),
                right + (mpmath.mpf("0.5") if self.star2 else 0),
                2 * h,
                z,
            )
        if parity != "odd":
            raise ValueError("parity must be 'even' or 'odd'")

        left_prefactor = left if self.star3 else 1
        right_prefactor = right if self.star2 else 1
        left_parameter = left + (1 if self.star3 else mpmath.mpf("0.5"))
        right_parameter = right + (1 if self.star2 else mpmath.mpf("0.5"))
        sign = -1
        return (
            sign
            * mpmath.sqrt(z)
            * left_prefactor
            * right_prefactor
            / (2 * h)
            * mpmath.hyp2f1(
                left_parameter,
                right_parameter,
                2 * h + 1,
                z,
            )
        )

    def recursive_z_block(
        self,
        z: Number,
        recursion_order: int,
        parity: Parity = "even",
    ) -> complex:
        """Evaluate the local block by the direct Zamolodchikov c-recursion.

        ``recursion_order=N`` retains every nested Kac-residue path whose
        accumulated physical null level is at most ``N``.  At every leaf the
        global osp(1|2) block is evaluated as an exact hypergeometric
        function.  Consequently this method does not truncate a local-z or
        elliptic-q series; increasing ``N`` only deepens the c-recursion.

        The returned block includes the conventional leading factor
        ``z**(h-h1-h2)`` (with the marked-field shift when applicable).
        It is intended for the channel domain ``|z| < 1``.
        """

        return self.recursive_z_blocks((z,), recursion_order, parity)[0]

    def recursive_z_blocks(
        self,
        z_values: Sequence[Number],
        recursion_order: int,
        parity: Parity = "even",
    ) -> tuple[complex, ...]:
        """Vectorized direct c-recursion on a cross-ratio grid.

        The recursion tree and all Kac residues are built once for the entire
        grid.  Only the exact hypergeometric leaves depend on ``z``.
        """

        if not isinstance(recursion_order, int) or recursion_order < 0:
            raise ValueError("recursion_order must be a nonnegative integer")
        if parity not in ("even", "odd"):
            raise ValueError("parity must be 'even' or 'odd'")
        points = tuple(mpmath.mpc(z) for z in z_values)
        if not points:
            raise ValueError("z_values must not be empty")
        if any(z == 0 for z in points):
            raise ValueError("the full block is singular or vanishing at z=0")

        cache = {}

        def recurse(twice_budget: int, sector: Parity, h, c):
            key = (twice_budget, sector, h, c)
            if key in cache:
                return cache[key]

            result = [self._global_z_block(z, h, sector) for z in points]
            sector_index = 0 if sector == "even" else 1
            for r in range(2, twice_budget + 1):
                for s in range(1, twice_budget // r + 1):
                    shift = r * s
                    if (r + s) % 2 or shift > twice_budget:
                        continue
                    residue, c_pole = self._residue(
                        sector_index, r, s, h
                    )
                    denominator = c - c_pole
                    scale = max(1, abs(c), abs(c_pole))
                    if abs(denominator) <= self.pole_tolerance * scale:
                        raise ZeroDivisionError(
                            f"c-recursion encountered the ({r},{s}) pole: "
                            f"c={c!r}, c_rs={c_pole!r}"
                        )
                    tail_sector: Parity = sector
                    if shift % 2:
                        tail_sector = "odd" if sector == "even" else "even"
                    tails = recurse(
                        twice_budget - shift,
                        tail_sector,
                        h + mpmath.mpf(shift) / 2,
                        c_pole,
                    )
                    coefficient = residue / denominator
                    shift_level = mpmath.mpf(shift) / 2
                    for index, z in enumerate(points):
                        result[index] += (
                            z**shift_level * coefficient * tails[index]
                        )
            cache[key] = tuple(result)
            return cache[key]

        reduced = recurse(
            2 * recursion_order,
            parity,
            self.internal_weight,
            self.c,
        )
        leading_power = self.internal_weight - self.h1 - self.exponent_h2
        return tuple(
            complex(z**leading_power * value)
            for z, value in zip(points, reduced)
        )

    @staticmethod
    def _elliptic_series_data(order: int) -> tuple[List[complex], List[complex], List[complex]]:
        theta3 = [0.0j] * (order + 1)
        theta3[0] = 1.0 + 0.0j
        n = 1
        while n * n <= order:
            theta3[n * n] += 2.0
            n += 1

        theta2_reduced = [0.0j] * (order + 1)
        n = 0
        while n * (n + 1) <= order:
            theta2_reduced[n * (n + 1)] += 1.0
            n += 1
        ratio = _series_mul(
            _series_pow(theta2_reduced, 4.0, order),
            _series_pow(theta3, -4.0, order),
            order,
        )
        z_series = [0.0j] * (order + 1)
        for power in range(1, order + 1):
            z_series[power] = 16.0 * ratio[power - 1]
        return theta3, ratio, z_series

    def elliptic_coefficients(self, order: int, parity: Parity = "even") -> Dict[int, complex]:
        """Return the truncated elliptic block H(q).

        Dictionary keys are twice the q-power.  Thus an even key 2n denotes
        q^n and an odd key 2n+1 denotes q^(n+1/2).  ``order`` coefficients
        are returned: q^0,...,q^(order-1) for the even block and
        q^(1/2),...,q^(order-1/2) for the odd block.
        """

        if order < 1:
            raise ValueError("order must be positive")
        cache_key = (order, parity)
        if cache_key in self._elliptic_coefficient_cache:
            return self._elliptic_coefficient_cache[cache_key]
        max_power = order - 1
        theta3, ratio, z_series = self._elliptic_series_data(max_power)
        q_background = self.background_charge
        q_squared = q_background * q_background
        alpha = self.internal_weight - q_squared / 8.0
        beta = q_squared / 8.0 - self.exponent_h2 - self.exponent_h3
        gamma = 1.5 * q_squared - 4.0 * (
            self.h1 + self.exponent_h2 + self.exponent_h3 + self.h4
        )

        one_minus_z = [-value for value in z_series]
        one_minus_z[0] += 1.0
        common = _series_mul(
            _series_mul(
                _series_pow(ratio, alpha, max_power),
                _series_pow(one_minus_z, -beta, max_power),
                max_power,
            ),
            _series_pow(theta3, -gamma, max_power),
            max_power,
        )

        if parity == "even":
            z_coefficients = [self.coefficient(2 * level) for level in range(order)]
            composed = _series_compose(z_coefficients, z_series, max_power)
            result = _series_mul(common, composed, max_power)
            coefficients = {2 * power: result[power] for power in range(order)}
            self._elliptic_coefficient_cache[cache_key] = coefficients
            return coefficients
        if parity == "odd":
            z_coefficients = [self.coefficient(2 * level + 1) for level in range(order)]
            composed = _series_compose(z_coefficients, z_series, max_power)
            reduced = _series_mul(
                _series_mul(common, _series_pow(ratio, 0.5, max_power), max_power),
                composed,
                max_power,
            )
            coefficients = {
                2 * power + 1: 4.0 * reduced[power]
                for power in range(order)
            }
            self._elliptic_coefficient_cache[cache_key] = coefficients
            return coefficients
        raise ValueError("parity must be 'even' or 'odd'")

    def elliptic_block(self, z: Number, order: int, parity: Parity = "even") -> complex:
        """Evaluate the full block using a truncated elliptic-q expansion."""

        z = complex(z)
        q = elliptic_nome(z)
        h_coefficients = self.elliptic_coefficients(order, parity)
        elliptic_part = sum(
            coefficient * q ** (twice_power / 2.0)
            for twice_power, coefficient in h_coefficients.items()
        )

        q_background = self.background_charge
        q_squared = q_background * q_background
        theta3 = complex(mpmath.jtheta(3, 0, mpmath.mpc(q)))
        prefactor = (
            (16.0 * q) ** (self.internal_weight - q_squared / 8.0)
            * z ** (q_squared / 8.0 - self.h1 - self.exponent_h2)
            * (1.0 - z) ** (q_squared / 8.0 - self.exponent_h2 - self.exponent_h3)
            * theta3
            ** (
                1.5 * q_squared
                - 4.0 * (self.h1 + self.exponent_h2 + self.exponent_h3 + self.h4)
            )
        )
        return prefactor * elliptic_part

    def bry_elliptic_block(
        self, z: Number, max_q_power: int, parity: Parity = "even"
    ) -> complex:
        """Evaluate a block through BRY's displayed order ``q^L``.

        For ``max_q_power=L``, the even block retains ``q^0,...,q^L``
        while the odd block retains ``q^(1/2),...,q^(L-1/2)``.  This differs
        from :meth:`elliptic_block`, whose ``order`` argument counts retained
        coefficients independently in each parity.
        """

        if not isinstance(max_q_power, int) or max_q_power < 1:
            raise ValueError("max_q_power must be a positive integer")
        if parity == "even":
            coefficient_count = max_q_power + 1
        elif parity == "odd":
            coefficient_count = max_q_power
        else:
            raise ValueError("parity must be 'even' or 'odd'")
        return self.elliptic_block(z, coefficient_count, parity)


class HighPrecisionNSSphereFourPointBlock(NSSphereFourPointBlock):
    """Arbitrary-precision version of :class:`NSSphereFourPointBlock`.

    BRY's continuum-momentum integral probes the neighborhood of P=0, where
    high-level c-recursion terms can be individually enormous before
    canceling.  This backend keeps those cancellations at mpmath precision
    while preserving the public interface of the double-precision class.
    """

    def __init__(
        self,
        *,
        c: Number,
        h1: Number,
        h2: Number,
        h3: Number,
        h4: Number,
        internal_weight: Number,
        star2: bool = False,
        star3: bool = False,
        working_precision: int = 60,
        pole_tolerance: float = 1.0e-30,
    ) -> None:
        if working_precision < 30:
            raise ValueError("working_precision must be at least 30 digits")
        self.working_precision = int(working_precision)
        with mpmath.workdps(self.working_precision):
            self.c = mpmath.mpc(c)
            self.h1 = mpmath.mpc(h1)
            self.h2 = mpmath.mpc(h2)
            self.h3 = mpmath.mpc(h3)
            self.h4 = mpmath.mpc(h4)
            self.internal_weight = mpmath.mpc(internal_weight)
        self.star2 = bool(star2)
        self.star3 = bool(star3)
        self.pole_tolerance = pole_tolerance
        self._coefficient_cache = {}
        self._elliptic_coefficient_cache = {}

    @property
    def background_charge(self):
        with mpmath.workdps(self.working_precision):
            return mpmath.sqrt(self.c / 3 - mpmath.mpf("0.5"))

    @staticmethod
    def _pole_data(r: int, s: int, h):
        pole = ns_c_pole_mp(r, s, h)
        # The base class historically stores dc/dh and inserts its minus sign
        # in ``_residue``.  The shared recipe exposes J=-dc/dh.
        return pole.b, pole.c, -pole.jacobian

    @staticmethod
    def _fusion_polynomial(
        r: int,
        s: int,
        b_pole,
        first_weight,
        second_weight,
        second_is_starred: bool,
    ):
        return ns_fusion_polynomial_mp(
            r=r,
            s=s,
            a=1 if second_is_starred else 0,
            first_weight=first_weight,
            second_weight=second_weight,
            b=b_pole,
        )

    @staticmethod
    def _a_factor(r: int, s: int, b_pole):
        """Multiprecision residue normalization at the active workdps."""

        return ns_inverse_null_slope_mp(r, s, b_pole)

    def _residue(self, twice_level: int, r: int, s: int, h):
        """Standard-frame sphere specialization of the graph NS recipe.

        ``twice_level mod 2`` is the current block sector.  The external top
        components shift the two endpoint three-form labels.  The final phase
        converts the component-ordered BRY residue to the human-note
        fixed-parity three-form.  This method uses the principal pole sheet
        and is not an analytic-continuation driver.
        """

        sector = int(twice_level) % 2
        left_sector = int(self.star2) ^ sector
        right_sector = int(self.star3) ^ sector
        pole, residue, child_sectors = ns_ordinary_edge_scalar_kernel_mp(
            r=r,
            s=s,
            internal_weight=h,
            left_weights=(self.h1, self.h2),
            right_weights=(self.h4, self.h3),
            left_sector=left_sector,
            right_sector=right_sector,
        )
        parity = (r * s) % 2
        expected_children = (
            left_sector ^ parity,
            right_sector ^ parity,
        )
        if child_sectors != expected_children:  # pragma: no cover
            raise AssertionError("ordinary-edge incidence transport changed")
        human_phase = (-1) ** (r * s)
        return human_phase * residue, pole.c

    def coefficient(self, twice_level: int):
        with mpmath.workdps(self.working_precision):
            return super().coefficient(twice_level)

    def z_block(self, z: Number, order: int, parity: Parity = "even") -> complex:
        with mpmath.workdps(self.working_precision):
            return complex(super().z_block(z, order, parity))

    def recursive_z_block(
        self,
        z: Number,
        recursion_order: int,
        parity: Parity = "even",
    ) -> complex:
        with mpmath.workdps(self.working_precision):
            return complex(
                super().recursive_z_block(z, recursion_order, parity)
            )

    def recursive_z_blocks(
        self,
        z_values: Sequence[Number],
        recursion_order: int,
        parity: Parity = "even",
    ) -> tuple[complex, ...]:
        with mpmath.workdps(self.working_precision):
            return tuple(
                complex(value)
                for value in super().recursive_z_blocks(
                    z_values, recursion_order, parity
                )
            )

    def elliptic_coefficients(self, order: int, parity: Parity = "even"):
        with mpmath.workdps(self.working_precision):
            return super().elliptic_coefficients(order, parity)

    def elliptic_block(self, z: Number, order: int, parity: Parity = "even") -> complex:
        with mpmath.workdps(self.working_precision):
            return complex(super().elliptic_block(z, order, parity))

    def bry_elliptic_block(
        self, z: Number, max_q_power: int, parity: Parity = "even"
    ) -> complex:
        with mpmath.workdps(self.working_precision):
            return complex(super().bry_elliptic_block(z, max_q_power, parity))


__all__ = [
    "HighPrecisionNSSphereFourPointBlock",
    "NSSphereFourPointBlock",
    "central_charge",
    "elliptic_nome",
    "ns_liouville_weight",
]
