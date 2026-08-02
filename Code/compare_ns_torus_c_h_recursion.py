"""Numerical c-versus-h recursion check for the NS torus one-point block.

The two calculations are independent:

* the h-recursion is the fixed-b toric recursion of
  Hadasz--Jaskolski--Suchanek;
* the c-recursion uses the large-c vacuum-times-osp(1|2) seed and the
  fixed-weight Kac poles of the genus-g recursion manuscript.

At the self-dual point b=1 (ordinary c=27/2, or hat c=9), individual Kac
terms are resonant.  The finite value is the constant Laurent coefficient
on a small circle in t=log(b).  Both NS temporal spin lifts xi=+1 and -1
are evaluated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Dict, Sequence

import mpmath


def _central_charge(b):
    return mpmath.mpf("1.5") + 3 * (b + 1 / b) ** 2


def _ns_weight(momentum, b):
    background = b + 1 / b
    return background**2 / 8 + momentum**2 / 2


def _rising(value, order: int):
    result = mpmath.mpc(1)
    for offset in range(order):
        result *= value + offset
    return result


def _falling(value, order: int):
    result = mpmath.mpc(1)
    for offset in range(order):
        result *= value - offset
    return result


def _ns_degenerate_weight(b, r: int, s: int):
    return (
        -mpmath.mpf(r * s - 1) / 4
        + mpmath.mpf(1 - r * r) * b * b / 8
        + mpmath.mpf(1 - s * s) / (8 * b * b)
    )


def _ns_a_factor(b, r: int, s: int):
    result = mpmath.mpc("0.5")
    sqrt_two = mpmath.sqrt(2)
    for p in range(1 - r, r + 1):
        for q in range(1 - s, s + 1):
            if (p + q) % 2 or (p, q) in ((0, 0), (r, s)):
                continue
            result /= (p * b + q / b) / sqrt_two
    return result


def _ns_ns_fusion_polynomial(
    *,
    b,
    r: int,
    s: int,
    lower_weight,
    upper_weight,
    starred: bool,
):
    background = b + 1 / b
    lower_lambda = mpmath.sqrt(background**2 - 8 * lower_weight)
    upper_lambda = mpmath.sqrt(background**2 - 8 * upper_weight)
    wanted_parity = 1 if starred else 0
    result = mpmath.mpc(1)
    denominator = 2 * mpmath.sqrt(2)
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


def _c_pole(weight, r: int, s: int):
    discriminant = mpmath.sqrt(
        16 * weight**2 + 8 * (r * s - 1) * weight + (r - s) ** 2
    )
    b_squared = -(
        4 * weight + r * s - 1 + discriminant
    ) / (r * r - 1)
    b_pole = mpmath.sqrt(b_squared)
    c_pole = mpmath.mpf("7.5") + 3 * b_squared + 3 / b_squared
    derivative_b_squared = -(
        4 + (16 * weight + 4 * (r * s - 1)) / discriminant
    ) / (r * r - 1)
    derivative_c = (
        3 * (1 - 1 / b_squared**2) * derivative_b_squared
    )
    return b_pole, c_pole, -derivative_c


@lru_cache(maxsize=None)
def _vacuum_character_coefficients(max_twice_level: int) -> tuple[int, ...]:
    """Large-c non-global NS vacuum character, with modes starting at 3/2 and 2."""

    coefficients = [0] * (max_twice_level + 1)
    coefficients[0] = 1
    for mode in range(2, max_twice_level // 2 + 2):
        bosonic_step = 2 * mode
        if bosonic_step <= max_twice_level:
            for level in range(bosonic_step, max_twice_level + 1):
                coefficients[level] += coefficients[level - bosonic_step]
        fermionic_step = 2 * mode - 1
        if fermionic_step <= max_twice_level:
            for level in range(max_twice_level, fermionic_step - 1, -1):
                coefficients[level] += coefficients[level - fermionic_step]
    return tuple(coefficients)


@lru_cache(maxsize=None)
def _ns_verma_character_coefficients(max_twice_level: int) -> tuple[int, ...]:
    """Generic NS Verma character in twice-level notation."""

    coefficients = [0] * (max_twice_level + 1)
    coefficients[0] = 1
    for mode in range(1, max_twice_level // 2 + 2):
        bosonic_step = 2 * mode
        if bosonic_step <= max_twice_level:
            for level in range(bosonic_step, max_twice_level + 1):
                coefficients[level] += coefficients[level - bosonic_step]
        fermionic_step = 2 * mode - 1
        if fermionic_step <= max_twice_level:
            for level in range(max_twice_level, fermionic_step - 1, -1):
                coefficients[level] += coefficients[level - fermionic_step]
    return tuple(coefficients)


@lru_cache(maxsize=None)
def _global_torus_coefficient(twice_level: int, weight, external_weight):
    """Coefficient of the global osp(1|2) torus one-point block."""

    epsilon = twice_level % 2
    occupation = (twice_level - epsilon) // 2
    h = mpmath.mpc(weight)
    d = mpmath.mpc(external_weight)
    result = mpmath.mpc(0)
    for contraction in range(occupation + 1):
        common = (
            math.comb(occupation, contraction)
            * _falling(occupation, contraction)
        )
        if epsilon:
            common *= _falling(
                2 * h + occupation, contraction
            )
            common *= 2 * h - d
        else:
            common *= _falling(
                2 * h + occupation - 1, contraction
            )
        result += (
            common
            * _rising(d, occupation - contraction)
            * _rising(
                d + contraction - occupation,
                occupation - contraction,
            )
        )
    norm = (
        math.factorial(occupation)
        * _rising(2 * h, occupation + epsilon)
    )
    return result / norm


@lru_cache(maxsize=None)
def _regular_torus_coefficient(twice_level: int, weight, external_weight):
    vacuum = _vacuum_character_coefficients(twice_level)
    return sum(
        vacuum[vacuum_level]
        * _global_torus_coefficient(
            twice_level - vacuum_level,
            weight,
            external_weight,
        )
        for vacuum_level in range(twice_level + 1)
    )


class TorusHRecursion:
    """Fixed-b NS torus one-point h-recursion."""

    def __init__(self, *, b, internal_weight, external_weight):
        self.b = mpmath.mpc(b)
        self.internal_weight = mpmath.mpc(internal_weight)
        self.external_weight = mpmath.mpc(external_weight)

    @lru_cache(maxsize=None)
    def _elliptic_coefficient(self, twice_level: int, internal_weight):
        result = mpmath.mpc(1 if twice_level == 0 else 0)
        for r in range(1, twice_level + 1):
            for s in range(1, twice_level // r + 1):
                product = r * s
                if product > twice_level or (r + s) % 2:
                    continue
                degenerate = _ns_degenerate_weight(self.b, r, s)
                shifted = degenerate + mpmath.mpf(product) / 2
                left = _ns_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    lower_weight=shifted,
                    upper_weight=self.external_weight,
                    starred=bool(product % 2),
                )
                right = _ns_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    lower_weight=degenerate,
                    upper_weight=self.external_weight,
                    starred=False,
                )
                sewing_sign = -1 if product % 2 else 1
                result += (
                    sewing_sign
                    * _ns_a_factor(self.b, r, s)
                    * left
                    * right
                    / (internal_weight - degenerate)
                    * self._elliptic_coefficient(
                        twice_level - product,
                        shifted,
                    )
                )
        return result

    def elliptic_coefficients(self, max_twice_level: int) -> Dict[int, complex]:
        return {
            level: self._elliptic_coefficient(
                level, self.internal_weight
            )
            for level in range(max_twice_level + 1)
        }

    def raw_coefficients(self, max_twice_level: int) -> Dict[int, complex]:
        elliptic = self.elliptic_coefficients(max_twice_level)
        character = _ns_verma_character_coefficients(max_twice_level)
        return {
            level: sum(
                character[offset] * elliptic[level - offset]
                for offset in range(level + 1)
            )
            for level in range(max_twice_level + 1)
        }


class TorusCRecursion:
    """Fixed-weight central-charge recursion for the NS torus one-point block."""

    def __init__(self, *, c, internal_weight, external_weight):
        self.c = mpmath.mpc(c)
        self.internal_weight = mpmath.mpc(internal_weight)
        self.external_weight = mpmath.mpc(external_weight)

    @lru_cache(maxsize=None)
    def _coefficient(self, twice_level: int, internal_weight, c):
        result = _regular_torus_coefficient(
            twice_level,
            internal_weight,
            self.external_weight,
        )
        for r in range(2, twice_level + 1):
            for s in range(1, twice_level // r + 1):
                product = r * s
                if product > twice_level or (r + s) % 2:
                    continue
                b_pole, c_pole, jacobian = _c_pole(
                    internal_weight, r, s
                )
                shifted = internal_weight + mpmath.mpf(product) / 2
                left = _ns_ns_fusion_polynomial(
                    b=b_pole,
                    r=r,
                    s=s,
                    lower_weight=shifted,
                    upper_weight=self.external_weight,
                    starred=bool(product % 2),
                )
                right = _ns_ns_fusion_polynomial(
                    b=b_pole,
                    r=r,
                    s=s,
                    lower_weight=internal_weight,
                    upper_weight=self.external_weight,
                    starred=False,
                )
                sewing_sign = -1 if product % 2 else 1
                residue = (
                    sewing_sign
                    * jacobian
                    * _ns_a_factor(b_pole, r, s)
                    * left
                    * right
                )
                result += (
                    residue
                    / (c - c_pole)
                    * self._coefficient(
                        twice_level - product,
                        shifted,
                        c_pole,
                    )
                )
        return result

    def raw_coefficients(self, max_twice_level: int) -> Dict[int, complex]:
        return {
            level: self._coefficient(
                level,
                self.internal_weight,
                self.c,
            )
            for level in range(max_twice_level + 1)
        }


def _finite_part_coefficients(
    *,
    radius: float,
    samples: int,
    max_twice_level: int,
    internal_momentum: float,
    external_momentum: float,
    working_precision: int,
) -> tuple[Dict[int, complex], Dict[int, complex]]:
    with mpmath.workdps(working_precision):
        c_totals = {
            level: mpmath.mpc(0)
            for level in range(max_twice_level + 1)
        }
        h_totals = dict(c_totals)
        p_internal = mpmath.mpf(str(internal_momentum))
        p_external = mpmath.mpf(str(external_momentum))
        for index in range(samples):
            angle = (
                2 * mpmath.pi
                * (mpmath.mpf(index) + mpmath.mpf("0.5"))
                / samples
            )
            b = mpmath.exp(
                mpmath.mpf(str(radius))
                * mpmath.exp(mpmath.j * angle)
            )
            c = _central_charge(b)
            h = _ns_weight(p_internal, b)
            d = _ns_weight(p_external, b)
            c_values = TorusCRecursion(
                c=c,
                internal_weight=h,
                external_weight=d,
            ).raw_coefficients(max_twice_level)
            h_values = TorusHRecursion(
                b=b,
                internal_weight=h,
                external_weight=d,
            ).raw_coefficients(max_twice_level)
            for level in range(max_twice_level + 1):
                c_totals[level] += c_values[level]
                h_totals[level] += h_values[level]
        return (
            {
                level: value / samples
                for level, value in c_totals.items()
            },
            {
                level: value / samples
                for level, value in h_totals.items()
            },
        )


def _relative(left, right):
    return abs(left - right) / max(
        abs(left), abs(right), mpmath.mpf("1e-300")
    )


def _evaluate(
    coefficients: Dict[int, complex],
    *,
    q: float,
    lift_sign: int,
    internal_weight,
    c,
):
    q_mp = mpmath.mpf(str(q))
    series = sum(
        value
        * lift_sign**level
        * q_mp ** (mpmath.mpf(level) / 2)
        for level, value in coefficients.items()
    )
    return q_mp ** (internal_weight - c / 24) * series


@dataclass(frozen=True)
class ComparisonRow:
    q: float
    lift_sign: int
    c_block: float
    h_block: float
    relative: float


def _generic_sanity(
    *,
    b_value: float,
    internal_momentum: float,
    external_momentum: float,
    max_twice_level: int,
    working_precision: int,
) -> dict:
    with mpmath.workdps(working_precision):
        b = mpmath.mpf(str(b_value))
        h = _ns_weight(mpmath.mpf(str(internal_momentum)), b)
        d = _ns_weight(mpmath.mpf(str(external_momentum)), b)
        c = _central_charge(b)
        c_values = TorusCRecursion(
            c=c,
            internal_weight=h,
            external_weight=d,
        ).raw_coefficients(max_twice_level)
        h_values = TorusHRecursion(
            b=b,
            internal_weight=h,
            external_weight=d,
        ).raw_coefficients(max_twice_level)
        relative = {
            level: _relative(c_values[level], h_values[level])
            for level in range(max_twice_level + 1)
        }
        direct_expected = {
            0: mpmath.mpc(1),
            1: (2 * h - d) / (2 * h),
            2: (2 * h + d * (d - 1)) / (2 * h),
        }
        direct_error = max(
            _relative(h_values[level], expected)
            for level, expected in direct_expected.items()
        )
        return {
            "b": b_value,
            "max_relative_c_h": float(max(relative.values())),
            "max_relative_direct_through_level_one": float(direct_error),
            "coefficient_relative_c_h": {
                str(level): float(value)
                for level, value in relative.items()
            },
        }


def build_comparison(
    *,
    internal_momentum: float = 0.7,
    external_momentum: float = 0.5,
    q_values: Sequence[float] = tuple(
        0.02 * (index + 1) for index in range(10)
    ),
    max_q_power: int = 10,
    radius: float = 0.035,
    check_radius: float = 0.045,
    samples: int = 32,
    working_precision: int = 60,
) -> dict:
    max_twice_level = 2 * max_q_power
    c_coefficients, h_coefficients = _finite_part_coefficients(
        radius=radius,
        samples=samples,
        max_twice_level=max_twice_level,
        internal_momentum=internal_momentum,
        external_momentum=external_momentum,
        working_precision=working_precision,
    )
    c_check, h_check = _finite_part_coefficients(
        radius=check_radius,
        samples=samples,
        max_twice_level=max_twice_level,
        internal_momentum=internal_momentum,
        external_momentum=external_momentum,
        working_precision=working_precision,
    )

    with mpmath.workdps(working_precision):
        b = mpmath.mpf(1)
        c = _central_charge(b)
        h = _ns_weight(mpmath.mpf(str(internal_momentum)), b)
        d = _ns_weight(mpmath.mpf(str(external_momentum)), b)
        rows = []
        for lift_sign in (1, -1):
            for q in q_values:
                c_value = _evaluate(
                    c_coefficients,
                    q=q,
                    lift_sign=lift_sign,
                    internal_weight=h,
                    c=c,
                )
                h_value = _evaluate(
                    h_coefficients,
                    q=q,
                    lift_sign=lift_sign,
                    internal_weight=h,
                    c=c,
                )
                rows.append(
                    ComparisonRow(
                        q=float(q),
                        lift_sign=lift_sign,
                        c_block=float(c_value.real),
                        h_block=float(h_value.real),
                        relative=float(_relative(c_value, h_value)),
                    )
                )

        coefficient_rows = []
        for level in range(max_twice_level + 1):
            c_value = c_coefficients[level]
            h_value = h_coefficients[level]
            coefficient_rows.append(
                {
                    "twice_level": level,
                    "q_power": level / 2,
                    "c": [float(c_value.real), float(c_value.imag)],
                    "h": [float(h_value.real), float(h_value.imag)],
                    "relative_c_h": float(_relative(c_value, h_value)),
                    "relative_c_radius": float(
                        _relative(c_value, c_check[level])
                    ),
                    "relative_h_radius": float(
                        _relative(h_value, h_check[level])
                    ),
                }
            )

        return {
            "parameters": {
                "hat_c": 9.0,
                "c": float(c.real),
                "b": 1.0,
                "internal_momentum": internal_momentum,
                "internal_weight": float(h.real),
                "external_momentum": external_momentum,
                "external_weight": float(d.real),
                "max_q_power": max_q_power,
                "max_twice_level": max_twice_level,
                "q_values": list(q_values),
                "lift_signs": [1, -1],
                "finite_part_radius": radius,
                "finite_part_check_radius": check_radius,
                "finite_part_samples": samples,
                "working_precision": working_precision,
            },
            "generic_sanity": _generic_sanity(
                b_value=1.27,
                internal_momentum=internal_momentum,
                external_momentum=external_momentum,
                max_twice_level=min(max_twice_level, 12),
                working_precision=working_precision,
            ),
            "rows": [asdict(row) for row in rows],
            "coefficients": coefficient_rows,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_comparison(
        max_q_power=args.order,
        samples=args.samples,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
