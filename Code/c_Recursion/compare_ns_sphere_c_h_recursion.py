"""Numerical c-versus-h recursion check for the NS sphere four-point block.

The c-recursive evaluator is the independent implementation in
``superconformal_blocks.py``.  The reference evaluator below implements the
fixed-b NS elliptic h-recursion directly.  At the self-dual Liouville point
``b=1`` (ordinary ``c=27/2``, or ``hat c=9``), individual h-recursion Kac
terms are resonant.  Their finite value is obtained by projecting the
constant Laurent coefficient on a small circle in ``t=log(b)``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Dict, Literal, Sequence

import mpmath

from superconformal_blocks import (
    HighPrecisionNSSphereFourPointBlock,
    central_charge,
    elliptic_nome,
    ns_liouville_weight,
)


Parity = Literal["even", "odd"]


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
    q_background = b + 1 / b
    lower_lambda = mpmath.sqrt(q_background * q_background - 8 * lower_weight)
    upper_lambda = mpmath.sqrt(q_background * q_background - 8 * upper_weight)
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


@dataclass(frozen=True)
class ComparisonRow:
    z: float
    q: float
    c_even: float
    h_even: float
    relative_even: float
    c_odd: float
    h_odd: float
    relative_odd: float


class NSSphereHRecursion:
    """NS four-point elliptic block from internal-weight recursion."""

    def __init__(
        self,
        *,
        b: complex,
        h1: complex,
        h2: complex,
        h3: complex,
        h4: complex,
        internal_weight: complex,
        working_precision: int = 80,
    ) -> None:
        self.working_precision = int(working_precision)
        with mpmath.workdps(self.working_precision):
            self.b = mpmath.mpc(b)
            self.c = mpmath.mpf("1.5") + 3 * (self.b + 1 / self.b) ** 2
            self.h1 = mpmath.mpc(h1)
            self.h2 = mpmath.mpc(h2)
            self.h3 = mpmath.mpc(h3)
            self.h4 = mpmath.mpc(h4)
            self.internal_weight = mpmath.mpc(internal_weight)

    @lru_cache(maxsize=None)
    def _series(
        self,
        max_twice_power: int,
        parity: Parity,
        internal_weight: complex,
    ) -> tuple[complex, ...]:
        result = [mpmath.mpc(0)] * (max_twice_power + 1)
        if parity == "even":
            # The large-internal-weight regular term is
            # theta_3(q^2)=sum_{n in Z} q^(2 n^2), not merely its constant
            # coefficient.  In twice-q-power notation it sits at 4 n^2.
            result[0] = mpmath.mpc(1)
            n = 1
            while 4 * n * n <= max_twice_power:
                result[4 * n * n] += 2.0
                n += 1

        for r in range(1, max_twice_power + 1):
            for s in range(1, max_twice_power // r + 1):
                shift = r * s
                if shift > max_twice_power or (r + s) % 2:
                    continue
                degenerate_weight = _ns_degenerate_weight(self.b, r, s)
                denominator = internal_weight - degenerate_weight

                next_parity: Parity = parity
                if shift % 2:
                    next_parity = "odd" if parity == "even" else "even"
                tail = self._series(
                    max_twice_power - shift,
                    next_parity,
                    degenerate_weight + mpmath.mpf(shift) / 2,
                )
                starred = parity == "odd"
                left = _ns_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    lower_weight=self.h4,
                    upper_weight=self.h3,
                    starred=starred,
                )
                right = _ns_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    lower_weight=self.h1,
                    upper_weight=self.h2,
                    starred=starred,
                )
                coefficient = (
                    mpmath.power(16, mpmath.mpf(shift) / 2)
                    * _ns_a_factor(self.b, r, s)
                    * left
                    * right
                    / denominator
                )
                for power, value in enumerate(tail):
                    result[power + shift] += coefficient * value
        return tuple(result)

    def elliptic_coefficients(
        self, max_q_power: int, parity: Parity
    ) -> Dict[int, complex]:
        if max_q_power < 1:
            raise ValueError("max_q_power must be positive")
        with mpmath.workdps(self.working_precision):
            max_twice_power = (
                2 * max_q_power if parity == "even" else 2 * max_q_power - 1
            )
            values = self._series(max_twice_power, parity, self.internal_weight)
            remainder = 0 if parity == "even" else 1
            return {
                power: values[power]
                for power in range(remainder, max_twice_power + 1, 2)
            }


def _finite_part_coefficients(
    *,
    radius: float,
    samples: int,
    max_q_power: int,
    weights: tuple[complex, complex, complex, complex, complex],
    working_precision: int,
) -> Dict[Parity, Dict[int, complex]]:
    keys = {
        "even": list(range(0, 2 * max_q_power + 1, 2)),
        "odd": list(range(1, 2 * max_q_power, 2)),
    }
    with mpmath.workdps(working_precision):
        totals = {
            parity: {key: mpmath.mpc(0) for key in parity_keys}
            for parity, parity_keys in keys.items()
        }
        h1, h2, h3, h4, internal_weight = map(mpmath.mpc, weights)
        for index in range(samples):
            angle = 2 * mpmath.pi * (mpmath.mpf(index) + mpmath.mpf("0.5")) / samples
            b = mpmath.exp(
                mpmath.mpf(str(radius)) * mpmath.exp(mpmath.j * angle)
            )
            block = NSSphereHRecursion(
                b=b,
                h1=h1,
                h2=h2,
                h3=h3,
                h4=h4,
                internal_weight=internal_weight,
                working_precision=working_precision,
            )
            for parity in ("even", "odd"):
                values = block.elliptic_coefficients(max_q_power, parity)
                for key in keys[parity]:
                    totals[parity][key] += values[key]
        return {
            parity: {
                key: value / samples for key, value in parity_totals.items()
            }
            for parity, parity_totals in totals.items()
        }


def _finite_part_c_coefficients(
    *,
    radius: float,
    samples: int,
    max_q_power: int,
    weights: tuple[complex, complex, complex, complex, complex],
    working_precision: int,
) -> Dict[Parity, Dict[int, complex]]:
    keys = {
        "even": list(range(0, 2 * max_q_power + 1, 2)),
        "odd": list(range(1, 2 * max_q_power, 2)),
    }
    with mpmath.workdps(working_precision):
        totals = {
            parity: {key: mpmath.mpc(0) for key in parity_keys}
            for parity, parity_keys in keys.items()
        }
        h1, h2, h3, h4, internal_weight = map(mpmath.mpc, weights)
        for index in range(samples):
            angle = 2 * mpmath.pi * (mpmath.mpf(index) + mpmath.mpf("0.5")) / samples
            b = mpmath.exp(
                mpmath.mpf(str(radius)) * mpmath.exp(mpmath.j * angle)
            )
            c = mpmath.mpf("1.5") + 3 * (b + 1 / b) ** 2
            block = HighPrecisionNSSphereFourPointBlock(
                c=c,
                h1=h1,
                h2=h2,
                h3=h3,
                h4=h4,
                internal_weight=internal_weight,
                working_precision=working_precision,
            )
            values = {
                "even": block.elliptic_coefficients(max_q_power + 1, "even"),
                "odd": block.elliptic_coefficients(max_q_power, "odd"),
            }
            for parity in ("even", "odd"):
                for key in keys[parity]:
                    totals[parity][key] += values[parity][key]
        return {
            parity: {
                key: value / samples for key, value in parity_totals.items()
            }
            for parity, parity_totals in totals.items()
        }


def _evaluate_series(coefficients: Dict[int, complex], z: float) -> complex:
    q = elliptic_nome(z)
    return sum(value * q ** (power / 2.0) for power, value in coefficients.items())


def _elliptic_prefactor(
    *,
    z: float,
    c: float,
    external_weights: tuple[float, float, float, float],
    internal_weight: float,
):
    with mpmath.workdps(60):
        z_mp = mpmath.mpf(str(z))
        q = mpmath.mpc(elliptic_nome(z))
        q_squared = mpmath.mpf(str(c)) / 3 - mpmath.mpf("0.5")
        h1, h2, h3, h4 = map(mpmath.mpf, external_weights)
        h = mpmath.mpf(internal_weight)
        theta3 = mpmath.jtheta(3, 0, q)
        return (
            (16 * q) ** (h - q_squared / 8)
            * z_mp ** (q_squared / 8 - h1 - h2)
            * (1 - z_mp) ** (q_squared / 8 - h2 - h3)
            * theta3 ** (mpmath.mpf("1.5") * q_squared - 4 * (h1 + h2 + h3 + h4))
        )


def build_comparison(
    *,
    momenta: tuple[float, float, float, float] = (0.5, 1.0 / 3.0, 0.25, 0.6),
    internal_momentum: float = 0.7,
    z_values: Sequence[float] = tuple(0.05 + 0.1 * index for index in range(10)),
    max_q_power: int = 10,
    radius: float = 0.035,
    check_radius: float = 0.045,
    samples: int = 32,
    working_precision: int = 60,
) -> dict:
    b = 1.0
    c = central_charge(b).real
    external_weights = tuple(ns_liouville_weight(p, b).real for p in momenta)
    internal_weight = ns_liouville_weight(internal_momentum, b).real
    weights = (*external_weights, internal_weight)

    c_coefficients = _finite_part_c_coefficients(
        radius=radius,
        samples=samples,
        max_q_power=max_q_power,
        weights=weights,
        working_precision=working_precision,
    )
    c_check_coefficients = _finite_part_c_coefficients(
        radius=check_radius,
        samples=samples,
        max_q_power=max_q_power,
        weights=weights,
        working_precision=working_precision,
    )
    h_coefficients = _finite_part_coefficients(
        radius=radius,
        samples=samples,
        max_q_power=max_q_power,
        weights=weights,
        working_precision=working_precision,
    )
    h_check_coefficients = _finite_part_coefficients(
        radius=check_radius,
        samples=samples,
        max_q_power=max_q_power,
        weights=weights,
        working_precision=working_precision,
    )

    rows = []
    for z in z_values:
        prefactor = _elliptic_prefactor(
            z=z,
            c=c,
            external_weights=external_weights,
            internal_weight=internal_weight,
        )
        values = {}
        for parity in ("even", "odd"):
            c_value = prefactor * _evaluate_series(c_coefficients[parity], z)
            h_value = prefactor * _evaluate_series(h_coefficients[parity], z)
            values[parity] = (
                c_value,
                h_value,
                abs(c_value - h_value) / max(abs(c_value), abs(h_value), 1.0e-300),
            )
        rows.append(
            ComparisonRow(
                z=float(z),
                q=float(elliptic_nome(z).real),
                c_even=float(values["even"][0].real),
                h_even=float(values["even"][1].real),
                relative_even=float(values["even"][2]),
                c_odd=float(values["odd"][0].real),
                h_odd=float(values["odd"][1].real),
                relative_odd=float(values["odd"][2]),
            )
        )

    coefficient_rows = {}
    for parity in ("even", "odd"):
        coefficient_rows[parity] = []
        for power in sorted(c_coefficients[parity]):
            c_value = complex(c_coefficients[parity][power])
            h_value = h_coefficients[parity][power]
            check_value = h_check_coefficients[parity][power]
            c_check_value = c_check_coefficients[parity][power]
            coefficient_rows[parity].append(
                {
                    "q_power": power / 2.0,
                    "c": [c_value.real, c_value.imag],
                    "h": [float(h_value.real), float(h_value.imag)],
                    "relative_c_h": float(
                        abs(c_value - h_value)
                        / max(abs(c_value), abs(h_value), 1.0e-300)
                    ),
                    "relative_h_radius": float(
                        abs(h_value - check_value)
                        / max(abs(h_value), abs(check_value), 1.0e-300)
                    ),
                    "relative_c_radius": float(
                        abs(c_value - c_check_value)
                        / max(abs(c_value), abs(c_check_value), 1.0e-300)
                    ),
                }
            )

    return {
        "parameters": {
            "hat_c": 9.0,
            "c": c,
            "b": b,
            "momenta": list(momenta),
            "external_weights": list(external_weights),
            "internal_momentum": internal_momentum,
            "internal_weight": internal_weight,
            "max_q_power": max_q_power,
            "z_values": list(z_values),
            "finite_part_radius": radius,
            "finite_part_check_radius": check_radius,
            "finite_part_samples": samples,
            "working_precision": working_precision,
        },
        "rows": [asdict(row) for row in rows],
        "coefficients": coefficient_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_comparison(max_q_power=args.order)
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
