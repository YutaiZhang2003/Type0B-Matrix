#!/usr/bin/env python3
"""Zhu-recursion operators for torus one-point descendant blocks.

This module is intentionally series-based.  A torus one-point block is stored as

    q^exponent * sum_{n=0}^N a_n q^n.

Then q d/dq and multiplication by Zhu-normalized Eisenstein series are finite
coefficient operations.  The first implemented descendant is the square-bracket
Virasoro descendant L[-2] acting on an external primary.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from functools import lru_cache

try:
    from torus_descendant_blocks import apply_virasoro_mode, state_level, torus_one_point_descendant_coefficients
    from virasoro_blocks import TorusOnePointVirasoroBlock
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.torus_descendant_blocks import (
        apply_virasoro_mode,
        state_level,
        torus_one_point_descendant_coefficients,
    )
    from plumbing.virasoro_blocks import TorusOnePointVirasoroBlock


def parse_complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def format_complex(value: complex) -> str:
    return f"{value.real:+.12e}{value.imag:+.12e}j"


def divisor_power_sum(n: int, power: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    total = 0
    for divisor in range(1, int(math.sqrt(n)) + 1):
        if n % divisor == 0:
            total += divisor**power
            partner = n // divisor
            if partner != divisor:
                total += partner**power
    return total


def bernoulli_number_even(index: int) -> float:
    values = {
        2: 1.0 / 6.0,
        4: -1.0 / 30.0,
        6: 1.0 / 42.0,
        8: -1.0 / 30.0,
        10: 5.0 / 66.0,
        12: -691.0 / 2730.0,
    }
    if index not in values:
        raise ValueError(f"Bernoulli B_{index} is not tabulated")
    return values[index]


def zhu_eisenstein_coefficients(weight: int, order: int) -> list[complex]:
    r"""Return coefficients of Zhu's E_weight(q) through q^order.

    The convention is

        E_{2k}(q) = -B_{2k}/(2k)! + 2/(2k-1)! sum_{n>=1} sigma_{2k-1}(n) q^n.

    Thus E_2(q) = -1/12 + 2 q + 6 q^2 + 8 q^3 + ...
    """
    if weight < 2 or weight % 2:
        raise ValueError("weight must be an even integer >= 2")
    if order < 0:
        raise ValueError("order must be non-negative")
    coeffs = [0.0 + 0.0j for _ in range(order + 1)]
    coeffs[0] = -bernoulli_number_even(weight) / math.factorial(weight)
    power = weight - 1
    scale = 2.0 / math.factorial(weight - 1)
    for n in range(1, order + 1):
        coeffs[n] = scale * divisor_power_sum(n, power)
    return coeffs


def zhu_eisenstein_product_coefficients(weight: int, series_order: int) -> list[complex]:
    return zhu_eisenstein_coefficients(weight, series_order)


@dataclass(frozen=True)
class ZhuSeries:
    exponent: complex
    coefficients: tuple[complex, ...]

    @property
    def order(self) -> int:
        return len(self.coefficients) - 1

    def value(self, q: complex) -> complex:
        q = complex(q)
        return (q**self.exponent) * sum(coeff * (q**idx) for idx, coeff in enumerate(self.coefficients))


def truncate(coefficients: list[complex], order: int) -> list[complex]:
    if len(coefficients) < order + 1:
        coefficients = coefficients + [0.0 + 0.0j] * (order + 1 - len(coefficients))
    return coefficients[: order + 1]


def add_series(left: ZhuSeries, right: ZhuSeries) -> ZhuSeries:
    if abs(left.exponent - right.exponent) > 1.0e-14:
        raise ValueError("can only add Zhu series with the same q exponent")
    order = min(left.order, right.order)
    return ZhuSeries(
        exponent=left.exponent,
        coefficients=tuple(left.coefficients[idx] + right.coefficients[idx] for idx in range(order + 1)),
    )


def scale_series(series: ZhuSeries, scale: complex) -> ZhuSeries:
    return ZhuSeries(series.exponent, tuple(scale * coeff for coeff in series.coefficients))


def q_derivative(series: ZhuSeries) -> ZhuSeries:
    """Return q d/dq of a ZhuSeries."""
    return ZhuSeries(
        series.exponent,
        tuple((series.exponent + idx) * coeff for idx, coeff in enumerate(series.coefficients)),
    )


def multiply_by_coefficients(series: ZhuSeries, multiplier: list[complex]) -> ZhuSeries:
    order = min(series.order, len(multiplier) - 1)
    out: list[complex] = []
    for n in range(order + 1):
        out.append(sum(multiplier[k] * series.coefficients[n - k] for k in range(n + 1)))
    return ZhuSeries(series.exponent, tuple(out))


def zero_series_like(series: ZhuSeries) -> ZhuSeries:
    return ZhuSeries(series.exponent, tuple(0.0 + 0.0j for _ in range(series.order + 1)))


def primary_torus_zhu_series(
    c: complex,
    internal_weight: complex,
    external_weight: complex,
    order: int,
    *,
    b: complex | None = None,
    external_lambda: complex | None = None,
    method: str = "recursion",
) -> ZhuSeries:
    """Return the primary torus one-point block as a ZhuSeries."""
    if method == "recursion":
        block = TorusOnePointVirasoroBlock(
            c,
            internal_weight,
            external_weight,
            b=b,
            external_lambda=external_lambda,
        )
        coefficients = tuple(block.descendant_coefficients(order))
    elif method == "direct":
        coefficients = tuple(
            torus_one_point_descendant_coefficients(
                c,
                internal_weight,
                external_weight,
                (),
                order,
            )
        )
    else:
        raise ValueError("method must be 'recursion' or 'direct'")
    return ZhuSeries(
        exponent=complex(internal_weight) - complex(c) / 24.0,
        coefficients=coefficients,
    )


def lminus2_primary_zhu_series(primary_series: ZhuSeries, external_weight: complex) -> ZhuSeries:
    r"""Return the Zhu-recursion series for Z_{L[-2] nu_d}.

    For a square-bracket primary nu_d,

        Z(L[-2] nu_d) = (q d/dq) Z(nu_d) + d E_2(q) Z(nu_d),

    where E_2 is Zhu-normalized.
    """
    e2_times = multiply_by_coefficients(
        primary_series,
        zhu_eisenstein_coefficients(2, primary_series.order),
    )
    return add_series(q_derivative(primary_series), scale_series(e2_times, external_weight))


def zhu_descendant_series(
    primary_series: ZhuSeries,
    state: tuple[int, ...],
    *,
    external_weight: complex,
    central_charge: complex,
) -> ZhuSeries:
    """Return the Zhu-recursion series for a Virasoro square-bracket state.

    ``state=(n1,n2,...)`` denotes ``L[-n1] L[-n2] ... nu_d`` in the order
    written.  The implementation uses the genus-one Virasoro Zhu recursion and
    is triangular in total descendant level.
    """
    external_weight = complex(external_weight)
    central_charge = complex(central_charge)

    @lru_cache(maxsize=None)
    def recurse(current_state: tuple[int, ...]) -> ZhuSeries:
        if not current_state:
            return primary_series
        first = current_state[0]
        rest = current_state[1:]
        if first <= 0:
            raise ValueError("Zhu descendant state entries must be positive")
        if first == 1:
            return zero_series_like(primary_series)

        total = zero_series_like(primary_series)
        if first == 2:
            total = add_series(total, q_derivative(recurse(rest)))

        max_k = (first + state_level(rest)) // 2
        for k_index in range(1, max_k + 1):
            mode = 2 * k_index - first
            mode_terms = apply_virasoro_mode(mode, rest, external_weight, central_charge)
            if not mode_terms:
                continue
            e_series = zhu_eisenstein_product_coefficients(2 * k_index, primary_series.order)
            mode_series = zero_series_like(primary_series)
            for next_state, coeff in mode_terms:
                mode_series = add_series(mode_series, scale_series(recurse(next_state), coeff))
            total = add_series(total, multiply_by_coefficients(mode_series, e_series))
        return total

    return recurse(tuple(state))


def run() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Zhu L[-2] torus one-point descendant series.")
    parser.add_argument("--c", type=parse_complex, required=True)
    parser.add_argument("--internal-weight", type=parse_complex, required=True)
    parser.add_argument("--external-weight", type=parse_complex, required=True)
    parser.add_argument("--q", type=parse_complex, required=True)
    parser.add_argument("--order", type=int, default=4)
    args = parser.parse_args()

    primary = primary_torus_zhu_series(
        args.c,
        args.internal_weight,
        args.external_weight,
        args.order,
    )
    descendant = lminus2_primary_zhu_series(primary, args.external_weight)

    print("Zhu torus one-point descendant series")
    print(f"  c={format_complex(args.c)}")
    print(f"  internal h={format_complex(args.internal_weight)}")
    print(f"  external h={format_complex(args.external_weight)}")
    print("  external state=L[-2] primary")
    print(f"  q={format_complex(args.q)}")
    print(f"  exponent={format_complex(descendant.exponent)}")
    print("  coefficients:")
    for level, coeff in enumerate(descendant.coefficients):
        print(f"    q^{level}: {format_complex(coeff)}")
    print(f"  value={format_complex(descendant.value(args.q))}")


if __name__ == "__main__":
    run()
