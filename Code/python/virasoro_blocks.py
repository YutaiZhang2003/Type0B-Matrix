#!/usr/bin/env python3
"""Virasoro torus one-point conformal blocks.

This module implements the Zamolodchikov/Poghossian recursion for the
non-trivial elliptic part of the genus-one one-point Virasoro block.  The CFT
data, such as three-point coefficients, are intentionally not included here:
this file only supplies the universal chiral block multiplying them.
"""

from __future__ import annotations

import argparse
import cmath
from dataclasses import dataclass
from functools import lru_cache


Number = complex


def parse_complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def _as_complex(value: complex | float | int) -> complex:
    return complex(value)


def central_charge_to_background_charge(c: complex | float) -> complex:
    """Return Q with c = 1 + 6 Q^2, using the principal square-root."""
    return cmath.sqrt((_as_complex(c) - 1.0) / 6.0)


def central_charge_to_b(c: complex | float, branch: int = 1) -> complex:
    """Return one solution of b + 1/b = Q, where c = 1 + 6 Q^2."""
    if branch not in {-1, 1}:
        raise ValueError("branch must be +1 or -1")
    q_background = central_charge_to_background_charge(c)
    return 0.5 * (q_background + branch * cmath.sqrt(q_background * q_background - 4.0))


def background_charge(b: complex) -> complex:
    return b + 1.0 / b


def weight_from_momentum(lam: complex, b: complex) -> complex:
    """Return Delta_lambda = (Q^2 - lambda^2)/4."""
    q_background = background_charge(b)
    return 0.25 * (q_background * q_background - lam * lam)


def momentum_from_weight(weight: complex | float, b: complex, sign: int = 1) -> complex:
    """Return lambda from Delta = (Q^2 - lambda^2)/4."""
    if sign not in {-1, 1}:
        raise ValueError("sign must be +1 or -1")
    q_background = background_charge(b)
    return sign * cmath.sqrt(q_background * q_background - 4.0 * _as_complex(weight))


def degenerate_momentum(r: int, s: int, b: complex) -> complex:
    return r * b + s / b


def degenerate_weight(r: int, s: int, b: complex) -> complex:
    return weight_from_momentum(degenerate_momentum(r, s, b), b)


def shifted_degenerate_weight(r: int, s: int, b: complex) -> complex:
    """Return Delta_{r,s}+rs = Delta_{r,-s}."""
    return degenerate_weight(r, -s, b)


def zamolodchikov_a_rs(r: int, s: int, b: complex) -> complex:
    """Universal A_rs factor for the Delta-recursion."""
    product = 1.0 + 0.0j
    for p in range(1 - r, r + 1):
        for ell in range(1 - s, s + 1):
            if (p, ell) in {(0, 0), (r, s)}:
                continue
            product *= 1.0 / (p * b + ell / b)
    return 0.5 * product


def fusion_polynomial(
    r: int,
    s: int,
    b: complex,
    lambda_top: complex,
    lambda_bottom: complex,
) -> complex:
    """Fusion polynomial P_c^{rs}[Delta_top / Delta_bottom].

    We use the momentum convention Delta_lambda = (Q^2 - lambda^2)/4.
    """
    product = 1.0 + 0.0j
    for p in range(1 - r, r, 2):
        for ell in range(1 - s, s, 2):
            shift = p * b + ell / b
            product *= 0.5 * (lambda_top + lambda_bottom + shift)
            product *= 0.5 * (lambda_top - lambda_bottom + shift)
    return product


def torus_one_point_residue(r: int, s: int, b: complex, external_lambda: complex) -> complex:
    """Residue multiplying the lower-level torus one-point block."""
    lambda_rs = degenerate_momentum(r, s, b)
    lambda_r_minus_s = degenerate_momentum(r, -s, b)
    return (
        zamolodchikov_a_rs(r, s, b)
        * fusion_polynomial(r, s, b, external_lambda, lambda_r_minus_s)
        * fusion_polynomial(r, s, b, external_lambda, lambda_rs)
    )


def partition_numbers(order: int) -> list[int]:
    """Coefficients of prod_{n>=1} (1-q^n)^(-1) through q^order."""
    if order < 0:
        raise ValueError("order must be non-negative")
    values = [0] * (order + 1)
    values[0] = 1
    for part in range(1, order + 1):
        for n in range(part, order + 1):
            values[n] += values[n - part]
    return values


def dedekind_eta(q: complex, *, tolerance: float = 1.0e-15, max_terms: int = 100000) -> complex:
    """Return eta(q)=q^(1/24) prod_{n>=1}(1-q^n) for |q|<1."""
    q = _as_complex(q)
    if not 0 < abs(q) < 1:
        raise ValueError("Dedekind eta expects 0 < |q| < 1")
    product = 1.0 + 0.0j
    q_power = q
    for _ in range(max_terms):
        product *= 1.0 - q_power
        if abs(q_power) < tolerance:
            return (q ** (1.0 / 24.0)) * product
        q_power *= q
    raise RuntimeError("Dedekind eta product did not converge")


@dataclass(frozen=True)
class TorusOnePointBlockParameters:
    c: complex
    internal_weight: complex
    external_weight: complex
    b: complex
    external_lambda: complex


class TorusOnePointVirasoroBlock:
    """Numerical torus one-point Virasoro block for a fixed channel."""

    def __init__(
        self,
        c: complex | float,
        internal_weight: complex | float,
        external_weight: complex | float,
        *,
        b: complex | float | None = None,
        external_lambda: complex | float | None = None,
        pole_tolerance: float = 1.0e-13,
    ) -> None:
        if b is None:
            b_value = central_charge_to_b(c)
        else:
            b_value = _as_complex(b)
        if external_lambda is None:
            external_lambda_value = momentum_from_weight(external_weight, b_value)
        else:
            external_lambda_value = _as_complex(external_lambda)

        self.params = TorusOnePointBlockParameters(
            c=_as_complex(c),
            internal_weight=_as_complex(internal_weight),
            external_weight=_as_complex(external_weight),
            b=b_value,
            external_lambda=external_lambda_value,
        )
        self.pole_tolerance = pole_tolerance

    @lru_cache(maxsize=None)
    def _h_coefficient(self, n: int, internal_weight: complex) -> complex:
        if n < 0:
            return 0.0j
        if n == 0:
            return 1.0 + 0.0j

        total = 0.0j
        b = self.params.b
        for r in range(1, n + 1):
            for s in range(1, n // r + 1):
                level = r * s
                pole = degenerate_weight(r, s, b)
                denominator = internal_weight - pole
                if abs(denominator) < self.pole_tolerance:
                    raise ZeroDivisionError(
                        "internal weight is too close to the degenerate pole "
                        f"Delta_({r},{s})={pole!r}"
                    )
                residue = torus_one_point_residue(r, s, b, self.params.external_lambda)
                shifted_weight = shifted_degenerate_weight(r, s, b)
                total += residue * self._h_coefficient(n - level, shifted_weight) / denominator
        return total

    def elliptic_coefficients(self, order: int) -> list[complex]:
        """Return H_n for H(q)=sum_n q^n H_n through q^order."""
        if order < 0:
            raise ValueError("order must be non-negative")
        return [self._h_coefficient(n, self.params.internal_weight) for n in range(order + 1)]

    def elliptic_block(self, q: complex, order: int) -> complex:
        """Return the non-trivial elliptic block H(q) truncated at q^order."""
        q = _as_complex(q)
        return sum(coeff * (q**n) for n, coeff in enumerate(self.elliptic_coefficients(order)))

    def descendant_coefficients(self, order: int) -> list[complex]:
        """Return coefficients of eta(q)^(-1) q^(1/24) H(q).

        Equivalently, these are coefficients of
        prod_{n>=1}(1-q^n)^(-1) H(q) through q^order.
        """
        h_coeffs = self.elliptic_coefficients(order)
        partitions = partition_numbers(order)
        out: list[complex] = []
        for n in range(order + 1):
            out.append(sum(partitions[n - k] * h_coeffs[k] for k in range(n + 1)))
        return out

    def chiral_block(self, q: complex, order: int, *, include_prefactor: bool = True) -> complex:
        """Return F(q) = q^(h-c/24) prod(1-q^n)^(-1) H(q), truncated."""
        q = _as_complex(q)
        series = sum(coeff * (q**n) for n, coeff in enumerate(self.descendant_coefficients(order)))
        if include_prefactor:
            series *= q ** (self.params.internal_weight - self.params.c / 24.0)
        return series

    def chiral_block_exact_eta(
        self,
        q: complex,
        order: int,
        *,
        eta_tolerance: float = 1.0e-15,
    ) -> complex:
        """Return q^(h-(c-1)/24) eta(q)^(-1) H(q), truncating only H(q)."""
        q = _as_complex(q)
        known_exponent = self.params.internal_weight - (self.params.c - 1.0) / 24.0
        return (
            q**known_exponent
            * self.elliptic_block(q, order)
            / dedekind_eta(q, tolerance=eta_tolerance)
        )


def torus_one_point_elliptic_block(
    c: complex | float,
    internal_weight: complex | float,
    external_weight: complex | float,
    q: complex | float,
    order: int,
    *,
    b: complex | float | None = None,
    external_lambda: complex | float | None = None,
) -> complex:
    return TorusOnePointVirasoroBlock(
        c,
        internal_weight,
        external_weight,
        b=b,
        external_lambda=external_lambda,
    ).elliptic_block(_as_complex(q), order)


def torus_one_point_chiral_block(
    c: complex | float,
    internal_weight: complex | float,
    external_weight: complex | float,
    q: complex | float,
    order: int,
    *,
    b: complex | float | None = None,
    external_lambda: complex | float | None = None,
    include_prefactor: bool = True,
) -> complex:
    return TorusOnePointVirasoroBlock(
        c,
        internal_weight,
        external_weight,
        b=b,
        external_lambda=external_lambda,
    ).chiral_block(_as_complex(q), order, include_prefactor=include_prefactor)


def torus_one_point_chiral_block_exact_eta(
    c: complex | float,
    internal_weight: complex | float,
    external_weight: complex | float,
    q: complex | float,
    order: int,
    *,
    b: complex | float | None = None,
    external_lambda: complex | float | None = None,
    eta_tolerance: float = 1.0e-15,
) -> complex:
    return TorusOnePointVirasoroBlock(
        c,
        internal_weight,
        external_weight,
        b=b,
        external_lambda=external_lambda,
    ).chiral_block_exact_eta(_as_complex(q), order, eta_tolerance=eta_tolerance)


def format_complex(z: complex) -> str:
    return f"{z.real:+.12e}{z.imag:+.12e}j"


def run() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a torus one-point Virasoro block.")
    parser.add_argument("--c", type=parse_complex, required=True)
    parser.add_argument("--internal-weight", type=parse_complex, required=True)
    parser.add_argument("--external-weight", type=parse_complex, required=True)
    parser.add_argument("--q", type=parse_complex, required=True)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--b", type=parse_complex)
    parser.add_argument("--external-lambda", type=parse_complex)
    args = parser.parse_args()

    block = TorusOnePointVirasoroBlock(
        args.c,
        args.internal_weight,
        args.external_weight,
        b=args.b,
        external_lambda=args.external_lambda,
    )
    print("torus one-point Virasoro block")
    print(f"  c={format_complex(block.params.c)}")
    print(f"  b={format_complex(block.params.b)}")
    print(f"  internal h={format_complex(block.params.internal_weight)}")
    print(f"  external h={format_complex(block.params.external_weight)}")
    print(f"  external lambda={format_complex(block.params.external_lambda)}")
    print(f"  q={format_complex(args.q)}")
    print(f"  order={args.order}")
    print("  elliptic H coefficients:")
    for n, coeff in enumerate(block.elliptic_coefficients(args.order)):
        print(f"    H[{n}]={format_complex(coeff)}")
    print(f"  H(q)={format_complex(block.elliptic_block(args.q, args.order))}")
    print(f"  F(q)={format_complex(block.chiral_block(args.q, args.order))}")


if __name__ == "__main__":
    run()
