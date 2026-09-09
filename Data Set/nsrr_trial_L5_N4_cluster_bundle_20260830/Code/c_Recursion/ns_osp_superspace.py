#!/usr/bin/env python3
"""Superspace derivation of the global NS osp(1|2) trinion.

In local supercoordinates about (infinity, 1, 0), put

    X = 1 + u (1 + v),   Y = 1 + u w,   Z = 1 + v - w.

The coefficient of theta_1^a theta_2^b theta_3^c in the fixed-parity
trilinear convention is a single shifted-power kernel.  Ordinary
derivatives of this kernel produce all global descendants.  This module
checks that compact superspace construction against the independent Ward
identity implementation in ``ns_global_osp_block.py``.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from typing import Mapping
from typing import Sequence

import sympy as sp

from ns_global_osp_block import osp_three_point
from ns_human_convention import (
    human_note_rho_sign,
    normalize_parity_triple,
    primary_parity_ward_sign,
)


u, v, w = sp.symbols("u v w")


@dataclass(frozen=True)
class _GrassmannPolynomial:
    """Sparse exterior polynomial with SymPy-valued coefficients."""

    terms: Mapping[int, sp.Expr]

    @staticmethod
    def scalar(value: sp.Expr) -> "_GrassmannPolynomial":
        value = sp.sympify(value)
        return _GrassmannPolynomial({0: value}) if value != 0 else _GrassmannPolynomial({})

    @staticmethod
    def generator(index: int) -> "_GrassmannPolynomial":
        return _GrassmannPolynomial({1 << index: sp.S.One})

    def __add__(self, other: "_GrassmannPolynomial") -> "_GrassmannPolynomial":
        result = dict(self.terms)
        for mask, coefficient in other.terms.items():
            result[mask] = sp.expand(result.get(mask, sp.S.Zero) + coefficient)
        return _GrassmannPolynomial(
            {mask: coefficient for mask, coefficient in result.items() if coefficient != 0}
        )

    def __neg__(self) -> "_GrassmannPolynomial":
        return self.scale(-1)

    def __sub__(self, other: "_GrassmannPolynomial") -> "_GrassmannPolynomial":
        return self + (-other)

    def __mul__(self, other: "_GrassmannPolynomial") -> "_GrassmannPolynomial":
        result: dict[int, sp.Expr] = {}
        for left_mask, left_coefficient in self.terms.items():
            for right_mask, right_coefficient in other.terms.items():
                if left_mask & right_mask:
                    continue
                inversions = 0
                remaining = left_mask
                while remaining:
                    lowest = remaining & -remaining
                    inversions += bin(right_mask & (lowest - 1)).count("1")
                    remaining ^= lowest
                sign = -1 if inversions % 2 else 1
                mask = left_mask | right_mask
                result[mask] = sp.expand(
                    result.get(mask, sp.S.Zero)
                    + sign * left_coefficient * right_coefficient
                )
        return _GrassmannPolynomial(
            {mask: coefficient for mask, coefficient in result.items() if coefficient != 0}
        )

    def scale(self, value: sp.Expr) -> "_GrassmannPolynomial":
        value = sp.sympify(value)
        return _GrassmannPolynomial(
            {mask: sp.expand(value * coefficient) for mask, coefficient in self.terms.items()}
        )

    def even_power(self, exponent: sp.Expr) -> "_GrassmannPolynomial":
        """Raise a three-generator even polynomial to an arbitrary power."""

        exponent = sp.sympify(exponent)
        if any(bin(mask).count("1") % 2 for mask in self.terms):
            raise ValueError("even_power requires an even Grassmann polynomial")
        body = self.terms.get(0, sp.S.Zero)
        if body == 0:
            raise ValueError("even_power requires a nonzero scalar body")
        result = _GrassmannPolynomial.scalar(body**exponent)
        for mask, coefficient in self.terms.items():
            if mask:
                result = result + _GrassmannPolynomial(
                    {mask: exponent * body ** (exponent - 1) * coefficient}
                )
        return result

    def coefficient(self, bits: Sequence[int]) -> sp.Expr:
        a, b, c = _validate_bits(bits)
        mask = a | (b << 1) | (c << 2)
        return self.terms.get(mask, sp.S.Zero)


def _validate_bits(bits: Sequence[int]) -> tuple[int, int, int]:
    if len(bits) != 3 or any(bit not in (0, 1) for bit in bits):
        raise ValueError("bits must be a length-three sequence of zeros and ones")
    return int(bits[0]), int(bits[1]), int(bits[2])


def component_prefactor(
    bits: Sequence[int],
    d1: sp.Expr,
    d2: sp.Expr,
    d3: sp.Expr,
    primary_parities: Sequence[int] = (0, 0, 0),
) -> sp.Expr:
    """Primary coefficient kappa_(abc) in the graded human convention."""

    a, b, c = _validate_bits(bits)
    primaries = normalize_parity_triple(
        primary_parities, name="primary_parities"
    )
    A = d2 + d3 - d1
    B = d1 + d2 - d3
    C = d1 - d2 + d3
    S = d1 + d2 + d3
    table = {
        (0, 0, 0): sp.S.One,
        (1, 0, 0): sp.S.One,
        (0, 1, 0): sp.S.One,
        (0, 0, 1): -sp.S.One,
        (1, 1, 0): B,
        (1, 0, 1): C,
        (0, 1, 1): -A,
        (1, 1, 1): -(S - sp.Rational(1, 2)),
    }
    # The table itself already contains the even-primary fixed-parity sign.
    parity_sign = primary_parity_ward_sign((a, b, c), primaries)
    return parity_sign * table[(a, b, c)]


def superspace_component_kernel(
    bits: Sequence[int],
    d1: sp.Expr,
    d2: sp.Expr,
    d3: sp.Expr,
    primary_parities: Sequence[int] = (0, 0, 0),
) -> sp.Expr:
    r"""Return the local coefficient function T^{abc}(u,v,w).

    The half-integer exponent shifts are the incidence combinations

        delta_X = (a+b-c)/2,
        delta_Y = (a+c-b)/2,
        delta_Z = (b+c-a)/2.
    """

    a, b, c = _validate_bits(bits)
    d1, d2, d3 = map(sp.sympify, (d1, d2, d3))
    A = d2 + d3 - d1
    B = d1 + d2 - d3
    C = d1 - d2 + d3
    delta_x = sp.Rational(a + b - c, 2)
    delta_y = sp.Rational(a + c - b, 2)
    delta_z = sp.Rational(b + c - a, 2)
    X = 1 + u * (1 + v)
    Y = 1 + u * w
    Z = 1 + v - w
    return (
        component_prefactor(
            (a, b, c), d1, d2, d3, primary_parities
        )
        * X ** (-B - delta_x)
        * Y ** (-C - delta_y)
        * Z ** (-A - delta_z)
    )


def localized_correlator_from_invariant(
    d1: sp.Expr,
    d2: sp.Expr,
    d3: sp.Expr,
    primary_parities: Sequence[int] = (0, 0, 0),
) -> _GrassmannPolynomial:
    r"""Expand the standard superspace three-point invariant locally.

    This starts from ``Z_ij`` and ``eta_123``, rather than from the compact
    component answer.  The factors ``-1/u`` in the two distances involving
    slot 1 cancel the BPZ superprimary Jacobian.  The returned formal sum has
    the even and odd primary structures both normalized to one.
    """

    primaries = normalize_parity_triple(
        primary_parities, name="primary_parities"
    )
    d1, d2, d3 = map(sp.sympify, (d1, d2, d3))
    A = d2 + d3 - d1
    B = d1 + d2 - d3
    C = d1 - d2 + d3
    theta1 = _GrassmannPolynomial.generator(0)
    theta2 = _GrassmannPolynomial.generator(1)
    theta3 = _GrassmannPolynomial.generator(2)
    X = 1 + u * (1 + v)
    Y = 1 + u * w
    Z = 1 + v - w

    # These are the normalized local superdistances after extracting the
    # common -1/u from Z_12 and Z_13.
    Z12 = _GrassmannPolynomial.scalar(X) - theta1 * theta2
    Z13 = _GrassmannPolynomial.scalar(Y) - theta1 * theta3
    Z23 = _GrassmannPolynomial.scalar(Z) - theta2 * theta3

    even_structure = (
        Z12.even_power(-B)
        * Z13.even_power(-C)
        * Z23.even_power(-A)
    )

    # After the same BPZ cancellation, the numerator of eta_123 becomes
    # Z23 theta1 - Z13 theta2 + Z12 theta3 + theta1 theta2 theta3.
    odd_numerator = (
        Z23 * theta1
        - Z13 * theta2
        + Z12 * theta3
        + theta1 * theta2 * theta3
    )
    odd_structure = (
        odd_numerator
        * Z12.even_power(-B - sp.Rational(1, 2))
        * Z13.even_power(-C - sp.Rational(1, 2))
        * Z23.even_power(-A - sp.Rational(1, 2))
    )
    superfield_components = even_structure + odd_structure

    # First convert superfield component maps to the component-ordered Ward
    # tensor by (-1)^(b(1-a)), then apply the canonical human-note
    # fixed-parity sign (-1)^(sector*c).
    converted: dict[int, sp.Expr] = {}
    for mask, coefficient in superfield_components.terms.items():
        a = mask & 1
        b = (mask >> 1) & 1
        c = (mask >> 2) & 1
        sign = (-1 if b * (1 - a) else 1) * human_note_rho_sign(
            (a, b, c), primaries
        )
        converted[mask] = sign * coefficient
    return _GrassmannPolynomial(converted)


def superspace_three_point(
    *,
    n1: int,
    n2: int,
    n3: int,
    epsilon1: int,
    epsilon2: int,
    epsilon3: int,
    d1: sp.Expr,
    d2: sp.Expr,
    d3: sp.Expr,
    primary_parities: Sequence[int] = (0, 0, 0),
) -> sp.Expr:
    r"""Extract a global trinion coefficient from the superspace kernel.

    The first local coordinate is related to the plane coordinate by
    z_1=-1/u, hence every L_-1 descendant in the infinity slot supplies
    ``-partial_u``.  The other two slots supply ``partial_v`` and
    ``partial_w``.
    """

    if min(n1, n2, n3) < 0:
        raise ValueError("descendant occupations must be non-negative")
    bits = _validate_bits((epsilon1, epsilon2, epsilon3))
    kernel = superspace_component_kernel(
        bits, d1, d2, d3, primary_parities
    )
    differentiated = sp.diff(kernel, u, n1, v, n2, w, n3)
    return sp.simplify((-1) ** n1 * differentiated.subs({u: 0, v: 0, w: 0}))


def _reference_value(
    n1: int,
    n2: int,
    n3: int,
    bits: tuple[int, int, int],
    weights: tuple[sp.Rational, sp.Rational, sp.Rational],
) -> complex:
    return osp_three_point(
        n1=n1,
        n2=n2,
        n3=n3,
        epsilon1=bits[0],
        epsilon2=bits[1],
        epsilon3=bits[2],
        d1=float(weights[0]),
        d2=float(weights[1]),
        d3=float(weights[2]),
    )


def run_checks(max_level: int = 3) -> None:
    """Compare all parities and descendants with the Ward-identity code."""

    weights_to_test = (
        (sp.Rational(7, 5), sp.Rational(11, 6), sp.Rational(13, 7)),
        (sp.Rational(9, 8), sp.Rational(5, 3), sp.Rational(17, 10)),
    )
    cases = 0
    largest_error = 0.0
    for weights in weights_to_test:
        from_invariant = localized_correlator_from_invariant(*weights)
        for bits in itertools.product((0, 1), repeat=3):
            invariant_coefficient = from_invariant.coefficient(bits)
            compact_coefficient = superspace_component_kernel(bits, *weights)
            if sp.simplify(invariant_coefficient - compact_coefficient) != 0:
                raise AssertionError(
                    "localized superspace invariant did not reduce to the "
                    f"compact kernel for weights={weights}, bits={bits}: "
                    f"{invariant_coefficient} != {compact_coefficient}"
                )
            expected_primary = component_prefactor(bits, *weights)
            actual_primary = superspace_three_point(
                n1=0,
                n2=0,
                n3=0,
                epsilon1=bits[0],
                epsilon2=bits[1],
                epsilon3=bits[2],
                d1=weights[0],
                d2=weights[1],
                d3=weights[2],
            )
            if sp.simplify(actual_primary - expected_primary) != 0:
                raise AssertionError(
                    f"primary mismatch for bits={bits}: "
                    f"{actual_primary} != {expected_primary}"
                )
            for n1, n2, n3 in itertools.product(
                range(max_level + 1), repeat=3
            ):
                actual = superspace_three_point(
                    n1=n1,
                    n2=n2,
                    n3=n3,
                    epsilon1=bits[0],
                    epsilon2=bits[1],
                    epsilon3=bits[2],
                    d1=weights[0],
                    d2=weights[1],
                    d3=weights[2],
                )
                expected = _reference_value(n1, n2, n3, bits, weights)
                error = abs(complex(sp.N(actual, 17)) - expected)
                largest_error = max(largest_error, error)
                cases += 1
                if error > 2.0e-10 * max(1.0, abs(expected)):
                    raise AssertionError(
                        "descendant mismatch: "
                        f"weights={weights}, bits={bits}, "
                        f"levels={(n1, n2, n3)}, "
                        f"superspace={actual}, Ward={expected}, error={error}"
                    )
    print(
        "PASS: direct exterior expansion of (Z_ij, eta_123) gives the compact "
        "superspace kernel, which agrees with the independent Ward-identity "
        f"trinion in {cases} descendant cases through occupation {max_level}; "
        f"largest absolute error={largest_error:.3e}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-level", type=int, default=3)
    args = parser.parse_args()
    run_checks(args.max_level)


if __name__ == "__main__":
    main()
