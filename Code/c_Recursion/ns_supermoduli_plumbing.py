#!/usr/bin/env python3
"""NS supermoduli plumbing and Berezin extraction of vertex sectors.

For a pants decomposition with only NS punctures, every internal tube has an
even plumbing parameter q (and discrete spin lift), while every three-NS
super-pants has one odd modulus nu.  The unintegrated chiral block is the
Grassmann generating function

    F(q, nu) = Sew[ product_v (rho_v^0 + nu_v rho_v^1),
                    product_e q_e^(L_0-h_e) I_e ].

This file verifies three points:

* the local odd differential operator squares to the even translation;
* Berezin integration over the vertex moduli extracts exactly the prescribed
  parity-homogeneous trinion sectors, including the ordering sign;
* a directly sewn global sphere four-point block agrees with that extraction
  for every external component and internal even/odd state tested.

The nu_v here are genuine SRS moduli.  They are distinct from the auxiliary
half-edge Grassmann variables used in ``ns_grassmann_sewing.py`` to implement
inverse-Gram contractions and Koszul signs.
"""

from __future__ import annotations

import argparse
import itertools
from typing import Sequence

import sympy as sp

from ns_global_osp_block import osp_norm, osp_three_point
from ns_grassmann_sewing import (
    ExteriorPolynomial,
    berezin_integral,
    ordered_product,
)


def superderivative(
    superfield: tuple[sp.Expr, sp.Expr], coordinate: sp.Symbol
) -> tuple[sp.Expr, sp.Expr]:
    r"""Apply D = partial_theta + theta partial_z to f(z)+theta g(z)."""

    even, odd = superfield
    return sp.sympify(odd), sp.diff(even, coordinate)


def vertex_modulus_monomial(sectors: Sequence[int]) -> ExteriorPolynomial:
    """Return nu_1^alpha_1 ... nu_V^alpha_V in canonical vertex order."""

    if any(sector not in (0, 1) for sector in sectors):
        raise ValueError("every vertex sector must be zero or one")
    return ordered_product(
        [
            ExteriorPolynomial.generator(vertex)
            for vertex, sector in enumerate(sectors)
            if sector
        ]
    )


def dual_sector_monomial(sectors: Sequence[int]) -> ExteriorPolynomial:
    r"""Return the Berezin-dual monomial to nu_1^alpha_1...nu_V^alpha_V.

    With measure d nu_V ... d nu_1, multiplication by this monomial and
    integration extracts the requested coefficient with positive sign.
    """

    alpha = tuple(int(sector) for sector in sectors)
    if any(sector not in (0, 1) for sector in alpha):
        raise ValueError("every vertex sector must be zero or one")
    complement = [index for index, sector in enumerate(alpha) if not sector]
    inversions = sum(
        alpha[left] * (1 - alpha[right])
        for left in range(len(alpha))
        for right in range(left + 1, len(alpha))
    )
    sign = -1 if inversions % 2 else 1
    return ordered_product(
        [ExteriorPolynomial.generator(index) for index in complement]
    ).scale(sign)


def extract_vertex_sector(
    generating_block: ExteriorPolynomial, sectors: Sequence[int]
) -> complex:
    """Extract one component block by integrating all genuine odd moduli."""

    vertex_count = len(sectors)
    integrand = dual_sector_monomial(sectors) * generating_block
    return berezin_integral(integrand, tuple(range(vertex_count)))


def sphere_four_point_generating_block_at_n(
    *,
    n: int,
    q: float,
    xi: int,
    internal_weight: float,
    external_weights: Sequence[float],
    external_parities: Sequence[int],
) -> ExteriorPolynomial:
    r"""Global four-point block at fixed bosonic occupation n.

    The left trinion has slots (d1,d2,h) and the right trinion has slots
    (h,d3,d4).  Its two genuine odd moduli are nu_L and nu_R.  Summing the
    internal epsilon=0,1 states produces a polynomial in those moduli.
    """

    if n < 0:
        raise ValueError("n must be non-negative")
    if len(external_weights) != 4 or len(external_parities) != 4:
        raise ValueError("four external weights and parities are required")
    if xi not in (-1, 1):
        raise ValueError("xi must be a spin-lift sign")
    if any(parity not in (0, 1) for parity in external_parities):
        raise ValueError("external parities must be zero or one")

    d1, d2, d3, d4 = map(float, external_weights)
    a, b, c, d = map(int, external_parities)
    h = float(internal_weight)
    result = ExteriorPolynomial.scalar(0)
    for epsilon in (0, 1):
        left = osp_three_point(
            n1=0,
            n2=0,
            n3=n,
            epsilon1=a,
            epsilon2=b,
            epsilon3=epsilon,
            d1=d1,
            d2=d2,
            d3=h,
        )
        right = osp_three_point(
            n1=n,
            n2=0,
            n3=0,
            epsilon1=epsilon,
            epsilon2=c,
            epsilon3=d,
            d1=h,
            d2=d3,
            d3=d4,
        )
        propagator = (
            xi**epsilon
            * q ** (n + 0.5 * epsilon)
            / osp_norm(h, n, epsilon)
        )
        sectors = ((a + b + epsilon) % 2, (epsilon + c + d) % 2)
        result = result + vertex_modulus_monomial(sectors).scale(
            propagator * left * right
        )
    return result


def sphere_four_point_component_at_n(
    *,
    sectors: Sequence[int],
    n: int,
    q: float,
    xi: int,
    internal_weight: float,
    external_weights: Sequence[float],
    external_parities: Sequence[int],
) -> complex:
    """Direct component sewing with the two trinion parity projectors."""

    d1, d2, d3, d4 = map(float, external_weights)
    a, b, c, d = map(int, external_parities)
    alpha_left, alpha_right = map(int, sectors)
    h = float(internal_weight)
    result = 0.0 + 0.0j
    for epsilon in (0, 1):
        if (a + b + epsilon) % 2 != alpha_left:
            continue
        if (epsilon + c + d) % 2 != alpha_right:
            continue
        left = osp_three_point(
            n1=0,
            n2=0,
            n3=n,
            epsilon1=a,
            epsilon2=b,
            epsilon3=epsilon,
            d1=d1,
            d2=d2,
            d3=h,
        )
        right = osp_three_point(
            n1=n,
            n2=0,
            n3=0,
            epsilon1=epsilon,
            epsilon2=c,
            epsilon3=d,
            d1=h,
            d2=d3,
            d3=d4,
        )
        result += (
            xi**epsilon
            * q ** (n + 0.5 * epsilon)
            / osp_norm(h, n, epsilon)
            * left
            * right
        )
    return result


def supermoduli_dimensions(genus: int, punctures: int) -> tuple[int, int]:
    """Return (even, odd) dimensions for a stable all-NS surface."""

    if genus < 0 or punctures < 0 or 2 * genus - 2 + punctures <= 0:
        raise ValueError("a stable all-NS surface is required")
    return 3 * genus - 3 + punctures, 2 * genus - 2 + punctures


def run_checks(max_n: int = 3) -> None:
    """Run algebra, dimension-counting, and component-sewing checks."""

    z = sp.symbols("z")
    field = (z**3 + 2 * z + 1, z**2 - 3 * z + 4)
    twice = superderivative(superderivative(field, z), z)
    translated = tuple(sp.diff(component, z) for component in field)
    if any(sp.simplify(left - right) != 0 for left, right in zip(twice, translated)):
        raise AssertionError("the odd local derivative does not square to translation")

    for genus, punctures in ((0, 3), (0, 4), (1, 1), (2, 0), (3, 2)):
        even_dimension, odd_dimension = supermoduli_dimensions(genus, punctures)
        edge_count = 3 * genus - 3 + punctures
        vertex_count = 2 * genus - 2 + punctures
        if even_dimension != edge_count or odd_dimension != vertex_count:
            raise AssertionError("pants coordinates do not match supermoduli dimension")

    parameters = dict(
        q=0.07,
        xi=-1,
        internal_weight=1.37,
        external_weights=(0.81, 1.12, 0.93, 1.26),
    )
    checked = 0
    largest_error = 0.0
    for external_parities in itertools.product((0, 1), repeat=4):
        for n in range(max_n + 1):
            generating = sphere_four_point_generating_block_at_n(
                n=n,
                external_parities=external_parities,
                **parameters,
            )
            for sectors in itertools.product((0, 1), repeat=2):
                integrated = extract_vertex_sector(generating, sectors)
                direct = sphere_four_point_component_at_n(
                    sectors=sectors,
                    n=n,
                    external_parities=external_parities,
                    **parameters,
                )
                error = abs(integrated - direct)
                largest_error = max(largest_error, error)
                checked += 1
                if error > 1.0e-12 * max(1.0, abs(direct)):
                    raise AssertionError(
                        "odd-moduli integration mismatch: "
                        f"external={external_parities}, n={n}, "
                        f"sectors={sectors}, integrated={integrated}, direct={direct}"
                    )
    print(
        "PASS: D^2=partial, pants coordinates match supermoduli dimensions, "
        f"and Berezin integration reproduces {checked} directly sewn sector "
        f"coefficients through n={max_n}; largest error={largest_error:.3e}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=3)
    args = parser.parse_args()
    run_checks(args.max_n)


if __name__ == "__main__":
    main()
