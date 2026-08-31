#!/usr/bin/env python3
"""Berezin sewing of the global NS parity sum.

At fixed bosonic occupations ``n_e``, attach one Grassmann variable to each
internal half-edge.  If edge ``e`` joins half-edges ``h`` and ``hbar``, its
kernel is

    theta_h theta_hbar + s_e xi_e q_e**(1/2) / (2 h_e + n_e),

where ``s_e`` is derived from the two half-edge frame signs and the lifted
plumbing transition.  Multiplication by the
vertex superpolynomials and ordered Berezin integration simultaneously

* forces the two endpoint fermion occupations to agree;
* supplies the odd inverse-Gram factor;
* produces the graph Koszul orientation sign.

This file implements a minimal exterior algebra and checks those statements
for every theta-graph parity assignment.  It is deliberately independent of
the component-level finite-c sewing code.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from ns_global_osp_block import osp_norm
from ns_regular_block import (
    PlumbingFrameLedger,
    PlumbingOrientation,
    THETA_ORIENTATION,
)


@dataclass(frozen=True)
class ExteriorPolynomial:
    """Sparse polynomial in ordered Grassmann generators."""

    terms: Mapping[int, complex]

    @staticmethod
    def scalar(value: complex) -> "ExteriorPolynomial":
        return ExteriorPolynomial({0: complex(value)})

    @staticmethod
    def generator(index: int) -> "ExteriorPolynomial":
        if index < 0:
            raise ValueError("Grassmann-generator index must be non-negative")
        return ExteriorPolynomial({1 << index: 1.0 + 0.0j})

    def __add__(self, other: "ExteriorPolynomial") -> "ExteriorPolynomial":
        result = dict(self.terms)
        for mask, value in other.terms.items():
            result[mask] = result.get(mask, 0.0 + 0.0j) + value
        return ExteriorPolynomial(
            {mask: value for mask, value in result.items() if value != 0}
        )

    def __mul__(self, other: "ExteriorPolynomial") -> "ExteriorPolynomial":
        result: dict[int, complex] = {}
        for left_mask, left_value in self.terms.items():
            for right_mask, right_value in other.terms.items():
                if left_mask & right_mask:
                    continue
                inversions = 0
                remaining = left_mask
                while remaining:
                    lowest = remaining & -remaining
                    left_index = lowest.bit_length() - 1
                    inversions += bin(right_mask & (lowest - 1)).count("1")
                    remaining ^= lowest
                sign = -1 if inversions % 2 else 1
                mask = left_mask | right_mask
                result[mask] = result.get(mask, 0.0 + 0.0j) + (
                    sign * left_value * right_value
                )
        return ExteriorPolynomial(
            {mask: value for mask, value in result.items() if value != 0}
        )

    def scale(self, value: complex) -> "ExteriorPolynomial":
        return ExteriorPolynomial(
            {mask: complex(value) * coefficient for mask, coefficient in self.terms.items()}
        )


def ordered_product(factors: Sequence[ExteriorPolynomial]) -> ExteriorPolynomial:
    result = ExteriorPolynomial.scalar(1)
    for factor in factors:
        result = result * factor
    return result


def berezin_integral(
    polynomial: ExteriorPolynomial, integration_order: Sequence[int]
) -> complex:
    r"""Integrate with measure ``d theta_cn ... d theta_c1``.

    ``integration_order=(c1,...,cn)`` means that the monomial
    ``theta_c1 ... theta_cn`` integrates to one.
    """

    order = tuple(int(index) for index in integration_order)
    if len(set(order)) != len(order):
        raise ValueError("integration_order contains a repeated generator")
    full_mask = sum(1 << index for index in order)
    coefficient = complex(polynomial.terms.get(full_mask, 0.0 + 0.0j))
    inversions = sum(
        order[left] > order[right]
        for left in range(len(order))
        for right in range(left + 1, len(order))
    )
    return (-coefficient) if inversions % 2 else coefficient


def edge_kernel(
    left: int,
    right: int,
    *,
    odd_weight: complex,
    linear_bit: int = 0,
) -> ExteriorPolynomial:
    odd_sign = -1 if int(linear_bit) % 2 else 1
    return (
        ExteriorPolynomial.generator(left) * ExteriorPolynomial.generator(right)
    ) + ExteriorPolynomial.scalar(odd_sign * complex(odd_weight))


def vertex_parity_monomial(
    orientation: PlumbingOrientation, edge_parities: Sequence[int]
) -> ExteriorPolynomial:
    """Return the vertex-slot-ordered monomial for an internal parity state."""

    if len(edge_parities) != orientation.edge_count:
        raise ValueError("edge_parities has the wrong length")
    half_edge_bits = [0] * len(orientation.contraction_order)
    for parity, (left, right) in zip(
        edge_parities, orientation.edge_half_edges
    ):
        half_edge_bits[left] = int(parity) % 2
        half_edge_bits[right] = int(parity) % 2
    factors = [
        ExteriorPolynomial.generator(index)
        for index, bit in enumerate(half_edge_bits)
        if bit
    ]
    return ordered_product(factors)


def sew_fixed_parity(
    *,
    orientation: PlumbingOrientation,
    edge_parities: Sequence[int],
    odd_weights: Sequence[complex],
    external_parities: Sequence[int] = (),
) -> complex:
    """Sew one fixed edge-parity assignment by a Berezin integral."""

    if len(odd_weights) != orientation.edge_count:
        raise ValueError("odd_weights has the wrong length")
    if len(external_parities) != orientation.external_count:
        raise ValueError("external_parities has the wrong length")
    zeros = (0,) * orientation.edge_count
    linear_bits = orientation.edge_linear_bits or (0,) * orientation.edge_count
    effective_bits = tuple(
        (
            linear_bit
            + orientation.polarized_exponent(
                tuple(int(index == edge) for index in range(orientation.edge_count)),
                zeros,
                external_parities,
            )
        )
        % 2
        for edge, linear_bit in enumerate(linear_bits)
    )
    kernels = [
        edge_kernel(
            left,
            right,
            odd_weight=weight,
            linear_bit=linear_bit,
        )
        for (left, right), weight, linear_bit in zip(
            orientation.edge_half_edges, odd_weights, effective_bits
        )
    ]
    integrand = ordered_product(
        kernels + [vertex_parity_monomial(orientation, edge_parities)]
    )
    internal_positions = {
        position for pair in orientation.edge_half_edges for position in pair
    }
    internal_contraction_order = tuple(
        position
        for position in orientation.contraction_order
        if position in internal_positions
    )
    external_sign = orientation.sign(zeros, external_parities)
    return external_sign * berezin_integral(
        integrand, internal_contraction_order
    )


def _self_check() -> None:
    weights = (2.0, 3.0, 5.0)
    for e0 in (0, 1):
        for e1 in (0, 1):
            for einf in (0, 1):
                parities = (e0, e1, einf)
                expected = THETA_ORIENTATION.sign(parities)
                for parity, weight in zip(parities, weights):
                    if parity:
                        expected *= weight
                actual = sew_fixed_parity(
                    orientation=THETA_ORIENTATION,
                    edge_parities=parities,
                    odd_weights=weights,
                )
                if actual != expected:
                    raise AssertionError(
                        f"theta Berezin sign mismatch at {parities}: "
                        f"{actual} != {expected}"
                    )

    h = 1.7
    q = 0.09
    xi = -1
    for n in range(5):
        even_prefactor = q**n / complex(osp_norm(h, n, 0))
        odd_weight = xi * q**0.5 / (2.0 * h + n)
        odd_from_kernel = even_prefactor * odd_weight
        odd_direct = xi * q ** (n + 0.5) / complex(osp_norm(h, n, 1))
        if abs(odd_from_kernel - odd_direct) > 1.0e-14:
            raise AssertionError("odd inverse-Gram edge weight changed")

    punctured = PlumbingFrameLedger(
        edge_half_edges=((0, 2),),
        external_half_edges=(1,),
        contraction_order=(0, 1, 2),
        half_edge_frame_signs=(1, -1, 1),
        edge_transition_signs=(-1,),
    ).orientation()
    for edge_parity in (0, 1):
        for external_parity in (0, 1):
            expected = punctured.sign((edge_parity,), (external_parity,))
            if edge_parity:
                expected *= 7
            actual = sew_fixed_parity(
                orientation=punctured,
                edge_parities=(edge_parity,),
                odd_weights=(7,),
                external_parities=(external_parity,),
            )
            if actual != expected:
                raise AssertionError("punctured Berezin orientation changed")


if __name__ == "__main__":
    _self_check()
    print("NS Grassmann/Berezin sewing checks: PASS")
